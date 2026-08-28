"""A reconstrução de `memories` — a parte capaz de quebrar o banco do usuário.

O defeito que este arquivo existe para pegar é da classe mais cruel deste repo: **verde
na suíte, quebrado no banco real**. `init_schema()` roda em toda invocação de CLI e todo
teste cria banco novo em `tmp_path`, onde o `CREATE TABLE` já traz o vocabulário novo —
então, sem estes testes, um `source='recon'` que o banco de produção RECUSA fica
invisível para 813 testes verdes.

Cada trava aqui vem com a mutação que a mata, e as três do rebuild foram EXECUTADAS.
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

from lithium.db import Store
from lithium.db.store import (
    MEMORY_KINDS,
    MEMORY_SOURCES,
    SCHEMA_PATH,
    memories_ddl,
)

DIM = 8

LEGACY_MEMORIES = """
CREATE TABLE memories (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('preference','context','constraint','fact',
       'dead_end','search_lesson','source_lesson','pattern')),
  rationale TEXT,
  source TEXT NOT NULL DEFAULT 'chat'
         CHECK (source IN ('chat','answer','manual','research')),
  provenance TEXT,
  confirmed INTEGER NOT NULL DEFAULT 0,
  active INTEGER NOT NULL DEFAULT 1,
  embedding BLOB,
  -- SEM DEFAULT, como no schema pré-Fase-A. É o que exige o COALESCE do rebuild.
  created_at TEXT NOT NULL,
  confirmed_at TEXT)
"""

LEGACY_ROWS = [
    (1, "prefere manhã", "preference", "chat"),
    (2, "resposta antiga", "fact", "answer"),
    (3, "anotado à mão", "context", "manual"),
    (4, "queries com novel voltam vazias", "search_lesson", "research"),
    (5, "fonte X não tem texto completo", "source_lesson", "research"),
    (6, "beco sem saída", "dead_end", "research"),
]


def _legacy_db(path, *, extra_rows=()) -> None:
    """Um banco PRÉ-Fase-C: `memories` com o CHECK antigo, sem `focus_id`.

    Recriar a tabela na forma antiga, e não `ALTER TABLE ... DROP COLUMN`: o DROP
    reescreve o DDL guardado e engasga com os comentários do CREATE TABLE. Recriar é o
    estado real de um banco criado antes desta coluna existir.
    """
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS memories")
    for view in ("recon_memories", "user_memories", "research_lessons",
                 "declined_memories", "live_memories"):
        conn.execute(f"DROP VIEW IF EXISTS {view}")
    conn.executescript(LEGACY_MEMORIES)
    conn.executescript("""
CREATE VIEW live_memories AS
  SELECT id,text,kind,source,provenance,created_at FROM memories
   WHERE confirmed=1 AND active=1;
CREATE VIEW user_memories AS
  SELECT id,text,kind,created_at FROM live_memories
   WHERE source IN ('chat','answer','manual');
CREATE VIEW research_lessons AS
  SELECT id,text,kind,provenance,created_at FROM live_memories
   WHERE source='research';
CREATE VIEW declined_memories AS
  SELECT id,text,kind,created_at FROM memories WHERE confirmed=0 AND active=0;
""")
    for i, text, kind, source in [*LEGACY_ROWS, *extra_rows]:
        conn.execute(
            "INSERT INTO memories(id,text,kind,source,confirmed,active,created_at) "
            "VALUES(?,?,?,?,1,1,?)", (i, text, kind, source, f"2024-01-01T0{i%10}:00:00Z"))
    conn.commit()
    conn.close()


def _fresh_then_legacy(tmp_path, *, extra_rows=()):
    """Banco criado pelo schema ATUAL e depois rebaixado à forma legada.

    Sem isto, o teste criaria um banco já migrado e passaria sem exercitar nada — a
    diferença entre testar a migração e testar que ela já rodou.
    """
    path = tmp_path / "legacy.db"
    s = Store(path, embedding_dim=DIM)
    s.init_schema()
    s.close()
    _legacy_db(path, extra_rows=extra_rows)
    return path


def _rows(conn) -> list[tuple]:
    return [tuple(r) for r in conn.execute(
        "SELECT id,text,kind,rationale,source,provenance,confirmed,active,created_at,"
        "       confirmed_at FROM memories ORDER BY id")]


# ══════════════════════════ o defeito central: o CHECK não migra sozinho


def test_the_recon_source_reaches_a_database_created_before_phase_c(tmp_path):
    """MUTAÇÃO: remover `_rebuild_memories_for_recon` de `init_schema`.

    MEDIDO: o DDL guardado num banco existente continua
    `source IN ('chat','answer','manual','research')`, o INSERT abaixo levanta
    `sqlite3.IntegrityError: CHECK constraint failed`, e num banco NOVO o MESMO INSERT
    passa — ou seja os 813 ficam verdes enquanto o banco real do usuário rejeita CADA
    aprovação de observação, com o usuário tendo dito "sim" explicitamente.
    """
    path = _fresh_then_legacy(tmp_path)
    before = _rows(sqlite3.connect(path))

    store = Store(path, embedding_dim=DIM)
    store.init_schema()

    assert store.recon_memory_available() is True
    store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active, focus_id) "
        "VALUES('lida na web', 'fact', 'recon', 1, 1, 1)")

    # (b) as quatro origens legadas seguem presentes, byte a byte.
    after = [r for r in _rows(store.conn) if r[0] <= len(LEGACY_ROWS)]
    assert after == before, "a reconstrução mexeu nas linhas"
    assert {r[4] for r in after} == {"chat", "answer", "manual", "research"}

    # (c) os dois índices de volta: o executescript recria um, `_migrate_indexes` o outro.
    names = {r["name"] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_memories%'")}
    assert names == {"idx_memories_live", "idx_memories_research_dedup"}, names

    # (d) as cinco views respondem.
    for view in ("live_memories", "user_memories", "research_lessons",
                 "declined_memories", "recon_memories"):
        store.conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM recon_memories").fetchone()["n"] == 1
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM user_memories").fetchone()["n"] == 3
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM research_lessons").fetchone()["n"] == 3
    store.close()


def test_the_rebuild_is_idempotent_and_the_pragmas_come_back(tmp_path):
    """MUTAÇÃO: restaurar `PRAGMA foreign_keys = ON` ANTES do ROLLBACK.

    MEDIDO: `PRAGMA foreign_keys=ON` dentro de uma transação aberta é NO-OP silencioso —
    `PRAGMA foreign_keys` continua devolvendo 0 depois, e o INSERT com FK inexistente é
    ACEITO. `Store` usa conexões thread-local de vida longa e `init_schema()` roda em
    toda invocação de CLI: seria a checagem de FK desligada pelo resto da vida do
    processo.
    """
    path = _fresh_then_legacy(tmp_path)
    store = Store(path, embedding_dim=DIM)
    store.init_schema()
    assert store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store.conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    n = store.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    store.close()

    again = Store(path, embedding_dim=DIM)
    again.init_schema()                                  # 2ª rodada: no-op
    assert again.conn.execute(
        "SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == n
    assert again.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    again.close()


def test_the_rebuild_survives_a_legacy_database_with_a_dangling_view(tmp_path, caplog):
    """MUTAÇÃO: remover `PRAGMA legacy_alter_table = ON`. Mata este teste.

    `ALTER TABLE ... RENAME` revalida o schema INTEIRO desde 3.25. Num banco legado o
    passo 0 de `init_schema` já dropou `hypotheses` e o executescript ainda não a
    recriou, então sem o pragma o RENAME levanta
    `error in view hypothesis_scoreboard: no such table: main.hypotheses` — um erro que
    não tem NADA a ver com memórias, no meio de uma migração de memórias.

    A matriz completa (dropar as views de memória antes × pragma) está medida no
    docstring de `_rebuild_memories_for_recon`. O resultado que mudou o código: o
    `DROP VIEW` que o desenho pedia matava **0 de 960** com o pragma ligado, e saiu.
    """
    path = _fresh_then_legacy(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS hypotheses")     # a view fica pendurada
    conn.commit()
    conn.close()

    store = Store(path, embedding_dim=DIM)
    with caplog.at_level(logging.ERROR):
        store.init_schema()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "\n".join(r.getMessage() for r in caplog.records)
    )
    assert store.recon_memory_available() is True
    store.close()


def test_the_rebuild_needs_the_coalesce_on_created_at(tmp_path):
    """MUTAÇÃO 3 das três EXECUTADAS: trocar `COALESCE(created_at, strftime(...))` por
    `created_at` → `NOT NULL constraint failed: memories_new.created_at`.

    O schema legado não tem default nessa coluna, e uma linha com `created_at` NULL é o
    que sobrou de um INSERT manual pelo `sqlite3`.
    """
    path = _fresh_then_legacy(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA writable_schema = ON")
    conn.execute(
        "UPDATE sqlite_master SET sql = replace(sql, "
        "  'created_at TEXT NOT NULL,', 'created_at TEXT,') "
        " WHERE type = 'table' AND name = 'memories'")
    conn.execute("PRAGMA writable_schema = OFF")
    conn.commit()
    conn.close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE memories SET created_at = NULL WHERE id = 2")
    conn.commit()
    conn.close()

    store = Store(path, embedding_dim=DIM)
    store.init_schema()
    assert store.recon_memory_available() is True
    assert store.conn.execute(
        "SELECT created_at FROM memories WHERE id = 2").fetchone()["created_at"]
    store.close()


def test_a_migrated_database_has_the_same_columns_as_a_fresh_one(tmp_path):
    """MUTAÇÃO: embutir uma segunda cópia do DDL de `memories` em store.py.

    Nada amarraria as duas cópias, e o dano só aparece na fase SEGUINTE: a Fase D
    acrescenta uma coluna ao schema, o rebuild cria `memories_new` a partir da cópia
    DESATUALIZADA, e a coluna some para sempre naquele banco. Invisível, porque todo
    teste cria banco novo em `tmp_path` — onde o rebuild nem roda.

    O conserto é derivar o DDL do próprio `schema.sql`; este teste é o que o segura.
    """
    fresh = Store(tmp_path / "fresh.db", embedding_dim=DIM)
    fresh.init_schema()
    expected = [r["name"] for r in fresh.conn.execute("PRAGMA table_info(memories)")]
    fresh.close()

    path = _fresh_then_legacy(tmp_path)
    migrated = Store(path, embedding_dim=DIM)
    migrated.init_schema()
    got = [r["name"] for r in migrated.conn.execute("PRAGMA table_info(memories)")]
    migrated.close()

    assert got == expected, f"colunas divergiram: {set(expected) ^ set(got)}"


def test_the_ddl_extractor_reads_the_real_schema():
    """Auto-verificação do extrator: o que ele devolve é DDL executável e renomeado."""
    ddl = memories_ddl(SCHEMA_PATH.read_text(encoding="utf-8"), table="memories_new")
    assert ddl.startswith("CREATE TABLE memories_new (")
    assert "'recon'" in ddl and "focus_id" in ddl
    conn = sqlite3.connect(":memory:")
    conn.execute(ddl)         # não pode levantar


def test_the_migration_vocabulary_matches_the_schema():
    """As duas cópias do vocabulário — a de `store.py` (pré-voo) e a de `schema.sql`
    (CHECK) — precisam concordar. Se o pré-voo aceitar o que o INSERT recusa, ele deixa
    de ser pré-voo."""
    ddl = memories_ddl(SCHEMA_PATH.read_text(encoding="utf-8"), table="t")
    for value in MEMORY_KINDS | MEMORY_SOURCES:
        assert f"'{value}'" in ddl, f"{value!r} está no pré-voo e não no schema"


# ═══════════════════ o pré-voo: adiar em vez de brickar o CLI


def test_the_rebuild_refuses_instead_of_bricking_the_cli(tmp_path, caplog):
    """MUTAÇÃO: remover o pré-voo.

    EXECUTADO: `IntegrityError` no `INSERT ... SELECT`, `init_schema()` morre — e ele
    roda em TODA invocação de CLI, então o usuário perde `lithium memories`, que é a
    única ferramenta que resolveria o problema. Mesmo deadlock que `_migrate_indexes`
    foi escrito para evitar, entrando por outra porta.
    """
    path = _fresh_then_legacy(tmp_path)
    conn = sqlite3.connect(path)
    # `writable_schema` para burlar o CHECK legado e criar exatamente a linha que o
    # pré-voo existe para detectar: um `kind` fora do vocabulário atual.
    conn.execute("PRAGMA writable_schema = ON")
    conn.execute("UPDATE sqlite_master SET sql = replace(sql, "
                 "  \"'pattern'))\", \"'pattern','antigo'))\") "
                 " WHERE type='table' AND name='memories'")
    conn.execute("PRAGMA writable_schema = OFF")
    conn.commit()
    conn.close()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO memories(id,text,kind,source,confirmed,active,created_at) "
                 "VALUES(99,'fora do vocabulario','antigo','chat',1,1,'2024-01-01Z')")
    conn.commit()
    conn.close()

    store = Store(path, embedding_dim=DIM)
    with caplog.at_level(logging.WARNING):
        store.init_schema()                          # NÃO pode levantar

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "#99" in blob, f"o aviso não nomeou o id: {blob}"
    assert store.recon_memory_available() is False
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == len(LEGACY_ROWS) + 1, (
        "nada pode ser apagado"
    )

    # `lithium memories` continua listando — e as VIEWS também respondem, que é a
    # metade que o teste ingênuo não mede: `ChatEngine.memories()` lê a TABELA BASE, e
    # ficaria verde sobre um banco em que o chat, o `ask` e a reflexão estão mortos.
    from lithium.chat import ChatEngine

    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from conftest import prod_profile

    engine = ChatEngine(store, None, None, profile=prod_profile())
    assert len(engine.memories()) == len(LEGACY_ROWS) + 1
    for view in ("live_memories", "user_memories", "research_lessons",
                 "declined_memories", "recon_memories"):
        store.conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
    assert engine._memory_block()
    assert engine._recon_notes()
    store.close()


def test_the_focus_id_column_arrives_even_when_the_rebuild_is_deferred(tmp_path):
    """A razão de `("memories", "focus_id", "INTEGER")` estar em `_ADDED_COLUMNS`.

    MUTAÇÃO: tirá-la de lá e deixar só o rebuild trazer a coluna. `CREATE VIEW` sobre
    coluna inexistente SUCEDE e só o SELECT levanta, então um banco com o rebuild adiado
    fica com `live_memories` e as três derivadas TODAS quebradas: `lithium chat`,
    `lithium ask` e o `reflect_tick` morrem com `no such column: focus_id` — uma
    mensagem sem nenhuma relação com o WARNING do pré-voo.
    """
    from lithium.db.store import Store as S

    assert ("memories", "focus_id", "INTEGER") in S._ADDED_COLUMNS

    path = _fresh_then_legacy(tmp_path)
    store = Store(path, embedding_dim=DIM)
    store._migrate_columns(store.conn)
    assert "focus_id" in {
        r["name"] for r in store.conn.execute("PRAGMA table_info(memories)")
    }, "o ALTER não trouxe a coluna e as views vão quebrar"
    store.close()


# ═════════════════════════════════ a fronteira entre as views de leitura


def test_recon_memories_stay_out_of_user_memories(tmp_path):
    """Gêmeo exato de `test_research_lessons_stay_out_of_user_memories`.

    MUTAÇÃO: incluir `'recon'` no `WHERE source IN (…)` de `user_memories`. MEDIDO na
    direção análoga: alargar `user_memories` para incluir `'research'` mata exatamente
    UM teste de 813 — a única trava que existe sobre a fronteira das views, e ela nomeia
    'research' explicitamente. Um `'recon'` acrescentado hoje mataria ZERO.

    O custo de não ter esta trava: o texto vindo da web aberta seria injetado sob o
    cabeçalho literal "## What you know about this user", apareceria na fila humana sob
    "o que você já me contou", e entraria no bloco `$existing` de `detect_memory.md` sob
    "Already remembered — do not propose again" — suprimindo propostas legítimas SUAS.
    Três afirmações falsas de proveniência de uma vez.
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from conftest import prod_profile

    from lithium.chat import ChatEngine

    store = Store(tmp_path / "views.db", embedding_dim=DIM)
    store.init_schema()
    store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active, focus_id, "
        "                     provenance) "
        "VALUES('a pagina do NICE diz X', 'fact', 'recon', 1, 1, 1, "
        "       '{\"url\": \"https://www.nice.org.uk/g\"}')")

    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM user_memories").fetchone()["n"] == 0
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM recon_memories").fetchone()["n"] == 1

    engine = ChatEngine(store, None, None, profile=prod_profile())
    block = engine._memory_block()
    assert "NICE diz X" not in block, (
        "prosa da web entrou em 'What you know about this user'"
    )
    notes = engine._recon_notes()
    assert "NICE diz X" in notes
    assert "nice.org.uk" in notes, "a nota precisa carregar a proveniência"
    store.close()


def test_a_failed_rebuild_leaves_foreign_key_enforcement_ON(tmp_path, caplog,
                                                            monkeypatch):
    """MUTAÇÃO: restaurar `PRAGMA foreign_keys = ON` ANTES do ROLLBACK.

    Sem esta trava a mutação matava **0 de 959** — `test_the_rebuild_is_idempotent…` só
    olha o caminho de SUCESSO, e o pragma só fica preso no de FALHA.

    EXECUTADO em sqlite: `PRAGMA foreign_keys=OFF` → `BEGIN IMMEDIATE` → exceção →
    `PRAGMA foreign_keys=ON` (ainda DENTRO da transação) → `PRAGMA foreign_keys` devolve
    **0**, e continua 0 depois do ROLLBACK. Em seguida um INSERT com FK inexistente é
    ACEITO. Aplicado ao lithium: qualquer falha do rebuild não coberta pelo pré-voo
    (disco cheio, `busy_timeout` estourado com o daemon escrevendo em paralelo) deixaria
    aquele processo rodando sem enforcement de `chunks.source_id`,
    `claim_directness.claim_id`, `focuses.scale_id` e da FK nova de
    `discoveries.focus_id` — silenciosamente, até o processo morrer. `Store` usa
    conexões thread-local de VIDA LONGA.
    """
    path = _fresh_then_legacy(tmp_path)
    store = Store(path, embedding_dim=DIM)

    import lithium.db.store as store_mod

    def boom(*a, **k):
        raise RuntimeError("disco cheio no meio do rebuild")

    monkeypatch.setattr(store_mod, "memories_ddl", boom)
    with caplog.at_level(logging.ERROR):
        # O MÉTODO, direto, e não `init_schema()`. MEDIDO: `init_schema` continua e o
        # `executescript` do passo 2 roda o `PRAGMA foreign_keys = ON` que abre o
        # schema.sql, mascarando o vazamento — o teste ficaria verde com o pragma preso
        # dentro da transação. A trava tem de morar na unidade que MEXE no pragma.
        assert store._rebuild_memories_for_recon(store.conn) is False

    assert store.recon_memory_available() is False
    assert store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1, (
        "a checagem de FK ficou DESLIGADA para o resto da vida desta conexão"
    )
    assert store.conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    # E a FK realmente volta a ser aplicada.
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO discoveries(focus_id, kind, query, url, title, summary) "
            "VALUES(999, 'observation', 'q', 'u', 't', 's')")
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == len(LEGACY_ROWS)
    store.close()
