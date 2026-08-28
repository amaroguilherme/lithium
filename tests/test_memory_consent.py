"""Consentimento e deduplicação de memória — os dois regimes, e o deadlock de migração.

A tabela `memories` carrega duas coisas com regras opostas. O que você diz exige
confirmação; o que o sistema aprende pesquisando é gravado sozinho. A separação é o
`source`, e quase todo teste aqui existe para garantir que ela não vaze:

* uma **lição de pesquisa** repetida não pode virar linha nova (o corpus de lições
  encheria de paráfrases do mesmo achado);
* uma **preferência sua** repetida **tem** que virar linha nova — deduplicar aqui seria
  o sistema engolir uma declaração sua em silêncio;
* uma memória **recusada** não pode ser reproposta, e não pode ser recusada para sempre.

O resto é sobre a migração do índice único, que é a parte capaz de brickar o sistema:
`init_schema()` roda em toda invocação de CLI, então um `CREATE UNIQUE INDEX` que
levanta derruba `lithium memories` — a ferramenta para resolver o que fez ele levantar.
"""

from __future__ import annotations

import sqlite3
import zlib

import numpy as np
import pytest

from lithium.chat import ChatEngine
from lithium.db import Store
from lithium.db.store import (
    MEMORY_DEDUP_INDEX,
    MEMORY_DEDUP_INDEX_SQL,
    normalize_memory_text,
)
from lithium.llm.schemas import MemoryProposal
from lithium.pipeline.reflect import Reflector

from conftest import prod_profile

# Perfil de PRODUÇÃO: o assunto deste arquivo é o vocabulário de segurança real
# (lítio, valproato, benzodiazepínico). Rodá-lo contra o perfil de teste — que declara
# `safety = false` — o deixaria verde afirmando sobre um ruleset que não existe.
PROFILE = prod_profile()


DIM = 8


class FakeEmbedder:
    """`zlib.crc32`, não `hash()`: o hash de str é salgado por processo e o repo não fixa
    `PYTHONHASHSEED`, então os vetores mudariam a cada execução."""

    dim = DIM

    async def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(zlib.crc32(t.encode()) % (2**32))
            v = rng.normal(size=DIM).astype(np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return out


class NullLLM:
    async def structured(self, *a, **k):
        raise AssertionError("nenhum teste aqui deve chamar o LLM")

    async def complete(self, *a, **k):
        raise AssertionError("nenhum teste aqui deve chamar o LLM")


class ProposingLLM:
    """Devolve sempre a mesma proposta.

    Necessário para exercitar `_detect_memory` de verdade. Afirmar sobre
    `declined_keys()` testaria o helper e deixaria passar a remoção do filtro — o
    detector é onde a supressão precisa acontecer.
    """

    def __init__(self, proposal: MemoryProposal) -> None:
        self.proposal = proposal

    async def structured(self, *a, **k):
        return self.proposal


class ApprovingPatternGate:
    """LLM mínimo que aprova todo `pattern` e CONTA as chamadas.

    Necessário porque `remember()` de um `pattern` agora passa pelo portão de derivação.
    Contar é o ponto: um teste que só aprova não distingue "o portão rodou e aprovou" de
    "o portão não existe mais".
    """

    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, messages, schema, **kw):
        from lithium.llm.schemas import PatternVerdict

        assert schema is PatternVerdict, f"schema inesperado: {schema}"
        self.calls += 1
        return PatternVerdict(
            follows_from_cited_claims_alone=True,
            population_scope_exceeded=False,
            contradicts_a_cited_claim_direction=False,
            restates_a_premise=False,
            reason="segue das claims citadas",
        )

    async def complete(self, *a, **k):
        raise AssertionError("não deve chamar complete")


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "m.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _engine(store):
    return ChatEngine(store, NullLLM(), FakeEmbedder(), profile=PROFILE)


def _reflector(store):
    return Reflector(store, NullLLM(), FakeEmbedder())


def _proposal(text: str, kind: str = "constraint") -> MemoryProposal:
    return MemoryProposal(
        worth_remembering=True, text=text, kind=kind, rationale="teste"
    )


def _legacy_db(path):
    """Reconstrói o banco PRÉ-migração: sem o índice e sem `text_key`.

    Sem isto, um teste de migração cria o banco já indexado e passa sem exercitar nada —
    é a diferença entre testar a migração e testar que ela já rodou.
    """
    conn = sqlite3.connect(path)
    conn.execute(f"DROP INDEX IF EXISTS {MEMORY_DEDUP_INDEX}")
    # Recria a tabela na forma antiga em vez de `ALTER TABLE ... DROP COLUMN`: o DROP
    # reescreve o DDL guardado e engasga com os comentários dentro do CREATE TABLE
    # ("incomplete input"). Recriar é mais fiel de todo jeito — é o estado real de um
    # banco criado antes desta coluna existir.
    conn.execute("ALTER TABLE memories RENAME TO memories_old")
    conn.execute(
        "CREATE TABLE memories ("
        " id INTEGER PRIMARY KEY, text TEXT NOT NULL, kind TEXT NOT NULL,"
        " rationale TEXT, source TEXT NOT NULL DEFAULT 'chat', provenance TEXT,"
        " confirmed INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,"
        " embedding BLOB,"
        " created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
        " confirmed_at TEXT)"
    )
    conn.execute(
        "INSERT INTO memories(id, text, kind, rationale, source, provenance, confirmed,"
        "                     active, embedding, created_at, confirmed_at) "
        "SELECT id, text, kind, rationale, source, provenance, confirmed, active,"
        "       embedding, created_at, confirmed_at FROM memories_old"
    )
    conn.execute("DROP TABLE memories_old")
    conn.commit()
    conn.close()


def _indexed(store) -> bool:
    return store.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (MEMORY_DEDUP_INDEX,),
    ).fetchone() is not None


def _active_lessons(store) -> int:
    return store.conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE source = 'research' AND active = 1"
    ).fetchone()["n"]


# ──────────────────────────────────────────────────── recusa: nem inerte, nem eterna


async def test_declined_memory_is_not_proposed_again(store):
    """O bug original: recusar não tinha efeito nenhum.

    `decline()` grava `confirmed = 0`, e `_detect_memory` lia `user_memories`, que
    filtra `confirmed = 1`. A recusa nunca chegava a lugar nenhum — o sistema
    reperguntava a mesma coisa a cada turno.
    """
    proposal = _proposal("O usuário evita fármacos com monitoramento sérico")
    engine = ChatEngine(store, ProposingLLM(proposal), FakeEmbedder(), profile=PROFILE)

    assert await engine._detect_memory("qualquer mensagem") is proposal   # antes
    engine.decline(proposal)
    assert await engine._detect_memory("qualquer mensagem") is None       # depois


async def test_decline_survives_case_and_accent_variation(store):
    """A rechamada quase nunca é byte-idêntica: o LLM reformula a caixa.

    E a comparação precisa ser em Python — `SELECT lower('LIÇÃO')` no SQLite devolve
    `'liÇÃo'`, então uma variante acentuada escaparia da supressão.
    """
    variant = _proposal("o usuário EVITA lítio por  monitoramento sérico")
    engine = ChatEngine(store, ProposingLLM(variant), FakeEmbedder(), profile=PROFILE)
    engine.decline(_proposal("O usuário evita Lítio por monitoramento sérico"))

    assert await engine._detect_memory("qualquer mensagem") is None


async def test_allowing_a_refusal_makes_the_proposal_come_back(store):
    """O ciclo completo: recusar suprime, `allow()` devolve. Sem este teste, `allow()`
    poderia apagar a linha sem que a proposta voltasse a passar pelo detector."""
    proposal = _proposal("O usuário evita monitoramento sérico")
    engine = ChatEngine(store, ProposingLLM(proposal), FakeEmbedder(), profile=PROFILE)

    memory_id = engine.decline(proposal)
    assert await engine._detect_memory("m") is None
    engine.allow(memory_id)
    assert await engine._detect_memory("m") is proposal


async def test_a_refusal_can_be_undone(store):
    """Recusar hoje não pode significar nunca mais poder dizer sim.

    Supressão permanente é esquecimento sem aviso — o modo de falha que o plano proíbe
    explicitamente para memórias do usuário.
    """
    engine = _engine(store)
    memory_id = engine.decline(_proposal("O usuário evita monitoramento sérico"))
    assert engine.declined_keys()

    assert engine.allow(memory_id) is True
    assert engine.declined_keys() == set()
    assert engine.allow(memory_id) is False       # idempotente


async def test_allow_refuses_to_touch_a_live_memory(store):
    """`allow()` apaga a linha — então tem que ser incapaz de apagar uma memória viva."""
    engine = _engine(store)
    live = await engine.remember(_proposal("O usuário prefere manhã"), source="chat")

    assert engine.allow(live) is False
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM user_memories"
    ).fetchone()["n"] == 1


async def test_a_declined_memory_never_enters_what_the_system_knows_about_you(store):
    """A razão de não alargar a view: o bloco `$memories` vai para o prompt de chat.

    Se a recusa entrasse ali para "não repropor", o texto rejeitado passaria a ser
    apresentado ao modelo como um fato sobre você — recusar viraria afirmar.
    """
    engine = _engine(store)
    engine.decline(_proposal("O usuário se recusa a tomar lítio"))

    assert "recusa a tomar lítio" not in engine._memory_block()
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM user_memories"
    ).fetchone()["n"] == 0


# ─────────────────────────────────────────── os dois regimes não podem se confundir


async def test_a_repeated_research_lesson_does_not_duplicate(store):
    """`ON CONFLICT DO NOTHING` era no-op: não havia UNIQUE nenhum sobre o texto."""
    r = _reflector(store)
    first = await r.remember("queries com 'novel' voltam vazias", "search_lesson", "t1")
    second = await r.remember("queries com 'novel' voltam vazias", "search_lesson", "t2")

    assert first is not None
    assert second is None
    assert _active_lessons(store) == 1


async def test_lesson_dedup_ignores_case_accent_and_spacing(store):
    r = _reflector(store)
    assert await r.remember("Ácido Valpróico é teratogênico", "search_lesson", "t") is not None
    assert await r.remember("ÁCIDO  VALPRÓICO É TERATOGÊNICO", "search_lesson", "t") is None
    assert _active_lessons(store) == 1


async def test_lesson_dedup_does_not_collapse_distinct_lessons(store):
    """O teste negativo. `NFKC` faria `10²` virar `102` e `Li₂CO₃` virar `Li2CO3`,
    descartando lições diferentes em silêncio. Por isso a chave usa `NFC`."""
    r = _reflector(store)
    assert await r.remember("meia-vida de 10² horas", "search_lesson", "t") is not None
    assert await r.remember("meia-vida de 102 horas", "search_lesson", "t") is not None
    assert await r.remember("Li₂CO₃ em dose baixa", "search_lesson", "t") is not None
    assert await r.remember("Li2CO3 em dose baixa", "search_lesson", "t") is not None
    assert _active_lessons(store) == 4


async def test_a_forgotten_lesson_can_be_learned_again(store):
    """Por que o índice tem `WHERE active = 1`.

    Sem esse termo a linha esquecida continua ocupando a chave e a lição fica
    permanentemente inaprendível — `--forget` viraria uma proibição.
    """
    r = _reflector(store)
    lesson = await r.remember("fonte X não tem texto completo", "source_lesson", "t")
    _engine(store).forget(lesson.id)

    again = await r.remember("fonte X não tem texto completo", "source_lesson", "t")
    assert again is not None and again.id != lesson.id


async def test_your_preference_is_never_deduplicated(store):
    """A assimetria deliberada, e o motivo dela.

    O índice é parcial em `source='research'`. Repetir uma preferência **tem** que
    gravar de novo: engolir uma declaração sua porque "já sabemos" é o sistema
    suprimindo o que você disse, que é a falha proibida pela invariante de que memória
    nunca filtra você.
    """
    engine = _engine(store)
    first = await engine.remember(_proposal("Prefiro evitar sedação diurna"), source="chat")
    second = await engine.remember(_proposal("Prefiro evitar sedação diurna"), source="chat")

    assert first != second
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM user_memories"
    ).fetchone()["n"] == 2


async def test_a_chat_memory_does_not_block_a_research_lesson_of_the_same_text(store):
    """O que o **predicado** do índice protege: os dois regimes coexistindo.

    (Não a coluna-chave: como o índice é parcial em `source='research'`, toda linha
    indexada já tem o mesmo `source`. Ver `MEMORY_DEDUP_INDEX_SQL`.)
    """
    text = "benzodiazepínico de uso contínuo é problema"
    await _engine(store).remember(_proposal(text), source="chat")
    lesson = await _reflector(store).remember(text, "search_lesson", "t")

    assert lesson is not None


async def test_writers_fill_the_dedup_key(store):
    """Nada aqui pode depender de um `init_schema()` posterior para preencher a chave.

    Se os escritores deixassem `text_key` NULL, o índice ficaria inerte até a próxima
    invocação de CLI — e `UNIQUE` trata NULLs como distintos, então a janela é
    exatamente quando o daemon está gravando.
    """
    engine = _engine(store)
    live = await engine.remember(_proposal("café à tarde atrapalha o sono"), source="chat")
    refused = engine.decline(_proposal("mora sozinho"))
    lesson = await _reflector(store).remember("PubMed limita a 10k", "search_lesson", "t")

    for memory_id in (live, refused, lesson.id):
        key = store.conn.execute(
            "SELECT text_key FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()["text_key"]
        assert key, f"memória #{memory_id} ficou sem text_key"


async def test_rederivation_widens_provenance_instead_of_losing_it(store):
    """Deduplicar não pode subdeclarar o suporte de um `pattern`.

    A primeira derivação congelaria `claim_ids`, e uma rederivação apoiada em mais
    claims verificadas sumiria — perdendo justamente o que o portão de `pattern` existe
    para exigir.
    """
    import json

    store.conn.execute(
        "INSERT INTO sources(kind, external_id, title, raw_json) "
        "VALUES('pubmed', '1', 't', '{}')"
    )
    for i in (1, 2):
        store.conn.execute(
            "INSERT INTO claims(id, source_id, chunk_ids, statement, direction, grade, "
            "  scale_id, confidence, verified) "
            "VALUES(?, 1, '[]', ?, 'positive', 'rct', 1, 0.8, 1)",
            (i, f"claim {i}"),
        )
        store.conn.execute(
            "INSERT INTO claim_directness(claim_id, focus_id, directness) "
            "VALUES(?, 1, 'direct')", (i,),
        )

    r = Reflector(store, ApprovingPatternGate(), FakeEmbedder())
    # `refs` são índices LOCAIS do bloco de atividade, não ids de claim: quem autoriza
    # é o conjunto mostrado, e o banco só confirma que o que foi mostrado segue
    # verificado. Sem `shown`, citar é erro de programação e levanta.
    from lithium.pipeline.reflect import Ref
    shown = {7: Ref("claim", 1, 'the claim "claim 1"'),
             8: Ref("claim", 2, 'the claim "claim 2"')}
    first = await r.remember("tônus glutamatérgico agudo falha", "pattern", "p1", [7],
                             shown=shown)
    assert first is not None
    assert await r.remember("tônus glutamatérgico agudo falha", "pattern", "p2", [7, 8],
                            shown=shown) is None

    prov = json.loads(store.conn.execute(
        "SELECT provenance FROM memories WHERE id = ?", (first.id,)
    ).fetchone()["provenance"])
    assert prov["claim_ids"] == [1, 2]


# ────────────────────────────────────────────── a migração que pode brickar o sistema


def test_schema_sql_does_not_create_a_unique_index_on_memories():
    """Regressão: mover o índice para dentro do `schema.sql` reabre o deadlock.

    Casa por regex sobre o texto sem comentários, não por substring: `CREATE UNIQUE\\n
    INDEX` é DDL válido e passaria por uma busca de `'unique index'`, e um comentário
    citando o termo daria falso positivo.
    """
    import re

    from lithium.db.store import SCHEMA_PATH

    body = re.sub(r"--[^\n]*", "", SCHEMA_PATH.read_text(encoding="utf-8"))
    offender = re.search(r"create\s+unique\s+index[\s\S]*?\bon\s+memories\b", body, re.I)
    assert offender is None, (
        "o índice único de memories tem que ser criado por `Store._migrate_indexes`, "
        "fora do executescript — dentro dele, um IntegrityError num banco com "
        "duplicatas aborta o script inteiro e derruba `lithium memories`"
    )
    # auto-verificação: o matcher pega a forma quebrada em duas linhas
    assert re.search(r"create\s+unique\s+index[\s\S]*?\bon\s+memories\b",
                     "CREATE UNIQUE\nINDEX IF NOT EXISTS x ON memories(a)", re.I)


def test_init_schema_survives_preexisting_duplicates(tmp_path):
    """O deadlock central, construído: banco antigo com duplicatas.

    `init_schema()` roda em toda invocação de CLI. Se ele levantar aqui, o usuário
    perde `lithium memories` — que é o único jeito de ver e resolver as duplicatas.
    """
    path = tmp_path / "legacy.db"
    s = Store(path, embedding_dim=DIM)
    s.init_schema()
    s.close()
    _legacy_db(path)

    conn = sqlite3.connect(path)
    for i in (1, 2, 3):
        conn.execute(
            "INSERT INTO memories(id, text, kind, source, confirmed, active) "
            "VALUES(?, 'a mesma lição', 'search_lesson', 'research', 1, 1)", (i,)
        )
    conn.commit()
    conn.close()

    s = Store(path, embedding_dim=DIM)
    s.init_schema()                                    # não pode levantar
    assert not _indexed(s), "o índice não pode subir com duplicatas vivas"
    assert s.conn.execute(
        "SELECT COUNT(*) AS n FROM memories"
    ).fetchone()["n"] == 3, "nada pode ser apagado"
    s.close()


def test_the_index_comes_up_after_the_duplicates_are_resolved(tmp_path):
    """O estado degradado tem que ser reversível — e a saída, praticável.

    Sem `--forget-duplicates`, 50 lições repetidas viram 50 comandos manuais sobre uma
    listagem ordenada por id, onde as duplicatas nem ficam adjacentes.
    """
    path = tmp_path / "dup.db"
    s = Store(path, embedding_dim=DIM)
    s.init_schema()
    s.close()
    _legacy_db(path)

    conn = sqlite3.connect(path)
    for i in (1, 2, 3):
        conn.execute(
            "INSERT INTO memories(id, text, kind, source, confirmed, active) "
            "VALUES(?, 'repetida', 'search_lesson', 'research', 1, 1)", (i,)
        )
    conn.commit()
    conn.close()

    s = Store(path, embedding_dim=DIM)
    s.init_schema()
    assert not _indexed(s)

    removed = s.forget_duplicate_lessons()
    assert removed == [2, 3]                      # mantém a de menor id
    assert _indexed(s), "o índice tem que subir sozinho depois da limpeza"
    assert _active_lessons(s) == 1
    s.close()


def test_backfill_tolerates_a_null_key_arriving_after_the_index(store):
    """Outra porta para o mesmo deadlock.

    Um binário antigo, ou um INSERT manual pelo `sqlite3`, deixa uma linha com
    `text_key` NULL depois de o índice existir. Um backfill em lote levantaria no meio
    e abortaria `init_schema()`.
    """
    assert _indexed(store)
    store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active, text_key) "
        "VALUES('lição viva', 'search_lesson', 'research', 1, 1, ?)",
        (normalize_memory_text("lição viva"),),
    )
    store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active) "
        "VALUES('lição viva', 'search_lesson', 'research', 1, 1)"      # text_key NULL
    )

    store.init_schema()                                # não pode levantar

    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE text_key IS NULL"
    ).fetchone()["n"] == 1, "a linha em conflito fica sem chave, mas intacta"


def test_an_index_with_a_stale_definition_is_rebuilt(tmp_path):
    """Decidir "já existe" só pelo nome deixa a definição errada sobreviver calada.

    Mesmo idioma de `_migrate_vector_table`, que já compara `sqlite_master.sql`.
    """
    path = tmp_path / "stale.db"
    s = Store(path, embedding_dim=DIM)
    s.init_schema()
    s.conn.execute(f"DROP INDEX {MEMORY_DEDUP_INDEX}")
    s.conn.execute(f"CREATE INDEX {MEMORY_DEDUP_INDEX} ON memories(text_key)")  # não único
    s.close()

    s = Store(path, embedding_dim=DIM)
    s.init_schema()
    sql = s.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (MEMORY_DEDUP_INDEX,)
    ).fetchone()["sql"]
    assert "UNIQUE" in sql.upper()
    s.close()


def test_the_canonical_index_definition_is_what_gets_created(store):
    """`MEMORY_DEDUP_INDEX_SQL` é a fonte da verdade para a comparação de definição —
    se ela divergir do que o SQLite guardou, todo `init_schema()` recria o índice."""
    from lithium.db.store import _sql_equal

    stored = store.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (MEMORY_DEDUP_INDEX,)
    ).fetchone()["sql"]
    assert _sql_equal(stored, MEMORY_DEDUP_INDEX_SQL)
