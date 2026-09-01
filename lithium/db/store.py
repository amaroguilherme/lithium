"""Camada de persistência — SQLite é a fonte de verdade e nunca sai da estação.

Duas decisões que carregam peso:

1. **Conexões thread-local.** SQLite em WAL permite N leitores concorrentes com um
   escritor. O worker pool é asyncio e chama isto via `asyncio.to_thread`, então cada
   thread ganha sua própria conexão em vez de compartilhar uma sob lock — leitura
   concorrente de verdade.

2. **`sqlite-vec` é opcional.** Se a build de Python do SO não permitir carregar
   extensões, caímos para cosseno por força bruta em numpy. Abaixo de ~100k chunks a
   diferença não é perceptível, e isso remove um risco de portabilidade Mac→Windows
   do caminho crítico.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from lithium.types import DIRECTNESS_WEIGHT, GRADE_WEIGHT

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
SCHEMA_VERSION = "2"

SCALE_SLUG = "clinical-evidence"
SCALE_DESCRIPTION = (
    "Calibração original de lithium: grade × directness, com directness definido "
    "contra bipolar I + TAG comórbido."
)

FOCUS_SLUG = "bipolar-tag"
FOCUS_TARGET = "bipolar I + comorbid GAD"
"""O alvo em prosa do foco SEMEADO. Na Fase B ele deixou de aparecer em
`extract_claims.md` — o prompt recebe `$target` do BANCO, via `focus["target"]`.

A duplicação que sobra é com `focuses/bipolar-tag/focus.toml`, e ela é deliberada:
`init_schema()` semeia o foco #1 em TODA abertura de banco e NÃO pode depender de
disco, senão um focus.toml com erro de sintaxe derrubaria `lithium status` e
`lithium focus`, que são os comandos de diagnóstico e o único caminho de saída.
`test_the_seeded_focus_target_comes_from_the_production_profile` trava as duas."""

LEGACY_HYPOTHESIS_INDEX = "idx_hypotheses_focus_statement"
LEGACY_HYPOTHESIS_INDEX_SQL = (
    f"CREATE UNIQUE INDEX {LEGACY_HYPOTHESIS_INDEX} ON hypotheses(focus_id, statement)"
)
"""ÚNICO, não comum. MEDIDO: com um índice comum, `INSERT ... ON CONFLICT(focus_id,
statement) DO NOTHING` de explore.py levanta `OperationalError: ON CONFLICT clause does
not match any PRIMARY KEY or UNIQUE constraint` em TODO insert de hipótese."""

MEMORY_DEDUP_INDEX = "idx_memories_research_dedup"

MEMORY_DEDUP_INDEX_SQL = (
    f"CREATE UNIQUE INDEX {MEMORY_DEDUP_INDEX} ON memories(text_key) "
    "WHERE source = 'research' AND active = 1"
)
"""A definição canônica. `_migrate_indexes` compara contra `sqlite_master.sql` e recria
quando diverge — mesmo idioma de `_migrate_vector_table`, e pela mesma razão: decidir
"já existe" pelo nome deixa um índice com a definição ERRADA sobreviver para sempre.

`text_key` sozinho na chave, não `(source, text_key)`. O índice é parcial em
`source = 'research'`, então toda linha indexada já tem o mesmo `source` — quem separa
os dois regimes de consentimento é o **predicado**, não a coluna-chave. Pôr `source` na
chave não faz mal, mas escrever que ele "protege a fronteira de consentimento" seria
falso, e um comentário falso em migração de banco é pior que nenhum."""


MEMORY_KINDS: frozenset[str] = frozenset({
    "preference", "context", "constraint", "fact",
    "dead_end", "search_lesson", "source_lesson", "pattern",
})
MEMORY_SOURCES: frozenset[str] = frozenset({
    "chat", "answer", "manual", "research", "recon",
})
"""O vocabulário de `memories`, repetido AQUI de propósito.

É o pré-voo de `_rebuild_memories_for_recon`: uma linha fora do vocabulário faz o
`INSERT ... SELECT` do rebuild levantar, e `init_schema()` roda em TODA invocação de
CLI — o usuário perderia `lithium memories`, que é a única ferramenta para resolver o
problema. Mesmo deadlock que `_migrate_indexes` foi escrito para evitar.

Duas cópias, uma aqui e uma em `schema.sql`, e elas são independentes de propósito:
`test_the_migration_vocabulary_matches_the_schema` compara as duas. Derivar esta do
arquivo faria o pré-voo aceitar exatamente o que o INSERT vai recusar.
"""

_MEMORIES_DDL_RX = re.compile(
    r"^CREATE TABLE IF NOT EXISTS memories \(.*?^\);", re.M | re.S
)


def memories_ddl(schema_sql: str, *, table: str) -> str:
    """O `CREATE TABLE` de `memories`, extraído do PRÓPRIO schema.sql, renomeado.

    Uma segunda cópia do DDL embutida em store.py seria a classe de defeito que este
    arquivo inteiro existe para evitar, entrando por outra porta: a Fase D acrescenta
    uma coluna a `memories` no schema, o rebuild cria a tabela nova a partir da cópia
    DESATUALIZADA, e a coluna some para sempre naquele banco. E nenhum teste veria —
    todo teste cria banco novo em `tmp_path`, onde o rebuild nem roda.

    Definição canônica em UM lugar, mesmo idioma de `MEMORY_DEDUP_INDEX_SQL`.
    """
    match = _MEMORIES_DDL_RX.search(schema_sql)
    if match is None:  # pragma: no cover - o schema é dado do repo
        raise RuntimeError("schema.sql não declara mais `CREATE TABLE ... memories (`")
    return match.group(0).replace(
        "CREATE TABLE IF NOT EXISTS memories (", f"CREATE TABLE {table} (", 1
    )


def _sql_equal(a: str | None, b: str) -> bool:
    """Compara DDL ignorando espaço e caixa — o SQLite guarda o texto como foi escrito."""
    norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().casefold()  # noqa: E731
    return norm(a) == norm(b)


def normalize_memory_text(text: str) -> str:
    """Chave de deduplicação de lição: NFC + casefold + colapso de espaço.

    Em Python, não em SQL. `SELECT lower('LIÇÃO')` no SQLite devolve `'liÇÃo'` — tanto
    `lower()` quanto `COLLATE NOCASE` são ASCII-only, e as lições deste sistema são em
    português. Uma coluna GENERATED herdaria a limitação exatamente onde ela dói.

    **NFC, não NFKC.** NFC reconcilia NFC vs NFD (texto colado no macOS chega
    decomposto: `sérico` tem 6 ou 7 code points conforme a origem). NFKC iria além e
    faria *compatibility folding* — `10²` viraria `102`, `Li₂CO₃` viraria `Li2CO3` —
    colapsando lições **semanticamente distintas** em silêncio. O ganho seria zero e o
    dano é perda de conteúdo sem log.

    **Limitação aceita, e ela é real:** `casefold()` colapsa `Bdnf` (símbolo de gene de
    camundongo) com `BDNF` (humano), que são coisas diferentes. Mantido mesmo assim
    porque a variação de caixa entre rederivações do mesmo achado por um 12B é o caso
    comum, e o símbolo gênico isolado como única diferença entre duas lições é o raro.
    Se algum dia doer, o conserto é chave sensível a caixa — não NFKC.
    """
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())

# `rank` é a ordem de força, só para exibição. Os pesos em si vivem em types.py —
# ver lá a justificativa da calibração.
GRADE_WEIGHTS: tuple[tuple[str, float, int], ...] = tuple(
    (grade.value, weight, i) for i, (grade, weight) in enumerate(GRADE_WEIGHT.items(), 1)
)

DIRECTNESS_WEIGHTS: tuple[tuple[str, float, int], ...] = tuple(
    (d.value, weight, i) for i, (d, weight) in enumerate(DIRECTNESS_WEIGHT.items(), 1)
)


class Store:
    """Acesso ao banco. Instancie uma vez por processo e compartilhe."""

    def __init__(self, db_path: Path, embedding_dim: int = 1024) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._local = threading.local()
        self._vec_available: bool | None = None

    # ────────────────────────────────────────────────────────────────── conexão

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # foreign_keys não persiste no arquivo — precisa ser ligado por conexão.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA synchronous = NORMAL")
        self._load_vec_extension(conn)
        return conn

    def _load_vec_extension(self, conn: sqlite3.Connection) -> bool:
        """Tenta carregar o sqlite-vec. Falhar aqui não é erro — é o modo fallback."""
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            available = True
        except Exception:
            available = False
        if self._vec_available is None:
            self._vec_available = available
        return available

    @property
    def vec_available(self) -> bool:
        if self._vec_available is None:
            _ = self.conn  # força a conexão, que resolve a detecção
        return bool(self._vec_available)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Transação explícita. `isolation_level=None` deixa o controle conosco."""
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # ─────────────────────────────────────────────────────────────────── schema

    def init_schema(self) -> None:
        """Migração idempotente. A ORDEM abaixo é forçada por fatos medidos do SQLite;
        cada passo diz qual."""
        conn = self.conn
        # 0. ANTES de tudo: um `hypotheses` legado e VAZIO é dropado para que o
        #    executescript o recrie com `UNIQUE (focus_id, statement)`. MEDIDO: o
        #    `UNIQUE(statement)` global sobrevive a qualquer índice novo, e com ele o
        #    segundo foco levanta `UNIQUE constraint failed: hypotheses.statement` —
        #    ou seja, o escopo por foco seria verdadeiro só em banco de teste.
        self._rebuild_empty_legacy_hypotheses(conn)
        # 1. ORDEM IMPORTA: as views referenciam colunas adicionadas depois da v1
        # (`hypotheses.tier`, por exemplo). Rodar o script antes de migrar faz o
        # CREATE VIEW falhar com "no such column" num banco antigo. Em banco novo o
        # migrador é no-op, porque nenhuma tabela existe ainda.
        self._migrate_columns(conn)
        # 1.5. Fase C. O CHECK de `memories.source` NÃO migra sozinho: num banco criado
        #      antes desta fase, o DDL guardado continua
        #      `source IN ('chat','answer','manual','research')` e todo `approve` de
        #      observação levanta IntegrityError — invisível para a suíte, porque todo
        #      teste cria banco novo em tmp_path. Só um rebuild da tabela resolve.
        #      DEPOIS de `_migrate_columns` (que garante `focus_id` mesmo se o rebuild
        #      for adiado) e ANTES do executescript (que recria as views).
        self._rebuild_memories_for_recon(conn)
        # 1.6. Fase D. `sources.kind` era `CHECK (kind IN (4 literais))` e passa a ser FK
        #      para `sources_registry(slug)`. `CREATE TABLE IF NOT EXISTS` não altera
        #      tabela existente, então num banco legado o CHECK sobreviveria e uma fonte
        #      nova continuaria impossível de NOMEAR — o escopo por registro seria
        #      verdadeiro só em banco de teste. Mesmo tratamento de `hypotheses` na Fase A:
        #      rebuild se VAZIA, limitação registrada se populada.
        self._rebuild_empty_legacy_sources(conn)
        self._rebuild_extraction_runs_for_unknown(conn)
        # 2. Cria tabelas novas e RECRIA todas as views. A ordem de declaração dentro do
        #    arquivo não importa para a criação (MEDIDO: `CREATE VIEW v AS SELECT * FROM
        #    nao_existe` SUCEDE e só falha no SELECT). O que importa é que `meta` está no
        #    topo do arquivo — ver a nota lá.
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        # 3. O REGISTRO DE FONTES, antes de qualquer coisa que insira em `sources`:
        #    `sources.kind` é FK para ele, e a FK é verificada na ESCRITA (medido: o
        #    `CREATE TABLE` com REFERENCES para tabela inexistente passa; o INSERT é que
        #    levanta). Também antes do portão de `fetch_source`, que agora lê a coluna
        #    `yields_evidence` em vez de uma frozenset compilada.
        self._seed_registry(conn)
        # 4. Depois do executescript porque lê o id da escala de volta: é exatamente
        #    por isso que a semeadura não pode morar no schema.sql. Antes do passo
        #    seguinte porque `focuses.scale_id` é FK.
        scale_id = self._seed_scale(conn)
        # 4. O foco #1 e `meta['active_focus']`.
        self._seed_focus(conn, scale_id)
        self._init_vector_table(conn)
        # 5. Depois do executescript: depende de `text_key` existir (banco novo) e de
        # `_migrate_columns` já ter rodado (banco antigo). E fora dele: um índice único
        # não pode viver dentro de um script atômico. Ver `_migrate_indexes`.
        self._migrate_source_index(conn)
        self._migrate_indexes(conn)
        # 6. O VOCABULÁRIO. Antes do passo 7 porque `claim_directness.directness` é FK
        #    para `directness_weight`.
        self._seed_weights(conn)
        # 7. Retro-carga. LÊ `claims.directness`, então tem de vir antes do passo 8.
        self._backfill_claim_judgments(conn, scale_id)
        # 8. ÚLTIMO passo de DDL, e a razão é medida e contra-intuitiva: `ALTER TABLE
        #    DROP COLUMN` revalida o SCHEMA INTEIRO. Com a `claim_weight` antiga viva:
        #    `error in view claim_weight after drop column: no such column: c.directness`.
        #    Com `claim_weight` DROPADA: `error in view hypothesis_scoreboard: no such
        #    table: main.claim_weight` — a sequência intuitiva DROP VIEW → DROP COLUMN →
        #    CREATE VIEW é IMPOSSÍVEL. Só funciona depois que o passo 2 recriou todas as
        #    views resolvíveis e o passo 1 criou `scale_id`.
        self._drop_legacy_directness_column(conn)
        # 9. O grito. As três mortes silenciosas de `active_focus` deixam `claim_weight`
        #    permanentemente vazia sem erro e sem log.
        self._warn_if_no_active_focus(conn)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (SCHEMA_VERSION,),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('embedding_dim', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(self.embedding_dim),),
        )

    def _migrate_vector_table(self, conn: sqlite3.Connection) -> None:
        """Recria `chunk_vec` se ela existir com a métrica ou a dimensão erradas.

        `CREATE VIRTUAL TABLE IF NOT EXISTS` não conserta uma tabela já criada — um
        banco feito antes de declararmos `distance_metric=cosine` continuaria em L2
        para sempre, calado. Aqui detectamos e recriamos; os embeddings se perdem e
        precisam ser recalculados, o que `Ingestor` faz sozinho na próxima passada.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'chunk_vec'"
        ).fetchone()
        if row is None:
            return

        sql = (row["sql"] or "").lower()
        needs_cosine = self.vec_available and "distance_metric=cosine" not in sql
        needs_dim = self.vec_available and f"float[{self.embedding_dim}]" not in sql
        if not (needs_cosine or needs_dim):
            return

        n = conn.execute("SELECT COUNT(*) AS n FROM chunk_vec").fetchone()["n"]
        reason = "métrica L2 em vez de cosseno" if needs_cosine else "dimensão diferente"
        log.warning(
            "recriando chunk_vec (%s); %d embeddings serão recalculados no próximo "
            "ingest", reason, n,
        )
        conn.execute("DROP TABLE chunk_vec")

    def _init_vector_table(self, conn: sqlite3.Connection) -> None:
        self._migrate_vector_table(conn)
        if self.vec_available:
            # `distance_metric=cosine` NÃO é o padrão do sqlite-vec — sem isto ele
            # devolve L2. E o fallback em numpy devolve cosseno. Os dois caminhos
            # precisam concordar na métrica, senão `search_vector` retorna números
            # com significados diferentes conforme a extensão tenha carregado ou não,
            # e qualquer limiar calibrado em cima disso fica errado em um dos dois.
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0("
                f"  chunk_id INTEGER PRIMARY KEY,"
                f"  embedding FLOAT[{self.embedding_dim}] distance_metric=cosine"
                f")"
            )
        else:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS chunk_vec ("
                "  chunk_id  INTEGER PRIMARY KEY"
                "            REFERENCES chunks(id) ON DELETE CASCADE,"
                "  embedding BLOB NOT NULL"
                ")"
            )

    # Colunas adicionadas depois da v1. `CREATE TABLE IF NOT EXISTS` não altera tabela
    # existente, então bancos antigos precisam do ALTER explícito.
    # Colunas acrescentadas depois da v1, na ordem em que surgiram. `CHECK` é omitido
    # de propósito: o `ALTER TABLE ADD COLUMN` do SQLite é restrito, e o schema novo já
    # traz a constraint para bancos criados do zero.
    _ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
        # A citação verbatim que o portão 1 validou. Ela existia só em memória: usada em
        # `extract.py` para a checagem de ancoragem e para o prompt do portão 2, e
        # descartada ~50 linhas depois. Sem ela não há como reverificar o portão 1
        # retroativamente nem mostrar ao revisor o trecho exato em que a claim se apoia —
        # `chunk_ids` aponta para o chunk inteiro, não para a frase. NÃO tem backfill:
        # cada extração que roda sem esta coluna perde a evidência para sempre.
        # Quando e por quem uma lição foi retirada. Ver METRICS.md, MF6: a âncora dessa
        # métrica é que quem retira é a PESSOA — `memories --forget` e `/esquecer` são os
        # únicos escritores, e o humano não é o sistema. Sem isto, `active = 0` diz que
        # saiu e não diz quando nem por quê, então "o modelo reinseriu a lição que a
        # pessoa negou" — a assinatura mais direta de delírio que o plano captura — não
        # tem como ser medida. NÃO reconstrói: a retirada é um evento.
        ("memories", "retired_at", "TEXT"),
        ("memories", "retired_by", "TEXT"),
        ("claims", "supporting_quote", "TEXT"),
        ("questions", "targets", "TEXT"),
        ("hypotheses", "tier", "TEXT NOT NULL DEFAULT 'evidence'"),
        ("hypotheses", "intervention_class", "TEXT"),
        ("hypotheses", "mechanism_target", "TEXT"),
        ("hypotheses", "route", "TEXT"),
        ("hypotheses", "combination", "TEXT"),
        ("hypotheses", "chain_json", "TEXT"),
        ("hypotheses", "falsifier", "TEXT"),
        ("hypotheses", "test_proposal", "TEXT"),
        ("hypotheses", "known_risks", "TEXT"),
        ("hypotheses", "novelty", "REAL"),
        ("hypotheses", "critique_json", "TEXT"),
        ("hypotheses", "survives_critique", "INTEGER"),
        ("hypotheses", "pursued_at", "TEXT"),
        ("hypotheses", "regrounded_at", "TEXT"),
        ("tasks", "origin", "TEXT NOT NULL DEFAULT 'on_demand'"),
        ("memories", "provenance", "TEXT"),
        # Coluna comum, não GENERATED: o SQLite rejeita `ADD COLUMN ... STORED`, e uma
        # VIRTUAL não aparece em `PRAGMA table_info` — que é o que o loop abaixo
        # consulta — então o ALTER seria retentado em toda invocação até levantar
        # "duplicate column name".
        ("memories", "text_key", "TEXT"),
        ("questions", "rounds", "INTEGER NOT NULL DEFAULT 0"),
        # Fase A. **SEM `REFERENCES`**, e isso é obrigatório, não estilo:
        #
        #   * neste instante `evidence_scales`/`focuses` ainda não existem, e MEDIDO em
        #     3.51.2 que `ADD COLUMN scale_id INTEGER REFERENCES evidence_scales(id)` é
        #     ACEITO mas deixa `claims` NÃO-ESCRIVÍVEL (INSERT e DELETE levantam
        #     `no such table: main.evidence_scales`) até o parent aparecer — janela
        #     visível para o daemon em outra conexão, e permanente se o executescript
        #     falhar nessa rodada.
        #   * para os `focus_id`, com `NOT NULL DEFAULT 1` o SQLite RECUSA a coluna com
        #     REFERENCES: `Cannot add a REFERENCES column with non-NULL default value`
        #     (MEDIDO, com foreign_keys=ON — que é o que `_connect` liga).
        #
        # Mesma concessão já registrada acima para o CHECK: banco novo ganha a
        # constraint pelo CREATE TABLE, banco legado não.
        ("claims", "scale_id", "INTEGER"),
        ("questions", "focus_id", "INTEGER NOT NULL DEFAULT 1"),
        ("hypotheses", "focus_id", "INTEGER NOT NULL DEFAULT 1"),
        # Fase B. O TERCEIRO veredito do relens: "julgada e IRRELEVANTE neste foco".
        # Não pode ser um 5º nível de directness (quebraria a FK e a gramática), e não
        # pode ser ausência de aresta (indistinguível de "ainda não julguei"). Por isso
        # é coluna à parte, excluída de `claim_weight` por predicado.
        ("claim_directness", "out_of_scope", "INTEGER NOT NULL DEFAULT 0"),
        ("focuses", "judgment_hash", "TEXT"),
        # Fase C. AQUI **e** no rebuild, e a redundância é obrigatória, não zelo:
        # `live_memories` passa a projetar `focus_id` e o executescript a recria SEMPRE,
        # mesmo quando o rebuild é ADIADO pelo pré-voo. `CREATE VIEW` sobre coluna
        # inexistente SUCEDE (é o mesmo fato registrado no passo 2 de `init_schema`) e
        # só o SELECT levanta — então, sem este ALTER, um banco com uma linha de `kind`
        # fora do vocabulário ficaria com `user_memories`, `research_lessons` e
        # `recon_memories` TODAS quebradas: `lithium chat`, `lithium ask` e o
        # `reflect_tick` morrem com `no such column: focus_id`, uma mensagem que não
        # tem relação nenhuma com o WARNING do pré-voo. O ALTER não traz os CHECKs —
        # quem os traz é o rebuild —, mas as views só precisam da COLUNA, e é
        # exatamente para isso que `_ADDED_COLUMNS` existe.
        ("memories", "focus_id", "INTEGER"),
    )

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        """Aplica ALTERs pendentes. Tabela inexistente é pulada — em banco novo o
        `executescript` que vem a seguir já a cria com todas as colunas."""
        for table, column, decl in self._ADDED_COLUMNS:
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            if not info:
                continue
            if column not in {r["name"] for r in info}:
                log.info("adicionando coluna %s.%s", table, column)
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _backfill_memory_keys(self, conn: sqlite3.Connection) -> None:
        """Preenche `text_key` onde está NULL, uma linha por vez.

        Linha a linha e não `executemany`: com o índice já criado, uma linha NULL que
        colida com uma ativa faria o UPDATE em lote levantar no meio, abortando
        `init_schema()` — o mesmo deadlock que `_migrate_indexes` existe para evitar,
        entrando por outra porta. A linha em conflito fica com NULL: não deduplicada,
        mas intacta e visível no `lithium memories`.
        """
        rows = conn.execute(
            "SELECT id, text FROM memories WHERE text_key IS NULL"
        ).fetchall()
        if not rows:
            return
        filled, blocked = 0, []
        for row in rows:
            try:
                conn.execute(
                    "UPDATE memories SET text_key = ? WHERE id = ?",
                    (normalize_memory_text(row["text"]), row["id"]),
                )
                filled += 1
            except sqlite3.IntegrityError:
                blocked.append(int(row["id"]))
        if filled:
            log.info("text_key preenchido para %d memória(s)", filled)
        if blocked:
            log.warning(
                "memória(s) %s repetem uma lição já ativa e ficaram sem chave de "
                "deduplicação. Nada foi apagado — use `lithium memories --forget <id>`.",
                ", ".join(f"#{i}" for i in blocked),
            )

    def _migrate_source_index(self, conn: sqlite3.Connection) -> None:
        """Cria o índice de `article_key` só quando a coluna existe.

        Fora do `executescript` pela mesma razão de `_migrate_indexes`, mas por um motivo
        diferente e medido: `CREATE INDEX` **valida a coluna**, enquanto `CREATE VIEW`
        aceita coluna inexistente e só falha no uso. Num banco legado com `sources`
        populada o rebuild é adiado de propósito, a coluna não nasce, e o índice dentro do
        script levantaria `no such column: article_key` — derrubando `lithium status` e
        `lithium sources`, que são as ferramentas para diagnosticar justamente isso.
        """
        cols = {r["name"] for r in conn.execute("PRAGMA table_xinfo(sources)")}
        if "article_key" not in cols:
            return
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_article ON sources(article_key)"
        )

    def _migrate_indexes(self, conn: sqlite3.Connection) -> int:
        """Cria o índice de deduplicação de lições — se e somente se for seguro.

        **Fora do `executescript`, de propósito.** `init_schema()` roda em TODA
        invocação de CLI. Um `CREATE UNIQUE INDEX` dentro do script, sobre um banco que
        já tem duplicatas, levanta `IntegrityError` e derruba `lithium memories` — que é
        justamente a ferramenta para inspecionar e resolver as duplicatas. Deadlock. E
        como `executescript` é atômico-por-script, tudo que viesse depois do índice (as
        views de memória, `sync_state`, `meta`) deixaria de ser aplicado, calado.

        Duplicata pré-existente **não é apagada**: apagar é destrutivo e ninguém
        autorizou. Aqui a gente degrada e avisa. O caminho de saída é
        `lithium memories --forget-duplicates`, porque o aviso sozinho não basta —
        50 lições repetidas viram 50 comandos manuais sobre uma listagem ordenada por
        id, onde as duplicatas nem ficam adjacentes.

        Devolve quantas chaves estão duplicadas (0 = índice ativo).
        """
        self._backfill_memory_keys(conn)

        # Pelo NOME **e** pela DEFINIÇÃO. Só pelo nome, um índice com a definição errada
        # (comum em vez de único, ou um predicado antigo) sobrevive para sempre e a
        # deduplicação fica inerte sem nenhum sinal. Mesmo idioma de
        # `_migrate_vector_table`.
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (MEMORY_DEDUP_INDEX,),
        ).fetchone()
        if existing is not None:
            if _sql_equal(existing["sql"], MEMORY_DEDUP_INDEX_SQL):
                return 0
            log.info("índice %s tem definição obsoleta, recriando", MEMORY_DEDUP_INDEX)
            conn.execute(f"DROP INDEX {MEMORY_DEDUP_INDEX}")

        # O predicado da guarda tem de ser IDÊNTICO ao do índice. Sem `active = 1` ela
        # continuaria vendo duplicatas depois do `--forget` e o índice nunca subiria;
        # sobre `text` cru não enxergaria variantes de caixa e o IntegrityError escapa.
        dupes = conn.execute(
            "SELECT text_key, COUNT(*) AS n FROM memories "
            " WHERE source = 'research' AND active = 1 AND text_key IS NOT NULL "
            " GROUP BY text_key HAVING n > 1"
        ).fetchall()
        if dupes:
            log.warning(
                "deduplicação de lições desligada: %d lição(ões) repetida(s) em %d "
                "linha(s) ativas. Nada foi apagado. Resolva com "
                "`lithium memories --forget-duplicates`; o índice sobe sozinho depois.",
                len(dupes), sum(int(r["n"]) for r in dupes),
            )
            return len(dupes)

        self._migrate_legacy_hypothesis_index(conn)

        try:
            conn.execute(MEMORY_DEDUP_INDEX_SQL)
        except sqlite3.IntegrityError as exc:
            # Corrida: outro processo inseriu uma duplicata entre o SELECT e o CREATE.
            # O daemon roda `init_schema` junto com o CLI, então isso acontece de fato.
            log.warning("deduplicação de lições desligada: %s", exc)
            return 1
        return 0

    def _migrate_legacy_hypothesis_index(self, conn: sqlite3.Connection) -> None:
        """Índice único composto para os bancos legados que sobreviveram ao passo 0.

        Aqui e não no schema.sql porque um `CREATE UNIQUE INDEX` no arquivo reprova
        `test_schema_sql_does_not_create_a_unique_index_on_memories` — mesma razão pela
        qual `MEMORY_DEDUP_INDEX` já mora aqui.

        Guardado pela EXISTÊNCIA DO ÍNDICE, não pelo SQL da tabela: senão o log repete a
        cada abertura.
        """
        if not self.hypotheses_are_globally_unique(conn):
            return
        existing = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
            (LEGACY_HYPOTHESIS_INDEX,),
        ).fetchone()
        if existing is not None:
            return
        conn.execute(LEGACY_HYPOTHESIS_INDEX_SQL)
        log.info(
            "índice %s criado: sem ele o `ON CONFLICT(focus_id, statement)` de "
            "explore.py levanta OperationalError neste banco", LEGACY_HYPOTHESIS_INDEX,
        )

    def forget_duplicate_lessons(self) -> list[int]:
        """Desativa lições de pesquisa repetidas, mantendo a de menor id.

        Existe porque sem ela o estado degradado é **autoalimentado**: enquanto o
        índice não sobe, o daemon continua pesquisando e gravando repetidas, então o
        backlog que o usuário precisa limpar cresce sozinho. Soft-delete, como
        `forget()` — o histórico do que foi lembrado importa.
        """
        rows = self.conn.execute(
            "SELECT id FROM memories WHERE source = 'research' AND active = 1 "
            "  AND text_key IS NOT NULL "
            "  AND id > (SELECT MIN(m2.id) FROM memories m2 "
            "             WHERE m2.text_key = memories.text_key "
            "               AND m2.source = 'research' AND m2.active = 1) "
            "ORDER BY id"
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        for memory_id in ids:
            self.conn.execute(
                "UPDATE memories SET active = 0 WHERE id = ?", (memory_id,)
            )
        if ids:
            log.info("%d lição(ões) repetida(s) desativada(s)", len(ids))
            self._migrate_indexes(self.conn)
        return ids

    def _rebuild_extraction_runs_for_unknown(self, conn: sqlite3.Connection) -> None:
        """Reconstrói `extraction_runs` quando os contadores de chunk são `NOT NULL`.

        A tabela nasceu com `chunks_annihilated INTEGER NOT NULL DEFAULT 0`, e isso
        impedia gravar DESCONHECIDO. A distinção importa: uma corrida retro-encaixada de
        log não tem as rejeições individuais, então não dá para saber quais chunks
        propuseram e perderam. Com `NOT NULL`, o retro-encaixe gravava 0 aniquilados e
        `chunks_seen` estéreis — e o primeiro relatório afirmou **137 chunks estéreis**
        num corpus que tem 76.

        Dropa em vez de `ALTER`: o SQLite não afrouxa `NOT NULL` sem rebuild, e o
        conteúdo é re-derivável — só entra aqui medição de primeira mão (que será
        regravada na próxima extração) ou retro-encaixe de log (que é de segunda mão por
        definição). Com medição de primeira mão presente, a limitação é REGISTRADA em vez
        de a linha ser perdida.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='extraction_runs'"
        ).fetchone()
        if row is None or "chunks_annihilated INTEGER NOT NULL" not in (row["sql"] or ""):
            return
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM extraction_runs WHERE origin = 'run'"
        ).fetchone()["n"]
        if n:
            already = conn.execute(
                "SELECT 1 FROM meta WHERE key = 'extraction_runs_notnull'").fetchone()
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('extraction_runs_notnull','legacy') "
                "ON CONFLICT(key) DO NOTHING")
            if already is None:
                log.warning(
                    "`extraction_runs` tem %d corrida(s) de primeira mão e a forma antiga "
                    "(NOT NULL): retro-encaixes de log nela reportam 0 aniquilados e "
                    "todos os chunks como estéreis, o que é FALSO. Corridas novas estão "
                    "corretas.", n)
            return
        log.info("reconstruindo `extraction_runs`: chunk desconhecido passa a ser NULL")
        conn.execute("DROP TABLE extraction_runs")

    def _rebuild_empty_legacy_sources(self, conn: sqlite3.Connection) -> None:
        """Dropa `sources` quando ela está na forma legada E vazia.

        A forma legada é `CHECK (kind IN ('pubmed', 'epmc', 'ctgov', 'fda'))`, e ela é o
        que impede uma fonte nova de ser NOMEADA: o INSERT é rejeitado antes de qualquer
        portão de política. `CREATE TABLE IF NOT EXISTS` não altera tabela existente, então
        sem este rebuild o registro de fontes funcionaria só em banco criado do zero — o
        modo de falha "verde na suíte, quebrado no banco do usuário".

        Mesmo tratamento e mesma razão de `_rebuild_empty_legacy_hypotheses`: dropar e
        deixar o `executescript` recriar. `chunks` e `claims` têm FK com ON DELETE CASCADE
        para `sources`, então uma `sources` vazia implica as duas vazias — o DROP não pode
        perder linha que exista. Com linhas, a limitação é REGISTRADA em vez de a operação
        mais perigosa da migração ser escrita para um banco que nunca colheu.

        `article_key` é GENERATED, e o SQLite recusa `ADD COLUMN ... GENERATED STORED`:
        ela só nasce em tabela criada do zero, o que faz deste rebuild o único caminho.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'sources'"
        ).fetchone()
        if row is None:
            return  # banco novo: o CREATE TABLE já traz a FK e o article_key
        if "sources_registry" in (row["sql"] or ""):
            return  # já está na forma nova
        n = conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"]
        if n:
            already = conn.execute(
                "SELECT 1 FROM meta WHERE key = 'sources_kind_check'"
            ).fetchone()
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('sources_kind_check', 'legacy') "
                "ON CONFLICT(key) DO NOTHING"
            )
            if already is not None:
                return  # avisa UMA vez, não em toda abertura de CLI
            log.warning(
                "`sources` está na forma legada com %d linha(s): o CHECK de quatro "
                "literais sobrevive, então fonte fora de "
                "('pubmed','epmc','ctgov','fda') será rejeitada no INSERT, e "
                "`article_key` não existe (duas linhas do mesmo DOI contam como duas "
                "citações). Registro de fontes funciona; fonte NOVA, não.", n,
            )
            return
        log.info("reconstruindo `sources` legada e vazia: CHECK de kind -> FK do registro")
        conn.execute("DROP TABLE sources")

    # As fontes que o sistema conhece de fábrica. Uma linha, não uma constante Python:
    # é a diferença entre "estas são as fontes" e "estas são as fontes que já vêm
    # cadastradas". Fonte nova entra por `lithium sources --approve`, sem tocar em código.
    BUILTIN_SOURCES: tuple[dict[str, Any], ...] = (
        {
            "slug": "pubmed",
            "description": "MEDLINE/PubMed via E-utilities do NCBI",
            "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            "yields_evidence": 1,
            "credential_ref": "sources.pubmed.api_key",
            "rate_per_s": 3.0,
            "adapter": "pubmed",
            "approved": True,
        },
    )
    """Só o PubMed vem aprovado, e é uma decisão medida, não conservadorismo.

    O item 9 mediu as outras três candidatas e RECUSOU as três: o Europe PMC não
    acrescenta um único artigo revisado que o PubMed já não traga (0 de 813); 75% dos
    registros do ClinicalTrials.gov não têm resultado e a prosa de registro passa o portão
    de citação literal porque a citação É literal — só o portão de TIPO distingue intenção
    de resultado; e uma bula não é desenho de estudo, então a escala de `grade` não tem
    lugar para ela.

    Essas medições valem para AQUELAS fontes naquele foco. O que elas não justificam é uma
    lista fixa e permanente — que era exatamente o que `EVIDENCE_KINDS` tinha virado.
    """

    def _seed_registry(self, conn: sqlite3.Connection) -> None:
        """Semeia as fontes de fábrica. Insert-if-absent, NUNCA update.

        Mesma razão de `_seed_weights` e `_seed_scale`: `init_schema()` roda em TODA
        invocação de CLI, então um `DO UPDATE` aqui desfaria em silêncio, a cada comando,
        qualquer coisa que o usuário tenha mudado — inclusive REVOGAR uma aprovação. Uma
        fonte que o usuário desaprovou voltaria a ser consultada no comando seguinte.
        """
        for src in self.BUILTIN_SOURCES:
            conn.execute(
                "INSERT INTO sources_registry"
                "  (slug, description, base_url, yields_evidence, credential_ref,"
                "   rate_per_s, adapter, approved_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, "
                "       CASE WHEN ? THEN strftime('%Y-%m-%dT%H:%M:%fZ','now') END) "
                "ON CONFLICT(slug) DO NOTHING",
                (src["slug"], src["description"], src["base_url"],
                 src["yields_evidence"], src.get("credential_ref"),
                 src["rate_per_s"], src["adapter"], bool(src.get("approved"))),
            )

    def propose_source(self, slug: str, description: str, base_url: str,
                       discovery_id: int | None = None) -> bool:
        """Registra uma fonte como PROPOSTA: não aprovada, sem produzir evidência.

        Mora aqui, e não em `lithium/recon/`, pela mesma razão que a ponte `recon_lead`
        mora em `worker/handlers.py`: a varredura de AST do pacote do batedor proíbe o nome
        `sources` justamente para que o código capaz de atravessar a fronteira não tenha
        onde ser escrito. Fazer o batedor falar SQL de `sources_registry` obrigaria a
        afrouxar a trava que protege o pacote inteiro — o portão pegou essa tentativa.

        `approved_at` NULL e `yields_evidence = 0` fazem a linha ficar fora de
        `active_sources`, logo fora do daemon e fora do portão de `fetch_source`. Aprovar
        a descoberta significa "vale investigar"; ativar a fonte é outra decisão.

        Devolve False quando o slug já existe — duas descobertas apontando para o mesmo
        domínio é o caso comum, não o excepcional.
        """
        exists = self.conn.execute(
            "SELECT 1 FROM sources_registry WHERE slug = ?", (slug,)
        ).fetchone()
        if exists is not None:
            return False
        self.conn.execute(
            "INSERT INTO sources_registry"
            "  (slug, description, base_url, yields_evidence, proposed_by_discovery) "
            "VALUES(?, ?, ?, 0, ?)",
            (slug, description, base_url, discovery_id),
        )
        return True

    def source_state(self, slug: str) -> str | None:
        """`None` (desconhecida), `'proposta'` ou `'ativa'`."""
        row = self.conn.execute(
            "SELECT approved_at FROM sources_registry WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return None
        return "ativa" if row["approved_at"] else "proposta"

    def record_reflect_tick(self, focus_id: int | None, texts: list[str]) -> int:
        """Persiste um tique de reflexão e conta quantas lições REINSERIRAM texto retirado.

        A reinserção é o sinal: `memories.retired_by` só é escrito por
        `lithium memories --forget` e por `/esquecer`, os dois caminhos HUMANOS. Uma
        lição nova cujo `text_key` casa uma retirada é o modelo reescrevendo o que a
        pessoa negou — ver METRICS.md, MF6.

        Casa por `text_key` (normalizado), não por texto cru: reinserir com outra
        pontuação continua sendo reinserir.
        """
        chaves = [normalize_memory_text(x) for x in texts]
        reinseridas = 0
        if chaves:
            marks = ",".join("?" * len(chaves))
            reinseridas = int(self.conn.execute(
                f"SELECT COUNT(DISTINCT text_key) AS n FROM memories "
                f" WHERE text_key IN ({marks}) AND retired_at IS NOT NULL",
                tuple(chaves)).fetchone()["n"])
        cur = self.conn.execute(
            "INSERT INTO reflect_ticks(focus_id, proposed, written, reinserted) "
            "VALUES(?,?,?,?) RETURNING id",
            (focus_id, len(texts), len(texts), reinseridas))
        return int(cur.fetchone()["id"])

    def record_extraction(self, result: Any, focus_id: int | None = None,
                          origin: str = "run") -> int:
        """Persiste o resultado INTEIRO de uma extração, incluindo as rejeições.

        Substitui um `log.info` mais um `log.debug(rejections[:5])`. Duas diferenças que
        importam: nada é truncado (o `[:5]` descartava silenciosamente a partir da sexta),
        e a linha sobrevive ao processo — o único handler de log deste projeto escreve no
        console. Ver METRICS.md, MF1.

        `origin='log'` marca retro-encaixe a partir de arquivo de log: medição de segunda
        mão, que nunca deve entrar numa média junto com a de primeira sem dizer.
        """
        with self.tx() as conn:
            cur = conn.execute(
                "INSERT INTO extraction_runs"
                "  (source_id, focus_id, proposed, anchored, verified, chunks_seen,"
                "   chunks_annihilated, chunks_sterile, origin) "
                "VALUES(?,?,?,?,?,?,?,?,?) RETURNING id",
                # Os dois contadores de chunk só têm sentido quando as rejeições
                # individuais existem. Num retro-encaixe de log elas não existem, e
                # derivá-los daria um número FALSO com cara de medição.
                (int(result.source_id), focus_id, result.proposed, result.anchored,
                 result.verified, result.chunks_seen,
                 result.chunks_annihilated if origin == "run" else None,
                 result.chunks_sterile if origin == "run" else None, origin),
            )
            run_id = int(cur.fetchone()["id"])
            for r in result.rejections:
                conn.execute(
                    "INSERT INTO extraction_rejections"
                    "  (run_id, chunk_id, gate, reason, statement, quote) "
                    "VALUES(?,?,?,?,?,?)",
                    (run_id, r.chunk_id, r.gate, r.reason[:500],
                     (r.statement or None), (r.quote or None)),
                )
        return run_id

    def active_sources(self) -> list[sqlite3.Row]:
        """As fontes aprovadas. É o que o daemon monta em `Context.sources`."""
        return list(self.conn.execute("SELECT * FROM active_sources ORDER BY slug"))

    def source_yields_evidence(self, slug: str) -> bool:
        """O PORTÃO, e ele é uma consulta em runtime em vez de um conjunto compilado.

        Substitui `EVIDENCE_KINDS`. A pergunta que ele responde não mudou — "isto pode
        virar claim?" — mas ela deixa de ser respondida por uma allowlist de duas APIs e
        passa a ser respondida pelo contrato: publica estudo com prosa citável verbatim e
        desenho graduável. Fail-closed em fonte desconhecida ou não aprovada.
        """
        row = self.conn.execute(
            "SELECT yields_evidence FROM active_sources WHERE slug = ?", (slug,)
        ).fetchone()
        return bool(row and row["yields_evidence"])

    def _seed_weights(self, conn: sqlite3.Connection) -> None:
        """O VOCABULÁRIO: o conjunto fechado de níveis legais, aplicado por FK.

        **Insert-if-absent, nunca UPDATE.** `init_schema()` roda em TODA abertura de CLI,
        então o `ON CONFLICT DO UPDATE` que estava aqui fazia uma edição em types.py
        reescrever retroativamente o peso de toda claim já colhida — e o ranking de ontem
        deixava de ser reproduzível, sem uma linha de log. Confirmado que a suíte inteira
        ficava verde nas duas versões; por isso a correção vem com
        `test_the_seed_never_overwrites_a_stored_weight`.

        As duas tabelas FICAM. Dropá-las é pior do que parece: com filhas populadas o
        DROP levanta `FOREIGN KEY constraint failed`, e — pior — com as filhas VAZIAS ele
        PASSA, deixando a FK pendurada e quebrando todo `INSERT INTO sources` com
        `no such table: main.grade_weight`, um diagnóstico que aponta para o ingest.
        Quem calibra é `scale_levels`.
        """
        conn.executemany(
            "INSERT INTO grade_weight(grade, weight, rank) VALUES(?, ?, ?) "
            "ON CONFLICT(grade) DO NOTHING",
            GRADE_WEIGHTS,
        )
        conn.executemany(
            "INSERT INTO directness_weight(directness, weight, rank) VALUES(?, ?, ?) "
            "ON CONFLICT(directness) DO NOTHING",
            DIRECTNESS_WEIGHTS,
        )

    # ──────────────────────────────────────────────────────── foco e escala

    @staticmethod
    def calibration_hash() -> str:
        """Impressão digital da CALIBRAÇÃO (os pesos), não do vocabulário nem da ordem.

        `rank` fica de fora de propósito: ele é ordem de força, é mantido em dia pelo
        seed, e mudá-lo não reescreve nenhum peso histórico.
        """
        payload = json.dumps(
            {
                "grade": {g.value: w for g, w in GRADE_WEIGHT.items()},
                "directness": {d.value: w for d, w in DIRECTNESS_WEIGHT.items()},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _seed_scale(self, conn: sqlite3.Connection) -> int:
        """Semeia a escala 'clinical-evidence' e seus 13 níveis. Devolve o id.

        **`weight` é congelado (DO NOTHING), `rank` é mantido em dia (DO UPDATE).** Os
        dois têm papéis diferentes e tratá-los igual introduz um defeito: `weight` é
        julgamento histórico e reescrevê-lo é o que esta fase existe para impedir, mas
        `rank` é derivado da ordem de declaração do enum em types.py — congelá-lo faria
        um nível NOVO entrar com rank duplicado e `MIN(rank)` passar a apontar para dois
        níveis ao mesmo tempo.
        """
        conn.execute(
            "INSERT INTO evidence_scales(id, slug, description) VALUES(1, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (SCALE_SLUG, SCALE_DESCRIPTION),
        )
        scale_id = int(
            conn.execute(
                "SELECT id FROM evidence_scales WHERE slug = ?", (SCALE_SLUG,)
            ).fetchone()["id"]
        )
        levels = [
            (scale_id, "grade", value, weight, rank)
            for value, weight, rank in GRADE_WEIGHTS
        ] + [
            (scale_id, "directness", value, weight, rank)
            for value, weight, rank in DIRECTNESS_WEIGHTS
        ]
        conn.executemany(
            "INSERT INTO scale_levels(scale_id, axis, value, weight, rank) "
            "VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(scale_id, axis, value) DO UPDATE SET rank = excluded.rank",
            levels,
        )
        return scale_id

    @staticmethod
    def judgment_hash(target: str, directness_definitions: str) -> str:
        """Hash do que define o SIGNIFICADO de um julgamento de directness.

        Separado de `calibration_hash` porque os dois derivam por motivos diferentes e
        exigem remédios OPOSTOS: calibração divergiu -> escala NOVA + foco NOVO (peso é
        julgamento congelado); `target`/`directness_definitions` divergiram -> RELENS,
        porque toda aresta daquele foco foi julgada contra a régua antiga; strategies ou
        taxonomy divergiram -> NADA, query não é julgamento.

        NÃO é lido por `_seed_focus`. Ver o comentário em `schema.sql`: compor as duas
        metades num hash só faria todo banco da Fase A gritar "a calibração divergiu" a
        cada comando de CLI, prescrevendo o remédio mais destrutivo que existe para um
        evento em que nada divergiu.
        """
        payload = json.dumps(
            {"target": target, "directness_definitions": directness_definitions},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _seed_focus(self, conn: sqlite3.Connection, scale_id: int) -> None:
        """Insere o foco #1 e aponta `meta['active_focus']` para ele.

        `meta` com **`DO NOTHING`**, jamais `DO UPDATE`: MEDIDO que com o padrão copiado
        de `schema_version` o foco escolhido pelo usuário volta a '1' a cada comando de
        CLI — é literalmente o defeito de `_seed_weights` reentrando pela chave nova.

        A chave AUSENTE é auto-curada aqui (um INSERT sem conflito), e isso é o
        comportamento certo: sem a chave o placar inteiro fica zerado. É por isso que
        `test_no_active_focus_zeroes_the_board` NÃO parametriza 'chave ausente' — esse
        cenário é impossível depois de uma abertura de CLI, e quem o trava é
        `test_a_missing_active_focus_key_heals_on_reopen`.
        """
        conn.execute(
            "INSERT INTO focuses(id, slug, target, scale_id, profile_hash) "
            "VALUES(1, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (FOCUS_SLUG, FOCUS_TARGET, scale_id, self.calibration_hash()),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('active_focus', '1') "
            "ON CONFLICT(key) DO NOTHING"
        )
        drifted = conn.execute(
            "SELECT slug, profile_hash FROM focuses "
            " WHERE profile_hash IS NOT NULL AND profile_hash <> ?",
            (self.calibration_hash(),),
        ).fetchall()
        for row in drifted:
            log.warning(
                "a calibração em types.py divergiu da escala congelada no foco %r "
                "(profile_hash %s ≠ %s). O peso das claims NÃO foi reescrito, de "
                "propósito. O caminho é escala NOVA + foco NOVO — nunca UPDATE em "
                "scale_levels, que reescreveria o julgamento de todo o histórico.",
                row["slug"], row["profile_hash"], self.calibration_hash(),
            )

    def active_focus(self) -> sqlite3.Row | None:
        """O foco ativo, ou None. `None` é um estado alcançável e silencioso — ver
        `_warn_if_no_active_focus`."""
        return self.conn.execute("SELECT * FROM active_focus").fetchone()

    def _warn_if_no_active_focus(self, conn: sqlite3.Connection) -> None:
        """Grita quando o placar está zerado por falta de foco, não por falta de corpus.

        MEDIDO: com `meta['active_focus']` = 'abc' (CAST vira 0), apontando para um id
        inexistente, ou para um foco com `retired_at` preenchido, `claim_weight` fica
        PERMANENTEMENTE VAZIA sem um erro e sem um log, e os 7 consumidores pontuam zero.
        Sem este passo, 'nenhum foco ativo' é indistinguível de 'ainda não colhemos nada'.
        """
        if conn.execute("SELECT COUNT(*) AS n FROM active_focus").fetchone()["n"]:
            return
        verified = conn.execute(
            "SELECT COUNT(*) AS n FROM claims WHERE verified = 1"
        ).fetchone()["n"]
        if not verified:
            return
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'active_focus'"
        ).fetchone()
        log.warning(
            "NENHUM FOCO ATIVO: meta['active_focus'] = %r não resolve para um foco vivo. "
            "%d claim(s) verificada(s) estão SEM PESO e todo placar do sistema vale "
            "zero. Escolha um com `lithium focus --use <slug>`.",
            row["value"] if row else None, verified,
        )

    def _backfill_claim_judgments(self, conn: sqlite3.Connection, scale_id: int) -> None:
        """Retro-carrega `claim_directness` a partir de `claims.directness`.

        Sem LLM: o valor guardado JÁ É o julgamento contra este alvo — a Fase A só passa
        a registrar contra QUAL alvo ele foi feito. Cópia 1-para-1, então nenhum peso
        muda. Uma migração que muda comportamento não é uma migração.

        Linha a linha com `try/except IntegrityError`, seguindo `_backfill_memory_keys`:
        um `executemany` que aborta no meio derruba `init_schema()` e com ele TODO comando
        de CLI, inclusive os de diagnóstico.

        Em banco novo é no-op puro (a coluna legada não existe).
        """
        info = {r["name"] for r in conn.execute("PRAGMA table_info(claims)")}
        if "directness" not in info:
            return
        focus = conn.execute(
            "SELECT id FROM focuses WHERE slug = ?", (FOCUS_SLUG,)
        ).fetchone()
        if focus is None:
            return
        focus_id = int(focus["id"])
        conn.execute(
            "UPDATE claims SET scale_id = ? WHERE scale_id IS NULL", (scale_id,)
        )
        rows = conn.execute(
            "SELECT id, directness FROM claims WHERE directness IS NOT NULL"
        ).fetchall()
        moved, blocked = 0, []
        for row in rows:
            try:
                cur = conn.execute(
                    "INSERT INTO claim_directness(claim_id, focus_id, directness) "
                    "VALUES(?, ?, ?) ON CONFLICT(claim_id, focus_id) DO NOTHING",
                    (int(row["id"]), focus_id, row["directness"]),
                )
                moved += cur.rowcount or 0
            except sqlite3.IntegrityError:
                blocked.append(int(row["id"]))
        if moved:
            log.info("julgamento de directness migrado para %d claim(s)", moved)
        if blocked:
            log.warning(
                "claim(s) %s carregam um directness fora do vocabulário e ficaram SEM "
                "PESO. Nada foi apagado; elas aparecem em `lithium status` como não "
                "julgadas.",
                ", ".join(f"#{i}" for i in blocked),
            )

    def _drop_legacy_directness_column(self, conn: sqlite3.Connection) -> None:
        """Remove `claims.directness`. Mandatório, não cosmético: a coluna é `NOT NULL`
        sem DEFAULT, então assim que `extract.py` para de escrevê-la TODO INSERT em
        claims num banco legado falharia com `NOT NULL constraint failed`."""
        info = {r["name"] for r in conn.execute("PRAGMA table_info(claims)")}
        if "directness" not in info:
            return
        conn.execute("ALTER TABLE claims DROP COLUMN directness")
        log.info("claims.directness removida; o julgamento agora vive em claim_directness")

    def _rebuild_empty_legacy_hypotheses(self, conn: sqlite3.Connection) -> None:
        """Dropa `hypotheses` quando ela está na forma legada E vazia.

        Dropar e deixar o `executescript` recriar, em vez de reconstruir com cópia: é a
        forma mais simples que existe e é suficiente porque o banco de produção nunca
        colheu. Tentar o rebuild com preservação de linhas é a operação mais perigosa da
        migração, e escrevê-la para um consumidor que não existe é o defeito deste repo.

        MEDIDO que o `ALTER TABLE ... RENAME TO hypotheses_old` do rebuild ingênuo
        REESCREVE a FK de `evidence_links` para `hypotheses_old` (comportamento padrão
        desde 3.25 com `legacy_alter_table=OFF`). MEDIDO também que DROP TABLE com as
        views apontando para ela é aceito, que as views sobrevivem, e que
        `PRAGMA foreign_key_check` volta vazio depois que o schema recria a tabela.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'hypotheses'"
        ).fetchone()
        if row is None:
            return  # banco novo: o CREATE TABLE já traz o UNIQUE composto
        if re.search(r"unique\s*\(\s*focus_id\s*,\s*statement\s*\)", row["sql"] or "",
                     re.I):
            return  # já está na forma nova
        n = conn.execute("SELECT COUNT(*) AS n FROM hypotheses").fetchone()["n"]
        if n:
            # Uma vez só. Repetir a cada abertura de CLI é ruído, e ruído recorrente é
            # como um aviso deixa de ser lido. A marca durável em `meta` é o que
            # `lithium focus` consulta para recusar o segundo foco.
            already = conn.execute(
                "SELECT 1 FROM meta WHERE key = 'hypotheses_focus_scope'"
            ).fetchone()
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('hypotheses_focus_scope', 'global') "
                "ON CONFLICT(key) DO NOTHING"
            )
            if already is not None:
                return
            log.warning(
                "`hypotheses` está na forma legada (UNIQUE global em `statement`) e tem "
                "%d linha(s), então NÃO foi reconstruída. Consequência registrada: neste "
                "banco dois focos não podem carregar o mesmo `statement`; "
                "`lithium focus` diz isso na listagem. O rebuild com preservação de "
                "linhas é da Fase B, onde existirá um segundo foco para exercitá-lo.", n,
            )
            return
        conn.execute("DROP TABLE hypotheses")
        log.info(
            "`hypotheses` legada e vazia foi dropada; o schema a recria com "
            "UNIQUE (focus_id, statement)"
        )

    # ─────────────────────────────────────────── Fase C: o rebuild de `memories`

    def recon_memory_available(self) -> bool:
        """True quando este banco aceita `memories.source = 'recon'`.

        Existe para o `approve` de uma observação dar uma MENSAGEM em vez de um
        `IntegrityError` cru quando o rebuild foi adiado pelo pré-voo. Compara a
        DEFINIÇÃO guardada, nunca o nome — mesmo idioma de `_migrate_indexes` e
        `_migrate_vector_table`.
        """
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memories'"
        ).fetchone()
        return row is not None and "'recon'" in (row["sql"] or "")

    def _rebuild_memories_for_recon(self, conn: sqlite3.Connection) -> bool:
        """Reconstrói `memories` para aceitar `source='recon'`. Devolve True se rodou.

        A operação mais perigosa desta migração — `_rebuild_empty_legacy_hypotheses`
        NÃO serve de modelo, porque ele só dropa tabela VAZIA. Aqui as linhas são
        preservadas, e cada precaução abaixo vem de uma falha OBSERVADA:

        * **pré-voo.** Uma linha com `kind` ou `source` fora do vocabulário faz o
          `INSERT ... SELECT` levantar e mata `init_schema()` — que roda em TODA
          invocação de CLI. O usuário perderia `lithium memories`, a única ferramenta
          para resolver o problema. Com o pré-voo: WARNING nomeando o id, nada apagado,
          CLI intacta, recon indisponível com mensagem.
        * **`legacy_alter_table = ON`.** `ALTER TABLE ... RENAME` revalida o schema
          INTEIRO desde 3.25. MEDIDO na matriz completa (dropar as views de memória
          antes × pragma), num banco legado onde o passo 0 já dropou `hypotheses`::

              drop_views=True   legacy=True   -> OK
              drop_views=True   legacy=False  -> error in view hypothesis_scoreboard:
                                                 no such table: main.hypotheses
              drop_views=False  legacy=True   -> OK
              drop_views=False  legacy=False  -> error in view live_memories:
                                                 no such table: main.memories

          Ou seja: **o pragma sozinho basta e o DROP VIEW não é a peça que faz isto
          funcionar.** Havia um `DROP VIEW` das quatro views aqui, escrito em cima da
          segunda linha da matriz; a mutação que o remove matava **0 de 960** testes,
          porque com o pragma ligado ele nunca chega a importar. Removido — fiação sem
          consumidor é o defeito que este repo conta. Quem recria as views é o
          `executescript` do passo 2, que já faz `DROP VIEW IF EXISTS` + `CREATE VIEW`.
        * **`COALESCE(created_at, …)`.** O schema legado não tem default nessa coluna e
          o `NOT NULL` da nova recusa a linha.
        * **ROLLBACK ANTES de restaurar os pragmas.** `PRAGMA foreign_keys = ON` dentro
          de uma transação aberta é NO-OP silencioso: a checagem de FK ficaria
          desligada pelo resto da vida daquela conexão — e `Store` usa conexões
          thread-local de vida longa.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memories'"
        ).fetchone()
        if row is None:
            return False                      # banco novo: o CREATE TABLE já traz tudo
        if "'recon'" in (row["sql"] or ""):
            return False                      # já reconstruída

        holes = ", ".join("?" * len(MEMORY_KINDS))
        s_holes = ", ".join("?" * len(MEMORY_SOURCES))
        bad = conn.execute(
            f"SELECT id, kind, source FROM memories "
            f" WHERE kind NOT IN ({holes}) OR source NOT IN ({s_holes})",
            (*sorted(MEMORY_KINDS), *sorted(MEMORY_SOURCES)),
        ).fetchall()
        if bad:
            log.warning(
                "memória(s) %s carregam kind/source fora do vocabulário atual, então a "
                "reconstrução de `memories` foi ADIADA e este banco não aceita "
                "observações de recon. NADA foi apagado e o resto do sistema segue "
                "intacto. Resolva com `lithium memories --forget <id>`; a reconstrução "
                "acontece sozinha na próxima abertura.",
                ", ".join(f"#{r['id']}" for r in bad),
            )
            return False

        before = int(
            conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        )
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("PRAGMA legacy_alter_table = ON")
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    memories_ddl(
                        SCHEMA_PATH.read_text(encoding="utf-8"), table="memories_new"
                    )
                )
                new_cols = [
                    r["name"] for r in conn.execute("PRAGMA table_info(memories_new)")
                ]
                old_cols = {
                    r["name"] for r in conn.execute("PRAGMA table_info(memories)")
                }
                shared = [c for c in new_cols if c in old_cols]
                select = ", ".join(
                    "COALESCE(created_at, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
                    if c == "created_at" else c
                    for c in shared
                )
                conn.execute(
                    f"INSERT INTO memories_new({', '.join(shared)}) "
                    f"SELECT {select} FROM memories"
                )
                after = int(
                    conn.execute(
                        "SELECT COUNT(*) AS n FROM memories_new"
                    ).fetchone()["n"]
                )
                if after != before:
                    raise RuntimeError(
                        f"reconstrução de memories perderia linhas: {before} → {after}"
                    )
                conn.execute("DROP TABLE memories")
                conn.execute("ALTER TABLE memories_new RENAME TO memories")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")
        except Exception:  # noqa: BLE001
            # Levantar aqui mataria `init_schema()` e com ele TODO comando de CLI,
            # inclusive os de diagnóstico. Degrada e grita — mesma doutrina de
            # `_migrate_indexes`.
            log.exception(
                "a reconstrução de `memories` falhou e foi revertida; este banco segue "
                "sem aceitar observações de recon. Nada foi perdido."
            )
            return False
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA legacy_alter_table = OFF")
        log.info(
            "`memories` reconstruída: %d linha(s) preservada(s), `source='recon'` e "
            "`focus_id` disponíveis", before,
        )
        return True

    @staticmethod
    def hypotheses_are_globally_unique(conn: sqlite3.Connection) -> bool:
        """True quando o `UNIQUE(statement)` legado ainda está vivo — isto é, quando este
        banco NÃO suporta dois focos com o mesmo statement. Lido por `lithium focus`."""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'hypotheses'"
        ).fetchone()
        if row is None:
            return False
        return not re.search(
            r"unique\s*\(\s*focus_id\s*,\s*statement\s*\)", row["sql"] or "", re.I
        )

    # ───────────────────────────────────────────────────────────────── embeddings

    @staticmethod
    def pack_embedding(vector: Sequence[float]) -> bytes:
        """float32 little-endian — o formato que o sqlite-vec espera."""
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def unpack_embedding(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def set_chunk_embedding(self, chunk_id: int, vector: Sequence[float]) -> None:
        if len(vector) != self.embedding_dim:
            raise ValueError(
                f"embedding com {len(vector)} dims, esperado {self.embedding_dim}"
            )
        blob = self.pack_embedding(vector)
        if self.vec_available:
            # vec0 não suporta ON CONFLICT; delete-then-insert é o padrão.
            self.conn.execute("DELETE FROM chunk_vec WHERE chunk_id = ?", (chunk_id,))
            self.conn.execute(
                "INSERT INTO chunk_vec(chunk_id, embedding) VALUES(?, ?)", (chunk_id, blob)
            )
        else:
            self.conn.execute(
                "INSERT INTO chunk_vec(chunk_id, embedding) VALUES(?, ?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET embedding = excluded.embedding",
                (chunk_id, blob),
            )

    def search_vector(self, query: Sequence[float], k: int = 20) -> list[tuple[int, float]]:
        """Vizinhos mais próximos.

        Retorna [(chunk_id, distância de cosseno)] — `1 - cos`, menor = mais similar,
        faixa [0, 2]. A métrica é cosseno nos DOIS caminhos (vec0 declarado com
        `distance_metric=cosine`, fallback calculando à mão), para que um limiar
        calibrado valha independentemente de a extensão ter carregado.
        """
        if self.vec_available:
            rows = self.conn.execute(
                "SELECT chunk_id, distance FROM chunk_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (self.pack_embedding(query), k),
            ).fetchall()
            return [(r["chunk_id"], r["distance"]) for r in rows]

        rows = self.conn.execute("SELECT chunk_id, embedding FROM chunk_vec").fetchall()
        if not rows:
            return []
        ids = np.fromiter((r["chunk_id"] for r in rows), dtype=np.int64, count=len(rows))
        mat = np.stack([self.unpack_embedding(r["embedding"]) for r in rows])
        q = np.asarray(query, dtype=np.float32)
        denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
        # Chunk com embedding degenerado não deve derrubar a busca inteira.
        cos = np.divide(mat @ q, denom, out=np.zeros(len(mat), dtype=np.float32),
                        where=denom > 0)
        dist = 1.0 - cos
        top = np.argsort(dist)[:k]
        return [(int(ids[i]), float(dist[i])) for i in top]

    def search_text(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """BM25 via FTS5. Retorna [(chunk_id, score)], menor = mais relevante."""
        rows = self.conn.execute(
            "SELECT rowid, bm25(chunk_fts) AS score FROM chunk_fts "
            "WHERE chunk_fts MATCH ? ORDER BY score LIMIT ?",
            (query, k),
        ).fetchall()
        return [(r["rowid"], r["score"]) for r in rows]

    # ──────────────────────────────────────────────────────────────────── helpers

    def upsert_source(self, *, kind: str, external_id: str, raw: dict[str, Any],
                      **fields: Any) -> int:
        """Insere ou atualiza uma fonte, devolvendo o id. Idempotente por (kind, external_id)."""
        cols = {"kind": kind, "external_id": external_id, "raw_json": json.dumps(raw),
                **{k: v for k, v in fields.items() if v is not None}}
        names = ", ".join(cols)
        holes = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{k} = excluded.{k}" for k in cols if k not in ("kind", "external_id"))
        cur = self.conn.execute(
            f"INSERT INTO sources({names}) VALUES({holes}) "
            f"ON CONFLICT(kind, external_id) DO UPDATE SET {updates} RETURNING id",
            tuple(cols.values()),
        )
        return int(cur.fetchone()["id"])

    def add_chunk(self, *, source_id: int, ord: int, text: str,
                  section: str | None = None, n_tokens: int | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO chunks(source_id, ord, section, text, n_tokens) "
            "VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(source_id, ord) DO UPDATE SET "
            "  text = excluded.text, section = excluded.section, n_tokens = excluded.n_tokens "
            "RETURNING id",
            (source_id, ord, section, text, n_tokens),
        )
        return int(cur.fetchone()["id"])

    def counts(self) -> dict[str, int]:
        """Contagens para `lithium status`."""
        tables = ("sources", "chunks", "claims", "questions", "hypotheses",
                  "findings", "tasks", "training_examples")
        out: dict[str, int] = {}
        for t in tables:
            out[t] = int(self.conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"])
        out["claims_verified"] = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM claims WHERE verified = 1"
            ).fetchone()["n"]
        )
        # A peça que torna o fail-closed HONESTO. Sem este número, 400 claims verificadas
        # sem julgamento e um corpus vazio produzem exatamente o mesmo placar zerado.
        #
        # `claims_unweighted` fica como o TOTAL e continua derivado da própria view
        # (`NOT EXISTS`), portanto EXAUSTIVO por construção. Os três números abaixo o
        # DECOMPÕEM, e a decomposição é deliberadamente parcial: existe um quarto modo
        # (escala casando, aresta presente, e mesmo assim sem peso — `scale_levels`
        # incompleta num dos eixos) que não tem remédio de usuário. Trocar o total
        # exaustivo por uma soma de subqueries enumeradas à mão faria a claim desse
        # quarto modo sumir de TODOS os contadores.
        out["claims_unweighted"] = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM claims c WHERE c.verified = 1 "
                "  AND NOT EXISTS (SELECT 1 FROM claim_weight cw "
                "                   WHERE cw.claim_id = c.id)"
            ).fetchone()["n"]
        )
        # ESCALA DIVERGENTE, e é independente de haver aresta. O `IS NOT` (em vez de
        # `<>`) é o que pega `scale_id` NULL — com `<>` a claim de escala NULL sumiria
        # dos dois. Vem ANTES do relens de propósito: com a definição inversa ("com
        # aresta E escala divergente"), uma claim ainda não julgada contaria como
        # `unjudged`, o `status` mandaria o usuário rodar horas de GPU, e depois do
        # relens o número apenas trocaria de coluna com `claim_weight` ainda vazia.
        out["claims_off_scale"] = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM claims c JOIN active_focus af "
                " WHERE c.verified = 1 AND c.scale_id IS NOT af.scale_id"
            ).fetchone()["n"]
        )
        # SEM JULGAMENTO neste foco — e só ISTO o relens resolve.
        out["claims_unjudged"] = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM claims c JOIN active_focus af "
                " WHERE c.verified = 1 AND c.scale_id IS af.scale_id "
                "   AND NOT EXISTS (SELECT 1 FROM claim_directness cd "
                "                    WHERE cd.claim_id = c.id AND cd.focus_id = af.id)"
            ).fetchone()["n"]
        )
        # JULGADA E IRRELEVANTE: aresta gravada com `out_of_scope = 1`. Distinguível de
        # "não julgada" de propósito. Sem este terceiro veredito, o relens de um foco
        # novo importaria o corpus inteiro do foco velho com peso POSITIVO — o piso
        # `extrapolated` vale 0.12, não 0, então dez claims de outro domínio superam
        # uma meta-análise perfeitamente no alvo e a tabela de cobertura vira do
        # domínio antigo.
        out["claims_out_of_scope"] = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM claims c JOIN active_focus af "
                "  JOIN claim_directness cd ON cd.claim_id = c.id AND cd.focus_id = af.id "
                " WHERE c.verified = 1 AND cd.out_of_scope = 1"
            ).fetchone()["n"]
        )
        out["questions_escalated"] = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM questions WHERE status = 'ESCALATED'"
            ).fetchone()["n"]
        )
        out["tasks_pending"] = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE status = 'pending'"
            ).fetchone()["n"]
        )
        return out
