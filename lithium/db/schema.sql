-- lithium :: schema
--
-- Convenções:
--   * timestamps são TEXT ISO-8601 UTC (SQLite não tem tipo datetime nativo)
--   * enums são CHECK constraints — erro de pipeline falha na escrita, não silenciosamente
--   * {{EMBEDDING_DIM}} é substituído por store.py a partir da config
--
-- O peso de uma evidência é f(grade, directness), NÃO f(grade). Mas `directness` não é
-- propriedade intrínseca da claim: é a aderência da população estudada a um ALVO. Com um
-- alvo só, guardar o valor na claim funcionava e escondia a pergunta "julgado contra o
-- quê?". A partir daqui o julgamento é uma ARESTA (`claim_directness`) que nomeia o foco,
-- e a calibração é um DADO (`scale_levels`), não uma constante compilada.
--
-- `grade_weight` e `directness_weight` continuam existindo, com papel NOVO e menor: são o
-- VOCABULÁRIO — o conjunto fechado de níveis legais, aplicado por FK, para que um erro de
-- pipeline falhe na escrita. Elas NÃO são a calibração. As colunas `weight`/`rank` delas
-- não têm leitor, e `test_no_second_source_computes_the_weight` reprova quem voltar a
-- lê-las. Quem calibra é `scale_levels`, por escala.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- `meta` mora no TOPO porque `active_focus` faz subselect nela e `executescript` NÃO é
-- atômico com isolation_level=None. MEDIDO em SQLite 3.51.2: num script
-- `CREATE a; INSERT; CREATE b; <erro>; CREATE c`, a, b e o INSERT SOBREVIVEM e
-- `in_transaction` volta False. Se `meta` continuasse sendo a última tabela do arquivo,
-- um erro no meio deixaria um banco onde TODO consumidor de peso levanta
-- "no such table: main.meta". `meta` não depende de ninguém, então subir é grátis.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);


-- ─────────────────────────────────────────────────────────────── pesos de evidência

CREATE TABLE IF NOT EXISTS grade_weight (
    grade   TEXT PRIMARY KEY,
    weight  REAL NOT NULL,
    rank    INTEGER NOT NULL          -- ordem de força, para exibição
);

CREATE TABLE IF NOT EXISTS directness_weight (
    directness  TEXT PRIMARY KEY,
    weight      REAL NOT NULL,
    rank        INTEGER NOT NULL
);


-- ──────────────────────────────────────────────────────────── foco e escala
--
-- Duas tabelas e não uma porque mudam em RITMOS diferentes:
--
--   focuses         — o alvo clínico. Muda quando a pergunta muda.
--   evidence_scales — a régua de peso. Muda quando a calibração muda.
--
-- Um foco APONTA para uma escala; recalibrar é criar escala nova e foco novo, nunca
-- editar in-place. Editar in-place reescreveria em silêncio o peso de toda claim já
-- colhida, e o ranking de ontem deixaria de ser reproduzível sem uma linha de log.

CREATE TABLE IF NOT EXISTS evidence_scales (
    id          INTEGER PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    description TEXT
);

-- Os níveis de uma escala, nos dois eixos.
--
-- PK composta, não `id` autoincrementado. Duas linhas para o mesmo nível fariam
-- `claim_weight` multiplicar a claim por fan-out do JOIN, e `hypothesis_scoreboard` SOMA:
-- a hipótese subiria no placar por causa de uma linha duplicada, sem erro nenhum.
-- MEDIDO: com a PK, o segundo INSERT levanta IntegrityError; sem ela, o peso dobra.
--
-- `rank` REINICIA EM 1 dentro de cada (scale_id, axis) e é ORDEM DE FORÇA, não
-- calibração: ele é derivado da ordem de declaração do enum em types.py, então
-- `_seed_scale` o mantém em dia (DO UPDATE) enquanto CONGELA `weight` (DO NOTHING).
-- Nenhum consumidor traduz rank por POSIÇÃO — `state.py` decodifica rank→value lendo
-- esta mesma tabela, justamente para que renumerar não devolva o nível errado.
CREATE TABLE IF NOT EXISTS scale_levels (
    scale_id   INTEGER NOT NULL REFERENCES evidence_scales(id),
    axis       TEXT NOT NULL CHECK (axis IN ('grade', 'directness')),
    value      TEXT NOT NULL,
    weight     REAL NOT NULL,
    rank       INTEGER NOT NULL,
    PRIMARY KEY (scale_id, axis, value)
);

-- `profile_hash` congela a CALIBRAÇÃO (os pesos) vigente quando
-- o foco nasceu: `init_schema` compara e AVISA quando types.py divergiu, porque a
-- semeadura de `weight` é insert-if-absent e sem esse aviso a escala fica congelada para
-- sempre em silêncio — trocaríamos uma falha muda por outra.
CREATE TABLE IF NOT EXISTS focuses (
    id           INTEGER PRIMARY KEY,
    slug         TEXT NOT NULL UNIQUE,
    -- CHECK de comprimento: `TEXT NOT NULL` aceita '' no SQLite, e um alvo vazio
    -- renderiza literalmente `matches ""` no prompt de extração. Como o BANCO vence
    -- para `target`, corrigir o TOML depois não conserta nada — o relens já teria
    -- julgado o corpus inteiro contra a string vazia com instrução fail-closed.
    target       TEXT NOT NULL CHECK (length(trim(target)) > 0),
    scale_id     INTEGER NOT NULL REFERENCES evidence_scales(id),
    profile_hash TEXT,
    -- DUAS colunas, não um hash composto, e a razão é medida: `_seed_focus` compara
    -- `profile_hash` contra `Store.calibration_hash()` a CADA abertura de banco, ou
    -- seja em todo comando de CLI. Compor as duas metades faria todo banco da Fase A
    -- reportar "a calibração divergiu" no primeiro comando — prescrevendo o remédio
    -- mais destrutivo que existe (escala NOVA + foco NOVO, jogando fora o peso do
    -- corpus) para um evento em que NADA divergiu, só a fórmula do hash.
    --
    -- E as duas exigem respostas OPOSTAS: calibração divergiu -> escala nova;
    -- `target`/`directness_definitions` divergiram -> relens; strategies/taxonomy
    -- divergiram -> nada, query não é julgamento. Um hash só forçaria a mensagem a
    -- dar o conselho errado em dois dos três casos.
    --
    -- NULLABLE e NUNCA lido por `_seed_focus`: `init_schema()` continua sem nenhuma
    -- dependência de disco, que é a propriedade que faz `lithium focus` sobreviver a
    -- um focus.toml com erro de sintaxe — o único caminho de saída do usuário.
    judgment_hash TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    retired_at   TEXT
);

-- O foco ativo. Uma view e não uma coluna: o valor vive em `meta`, então trocar de foco é
-- uma escrita de uma linha e todo consumidor de peso segue junto sem alteração.
--
-- **Fail-closed em três direções, todas medidas:** valor não-numérico (CAST('abc') = 0),
-- id inexistente, e foco aposentado — nos três a view devolve ZERO linhas e
-- `claim_weight` fica vazia. (A quarta, chave ausente, é AUTO-CURADA por `_seed_focus`.)
-- Isso é o comportamento certo e é SILENCIOSO, então `init_schema` grita quando a view
-- está vazia e existem claims verificadas.
--
-- As colunas são ENUMERADAS, não `f.*`: `created_at` e `retired_at` ficam FORA da
-- projeção de propósito. `claim_weight` dá JOIN nesta view, e o que ela não expõe não
-- pode virar decaimento por idade lá dentro. `retired_at` no predicado é legítimo — é
-- `IS NULL`, não ordenação.
DROP VIEW IF EXISTS active_focus;
CREATE VIEW active_focus AS
SELECT f.id, f.slug, f.target, f.scale_id, f.profile_hash, f.judgment_hash
  FROM focuses f
 WHERE f.id = (SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'active_focus')
   AND f.retired_at IS NULL;


-- ─────────────────────────────────────────────────────────────── registro de fontes

-- Que fontes existem, e qual delas pode produzir EVIDÊNCIA.
--
-- Antes disto a lista era `EVIDENCE_KINDS = {PUBMED}` em Python mais um `CHECK (kind IN
-- (...))` com quatro literais no schema. Duas consequências, e as duas doem: uma quinta
-- fonte não conseguia nem ser NOMEADA (o CHECK rejeitava o INSERT), e a decisão "isto
-- vale como evidência" era uma allowlist de API em vez do contrato que ela deveria
-- expressar. As medições do item 9 continuam válidas para AQUELAS fontes naquele foco;
-- elas não justificam uma lista fixa e permanente.
--
-- `yields_evidence = 1` significa: publica estudo com prosa citável verbatim e desenho
-- graduável na escala do foco. Uma página web é 0 — e é por isso que o portão passa a
-- ser uma COLUNA consultada em runtime, não um conjunto compilado.
--
-- `credential_ref` guarda o NOME de uma variável de ambiente ou de uma chave em
-- `config.local.toml`, NUNCA o segredo. O movimento óbvio é uma coluna com a chave
-- dentro; `config.local.toml` está no `.gitignore` e uma coluna do banco não está
-- protegida por nada — e é o banco que o item 8 sincroniza para o Hugging Face.
CREATE TABLE IF NOT EXISTS sources_registry (
    id                     INTEGER PRIMARY KEY,
    slug                   TEXT NOT NULL UNIQUE,
    description            TEXT NOT NULL,
    base_url               TEXT NOT NULL,
    -- Como buscar e como buscar o registro completo. JSON e não colunas porque a forma
    -- varia por API (E-utilities usa `db=`+`term=`, OpenAlex usa `filter=`), e uma
    -- coluna por parâmetro viraria dezenas de NULLs.
    search_spec_json       TEXT,
    fetch_spec_json        TEXT,
    yields_evidence        INTEGER NOT NULL DEFAULT 0
                           CHECK (yields_evidence IN (0, 1)),
    credential_ref         TEXT,
    rate_per_s             REAL NOT NULL DEFAULT 1.0 CHECK (rate_per_s > 0),
    -- NULL = proposta pendente. O batedor da web (Fase C) propõe; você aprova. Uma
    -- fonte não aprovada não é montada em `Context.sources` e portanto não é consultada.
    approved_at            TEXT,
    -- De qual descoberta esta fonte nasceu, quando nasceu de uma. Sem FK de propósito:
    -- `discoveries` expira em 14 dias e a fonte aprovada tem de sobreviver à expiração.
    proposed_by_discovery   INTEGER,
    -- Parser dedicado, quando o genérico não serve. `pubmed` tem um porque o XML das
    -- E-utilities é irregular demais para spec declarativa; fonte nova usa o genérico.
    adapter                TEXT NOT NULL DEFAULT 'http'
                           CHECK (adapter IN ('http', 'pubmed'))
);

CREATE INDEX IF NOT EXISTS idx_registry_active
    ON sources_registry(approved_at, yields_evidence);

-- As fontes utilizáveis AGORA: aprovadas. É o que o daemon monta e o que o portão lê.
DROP VIEW IF EXISTS active_sources;
CREATE VIEW active_sources AS
SELECT slug, description, base_url, search_spec_json, fetch_spec_json,
       yields_evidence, credential_ref, rate_per_s, adapter
  FROM sources_registry
 WHERE approved_at IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────── fontes

CREATE TABLE IF NOT EXISTS sources (
    id              INTEGER PRIMARY KEY,
    -- FK para o registro, NÃO um CHECK de literais. O CHECK antigo tinha quatro valores
    -- fixos e impedia que uma fonte nova fosse sequer nomeada; a FK diz a mesma coisa que
    -- ele queria dizer ("kind é uma fonte conhecida") sem congelar QUAIS são.
    kind            TEXT NOT NULL REFERENCES sources_registry(slug),
    external_id     TEXT NOT NULL,          -- PMID / PMCID / NCT / SPL set id
    title           TEXT,
    year            INTEGER,
    journal         TEXT,
    doi             TEXT,
    url             TEXT,
    design          TEXT REFERENCES grade_weight(grade),
    sample_n        INTEGER,
    -- MENTIRA CORRIGIDA: o nome diz "população" e o comentário antigo dizia
    -- 'rótulo livre: "bipolar I + GAD"', mas `handlers.py` grava aqui o
    -- `expected_directness` da estratégia de busca ('indirect', 'extrapolated', ...).
    -- É write-only: nada em lithium/ lê esta coluna. O conserto certo é REMOVER, e
    -- remover exige rebuild de `sources`; por ora o comentário diz a verdade.
    population_tag  TEXT,
    -- MENTIRA CORRIGIDA (Fase B): o comentário antigo dizia "payload original da API,
    -- para reprocessar sem refetch" e isso é FALSO para a única fonte implementada.
    -- MEDIDO contra a fixture real (PMID 30712879): 374 bytes e três chaves —
    -- {pmid, publication_types, mesh}. NÃO tem abstract, título nem ano. O TEXTO mora
    -- em `chunks.text` (7.875 bytes para a mesma fonte), e é de lá que qualquer
    -- reprocessamento tem de partir. Um `--regrade` desenhado sobre esta coluna
    -- descobriria em runtime que não há o que reprocessar, e o conserto de emergência
    -- viraria refetch do corpus inteiro a 3 req/s.
    raw_json        TEXT NOT NULL,          -- metadados de indexação da API (NÃO o texto)
    fetched_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- Identidade de ARTIGO, não de linha. Com uma fonte só, `(kind, external_id)`
    -- bastava; com duas, o MESMO paper entra como duas linhas e chega ao juiz de
    -- suficiência como DUAS FONTES INDEPENDENTES CONCORDANDO — satisfazendo o piso de
    -- citações com um artigo só. Medido no item 9: 100% dos PMIDs que o PubMed colhe
    -- neste domínio também estão no Europe PMC.
    --
    -- GENERATED e não coluna comum: derivada não pode divergir da origem, e o SQLite
    -- recusa `ADD COLUMN ... GENERATED STORED`, então ela só existe em tabela criada do
    -- zero — que é exatamente o caminho do rebuild.
    article_key     TEXT NOT NULL GENERATED ALWAYS AS (
                        COALESCE(NULLIF(TRIM(LOWER(doi)), ''), kind || ':' || external_id)
                    ) VIRTUAL,
    UNIQUE (kind, external_id)
);

-- O índice de `article_key` NÃO mora aqui: `CREATE INDEX` VALIDA a coluna (ao
-- contrário de `CREATE VIEW`, que aceita coluna inexistente e só falha no uso), e num
-- banco legado POPULADO o rebuild de `sources` é adiado — a coluna não existe, o
-- índice levanta, e como `executescript` é atômico-por-script tudo que vem depois
-- deixa de ser aplicado, calado. Criado em `Store._migrate_source_index`, guardado.

CREATE INDEX IF NOT EXISTS idx_sources_design ON sources(design);
CREATE INDEX IF NOT EXISTS idx_sources_year   ON sources(year);


-- ─────────────────────────────────────────────────────────────────────── chunks

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,           -- posição no documento
    section     TEXT,                       -- abstract / methods / results / label section
    text        TEXT NOT NULL,
    n_tokens    INTEGER,
    UNIQUE (source_id, ord)
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);

-- A tabela de busca vetorial (`chunk_vec`) é criada por store.py, não aqui: a forma
-- dela depende de o `sqlite-vec` ter carregado (tabela vec0 nativa) ou não (tabela
-- comum + cosseno em numpy). Ver Store._init_vector_table.

-- Busca keyword (BM25). External-content: o texto vive em `chunks`, aqui só o índice.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text,
    content = 'chunks',
    content_rowid = 'id',
    tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunk_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunk_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunk_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;


-- ─────────────────────────────────────────────────────────────────────── claims
--
-- Uma claim só existe se for rastreável a chunks concretos. `chunk_ids` é JSON array
-- e `verified` só vira 1 depois que o passe verificador confirmou que o `statement`
-- é sustentado por aquele texto. Claim não verificada nunca entra em síntese,
-- relatório ou dataset de treino.

CREATE TABLE IF NOT EXISTS claims (
    id           INTEGER PRIMARY KEY,
    source_id    INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_ids    TEXT NOT NULL,             -- JSON array de chunk.id
    statement    TEXT NOT NULL,
    -- A citação VERBATIM que o portão 1 validou. Ela existia só em memória: usada
    -- para a checagem de ancoragem e para o prompt do portão 2, e descartada logo
    -- depois. `chunk_ids` aponta para o chunk INTEIRO — esta coluna aponta para a
    -- frase. Sem ela o portão 1 não é reverificável e o revisor não vê o trecho.
    supporting_quote TEXT,
    population   TEXT,
    intervention TEXT,
    comparator   TEXT,
    outcome      TEXT,
    direction    TEXT CHECK (direction IN ('positive', 'negative', 'null', 'mixed')),
    effect       TEXT,                      -- texto livre: "HR 0.62 (0.41–0.94)"
    grade        TEXT NOT NULL REFERENCES grade_weight(grade),
    -- Sob QUAL escala este `grade` foi atribuído. Não é redundante com o foco: recalibrar
    -- cria escala nova, e uma claim graduada sob a escala velha não pode ser pesada pelos
    -- números da nova. É a exclusão nº 1 de `claim_weight`, e ela é invisível hoje porque
    -- só existe uma escala.
    scale_id     INTEGER REFERENCES evidence_scales(id),
    confidence   REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    verified     INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    extracted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_claims_source     ON claims(source_id);
CREATE INDEX IF NOT EXISTS idx_claims_verified   ON claims(verified);
CREATE INDEX IF NOT EXISTS idx_claims_interv     ON claims(intervention);

-- O JULGAMENTO: quão bem a população desta claim adere ao alvo DESTE foco.
--
-- É a aresta que esta fase existe para criar. O valor é o mesmo que o LLM já produzia em
-- `claims.directness`; o que muda é que agora ele diz contra QUAL alvo foi julgado — e
-- que a mesma claim pode carregar julgamentos diferentes sob focos diferentes, que é
-- justamente o que uma coluna não consegue guardar.
--
-- **Por claim e não por população.** A tentação é `populations(text_key) ← julgamento`,
-- deduplicando `claims.population` entre fontes. MEDIDO como perda muda de evidência:
-- `claims.population` é texto livre de um 12B e as strings mais frequentes são genéricas
-- ("adults", "not specified"), então um estudo pré-clínico que escreve a mesma string que
-- uma meta-análise REBAIXA o peso dela — 1.00 → 0.12 — sem nenhum erro e sem alterar
-- `claims_unweighted`. Identidade de população exige julgamento de LLM, não igualdade de
-- string; enquanto isso não existir, a aresta é por claim e a semântica de hoje é
-- preservada bit a bit.
--
-- PK (claim_id, focus_id) pela mesma razão da PK de `scale_levels`: sem ela, dois
-- julgamentos do mesmo par multiplicam a claim em `claim_weight`.
--
-- FK para `directness_weight`: é exatamente a constraint que estava em
-- `claims.directness` e que muda de casa junto com o valor. Sem ela um typo ('directt')
-- não levanta nada — só deixa de casar o JOIN, e o fail-closed converte erro de escrita
-- em perda muda de evidência.
--
-- `judged_at` NÃO é dado de ranking, e é por isso que `claim_directness` entra em
-- `CLAIM_TABLES` no test_invariants: esta coluna é o próximo desempate tentador
-- ("julgamento mais recente primeiro") e recência de julgamento é o mesmo proxy de ordem
-- de colheita que a TRAVA 3 proíbe.
CREATE TABLE IF NOT EXISTS claim_directness (
    claim_id   INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    focus_id   INTEGER NOT NULL REFERENCES focuses(id),
    directness TEXT NOT NULL REFERENCES directness_weight(directness),
    judged_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    rationale  TEXT,
    -- O TERCEIRO veredito: "julgada e IRRELEVANTE neste foco". Existe porque os dois
    -- estados anteriores não conseguem dizê-lo. Ausência de aresta significa "ainda
    -- não julguei" e é o que `claims_unjudged` conta; um 5º nível de directness
    -- quebraria a FK acima e o StrEnum de 4 membros da gramática. Sem esta coluna, o
    -- relens de um foco novo importa o corpus inteiro do foco velho com peso POSITIVO
    -- (o piso `extrapolated` vale 0.12, não 0), e como a cobertura ordena por peso
    -- SOMADO, dez claims fora de domínio superam uma meta-análise no alvo.
    out_of_scope INTEGER NOT NULL DEFAULT 0 CHECK (out_of_scope IN (0, 1)),
    PRIMARY KEY (claim_id, focus_id)
);

CREATE INDEX IF NOT EXISTS idx_claim_directness_focus
    ON claim_directness(focus_id, directness);

-- Peso final de cada claim, pronto para agregação. Continua sendo a ÚNICA origem do peso
-- no sistema, e a fórmula continua sendo exatamente `grade × directness × confidence`.
--
-- O que mudou é de ONDE vêm os dois primeiros fatores. Três exclusões acontecem por JOIN,
-- sem nenhum filtro escrito, e as três significam a mesma coisa — "sem peso":
--
--   1. nenhum foco ativo                                     (JOIN active_focus)
--   2. claim graduada numa escala que não é a do foco ativo  (c.scale_id = af.scale_id)
--   3. claim não julgada para este foco                      (JOIN claim_directness)
--   4. claim julgada FORA DE ESCOPO neste foco                (cd.out_of_scope = 0)
--
-- Fail-closed é deliberado: peso zero é a resposta certa quando não se sabe contra quem a
-- evidência foi medida. Mas é SILENCIOSO, e por isso `Store.counts()` expõe
-- `claims_unweighted` e `lithium status` o mostra — sem esse número, "não julgamos nada"
-- lê exatamente igual a "não colhemos nada".
--
-- Os aliases `gw`/`dw` são deliberados e NÃO podem ser renomeados: eles são parte do
-- literal travado em `test_the_weight_expression_is_exactly_the_product_of_the_three_factors`
-- e do `.replace()` que injeta os CONSTRUCTED_DEFECTS. Com `g`/`d`, o replace vira no-op e
-- os três testes de defeito construído passam VERDES sem ter injetado defeito nenhum.
DROP VIEW IF EXISTS claim_weight;
CREATE VIEW claim_weight AS
SELECT c.id            AS claim_id,
       c.intervention,
       c.direction,
       gw.weight * dw.weight * c.confidence AS weight
  FROM claims c
  JOIN active_focus af
  JOIN scale_levels gw ON gw.scale_id = af.scale_id AND gw.axis = 'grade'
                      AND gw.value = c.grade AND c.scale_id = af.scale_id
  JOIN claim_directness cd ON cd.claim_id = c.id AND cd.focus_id = af.id
                          AND cd.out_of_scope = 0
  JOIN scale_levels dw ON dw.scale_id = af.scale_id AND dw.axis = 'directness'
                      AND dw.value = cd.directness
 WHERE c.verified = 1;


-- ────────────────────────────────────────────────────────────────────── perguntas

CREATE TABLE IF NOT EXISTS questions (
    id            INTEGER PRIMARY KEY,
    focus_id      INTEGER NOT NULL REFERENCES focuses(id),
    text          TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN (
                      'FACTUAL', 'SYNTHESIS', 'PREFERENCE', 'CONTEXT', 'METHODOLOGICAL')),
    status        TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN (
                      'OPEN', 'RESEARCHING', 'ANSWERED_AUTO', 'ESCALATED',
                      'ANSWERED_HUMAN', 'CLOSED')),
    priority      REAL NOT NULL DEFAULT 0.5,
    parent_id     INTEGER REFERENCES questions(id) ON DELETE SET NULL,
    origin        TEXT NOT NULL DEFAULT 'auto' CHECK (origin IN ('auto', 'human')),
    rounds        INTEGER NOT NULL DEFAULT 0,
    targets       TEXT,                     -- intervenção/tópico; escopo da deduplicação
    stuck_reason  TEXT CHECK (stuck_reason IN (
                      'NEEDS_CONTEXT', 'NEEDS_VALUE_JUDGMENT',
                      'IRRECONCILABLE_CONFLICT', 'INSUFFICIENT_EVIDENCE')),
    partial_work  TEXT,                     -- o que já achou / descartou, mostrado no Space
    answer        TEXT,
    answer_origin TEXT CHECK (answer_origin IN ('auto', 'human')),
    embedding     BLOB,                     -- para dedup semântico contra perguntas existentes
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    escalated_at  TEXT,
    answered_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_questions_focus
    ON questions(focus_id, status, priority DESC);

-- Teto da fila humana: o Space lê daqui, e o limite fica no consumidor, não aqui.
--
-- Escopada por foco AQUI, num ponto só: as cinco vagas de `human_queue_limit` são um
-- recurso do foco, não do banco. Com dois focos e a view global, perguntas do foco A
-- ocupam a fila e as do foco B ficam represadas em OPEN com `stuck_reason` para sempre,
-- sem log dizendo que foi por causa de outro foco.
DROP VIEW IF EXISTS escalated_queue;
CREATE VIEW escalated_queue AS
SELECT q.id, q.text, q.kind, q.priority, q.stuck_reason, q.partial_work, q.escalated_at
  FROM questions q
  JOIN active_focus af ON af.id = q.focus_id
 WHERE q.status = 'ESCALATED'
 ORDER BY q.priority DESC, q.escalated_at ASC;


-- ────────────────────────────────────────────────────────────────────── hipóteses

-- Duas trilhas, uma tabela. `tier` as separa:
--
--   'evidence'    — sustentada por claims verificadas, ordenada por peso de evidência
--   'speculative' — hipótese mecanística, ordenada por plausibilidade × ineditismo
--
-- Estarem na mesma tabela é o que permite a promoção `speculative → evidence` ser uma
-- transição de estado quando a evidência aparece, em vez de migração de dados. Mas os
-- rankings NUNCA se misturam num mesmo relatório: com peso de evidência, um candidato
-- novo (pré-clínico × extrapolado ≈ 0.01) some ao lado de qualquer RCT. Rankear as duas
-- juntas ou enterra o inédito, ou tira precedência do estabelecido.
CREATE TABLE IF NOT EXISTS hypotheses (
    id          INTEGER PRIMARY KEY,
    focus_id    INTEGER NOT NULL REFERENCES focuses(id),
    statement   TEXT NOT NULL,          -- a unicidade é POR FOCO, ver o UNIQUE no fim
    status      TEXT NOT NULL DEFAULT 'active' CHECK (status IN (
                    'active', 'supported', 'refuted', 'parked')),
    tier        TEXT NOT NULL DEFAULT 'evidence' CHECK (tier IN (
                    'evidence', 'speculative')),

    -- Campos só da trilha especulativa.
    intervention_class TEXT,
    mechanism_target   TEXT,
    route              TEXT,   -- via de entrega; mecanisticamente carregada, ver mechanism.ROUTES
    combination        TEXT,   -- componentes, quando a hipótese é uma associação
    pursued_at         TEXT,   -- quando gerou buscas dirigidas
    regrounded_at      TEXT,   -- quando a cadeia foi reavaliada contra o corpus atual
    chain_json         TEXT,   -- [{claim, status: supported|assumed, evidence}]
    falsifier          TEXT,   -- o que refutaria isto. Sem falsificador é prosa.
    test_proposal      TEXT,   -- próxima busca ou experimento concreto
    known_risks        TEXT,
    novelty            REAL CHECK (novelty IS NULL OR novelty BETWEEN 0 AND 1),
    critique_json      TEXT,
    survives_critique  INTEGER CHECK (survives_critique IN (0, 1)),

    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    -- "agonismo sigma-1 reduz ansiedade" não é específico de um alvo: dois focos podem
    -- legitimamente querer a mesma hipótese mecanística. Com UNIQUE global, o segundo
    -- recebe None, o Explorer conta como "já existia", ela some do speculation_board dele
    -- e `pending_pursuit` nunca a persegue. Nada no sistema reporta hipótese que NUNCA
    -- foi gerada — a falha não tem assinatura observável.
    UNIQUE (focus_id, statement)
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_tier ON hypotheses(tier, status);

CREATE TABLE IF NOT EXISTS evidence_links (
    claim_id      INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    polarity      TEXT NOT NULL CHECK (polarity IN ('support', 'contra')),
    weight        REAL NOT NULL DEFAULT 1.0,   -- modulador manual; o peso base vem de claim_weight
    PRIMARY KEY (claim_id, hypothesis_id)
);

-- O placar da trilha de EVIDÊNCIA. Ordenado por peso; é o que o relatório e a aba
-- "Hipóteses" do Space renderizam.
DROP VIEW IF EXISTS hypothesis_scoreboard;
CREATE VIEW hypothesis_scoreboard AS
SELECT h.id,
       -- Projetado para o leitor poder FILTRAR, espelhando `speculation_board`. Sem
       -- ele, `build_state` mostrava ao foco NOVO as hipóteses do foco VELHO, com o
       -- sufixo "N linked claim(s) unjudged in this focus" — um convite a relensar
       -- claims de uma hipótese que não é dele, no prompt que decide a agenda.
       h.focus_id,
       h.statement,
       h.status,
       COALESCE(SUM(CASE WHEN el.polarity = 'support'
                         THEN cw.weight * el.weight END), 0) AS support,
       COALESCE(SUM(CASE WHEN el.polarity = 'contra'
                         THEN cw.weight * el.weight END), 0) AS contra,
       COUNT(el.claim_id) AS n_claims,
       -- `n_claims` conta ELOS (fora do JOIN em claim_weight); `support`/`contra` somam
       -- PESOS (dentro dele). Antes desta fase as duas contagens sempre coincidiam. Com o
       -- fail-closed deixam de coincidir, e "n_claims = 3, support = 0.00" lê como
       -- REFUTAÇÃO quando o fato é AUSÊNCIA DE JULGAMENTO — e `state.render()` injeta isso
       -- no prompt do gerador de especulação. Esta coluna é o que separa as duas leituras,
       -- e ela é LIDA por state.py: sem leitor seria fiação para consumidor inexistente.
       SUM(CASE WHEN el.claim_id IS NOT NULL AND cw.claim_id IS NULL
                THEN 1 ELSE 0 END) AS n_unweighted,
       h.updated_at
  FROM hypotheses h
  LEFT JOIN evidence_links el ON el.hypothesis_id = h.id
  LEFT JOIN claim_weight   cw ON cw.claim_id      = el.claim_id
 WHERE h.tier = 'evidence'
 GROUP BY h.id;

-- O placar da trilha ESPECULATIVA. Critério diferente de propósito.
--
-- `plausibility` = fração de elos da cadeia que estão ancorados em evidência, não
-- assumidos. Uma cadeia de 5 elos com 4 citados vale mais que uma com 1 citado —
-- e isso é medível, ao contrário de "parece razoável".
--
-- A ordenação é plausibilidade × ineditismo: o mais plausível que ninguém testou.
-- Ordenar só por plausibilidade devolveria o óbvio; só por ineditismo, delírio.
DROP VIEW IF EXISTS speculation_board;
CREATE VIEW speculation_board AS
SELECT h.id,
       h.focus_id,   -- aditivo; é o que permite os leitores com teto escoparem por foco
       h.statement,
       h.status,
       h.intervention_class,
       h.mechanism_target,
       h.route,
       h.combination,
       h.pursued_at,
       h.regrounded_at,
       h.falsifier,
       h.test_proposal,
       h.known_risks,
       h.novelty,
       h.survives_critique,
       h.critique_json,   -- o elo mais fraco é o que o revisor quer ver primeiro
       json_array_length(COALESCE(h.chain_json, '[]')) AS chain_length,
       -- Um elo só conta como ancorado se `supported` for verdadeiro E houver
       -- citação. Marcar supported sem citar nada é o atalho óbvio para inflar a
       -- pontuação. Espelha exatamente `explore.plausibility` — test_pipeline_explore
       -- compara as duas implementações.
       (SELECT COUNT(*) FROM json_each(COALESCE(h.chain_json, '[]'))
         WHERE json_extract(value, '$.supported') = 1
           AND COALESCE(json_extract(value, '$.evidence'), '') <> '') AS supported_links,
       CASE WHEN json_array_length(COALESCE(h.chain_json, '[]')) = 0 THEN 0.0
            ELSE (SELECT COUNT(*) FROM json_each(h.chain_json)
                   WHERE json_extract(value, '$.supported') = 1
                     AND COALESCE(json_extract(value, '$.evidence'), '') <> '') * 1.0
                 / json_array_length(h.chain_json)
       END AS plausibility,
       h.updated_at
  FROM hypotheses h
 WHERE h.tier = 'speculative';


-- ─────────────────────────────────────────────────────────────────────── findings

CREATE TABLE IF NOT EXISTS findings (
    id            INTEGER PRIMARY KEY,
    question_id   INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    citations_json TEXT NOT NULL DEFAULT '[]',   -- [{claim_id, source_id, external_id}]
    critique_json  TEXT,                          -- veredito do passe adversarial
    safety_json    TEXT,                          -- alertas anexados por safety/check.py
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_findings_question ON findings(question_id);


-- ─────────────────────────────────────────────────────────────────────────── fila

CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                     'pending', 'running', 'done', 'failed', 'dead')),
    priority     REAL NOT NULL DEFAULT 0.5,
    attempts     INTEGER NOT NULL DEFAULT 0,
    -- 'scheduled' = o daemon decidiu sozinho; 'on_demand' = você pediu.
    -- Com o modo pesquisa desligado, os workers só reivindicam 'on_demand'.
    origin       TEXT NOT NULL DEFAULT 'on_demand' CHECK (origin IN (
                     'scheduled', 'on_demand')),
    dedup_key    TEXT UNIQUE,               -- evita enfileirar o mesmo trabalho duas vezes
    scheduled_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    claimed_at   TEXT,
    finished_at  TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Índice que sustenta o claim atômico do worker pool.
CREATE INDEX IF NOT EXISTS idx_tasks_claim
    ON tasks(status, scheduled_at, priority DESC);


-- ────────────────────────────────────────────────────────────────────── relatórios

CREATE TABLE IF NOT EXISTS reports (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);


-- ─────────────────────────────────────────────── medição dos portões de extração

-- O NÃO de cada portão, que antes não existia em lugar nenhum.
--
-- Todo portão do sistema gravava o SIM e jogava o NÃO num logger cujo único handler é
-- `RichHandler(console)` (`cli.py:_setup_logging`) — sem arquivo. As taxas dos portões
-- que motivaram esta tabela (219/247 e 186/219) só puderam ser medidas porque uma sessão
-- redirecionou stdout POR ACASO, e o arquivo estava em `/tmp`, que o macOS apaga no boot
-- — como de fato apagou o log das cinco primeiras fontes, 27 claims, 12,7% do corpus.
--
-- Sem isto, "os portões afrouxaram?" não tem resposta no banco. E portão afrouxando é o
-- mecanismo de acumular delírio, que é a tese do projeto. Ver METRICS.md, MF1.
CREATE TABLE IF NOT EXISTS extraction_runs (
    id            INTEGER PRIMARY KEY,
    source_id     INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    focus_id      INTEGER REFERENCES focuses(id),
    proposed      INTEGER NOT NULL,
    anchored      INTEGER NOT NULL,
    verified      INTEGER NOT NULL,
    chunks_seen   INTEGER NOT NULL,
    -- Propôs e perdeu tudo (o paper certo, a citação ruim) vs não propôs nada (o paper
    -- errado). Ações OPOSTAS — apertar o extrator vs trocar a frente de busca — e hoje
    -- as duas são a mesma ausência de linha em 76 dos 160 chunks.
    chunks_annihilated INTEGER NOT NULL DEFAULT 0,
    chunks_sterile     INTEGER NOT NULL DEFAULT 0,
    -- 'run' = medido de primeira mão. 'log' = retro-encaixado de um arquivo de log, e
    -- portanto de segunda mão. Nunca misturar os dois numa média sem dizer.
    origin        TEXT NOT NULL DEFAULT 'run' CHECK (origin IN ('run', 'log')),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_source ON extraction_runs(source_id);

-- Uma linha por claim que NÃO entrou. `gate` é o que separa "o gerador fabricou citação"
-- de "a citação era literal mas não sustentava" — dois defeitos diferentes, com consertos
-- diferentes, que a string formatada antiga misturava.
CREATE TABLE IF NOT EXISTS extraction_rejections (
    id         INTEGER PRIMARY KEY,
    run_id     INTEGER NOT NULL REFERENCES extraction_runs(id) ON DELETE CASCADE,
    chunk_id   INTEGER,
    gate       TEXT NOT NULL CHECK (gate IN ('anchor', 'entailment', 'error')),
    reason     TEXT NOT NULL,
    statement  TEXT,
    -- A citação REJEITADA. É o material para responder "o gerador está parafraseando ou
    -- está colando três palavras seguras?" — que MF1 não consegue distinguir só com taxa.
    quote      TEXT
);

CREATE INDEX IF NOT EXISTS idx_rejections_run ON extraction_rejections(run_id);

-- ──────────────────────────────────────────────────────────────── dataset de treino
--
-- Alimentado só por material verificado. Peso 3.0 nas respostas humanas: é o sinal
-- mais escasso do sistema. `question_id` existe para o split hold-out ser POR
-- PERGUNTA — separar por exemplo vazaria paráfrases entre treino e teste.

CREATE TABLE IF NOT EXISTS training_examples (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN (
                      'finding', 'human_answer', 'extraction', 'safety_probe')),
    question_id   INTEGER REFERENCES questions(id) ON DELETE SET NULL,
    messages_json TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    source_ref    TEXT,
    verified      INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_training_verified ON training_examples(verified, kind);


-- ────────────────────────────────────────────────────── custo de inferência

-- Uma linha por **POST HTTP**, não por chamada lógica. `structured()` permite
-- `repair_attempts=2`, e cada reparo *anexa* a saída ruim mais o erro à conversa —
-- então a tentativa 3 carrega o prompt inteiro mais dois rounds falhos. Contar por
-- chamada lógica esconderia exatamente o custo que mais dói.
--
-- `elapsed_ms` é obrigatório, não conveniência: a faixa medida de 45–70 s não pode
-- valer ao mesmo tempo para `verify_citation` (433 tokens de entrada, ~40 de saída) e
-- para `generate_speculation` (3072 de saída). Sem tempo por label, todo número de
-- wall-clock em qualquer estimativa é infalsificável.
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY,
    label             TEXT NOT NULL,
    attempt           INTEGER NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    elapsed_ms        INTEGER NOT NULL,
    finish_reason     TEXT,
    ok                INTEGER NOT NULL CHECK (ok IN (0, 1)),
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_label ON llm_calls(label, created_at);


-- ──────────────────────────────────────────────────────────────── conversa

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(id DESC);

-- Memórias: o que VOCÊ contou, não o que o sistema extraiu da literatura.
--
-- Deliberadamente separado de `claims`. Uma claim é uma afirmação sobre o mundo,
-- rastreável a um PMID e graduável por desenho de estudo. Uma memória é sobre você:
-- preferência, restrição, histórico, contexto do caso. Nada disso tem citação, e
-- misturar as duas contaminaria o peso de evidência com material não-publicado.
--
-- **Consentimento depende da origem**, por decisão explícita do usuário:
--
--   source='chat'     → PROPÕE e espera confirmação. É sobre você; acumular fatos
--                       sobre a pessoa sem autorização é diferente de acumular papers.
--   source='research' → grava sozinho. É o sistema aprendendo sobre o próprio
--                       trabalho, e pedir permissão para isso seria fricção sem ganho.
--
-- Duas categorias de aprendizado de pesquisa, com portões diferentes:
--
--   processo (`dead_end`, `search_lesson`, `source_lesson`) — sobre COMO pesquisar.
--       Não afirma nada sobre biologia, então não precisa de citação.
--
--   substantivo (`pattern`) — um padrão derivado sobre o domínio, tipo "aumentar tônus
--       glutamatérgico agudamente foi o modo de falha comum em três hipóteses".
--       Exige `provenance.claim_ids` apontando para claims VERIFICADAS, e a escrita
--       valida que existem. Mesma disciplina da reancoragem, que só aceita PMID
--       presente na evidência recuperada.
--
-- Sem o portão, uma memória "sigma-1 não funciona em bipolar" entraria sem fonte e
-- seria injetada em todo prompt seguinte — o sistema ensinando as próprias suposições
-- a si mesmo, com nenhum portão pegando. É a falha que a extração previne, entrando
-- pela porta de trás.
-- Uma QUARTA origem entra na Fase C, e ela é do PRIMEIRO regime:
--
--   source='recon'    → PROPÕE e espera confirmação, igual a 'chat'. Não é o sistema
--                       aprendendo sobre o próprio trabalho; é conteúdo EXTERNO, lido
--                       na web aberta, entrando no que ele passa a acreditar. A lição
--                       de uma REJEIÇÃO continua indo com `source='research'`, por
--                       razão mecânica: `lessons()` lê a view `research_lessons`, e
--                       uma lição gravada como 'recon' seria invisível para a máquina
--                       de lições inteira.
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN (
                   'preference', 'context', 'constraint', 'fact',
                   -- aprendizado de pesquisa sobre o PROCESSO
                   'dead_end', 'search_lesson', 'source_lesson',
                   -- padrão SUBSTANTIVO derivado; exige claim_ids em `provenance`
                   'pattern')),
    rationale  TEXT,                      -- por que o sistema achou que valia guardar
    source     TEXT NOT NULL DEFAULT 'chat' CHECK (source IN (
                   'chat', 'answer', 'manual', 'research', 'recon')),
    provenance TEXT,                      -- JSON: o evento do banco que gerou isto
    confirmed  INTEGER NOT NULL DEFAULT 0 CHECK (confirmed IN (0, 1)),
    active     INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    embedding  BLOB,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    confirmed_at TEXT,
    -- Chave de deduplicação de lições de pesquisa. Normalizada em Python (ver
    -- `normalize_memory_text`), NUNCA em SQL: `SELECT lower('LIÇÃO')` devolve
    -- 'liÇÃo' — o `lower()` do SQLite e o `COLLATE NOCASE` são ASCII-only, e as
    -- lições deste sistema são em português.
    text_key   TEXT,
    -- NULL = global. Memória sobre a PESSOA atravessa focos (é sobre ela, e ela não
    -- muda quando a lente muda). Uma observação lida na web sob `bipolar-tag` é
    -- específica da LENTE: injetá-la depois de `focus --use onco-…` é ruído e, no pior
    -- caso, contexto enganoso. Por isso o escopo é obrigatório só para 'recon'.
    focus_id   INTEGER,
    CHECK (source <> 'recon' OR focus_id IS NOT NULL),
    -- Uma observação da web NUNCA pode ser 'constraint' nem 'preference'. Sem este
    -- CHECK, se alguém um dia alargar `user_memories`, `_constraint_notes` passa a
    -- emitir "colide com a restrição DECLARADA: <prosa de blog>" — o sistema
    -- atribuindo a VOCÊ uma restrição que veio de uma página web, no único bloco que
    -- existe para o modelo não omitir evidência em silêncio.
    CHECK (source <> 'recon' OR kind = 'fact')
);

CREATE INDEX IF NOT EXISTS idx_memories_live ON memories(confirmed, active, kind);

-- O índice ÚNICO sobre `text_key` NÃO mora aqui, de propósito: um IntegrityError num
-- banco com duplicatas abortaria o `executescript` e tudo que viesse DEPOIS dele no
-- arquivo deixaria de ser aplicado — inclusive as views logo abaixo. (O que NÃO acontece
-- é rollback: MEDIDO em 3.51.2, `executescript` com isolation_level=None NÃO é atômico e
-- os statements anteriores ao erro persistem. Este comentário afirmava o contrário.)
-- Ele é criado por `Store._migrate_indexes`, guardado e reversível. Ver o docstring de lá.
--
-- Segundo motivo, independente: o regex de
-- `test_memory_consent::test_schema_sql_does_not_create_a_unique_index_on_memories` usa
-- `[\s\S]*?` ilimitado e atravessa o arquivo até o `ON memories` de `idx_memories_live`.
-- Um `CREATE UNIQUE INDEX` em QUALQUER ponto acima dele reprova o teste com uma mensagem
-- que aponta para o lugar errado. Por isso `focuses.slug` usa `UNIQUE` inline.

DROP VIEW IF EXISTS live_memories;
CREATE VIEW live_memories AS
SELECT id, text, kind, source, provenance, focus_id, created_at
  FROM memories
 WHERE confirmed = 1 AND active = 1
 ORDER BY id;

-- Separadas nas leituras porque servem a prompts diferentes: o que o sistema sabe
-- sobre VOCÊ entra no chat e no gerador de perguntas; o que ele aprendeu sobre o
-- PRÓPRIO TRABALHO entra na geração de especulações e no planejamento de buscas.
DROP VIEW IF EXISTS user_memories;
CREATE VIEW user_memories AS
SELECT id, text, kind, created_at FROM live_memories
 WHERE source IN ('chat', 'answer', 'manual');

DROP VIEW IF EXISTS research_lessons;
CREATE VIEW research_lessons AS
SELECT id, text, kind, provenance, created_at FROM live_memories
 WHERE source = 'research';

-- A TERCEIRA view de leitura, e ela existe porque as duas alternativas foram MEDIDAS e
-- são piores. Pôr 'recon' dentro de `user_memories` injetaria texto lido na web aberta
-- sob o cabeçalho literal "## What you know about this user" (chat.py), o faria aparecer
-- na fila humana sob "o que você já me contou" (question.py) e o poria no bloco
-- $existing de detect_memory.md sob "Already remembered — do not propose again",
-- suprimindo propostas legítimas SUAS: três afirmações falsas de proveniência de uma
-- vez. Deixá-la fora de toda view grava um consentimento que ninguém lê.
--
-- FAIL-CLOSED por construção: sem foco ativo o subselect é NULL, `focus_id = NULL` é
-- NULL, e a view devolve zero linhas — a mesma doutrina de `claim_weight`.
DROP VIEW IF EXISTS recon_memories;
CREATE VIEW recon_memories AS
SELECT id, text, provenance, created_at FROM live_memories
 WHERE source = 'recon'
   AND focus_id = (SELECT id FROM active_focus);

-- Recusadas. `confirmed = 0 AND active = 0` é a assinatura exata de `decline()` e de
-- mais nada: `remember()` grava (1, 1), `Reflector.remember()` grava (1, 1), e
-- `forget()` deixa (1, 0). A distinção entre (0,0) e (1,0) é o que separa "você
-- recusou isto" de "isto foi esquecido" — sem ela, esquecer viraria suprimir a
-- proposta para sempre.
DROP VIEW IF EXISTS declined_memories;
CREATE VIEW declined_memories AS
SELECT id, text, kind, text_key, created_at FROM memories
 WHERE confirmed = 0 AND active = 0
 ORDER BY id;


-- ──────────────────────────────────────────────────────────── reconhecimento (web)
--
-- O canal de RECONHECIMENTO. **Não é o canal de evidência e não tem aresta para ele:**
-- a única FK desta tabela é `focuses` (verificável com `PRAGMA foreign_key_list`).
--
-- A regra que é a espinha da fase: UMA DESCOBERTA NUNCA VIRA CLAIM. Prosa da web aberta
-- não tem desenho graduável nem identidade citável estável. Ela pode APONTAR para um
-- artigo — e aí o caminho é: descoberta → você autoriza → o ARTIGO é recolhido do
-- PubMed e passa pelos dois portões de extração como qualquer outro.
--
-- A ÚNICA coluna que o canal de evidência lê é `lead_external_id`, e os CHECKs de FORMA
-- abaixo são o portão da fase inteira: uma URL, uma frase ou um resumo NÃO CABEM ali.
-- Para levar prosa da web ao canal de evidência é preciso EDITAR UM CHECK NESTE
-- ARQUIVO — um ato visível e diffável, não um esquecimento.
--
-- `title`/`summary`/`url`/`payload_json` — os campos que carregam prosa — não têm
-- caminho nenhum para lá.
CREATE TABLE IF NOT EXISTS discoveries (
    id           INTEGER PRIMARY KEY,
    focus_id     INTEGER NOT NULL REFERENCES focuses(id),
    kind         TEXT NOT NULL CHECK (kind IN ('lead', 'source', 'observation')),
    -- pending  → esperando VOCÊ. É o único estado que o notify anuncia e o único do
    --            qual se pode sair por decisão. Nada autônomo tira uma descoberta daqui.
    -- queued   → você aprovou um lead; o ARTIGO foi enfileirado (não o texto).
    -- deferred → você aprovou um source; não há registro de fontes até a Fase D.
    -- approved → você aprovou uma observation; a memória 'recon' existe.
    -- rejected → você recusou; lead/source viraram lição de pesquisa.
    -- expired  → dias sem decisão. É a ÚNICA transição automática, e é uma
    --            NÃO-decisão sua virando terminal.
    status       TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                     'pending', 'queued', 'deferred',
                     'approved', 'rejected', 'expired')),
    query        TEXT NOT NULL,           -- a busca que a produziu
    url          TEXT NOT NULL,           -- SEMPRE da resposta da API, nunca do LLM
    title        TEXT NOT NULL,
    summary      TEXT NOT NULL,
    -- A PONTE. Só o `lead` a tem, e ela carrega IDENTIFICADOR, nunca texto. O valor é
    -- DERIVADO EM CÓDIGO da URL do hit (ver `lithium/recon/leads.py`), nunca emitido
    -- pelo modelo: um 12B a quem se pede um PMID troca um dígito, e um PMID vizinho é
    -- quase sempre um artigo real e sem relação — a autorização iria para um artigo e
    -- o corpus receberia outro, sem nada ligando as duas linhas.
    lead_kind        TEXT CHECK (lead_kind IS NULL OR lead_kind IN ('pubmed', 'doi')),
    lead_external_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',   -- snippet, domínio, truncated, http
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    decided_at   TEXT,
    CHECK (kind <> 'lead' OR (lead_kind IS NOT NULL AND lead_external_id IS NOT NULL)),
    CHECK (kind =  'lead' OR (lead_kind IS NULL AND lead_external_id IS NULL)),
    -- O PORTÃO, em forma de gramática. INCONDICIONAL primeiro: nada de espaço, tab,
    -- quebra de linha ou byte não-ASCII-imprimível em NENHUM ramo. Sem esta linha o
    -- ramo DOI aceitaria ~190 caracteres de prosa arbitrária depois do prefixo, e a
    -- garantia declarada acima seria falsa para metade da ponte (MEDIDO).
    CHECK (lead_external_id IS NULL OR (lead_external_id NOT GLOB '*[^!-~]*'
                                        AND length(lead_external_id) BETWEEN 1 AND 100)),
    CHECK (lead_kind <> 'pubmed' OR (lead_external_id NOT GLOB '*[^0-9]*'
                                     AND length(lead_external_id) BETWEEN 1 AND 9)),
    -- DOI: prefixo `10.NNNN`, uma barra com pelo menos um caractere depois, e um
    -- vocabulário de caractere fechado — `<` e `>` ficam de fora de propósito.
    CHECK (lead_kind <> 'doi' OR (lead_external_id GLOB '10.[0-9][0-9][0-9][0-9]*/?*'
                                  AND lead_external_id NOT GLOB '*[^0-9A-Za-z./_():;+-]*'
                                  AND length(lead_external_id) BETWEEN 8 AND 100))
);

-- Uma página, uma descoberta, por foco. É o "não me mostre isto de novo" — e ele mora
-- AQUI, não em `declined_memories`, porque `declined_keys()` é global por TEXTO e
-- recusar uma leitura da web suprimiria uma proposta de chat idêntica.
--
-- **PARCIAL, excluindo `expired`**, e a exclusão não é detalhe: com o UNIQUE total, uma
-- descoberta que ninguém decidiu por estar de férias vira `expired` e aquela URL NUNCA
-- MAIS é proposta naquele foco — sem log, sem toast, sem linha em `lithium
-- discoveries`. Para um `lead`, isso é perder um artigo do corpus por ter estado
-- ocupado. Supressão permanente fica só para a decisão DELIBERADA.
--
-- Índice e não `UNIQUE` inline por causa da forma PARCIAL, que a sintaxe de coluna não
-- expressa. Ele mora DEPOIS do último `ON memories` do arquivo de propósito: o regex de
-- `test_memory_consent::test_schema_sql_does_not_create_a_unique_index_on_memories` usa
-- `[\s\S]*?` ilimitado e casaria um `CREATE UNIQUE INDEX` colocado ACIMA de
-- `idx_memories_live`, reprovando com uma mensagem que aponta para o lugar errado.
CREATE UNIQUE INDEX IF NOT EXISTS idx_discoveries_url
    ON discoveries(focus_id, url) WHERE status <> 'expired';

CREATE INDEX IF NOT EXISTS idx_discoveries_pending
    ON discoveries(focus_id, status, id);


-- Consumo de recon, DURÁVEL. Não pode ser o `RateLimiter`: ele guarda `_tokens` em
-- estado de INSTÂNCIA e zera a cada restart do daemon — sob launchd, um processo que
-- reinicia em laço zera a cota a cada subida, que é exatamente o cenário em que ela é a
-- única proteção (uso pessoal, ninguém olhando dashboard, chave com fatura).
--
-- TRÊS colunas, não uma. `search_calls` é o que TEM FATURA e é a que tem teto duro.
-- `page_fetches` conta LEITURA de página. `robots_fetches` conta robots.txt —
-- separado porque conflatá-lo com a leitura faria o teto de páginas ser consumido por
-- requisições que não leram nada, e o número que o usuário lê como "quanto isto me
-- custou" ser mentira nas duas direções.
--
-- Debitado ANTES da requisição e NUNCA estornado: `queue.fail` repete até 3 vezes, e
-- uma falha DEPOIS de o provedor responder (parse, timeout de leitura) gasta cota que
-- um contador de sucessos registraria como zero.
CREATE TABLE IF NOT EXISTS recon_budget (
    day            TEXT PRIMARY KEY,       -- 'YYYY-MM-DD' UTC
    search_calls   INTEGER NOT NULL DEFAULT 0,
    page_fetches   INTEGER NOT NULL DEFAULT 0,
    robots_fetches INTEGER NOT NULL DEFAULT 0
);


-- ───────────────────────────────────────────────────────────────────────── sync

CREATE TABLE IF NOT EXISTS sync_state (
    key             TEXT PRIMARY KEY,
    last_local_rev  TEXT,
    last_remote_rev TEXT,
    synced_at       TEXT
);

-- (`meta` está no topo do arquivo; ver a nota lá.)
