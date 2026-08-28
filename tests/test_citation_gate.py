"""O portão de citação da trilha especulativa.

**O achado que motivou esta fase, medido em operação real.** Com gemma-4-12b, 9 de 12
elos de cadeia vieram marcados `supported` — cada um com um PMID **verdadeiro do PubMed**
que não tem relação nenhuma com a alegação. Os títulos reais dos nove, consultados na
NCBI:

    28544150  matriz dérmica acelular humana em feridas crônicas
    25115112  microarray de retrovírus endógeno suíno
    23731151  clado sul-africano de peixes-cachimbo costeiros (Syngnathus spp.)
    24141515  doenças priônicas
    30333051  exposição à luz do dia e comunidades bacterianas em poeira doméstica
    21441130  osteopenia em homens com litíase renal
    28155044  amiloidose cardíaca por transtirretina e estenose aórtica
    28214153  displasia arritmogênica de ventrículo direito
    26346444  composto antimalárico

Nenhum sobre sigma-1, cetose, ansiedade ou bipolar. E `plausibility` reportava **0,75**
para as três hipóteses — o número que o quadro mostra ao psiquiatra, ao lado de
`[supported: PMID:xxx]`.

Alucinar um identificador *plausível e real* é pior que inventar um: ele sobrevive a
qualquer checagem de formato e só cai contra o corpus. Por isso as três decisões desta
fase são as que a medição impôs, não as óbvias:

* **formato não basta** — pega 0 de 9;
* **formato estrito apaga a trilha** — `^PMID:\\d+$` rejeita 9 de 9, porque o modelo
  escreve `PMID: 28544150` com espaço;
* **existência no corpus pega 9 de 9**, por 10,5 µs de query indexada.
"""

from __future__ import annotations

import json

import pytest

from lithium.db import Store
from lithium.pipeline.explore import (

    Explorer,
    canonical_pmid,
    gate_citations,
    plausibility,
)

from conftest import onco_profile

PROFILE = onco_profile()


# Os nove PMIDs reais que o modelo citou em produção, com o assunto verdadeiro deles.
HALLUCINATED = {
    "28544150": "matriz dérmica acelular em feridas crônicas",
    "23731151": "peixes-cachimbo costeiros sul-africanos",
    "30333051": "poeira doméstica e luz do dia",
    "28155044": "amiloidose cardíaca por transtirretina",
    "26346444": "composto antimalárico",
}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "g.db", embedding_dim=8)
    s.init_schema()
    yield s
    s.close()


def _seed(store, pmid: str) -> None:
    store.upsert_source(kind="pubmed", external_id=pmid, raw={}, title="no corpus")


def _step(claim: str, *, supported: bool, evidence: str = "") -> dict:
    return {"claim": claim, "supported": supported, "evidence": evidence}


# ────────────────────────────────────── extrair o PMID do que o modelo escreve


@pytest.mark.parametrize("written,expected", [
    ("PMID: 28544150", "28544150"),      # a forma real, COM espaço
    ("PMID:28544150", "28544150"),
    ("pmid 28544150", "28544150"),
    ("28544150", "28544150"),
    ("PMID:1", "1"),                     # PMIDs antigos são curtos
    ("well established", None),
    ("", None),
    (None, None),
])
def test_the_pmid_is_extracted_from_what_the_model_actually_writes(written, expected):
    """Tolerante por medição, não por preguiça.

    Um regex estrito rejeitaria **9 de 9** citações reais, porque o modelo escreve
    `PMID: 28544150` com espaço depois do dois-pontos, sempre. A segurança não vem daqui
    — vem da existência no corpus.
    """
    assert canonical_pmid(written) == expected


# ────────────────────────────────────────────── o portão: rebaixa, não rejeita


def test_a_hallucinated_but_real_pmid_is_refused():
    """O caso exato de produção: PMID verdadeiro, assunto errado, fora do corpus."""
    chain = [_step("sigma-1 reduz ansiedade", supported=True, evidence="PMID: 28544150")]
    gated, refused = gate_citations(chain, known=set())

    assert refused == 1
    assert gated[0]["supported"] is False
    assert gated[0]["citation_refused"] == "PMID: 28544150", (
        "o que foi recusado tem que ficar registrado — sumir em silêncio é pior"
    )


def test_a_citation_that_exists_in_the_corpus_survives():
    chain = [_step("x", supported=True, evidence="PMID: 28544150")]
    gated, refused = gate_citations(chain, known={"28544150"})

    assert refused == 0
    assert gated[0]["supported"] is True
    assert gated[0]["evidence"] == "PMID:28544150", "a citação é canonizada"


def test_an_assumed_link_is_left_alone():
    """O portão julga citação, não mérito. Elo já assumido não tem o que recusar."""
    chain = [_step("x", supported=False, evidence="")]
    gated, refused = gate_citations(chain, known=set())

    assert refused == 0
    assert gated[0]["supported"] is False
    assert "citation_refused" not in gated[0]


def test_prose_instead_of_a_citation_is_refused():
    chain = [_step("x", supported=True, evidence="well established in the literature")]
    _, refused = gate_citations(chain, known={"28544150"})
    assert refused == 1


def test_the_gate_never_promotes():
    """Ele só pode rebaixar. Um elo assumido não vira ancorado por passar aqui."""
    chain = [_step("a", supported=False, evidence="PMID:28544150")]
    gated, _ = gate_citations(chain, known={"28544150"})
    assert gated[0]["supported"] is False


def test_plausibility_falls_to_the_honest_number():
    """A consequência medida: 0,75 → 0,00 nas três hipóteses reais.

    Não é "metade dos elos rebaixada" — é todos. O número anterior era inflado, e o
    quadro do psiquiatra o mostrava como se fosse ancoragem.
    """
    chain = [
        _step(f"elo {i}", supported=True, evidence=f"PMID: {pmid}")
        for i, pmid in enumerate(HALLUCINATED)
    ] + [_step("elo assumido", supported=False)]

    assert plausibility(chain) == pytest.approx(5 / 6)
    gated, refused = gate_citations(chain, known=set())
    assert refused == 5
    assert plausibility(gated) == 0.0


# ───────────────────────────────────── a ligação: o portão roda na gravação


async def test_the_gate_runs_when_a_speculation_is_stored(store):
    """Wiring. Reverter a chamada em `evaluate_and_store` deixa a inflação passar
    inteira — e é a quinta vez que esta classe aparece no projeto."""
    from tests.test_pipeline_explore import ScriptedLLM, _batch, _ok_critique, _spec

    chain = [
        {"claim": "a", "supported": True, "evidence": "PMID: 28544150"},
        {"claim": "b", "supported": True, "evidence": "PMID: 23731151"},
    ]
    llm = ScriptedLLM([_batch(_spec(statement="hipótese", chain=chain))],
                      [_ok_critique()])
    [record] = await Explorer(store, llm, profile=PROFILE).generate(max_items=1)

    assert record.plausibility == 0.0, (
        "a cadeia foi gravada com plausibilidade inflada: o portão não rodou"
    )
    stored = json.loads(store.conn.execute(
        "SELECT chain_json FROM hypotheses WHERE id = ?", (record.id,)
    ).fetchone()["chain_json"])
    assert all(not s["supported"] for s in stored)
    assert all("citation_refused" in s for s in stored)


async def test_a_real_citation_survives_the_whole_path(store):
    """A contrapartida: o portão não pode ser "recusa tudo". Sem este teste, um portão
    quebrado que zera toda cadeia passaria com o teste acima verde."""
    from tests.test_pipeline_explore import ScriptedLLM, _batch, _ok_critique, _spec

    _seed(store, "28544150")
    _seed(store, "23731151")
    chain = [{"claim": "a", "supported": True, "evidence": "PMID: 28544150"},
             {"claim": "b", "supported": True, "evidence": "PMID: 23731151"}]
    llm = ScriptedLLM([_batch(_spec(statement="hipótese", chain=chain))],
                      [_ok_critique()])
    [record] = await Explorer(store, llm, profile=PROFILE).generate(max_items=1)

    assert record.plausibility == 1.0


async def test_the_hypothesis_survives_a_refused_citation(store):
    """Rebaixar, não rejeitar. Medido: rejeitar mataria 3 de 3 das hipóteses reais, e
    os portões que julgam mérito já existem (falsificador, crítica adversarial)."""
    from tests.test_pipeline_explore import ScriptedLLM, _batch, _ok_critique, _spec

    chain = [{"claim": "a", "supported": True, "evidence": "PMID: 28544150"},
             {"claim": "b", "supported": False, "evidence": ""}]
    llm = ScriptedLLM([_batch(_spec(statement="sobrevive", chain=chain))],
                      [_ok_critique()])
    records = await Explorer(store, llm, profile=PROFILE).generate(max_items=1)

    assert len(records) == 1
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM hypotheses"
    ).fetchone()["n"] == 1


def test_the_corpus_lookup_asks_for_the_right_index(store):
    """`kind` na cláusula não é decoração: o índice é `UNIQUE(kind, external_id)`, e sem
    a primeira coluna o SQLite troca SEARCH por SCAN."""
    _seed(store, "28544150")
    explorer = Explorer(store, None, profile=PROFILE)
    chain = [_step("a", supported=True, evidence="PMID: 28544150"),
             _step("b", supported=True, evidence="PMID: 99999999")]

    assert explorer.corpus_pmids(chain) == {"28544150"}

    plan = store.conn.execute(
        "EXPLAIN QUERY PLAN SELECT external_id FROM sources "
        " WHERE kind = 'pubmed' AND external_id IN ('1')"
    ).fetchall()
    assert any("SEARCH" in str(r["detail"]) for r in plan), (
        f"a consulta virou SCAN: {[str(r['detail']) for r in plan]}"
    )


def test_a_chain_with_no_citations_costs_no_query(store):
    explorer = Explorer(store, None, profile=PROFILE)
    assert explorer.corpus_pmids([_step("a", supported=False)]) == set()


# ═══════════════════════════ `seeking`: o campo que era gerado e descartado


def test_seeking_survives_into_the_task_payload():
    """O ponto único de perda era o call site, não o pipeline.

    `plan_queries` devolvia `SourceQuery` com `seeking` intacto; quem descartava era o
    dict do payload em `pursue_speculation`. Sem ele, a lição que sai de uma busca
    estéril é "esta query não retornou nada" — verdadeira e inútil.
    """
    import inspect

    from lithium.worker import handlers

    body = inspect.getsource(handlers.pursue_speculation)
    assert '"seeking"' in body, "o payload voltou a descartar o elo buscado"


def test_the_seeking_label_carries_no_bracket_or_bare_digit():
    """A colisão de namespace que a Fase 2 fechou, defendida aqui também.

    O bloco de atividade usa `[k]` como espaço de referências, e um número renderizado
    ali resolve COM SUCESSO para o objeto errado — pior que não resolver. `seeking` é
    string livre, e "elo 3" põe um dígito exatamente nessa posição.
    """
    from lithium.pipeline.reflect import _plain

    assert _plain("ancorar o elo 3 [sigma-1]") == "ancorar o elo sigma-1"
    assert _plain("sigma-1 agonism -> anxiolysis") == "sigma-1 agonism -> anxiolysis"
    assert "[" not in _plain("[MeSH] termo")
    assert _plain(None) == ""


def test_the_seeking_label_is_capped():
    from lithium.pipeline.reflect import _plain

    assert len(_plain("palavra " * 80)) <= 90


async def test_the_reflection_block_shows_what_the_search_was_anchoring(store):
    """A ligação: o campo no payload precisa CHEGAR ao prompt de reflexão."""
    from lithium.pipeline.reflect import Reflector

    class NullEmbedder:
        async def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

    store.conn.execute(
        "INSERT INTO tasks(kind, payload_json, status, priority) "
        "VALUES('harvest_query', ?, 'done', 0.5)",
        (json.dumps({
            "query": "sigma-1 receptor anxiolysis human trial",
            "label": "spec:41",
            "seeking": "sigma-1 agonism -> anxiolysis in humans",
        }),),
    )
    activity = Reflector(store, None, NullEmbedder()).recent_activity()

    assert "aiming to anchor: sigma-1 agonism -> anxiolysis in humans" in activity.text
    assert "spec:41" not in activity.text, (
        "o id real da hipótese voltou ao prompt de índices locais"
    )


# ════════════════════ a ordenação da trilha: comportamento, não literal SQL


def _hypothesis(store, statement, *, novelty, anchored, total, target="sigma-1"):
    chain = (
        [{"claim": f"a{i}", "supported": True, "evidence": "PMID:1"} for i in range(anchored)]
        + [{"claim": f"b{i}", "supported": False, "evidence": ""}
           for i in range(total - anchored)]
    )
    cur = store.conn.execute(
        "INSERT INTO hypotheses(focus_id, statement, tier, status, mechanism_target, "
        "  chain_json, "
        "  falsifier, novelty, survives_critique, updated_at) "
        "VALUES(1, ?, 'speculative', 'active', ?, ?, 'f', ?, 1, '2024-01-01') "
        "RETURNING id",
        (statement, target, json.dumps(chain), novelty),
    )
    return int(cur.fetchone()["id"])


def test_the_pursuit_queue_is_ordered_by_merit_not_by_insertion(store):
    """`pending_pursuit` decide QUEM gasta chamadas de LLM.

    Sem ordem por mérito, o SQLite devolve ordem de rowid — ordem de colheita, o mesmo
    critério que a Trava 3 proíbe para claims. A hipótese mais forte é inserida por
    ÚLTIMO de propósito: um teste que a insere primeiro passa mesmo sem `ORDER BY`.
    """
    _seed(store, "1")
    fraca = _hypothesis(store, "fraca", novelty=0.2, anchored=1, total=4)
    forte = _hypothesis(store, "forte", novelty=0.9, anchored=3, total=4)

    assert Explorer(store, None, profile=PROFILE).pending_pursuit(limit=2) == [forte, fraca]


def test_a_second_hypothesis_of_the_same_target_is_not_demoted(store):
    """O guard anti-thrash como FATOR é o que a fase proíbe, e ele entraria por Python
    com o literal SQL intacto.

    Duas hipóteses do mesmo alvo, a segunda melhor que uma terceira de outro alvo. Um
    guard multiplicativo rebaixaria a segunda abaixo da terceira — e o `ORDER BY` no
    texto continuaria byte-idêntico, então uma trava que só lê o literal não veria nada.
    """
    _seed(store, "1")
    # A fixture tem de DISCRIMINAR: a terceira precisa valer entre o mérito real da
    # segunda (0,85) e o valor que um guard multiplicativo lhe daria (0,85 × 0,3 =
    # 0,255). Com a terceira em 0,075 o rebaixamento não inverte nada e o teste passa
    # com o defeito presente — foi o que a primeira versão deste teste fazia.
    primeira = _hypothesis(store, "sigma-1 A", novelty=0.9, anchored=4, total=4)
    segunda = _hypothesis(store, "sigma-1 B", novelty=0.85, anchored=4, total=4)
    outra = _hypothesis(store, "orexina", novelty=1.0, anchored=2, total=4,
                        target="orexin")

    assert Explorer(store, None, profile=PROFILE).pending_pursuit(limit=2) == [primeira, segunda], (
        "a segunda hipótese do mesmo alvo foi rebaixada: um guard anti-thrash entrou "
        "como fator no ranking, que é o que esta fase proíbe"
    )
    assert outra


def test_the_board_is_ordered_by_merit(store):
    _seed(store, "1")
    fraca = _hypothesis(store, "fraca", novelty=0.2, anchored=1, total=4)
    forte = _hypothesis(store, "forte", novelty=0.9, anchored=3, total=4)

    board = Explorer(store, None, profile=PROFILE).board()
    assert [r["id"] for r in board][:2] == [forte, fraca]
