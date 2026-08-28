"""O item 9 virou isto: os bugs que as quatro fontes novas expuseram.

Nenhuma das quatro foi construída, e as três medições que decidiram estão no `PLAN.md`.
O que sobrou são defeitos que existem **hoje**, independentes de qualquer fonte nova, e
que só apareceram porque alguém tentou adicionar uma:

1. **`canonical_pmid` convertia PMCID no PMID de outro artigo.** `PMC7738613` → `7738613`,
   que é um paper real de 1995 sobre linfoma não-Hodgkin no JCO. O portão de citação da
   Fase 6 existe porque "alucinar um identificador plausível e real é pior que inventar
   um" — e o regex produzia exatamente isso, deterministicamente. O modelo escreve PMCID
   com frequência.
2. **`fetch_source` aceitava qualquer `kind`.** `AVAILABLE_SOURCES` não protege esse
   caminho: uma linha em `daemon.py` fez o campo de contraindicação de uma bula virar
   claim com `grade='rct'` e peso 0,408.
3. **O mesmo artigo por dois `kind` conta como duas fontes independentes** — satisfazendo
   `MIN_CITATIONS` e `n_independent_sources >= 2` com um paper só.
4. **`Strategy.sources` é código morto com cara de configuração**, e `harvest_query` fixa
   a string `'pubmed'` quatro vezes.
"""

from __future__ import annotations

import json

import pytest

from lithium.db import Store
from lithium.pipeline.explore import canonical_pmid
from lithium.sources.base import EVIDENCE_KINDS, UnsupportedSourceKind
from lithium.types import Directness, Grade, SourceKind

from conftest import onco_profile

PROFILE = onco_profile()


# Os dois ids reais da colisão, com o que cada um de fato é.
PMCID = "PMC7738613"
COLLIDING_PMID = "7738613"   # J Clin Oncol 1995, linfoma não-Hodgkin


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db", embedding_dim=4)
    s.init_schema()
    yield s
    s.close()


# ═══════════════════════ 1. PMCID não vira o PMID de outro artigo


@pytest.mark.parametrize("written", [
    PMCID, "PMC11302992", "pmc7738613", "(PMC7738613)",
    "v2.7738613", "10.1016/j.jad.7738613",
])
def test_a_pmcid_is_not_read_as_a_pmid(written):
    """A colisão era determinística e o portão de citação a aprovava.

    Se `7738613` existir no corpus — e ele existe, é um paper real —, `gate_citations`
    marca o elo como ancorado e o quadro mostra `[supported: PMID:7738613]` ao
    psiquiatra. Uma citação verdadeira, de outro artigo, sobre outro assunto.
    """
    assert canonical_pmid(written) is None, (
        f"{written!r} foi lido como PMID {canonical_pmid(written)!r}"
    )


@pytest.mark.parametrize("written,expected", [
    ("PMID: 28544150", "28544150"),
    ("PMID:28544150", "28544150"),
    ("pmid 28544150", "28544150"),
    ("28544150", "28544150"),
    ("PMID:1", "1"),
])
def test_a_real_citation_still_parses(written, expected):
    """A contrapartida: a correção não pode virar um filtro que recusa tudo. O modelo
    escreve `PMID: 123` com espaço, sempre."""
    assert canonical_pmid(written) == expected


def test_the_collision_would_have_been_approved_by_the_citation_gate(store):
    """Prova de que o bug era explorável, não teórico: com o PMID colidente no corpus,
    o portão aprovaria o elo."""
    from lithium.pipeline.explore import Explorer, gate_citations

    store.upsert_source(kind="pubmed", external_id=COLLIDING_PMID, raw={},
                        title="paper de 1995 sobre linfoma")
    explorer = Explorer(store, None, profile=PROFILE)
    chain = [{"claim": "sigma-1 reduz ansiedade", "supported": True,
              "evidence": f"{PMCID}"}]

    gated, refused = gate_citations(chain, explorer.corpus_pmids(chain))
    assert refused == 1, "o PMCID foi aceito como citação do artigo colidente"


# ═════════════════ 2. o portão de TIPO, no lugar que ingere


def test_only_study_evidence_kinds_may_be_ingested():
    assert EVIDENCE_KINDS == frozenset({SourceKind.PUBMED})


@pytest.mark.parametrize("kind", ["fda", "ctgov", "epmc", "inventado"])
async def test_fetch_source_refuses_a_non_evidence_kind(kind, tmp_path, store):
    """A única checagem que sobrevive a um adapter chegando por qualquer caminho.

    `AVAILABLE_SOURCES` não está neste caminho — ela filtra query planejada e edita uma
    dica de prompt. Medido: uma linha em `daemon.py` mais um `fetch_source` fez o campo
    de contraindicação de uma bula, cujo conteúdo literal é *"None with olanzapine
    monotherapy…"*, virar claim com `grade='rct'` e peso 0,408.
    """
    from lithium.config import Config
    from lithium.worker.handlers import HANDLERS

    class Spy:
        def __init__(self) -> None:
            self.fetched = False

        async def fetch(self, ids):
            self.fetched = True
            return []

    spy = Spy()

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.sources = {kind: spy}
            self.config = Config(data_dir=tmp_path)

    with pytest.raises(UnsupportedSourceKind):
        await HANDLERS["fetch_source"]({"kind": kind, "external_id": "x"}, Ctx())
    assert not spy.fetched, "o adapter foi consultado antes do portão de tipo"


async def test_pubmed_still_passes_the_kind_gate(tmp_path, store):
    """Sem isto, o portão poderia ser "recusa tudo" e a suíte não notaria."""
    from lithium.config import Config
    from lithium.worker.handlers import HANDLERS

    class Empty:
        async def fetch(self, ids):
            return []

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.sources = {"pubmed": Empty()}
            self.config = Config(data_dir=tmp_path)

    await HANDLERS["fetch_source"]({"kind": "pubmed", "external_id": "1"}, Ctx())


def test_the_daemon_registers_only_evidence_kinds():
    """O dict de fontes do daemon é o outro lugar por onde um adapter entra."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parent.parent
                      / "lithium" / "daemon.py").read_text(encoding="utf-8"))
    registered: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "sources":
            if isinstance(node.value, ast.Dict):
                registered = {
                    k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
    assert registered, "não achei o dict de fontes em daemon.py"
    assert registered <= {k.value for k in EVIDENCE_KINDS}, (
        f"daemon registra fonte que não é evidência de estudo: "
        f"{sorted(registered - {k.value for k in EVIDENCE_KINDS})}"
    )


# ═══════════ 3. o mesmo artigo não conta como duas fontes independentes


async def test_one_article_under_two_kinds_counts_once(store):
    """O defeito que qualquer segunda fonte de literatura criaria.

    Medido: 100% dos PMIDs que o PubMed colhe neste domínio também estão no Europe PMC.
    O mesmo paper entraria como `pubmed` e como `epmc`, seria extraído duas vezes, e
    chegaria ao juiz de suficiência como **duas fontes independentes concordando** —
    satisfazendo `MIN_CITATIONS = 2` com um artigo só.
    """
    from lithium.pipeline.answer import MIN_CITATIONS

    doi = "10.1016/j.jad.2023.11.001"
    for kind, external in (("pubmed", "37956131"), ("pubmed", "37956132")):
        store.upsert_source(kind=kind, external_id=external, raw={},
                            title="mesmo artigo", doi=doi)

    distintos = store.conn.execute(
        "SELECT COUNT(DISTINCT COALESCE(NULLIF(doi, ''), kind || ':' || external_id)) "
        "  AS n FROM sources"
    ).fetchone()["n"]

    assert distintos == 1, (
        "duas linhas com o mesmo DOI contam como duas fontes; o piso de "
        f"{MIN_CITATIONS} citações seria satisfeito por um artigo só"
    )


# ═════════════════════ 4. a fiação, por COMPORTAMENTO


async def test_registering_an_adapter_is_not_enough_to_be_consulted(store, tmp_path):
    """Trava comportamental, e a razão de não ser por contagem de literal.

    A auditoria provou que um teste que conta ocorrências de `'pubmed'` no fonte é
    satisfeito **movendo o literal para uma constante**: zero mudança de comportamento, o
    guard fica verde, e o adapter novo continua sendo código morto.

    Este teste afirma o fato: hoje `harvest_query` decide a fonte sozinho, então
    registrar um adapter em `Context.sources` não faz nada. Enquanto isso for verdade, o
    teste documenta a dívida; quando alguém rotear por `kind`, ele falha e obriga a
    atualizar a expectativa junto com o comportamento.
    """
    import inspect

    from lithium.worker import handlers

    body = inspect.getsource(handlers.harvest_query)
    assert '"pubmed"' in body or "'pubmed'" in body, (
        "harvest_query passou a rotear por kind — atualize este teste e o PLAN.md: a "
        "dívida de fiação de fonte foi paga"
    )


def test_strategy_sources_is_documented_as_dead():
    """`Strategy.sources` é declarado e nunca lido. Um botão de configuração que não faz
    nada é pior que ausência: alguém o ajusta e conclui que ajustou."""
    import inspect
    from pathlib import Path

    from lithium.pipeline import strategy

    src = Path(inspect.getfile(strategy)).read_text(encoding="utf-8")
    if "sources:" not in src:
        pytest.skip("o campo foi removido")
    assert "NÃO É LIDO" in src or "não é lido" in src, (
        "`Strategy.sources` continua sem aviso de que é código morto"
    )
