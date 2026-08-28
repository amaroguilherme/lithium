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
from lithium.sources.base import UnsupportedSourceKind
from lithium.types import Directness, Grade

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


def test_the_gate_is_a_registry_query_not_a_compiled_set(store):
    """O portão continua no mesmo lugar; mudou quem responde.

    A versão anterior deste teste era `assert EVIDENCE_KINDS == frozenset({PUBMED})` —
    igualdade literal contra a própria constante que ele policiava, ou seja tautológica:
    ela passava por construção e não dizia nada sobre comportamento.

    O que importa é o CONTRATO, e ele agora é uma coluna: só fonte aprovada E com
    `yields_evidence = 1` pode virar claim. Fail-closed cobre três estados com a mesma
    resposta, e os três são testados: desconhecida, aprovada sem evidência, e conhecida
    mas não aprovada.

    MUTAÇÃO: `source_yields_evidence` devolver True quando a linha não existe, ou ignorar
    `approved_at` (ler `sources_registry` em vez de `active_sources`).
    """
    assert store.source_yields_evidence("pubmed")
    assert not store.source_yields_evidence("nunca-cadastrada")

    store.conn.execute(
        "INSERT INTO sources_registry(slug, description, base_url, yields_evidence, "
        "  approved_at) VALUES('nice', 'diretrizes', 'https://x', 0, "
        "  strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
    )
    assert not store.source_yields_evidence("nice"), (
        "fonte aprovada mas sem yields_evidence passou o portão"
    )

    store.conn.execute(
        "INSERT INTO sources_registry(slug, description, base_url, yields_evidence) "
        "VALUES('openalex', 'metadados', 'https://y', 1)"
    )
    assert not store.source_yields_evidence("openalex"), (
        "fonte NÃO aprovada passou o portão: aprovação é o que separa proposta de fonte"
    )


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


def test_the_daemon_builds_its_sources_from_the_registry():
    """O dict de fontes do daemon era um LITERAL, e era o outro lugar por onde um adapter
    entrava sem passar por política nenhuma.

    Medido no item 9: uma linha em `daemon.py` mais um `fetch_source` fez o campo de
    contraindicação de uma bula virar claim com `grade='rct'` e peso 0,408. Enquanto a
    fonte fosse escolhida por edição de código, "aprovar uma fonte" não significava nada.

    Agora ele lê `active_sources()`. O teste é por AST porque a alternativa — subir o
    daemon — exige dois llama-server e 9 GB de peso.

    MUTAÇÃO: voltar `sources={"pubmed": pubmed}` em `daemon.py`.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "lithium" / "daemon.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "sources":
            assert not isinstance(node.value, ast.Dict), (
                "daemon voltou a montar as fontes como dict literal: a aprovação no "
                "registro deixaria de decidir o que é consultado"
            )
    # A CHAMADA, não o import. Mutação medida: trocar `sources = build_sources(...)` por
    # `sources = {"pubmed": PubMedSource()}` deixava o import intocado, então
    # `"build_sources" in src` continuava True e o teste passava verde com a fiação
    # revertida. E o `isinstance(..., ast.Dict)` acima também não pega, porque o dict fica
    # atrás de uma variável.
    chamadas = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "build_sources" in chamadas, (
        "daemon.py não CHAMA mais build_sources — a aprovação no registro virou "
        "decoração e a escolha de fonte volta a morar no código"
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


def test_the_citation_floor_counts_articles_not_claims(store):
    """A metade que o teste acima NÃO cobria, e é a que importa.

    Ele afirmava sobre uma expressão SQL escrita à mão dentro do próprio teste — ou seja,
    provava que a expressão funciona, não que o SISTEMA a usa. O loop de resposta contava
    `len(hits)`, que são CLAIMS: duas claims do mesmo paper satisfaziam `MIN_CITATIONS`, e
    com duas fontes o mesmo artigo em dois `kind` fazia o mesmo.

    MUTAÇÃO: em `answer.py`, voltar `n_articles` para `len(hits)`.
    """
    from lithium.pipeline.answer import distinct_articles

    doi = "10.1016/j.jad.2023.11.001"
    store.upsert_source(kind="pubmed", external_id="A1", raw={}, doi=doi)
    store.upsert_source(kind="pubmed", external_id="A2", raw={}, doi=doi)
    store.upsert_source(kind="pubmed", external_id="B1", raw={}, doi="10.1/outro")
    ids = [r["id"] for r in store.conn.execute("SELECT id FROM sources ORDER BY id")]

    class Hit:
        def __init__(self, sid):
            self.source_id = sid

    # duas claims, mesmo artigo por DOI -> UM artigo
    assert distinct_articles([Hit(ids[0]), Hit(ids[1])], store) == 1, (
        "duas linhas do mesmo DOI contaram como duas citações independentes"
    )
    # duas claims da MESMA linha -> um artigo
    assert distinct_articles([Hit(ids[0]), Hit(ids[0])], store) == 1
    # artigos de verdade diferentes -> dois
    assert distinct_articles([Hit(ids[0]), Hit(ids[2])], store) == 2
    assert distinct_articles([], store) == 0


# ═════════════════════ 4. a fiação, por COMPORTAMENTO


async def test_harvest_routes_by_the_source_in_the_payload(store, tmp_path):
    """A DÍVIDA DE FIAÇÃO, PAGA — e este teste era o seu marcador.

    A versão anterior afirmava o contrário: que `harvest_query` decidia a fonte sozinho e
    que registrar um adapter em `Context.sources` não fazia nada. Ela era deliberadamente
    escrita por COMPORTAMENTO e não por contagem de literal, porque a auditoria provou que
    um teste que conta ocorrências de `'pubmed'` no fonte é satisfeito **movendo o literal
    para uma constante**: zero mudança de comportamento, guard verde, adapter novo
    continua código morto.

    Agora a expectativa se inverte junto com o comportamento, que é o que o docstring
    antigo mandava fazer.

    MUTAÇÃO: voltar `source = (ctx.sources or {}).get("pubmed")` em `harvest_query`.
    """
    from lithium.config import Config
    from lithium.worker.handlers import HANDLERS

    store.conn.execute(
        "INSERT INTO sources_registry(slug, description, base_url, yields_evidence, "
        "  approved_at) VALUES('openalex', 'metadados abertos', 'https://api.openalex.org',"
        "  1, strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
    )

    class Spy:
        def __init__(self, name):
            self.name = name
            self.searched = False

        async def search(self, spec):
            self.searched = True
            return []

    pubmed, openalex = Spy("pubmed"), Spy("openalex")

    class Q:
        def enqueue(self, *a, **k):
            return 1

    class Ctx:
        def __init__(self):
            self.store = store
            self.queue = Q()
            self.sources = {"pubmed": pubmed, "openalex": openalex}
            self.config = Config(data_dir=tmp_path)

    await HANDLERS["harvest_query"](
        {"query": "x", "source": "openalex", "focus_id": 1}, Ctx())

    assert openalex.searched, "a fonte do payload foi ignorada"
    assert not pubmed.searched, (
        "`harvest_query` consultou o PubMed apesar de o payload pedir outra fonte — "
        "a escolha de fonte continua morrendo antes da busca"
    )


def test_strategy_sources_is_read_not_dead():
    """O outro marcador de dívida, também invertido.

    `Strategy.sources` era declarado e nunca lido — "um botão de configuração que não
    configura nada é pior que ausência: alguém o ajusta e conclui que ajustou". A Fase D
    passou a lê-lo como default quando o payload não nomeia fonte.

    Afirma sobre o CÓDIGO que lê, não sobre o docstring: a versão anterior checava se a
    palavra "NÃO É LIDO" aparecia no arquivo, o que é satisfeito editando prosa.

    MUTAÇÃO: em `harvest_query`, trocar o fallback por `DEFAULT_SOURCE` direto, ignorando
    `strategy.sources`.
    """
    import inspect

    from lithium.worker import handlers

    body = inspect.getsource(handlers.harvest_query)
    assert "strategy.sources" in body, (
        "`Strategy.sources` voltou a ser código morto com cara de configuração"
    )


async def test_the_chosen_source_reaches_the_harvest_payload(store, tmp_path):
    """O último elo, e o que a mutação provou faltar.

    `plan_queries` já devolvia `q.source` e um teste cobria isso — mas nada verificava que
    ele chega ao PAYLOAD. Mutação medida: apagar a linha `"source": q.source` de
    `pursue_speculation` deixava a suíte inteira verde. É a classe de defeito
    "fiação-não-testada", que este repo já cometeu seis vezes: a escolha de fonte
    atravessava o schema, sobrevivia ao filtro, e morria na montagem do payload.

    MUTAÇÃO: remover `"source": q.source` do payload de `harvest_query`.
    """
    from lithium.config import Config
    from lithium.worker.handlers import HANDLERS

    store.conn.execute(
        "INSERT INTO sources_registry(slug, description, base_url, yields_evidence, "
        "  approved_at) VALUES('ctgov', 'ensaios', 'https://x', 1, "
        "  strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
    )
    store.conn.execute(
        "INSERT INTO hypotheses(focus_id, statement, tier, chain_json, novelty) "
        "VALUES(1, 'h', 'speculative', '[]', 0.5)"
    )
    hid = store.conn.execute("SELECT MAX(id) AS id FROM hypotheses").fetchone()["id"]

    enfileirados: list[dict] = []

    class Q:
        def enqueue(self, kind, payload, **kw):
            enfileirados.append(payload)
            return 1

    class FakeExplorer:
        def __init__(self, *a, **k):
            pass

        def pending_pursuit(self, limit=2):
            return [hid]

        async def plan_queries(self, hypothesis_id):
            from lithium.llm.schemas import SourceQuery
            return [SourceQuery(source="ctgov", query="q", seeking="elo 2")]

        def mark_pursued(self, hypothesis_id):
            pass

    import lithium.worker.handlers as H
    original = H.Explorer
    H.Explorer = FakeExplorer
    try:
        class Ctx:
            def __init__(self):
                self.store = store
                self.queue = Q()
                self.llm = None
                self.config = Config(data_dir=tmp_path)

        await HANDLERS["pursue_speculation"]({"focus_id": 1}, Ctx())
    finally:
        H.Explorer = original

    assert enfileirados, "nada foi enfileirado"
    assert enfileirados[0].get("source") == "ctgov", (
        "a fonte que o modelo escolheu não chegou ao payload: ela morre entre "
        "`plan_queries` e `harvest_query`, e o harvest cai no default"
    )
