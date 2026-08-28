"""Adapter do PubMed, testado contra XML real gravado de eutils.

As fixtures em tests/fixtures/ são respostas de verdade do NCBI (4 PMIDs sobre
quetiapina/TAG). Testar contra XML inventado esconde exatamente os casos que
quebram na prática: abstract não estruturado, múltiplos PublicationType no mesmo
artigo, PubDate sem <Year>.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from lithium.sources.base import SearchSpec
from lithium.sources.pubmed import PubMedSource, _pick_strongest
from lithium.types import Grade, SourceKind

FIXTURES = Path(__file__).parent / "fixtures"
EFETCH_XML = (FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8")
ESEARCH_JSON = json.loads((FIXTURES / "pubmed_esearch.json").read_text(encoding="utf-8"))


def _source(handler, **kw) -> PubMedSource:
    return PubMedSource(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw
    )


@pytest.fixture
def records():
    async def _fetch():
        s = _source(lambda _: httpx.Response(200, text=EFETCH_XML))
        return await s.fetch(["30712879"])

    import asyncio

    return asyncio.run(_fetch())


# ──────────────────────────────────────────────────────────────────────── busca


async def test_search_returns_pmids():
    s = _source(lambda _: httpx.Response(200, json=ESEARCH_JSON))
    ids = await s.search(SearchSpec(query="quetiapine generalized anxiety"))
    assert ids == ["30712879", "37956131", "26834458", "21403524"]


async def test_search_sends_expected_params():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=ESEARCH_JSON)

    s = _source(handler)
    await s.search(SearchSpec(query="lítio", limit=7))
    assert seen["db"] == "pubmed"
    assert seen["term"] == "lítio"
    assert seen["retmax"] == "7"
    assert seen["sort"] == "relevance"
    assert "api_key" not in seen


async def test_api_key_is_forwarded_and_raises_rate_ceiling():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=ESEARCH_JSON)

    s = _source(handler, api_key="segredo", rate_per_s=3.0)
    await s.search(SearchSpec(query="x"))
    assert seen["api_key"] == "segredo"
    assert s.limiter.rate == 10.0, "com API key o NCBI libera 10 req/s"


async def test_fetch_empty_list_skips_request():
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("não deveria chamar a API")

    assert await _source(handler).fetch([]) == []


# ─────────────────────────────────────────────────────────────────────── parsing


def test_parses_all_records_from_real_xml(records):
    assert len(records) == 4
    assert {r.external_id for r in records} == {
        "30712879", "37956131", "26834458", "21403524"
    }
    assert all(r.kind is SourceKind.PUBMED for r in records)


def test_structured_abstract_becomes_one_passage_per_section(records):
    lancet = next(r for r in records if r.external_id == "30712879")
    assert [p.section for p in lancet.passages] == [
        "background", "methods", "findings", "interpretation", "funding",
    ]
    assert all(p.text for p in lancet.passages)


def test_unstructured_abstract_becomes_single_unlabelled_passage(records):
    rct = next(r for r in records if r.external_id == "21403524")
    assert len(rct.passages) == 1
    assert rct.passages[0].section is None


def test_network_meta_analysis_maps_to_meta_analysis(records):
    """Regressão: a NMA do Lancet caía para systematic_review porque
    'Network Meta-Analysis' não estava no mapa. NMA é o desenho que melhor
    responde "qual alternativa" — não pode ser rebaixado."""
    lancet = next(r for r in records if r.external_id == "30712879")
    assert "Network Meta-Analysis" in lancet.raw["publication_types"]
    assert lancet.design is Grade.META_ANALYSIS


def test_guideline_is_graded_as_synthesis(records):
    """Diretriz da ABP: âncora de alto valor, não pode ficar sem grade."""
    guideline = next(r for r in records if r.external_id == "37956131")
    assert guideline.design is Grade.SYSTEMATIC_REVIEW


def test_rct_and_meta_analysis_are_graded(records):
    assert next(r for r in records if r.external_id == "21403524").design is Grade.RCT
    assert next(r for r in records if r.external_id == "26834458").design is Grade.META_ANALYSIS


def test_metadata_is_extracted(records):
    lancet = next(r for r in records if r.external_id == "30712879")
    assert lancet.year == 2019
    assert lancet.journal == "Lancet"
    assert lancet.doi == "10.1016/S0140-6736(18)31793-8"
    assert lancet.url == "https://pubmed.ncbi.nlm.nih.gov/30712879/"
    assert "Anti-Anxiety Agents" in lancet.keywords


# ────────────────────────────────────────────── seleção de grade e casos-limite


def test_strongest_grade_wins_among_multiple_types():
    assert _pick_strongest([Grade.OPINION, Grade.META_ANALYSIS, Grade.RCT]) is Grade.META_ANALYSIS
    assert _pick_strongest([]) is None


def test_unindexed_design_is_none_not_opinion():
    """`None` significa "o NCBI não indexou", não "evidência fraca".

    Fixar `opinion` como default esmagaria coorte e caso-controle legítimos — que
    é exatamente a evidência que sobra num domínio sem RCT direto. Com None, quem
    julga é o LLM lendo o texto.
    """
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>999</PMID><Article>
        <Journal><Title>J Test</Title></Journal>
        <ArticleTitle>Um estudo qualquer</ArticleTitle>
        <Abstract><AbstractText>Achamos algo relevante em 40 pacientes.</AbstractText></Abstract>
        <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""

    async def go():
        return await _source(lambda _: httpx.Response(200, text=xml)).fetch(["999"])

    import asyncio

    [record] = asyncio.run(go())
    assert record.design is None


async def test_article_without_abstract_is_dropped():
    """Título sozinho não sustenta claim — indexar sem abstract só gera ruído."""
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>111</PMID><Article>
        <Journal><Title>J</Title></Journal><ArticleTitle>Só título</ArticleTitle>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    assert await _source(lambda _: httpx.Response(200, text=xml)).fetch(["111"]) == []


async def test_malformed_record_does_not_kill_the_batch():
    """Um registro quebrado no meio de 20 não pode custar o lote inteiro."""
    xml = (
        "<PubmedArticleSet>"
        "<PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle>"
        + EFETCH_XML.split("<PubmedArticleSet>", 1)[1]
    )
    records = await _source(lambda _: httpx.Response(200, text=xml)).fetch(["1"])
    assert len(records) == 4


async def test_medline_date_fallback_for_year():
    """PubDate às vezes traz só <MedlineDate>2019 Jan-Feb</MedlineDate>."""
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>222</PMID><Article>
        <Journal><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate>
        </JournalIssue><Title>J</Title></Journal>
        <ArticleTitle>T</ArticleTitle>
        <Abstract><AbstractText>Texto.</AbstractText></Abstract>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    [record] = await _source(lambda _: httpx.Response(200, text=xml)).fetch(["222"])
    assert record.year == 2019


async def test_inline_markup_is_flattened():
    """Abstracts trazem <i>, <sup>; texto partido quebraria a citação verbatim."""
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>333</PMID><Article>
        <Journal><Title>J</Title></Journal><ArticleTitle>T</ArticleTitle>
        <Abstract><AbstractText>O valor de <i>p</i> foi 0,03.</AbstractText></Abstract>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    [record] = await _source(lambda _: httpx.Response(200, text=xml)).fetch(["333"])
    assert record.passages[0].text == "O valor de p foi 0,03."


# ═══════════════════════════════════════════════════════ a credencial não vaza


async def test_the_api_key_never_reaches_the_log(caplog):
    """MUTAÇÃO: remover `logging.getLogger("httpx").setLevel(logging.WARNING)` de
    `_setup_logging`. O httpx loga em INFO
    `HTTP Request: GET .../esearch.fcgi?db=pubmed&tool=lithium&api_key=<CHAVE>&term=...`,
    uma vez por query, 19 por harvest_sweep — para o console e para qualquer handler de
    arquivo. Vazamento de credencial não tem teste que falhe sozinho."""
    import logging

    from lithium.cli import _setup_logging

    secret = "chave-secretissima-da-ncbi"
    _setup_logging(verbose=False)
    source = _source(lambda _: httpx.Response(200, json=ESEARCH_JSON), api_key=secret)
    with caplog.at_level(logging.DEBUG):
        await source.search(SearchSpec(query="quetiapine AND anxiety", limit=3))

    leaked = [r.getMessage() for r in caplog.records if secret in r.getMessage()]
    assert not leaked, f"a api_key da NCBI apareceu no log: {leaked[:1]}"
