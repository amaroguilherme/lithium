"""Adapter do PubMed via E-utilities do NCBI.

Duas coisas valem destaque:

**`PublicationType` dá o grade de graça — mas só às vezes.** O NCBI indexa
confiavelmente as categorias fortes (Meta-Analysis, Systematic Review, RCT, Case
Reports). Para o resto, o campo diz "Journal Article" e nada mais. Nesse caso
devolvemos `design=None`, não `opinion`: fixar o piso esmagaria coorte e
caso-controle legítimos, que são justamente a evidência que sobra num domínio sem
RCT direto. Quem julga aí é o LLM, lendo o texto.

**Abstract estruturado já vem seccionado.** Os `<AbstractText Label="RESULTS">` do
PubMed são fronteiras de chunk melhores que qualquer heurística de contagem de
caracteres — e `section` depois entra no metadado do chunk.
"""

from __future__ import annotations

import logging
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from lithium.sources.base import Passage, SearchSpec, SourceRecord
from lithium.rate_limit import RateLimiter
from lithium.types import Grade, SourceKind

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Só mapeamos o que o NCBI indexa com confiança. Ausência vira None de propósito.
PUBLICATION_TYPE_TO_GRADE: dict[str, Grade] = {
    "Meta-Analysis": Grade.META_ANALYSIS,
    # NMA compara múltiplos tratamentos por evidência indireta — é o desenho que
    # melhor responde "qual alternativa", que é a pergunta deste projeto.
    "Network Meta-Analysis": Grade.META_ANALYSIS,
    "Systematic Review": Grade.SYSTEMATIC_REVIEW,
    # Diretrizes (CANMAT/ISBD, NICE, ABP) são âncoras de alto valor: sintetizam
    # evidência e já resolvem o julgamento clínico que falta ao paper isolado.
    "Practice Guideline": Grade.SYSTEMATIC_REVIEW,
    "Guideline": Grade.SYSTEMATIC_REVIEW,
    "Consensus Development Conference": Grade.SYSTEMATIC_REVIEW,
    "Randomized Controlled Trial": Grade.RCT,
    "Controlled Clinical Trial": Grade.RCT,
    "Case Reports": Grade.CASE_REPORT,
    "Observational Study": Grade.COHORT,
    "Comparative Study": Grade.COHORT,
    "Editorial": Grade.OPINION,
    "Comment": Grade.OPINION,
    "Letter": Grade.OPINION,
    "Review": Grade.OPINION,  # revisão narrativa; a sistemática tem tipo próprio
}

# Quando um artigo carrega vários tipos, vence o mais forte — um paper marcado
# "Meta-Analysis" + "Review" é meta-análise.
_GRADE_RANK = {g: i for i, g in enumerate(Grade)}


def _pick_strongest(grades: list[Grade]) -> Grade | None:
    return min(grades, key=lambda g: _GRADE_RANK[g]) if grades else None


def _text(node: ET.Element | None) -> str:
    """Texto de um nó incluindo filhos inline (<i>, <sup> aparecem em abstracts)."""
    return "".join(node.itertext()).strip() if node is not None else ""


class PubMedSource:
    kind = SourceKind.PUBMED

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rate_per_s: float = 3.0,
        client: httpx.AsyncClient | None = None,
        tool: str = "lithium",
        email: str | None = None,
    ) -> None:
        # Com API key o NCBI libera 10 req/s; sem, 3.
        self.api_key = api_key
        self.limiter = RateLimiter(rate_per_s if not api_key else max(rate_per_s, 10.0))
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = client is None
        self.tool = tool
        self.email = email

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _params(self, **extra: Any) -> dict[str, Any]:
        params: dict[str, Any] = {"db": "pubmed", "tool": self.tool, **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    async def _get(self, endpoint: str, params: dict[str, Any]) -> httpx.Response:
        await self.limiter.acquire()
        r = await self._client.get(f"{EUTILS}/{endpoint}", params=params)
        r.raise_for_status()
        return r

    # ──────────────────────────────────────────────────────────────────── busca

    async def search(self, spec: SearchSpec) -> list[str]:
        r = await self._get(
            "esearch.fcgi",
            self._params(
                term=spec.query, retmode="json", retmax=spec.limit, sort="relevance"
            ),
        )
        return list(r.json().get("esearchresult", {}).get("idlist", []))

    async def fetch(self, external_ids: list[str]) -> list[SourceRecord]:
        if not external_ids:
            return []
        r = await self._get(
            "efetch.fcgi", self._params(id=",".join(external_ids), retmode="xml")
        )
        root = ET.fromstring(r.text)

        records: list[SourceRecord] = []
        for article in root.findall(".//PubmedArticle"):
            try:
                record = self._parse_article(article)
            except Exception:  # noqa: BLE001
                # Um registro malformado não pode derrubar o lote inteiro.
                log.exception("falha ao parsear PubmedArticle, pulando")
                continue
            if record is not None:
                records.append(record)
        return records

    # ─────────────────────────────────────────────────────────────────── parsing

    def _parse_article(self, article: ET.Element) -> SourceRecord | None:
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            return None

        art = article.find(".//MedlineCitation/Article")
        if art is None:
            return None

        title = _text(art.find("ArticleTitle"))
        journal = _text(art.find("Journal/ISOAbbreviation")) or _text(
            art.find("Journal/Title")
        )

        passages = self._abstract_passages(art)
        if not passages:
            # Sem abstract não há o que extrair; o título sozinho não sustenta claim.
            return None

        types = [_text(t) for t in art.findall("PublicationTypeList/PublicationType")]
        design = _pick_strongest(
            [PUBLICATION_TYPE_TO_GRADE[t] for t in types if t in PUBLICATION_TYPE_TO_GRADE]
        )

        doi = next(
            (
                _text(node)
                for node in article.findall(".//ArticleIdList/ArticleId")
                if node.get("IdType") == "doi"
            ),
            None,
        )

        mesh = [
            _text(d) for d in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        ]

        return SourceRecord(
            kind=SourceKind.PUBMED,
            external_id=pmid,
            title=title,
            passages=passages,
            raw={"pmid": pmid, "publication_types": types, "mesh": mesh},
            year=self._year(art),
            journal=journal or None,
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            design=design,
            keywords=mesh,
        )

    @staticmethod
    def _abstract_passages(art: ET.Element) -> list[Passage]:
        """Abstract estruturado vira uma passagem por seção; o não estruturado, uma só."""
        nodes = art.findall("Abstract/AbstractText")
        passages: list[Passage] = []
        for node in nodes:
            text = _text(node)
            if not text:
                continue
            label = node.get("Label") or node.get("NlmCategory")
            passages.append(
                Passage(text=text, section=label.lower() if label else None)
            )
        return passages

    @staticmethod
    def _year(art: ET.Element) -> int | None:
        """PubDate pode trazer <Year>, ou só <MedlineDate> como "2019 Jan-Feb"."""
        pub_date = art.find("Journal/JournalIssue/PubDate")
        if pub_date is None:
            return None
        year = _text(pub_date.find("Year"))
        if year.isdigit():
            return int(year)
        medline = _text(pub_date.find("MedlineDate"))
        head = medline[:4]
        return int(head) if head.isdigit() else None
