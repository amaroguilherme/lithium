"""Contrato comum das fontes de evidência.

Todo adapter devolve `SourceRecord`. O campo que exige atenção é `design`: ele pode
ser `None`, e `None` significa "a fonte não indexou o desenho do estudo", não
"desenho fraco". Chutar `opinion` como default esmagaria evidência real — em
PubMed, a maioria dos artigos primários não carrega um PublicationType útil. Quando
vem `None`, quem julga é o LLM, a partir do texto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from lithium.types import Directness, Grade


@dataclass(slots=True)
class Passage:
    """Um pedaço citável do documento. `section` vira metadado de chunk."""

    text: str
    section: str | None = None


@dataclass(slots=True)
class SourceRecord:
    kind: str
    external_id: str
    title: str
    passages: list[Passage]
    raw: dict[str, Any]
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    url: str | None = None
    design: Grade | None = None
    sample_n: int | None = None
    population_tag: str | None = None
    keywords: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.passages)


@dataclass(slots=True)
class SearchSpec:
    """Uma busca planejada. `expected_directness` é atribuído ANTES de ver o
    resultado — é o que permite depois raciocinar "RCT forte, população errada"."""

    query: str
    expected_directness: Directness = Directness.INDIRECT
    limit: int = 20


# `EVIDENCE_KINDS` foi REMOVIDO na Fase D. O portão continua existindo no mesmo lugar
# (`fetch_source`, onde ingere) mas passou a CONSULTAR `sources_registry.yields_evidence`
# em vez de comparar contra uma frozenset compilada.
#
# A pergunta não mudou: "isto pode virar claim?". Mudou quem responde. Uma allowlist de
# duas APIs em código expressa uma decisão de POLÍTICA como se fosse um fato sobre o
# mundo; a coluna expressa o CONTRATO — publica estudo com prosa citável verbatim e
# desenho graduável na escala do foco.
#
# As medições do item 9 continuam válidas e continuam registradas no PLAN.md: elas dizem
# que aquelas três fontes não devem produzir evidência NESTE foco, e é por isso que só o
# PubMed nasce aprovado. O que elas não justificam é congelar a lista para sempre.


class UnsupportedSourceKind(RuntimeError):
    """Tentativa de ingerir um tipo de fonte que não é evidência de estudo."""


class Source(Protocol):
    kind: str

    async def search(self, spec: SearchSpec) -> list[str]:
        """Devolve identificadores externos, ordenados por relevância."""
        ...

    async def fetch(self, external_ids: list[str]) -> list[SourceRecord]:
        """Busca os registros completos. Deve tolerar ids inexistentes, pulando-os."""
        ...
