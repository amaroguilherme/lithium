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

from lithium.types import Directness, Grade, SourceKind


@dataclass(slots=True)
class Passage:
    """Um pedaço citável do documento. `section` vira metadado de chunk."""

    text: str
    section: str | None = None


@dataclass(slots=True)
class SourceRecord:
    kind: SourceKind
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


EVIDENCE_KINDS: frozenset[SourceKind] = frozenset({SourceKind.PUBMED})
"""Quais tipos de fonte podem virar `claims`. É um portão de RUNTIME, e a razão é medida.

`AVAILABLE_SOURCES` (em explore.py) **não** protege este caminho: ele só filtra queries
planejadas e edita uma dica de prompt. Um adapter registrado em `Context.sources` chega a
`claims` sem passar por ele — construído e medido: uma linha em `daemon.py` mais um
`fetch_source {"kind": "fda"}` fez o campo de contraindicação de uma bula, cujo conteúdo
literal é *"None with olanzapine monotherapy…"*, virar uma claim com `grade='rct'`,
`directness='partial'` e peso 0,408 na view `claim_weight`.

Um fato regulatório não é desenho de estudo, e a escala de `Grade` não tem lugar para ele.
Mais grave: **a prosa de um registro de ensaio é INTENÇÃO e passa o portão 1** — o
`quote_is_anchored` foi rodado contra texto literal da API do ClinicalTrials.gov e
aprovou, porque a citação *é* literal. O portão de citação não distingue intenção de
resultado; só este portão de tipo distingue.

Por isso a checagem é aqui, no handler que ingere, e não numa frozenset que edita prompt:
esta é a única que sobrevive a um adapter chegando por CLI, por handler novo ou por
`Context.sources`.
"""


class UnsupportedSourceKind(RuntimeError):
    """Tentativa de ingerir um tipo de fonte que não é evidência de estudo."""


class Source(Protocol):
    kind: SourceKind

    async def search(self, spec: SearchSpec) -> list[str]:
        """Devolve identificadores externos, ordenados por relevância."""
        ...

    async def fetch(self, external_ids: list[str]) -> list[SourceRecord]:
        """Busca os registros completos. Deve tolerar ids inexistentes, pulando-os."""
        ...
