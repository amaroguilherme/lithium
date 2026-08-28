"""As frentes de busca — vindas do PERFIL, não de constante de módulo.

Buscar só a interseção "bipolar I + TAG" volta quase vazio, e por um motivo
estrutural, não por falta de jeito: RCTs de TAG excluem bipolares no critério de
elegibilidade, e RCTs de bipolar tratam ansiedade como desfecho secundário. A
literatura direta é escassa porque o desenho dos estudos a impede de existir.

Então o harvest cobre cinco frentes deliberadamente diferentes, e cada uma carrega um
`expected_directness` atribuído **antes** de ver o resultado. É isso que permite ao
sistema raciocinar "esta é uma meta-análise forte, mas em unipolar — vale menos do
que o tamanho dela sugere", em vez de tratar todo RCT como equivalente.

O `expected_directness` é um prior, não um veredito: a extração pode rebaixar ou
promover ao ler a população real do estudo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lithium.focus import FocusProfile
from lithium.types import Directness


DEFAULT_SOURCE = "pubmed"
"""A fonte usada quando ninguém nomeia uma.

Um default e não uma lista fechada. Ele existe porque as frentes fixas do perfil não
declaram fonte (ver `Strategy.sources`) e porque uma busca vinda do LLM pode omitir o
campo. Que ele seja `pubmed` é consequência das medições do item 9, não uma preferência:
as outras três candidatas foram recusadas por medição, e a única fonte de evidência
aprovada de fábrica é esta.
"""


@dataclass(slots=True, frozen=True)
class Strategy:
    name: str
    rationale: str
    expected_directness: Directness
    queries: tuple[str, ...]
    sources: tuple[str, ...] = (DEFAULT_SOURCE,)
    """A dívida de fiação foi PAGA na Fase D — este campo passou a ser lido.

    O docstring anterior dizia, com razão, que ele era "um botão de configuração que não
    configura nada": `harvest_query` fixava a string `'pubmed'` quatro vezes e resolvia a
    fonte sozinho. Agora `harvest_query` usa `sources[0]` quando o payload não nomeia uma,
    e `tests/test_source_kinds.py` trava isso por COMPORTAMENTO.

    `tuple[str, ...]` e não `tuple[SourceKind, ...]`: os slugs vivem em
    `sources_registry`, então o conjunto não é fechado em tempo de compilação. O que
    valida é a FK de `sources.kind`, na escrita.

    Continua NÃO vindo do TOML do perfil. A razão mudou: antes era código morto, agora é
    que a escolha de fonte por busca pertence a quem planeja a busca — o LLM a preenche em
    `SourceQuery.source` para as buscas dirigidas por especulação, e as frentes fixas do
    perfil não têm por que divergir do default."""
    limit: int = 20
    priority: float = 0.5
    tags: frozenset[str] = field(default_factory=frozenset)

def all_strategies(profile: FocusProfile) -> tuple[Strategy, ...]:
    """As frentes do perfil, na ordem declarada.

    `sources` e `tags` NÃO vêm do TOML. `sources` está declarado morto no docstring
    acima; `tags` é pior — grep em `lithium/` e `tests/` devolve UMA ocorrência, a
    própria declaração, sem aviso de código morto e sem teste. Transcrever um botão
    inerte para um arquivo que o usuário EDITA À MÃO cria o pior formato possível: um
    campo editável que não faz nada.
    """
    return tuple(
        Strategy(
            name=s.name,
            rationale=s.rationale,
            expected_directness=Directness(s.expected_directness),
            queries=tuple(s.queries),
            limit=s.limit,
            priority=s.priority,
        )
        for s in profile.strategies.strategies
    )


def strategy_by_name(profile: FocusProfile) -> dict[str, Strategy]:
    """Resolvido em CALL TIME, sempre.

    MEDIDO por que isto não pode ser um dict de módulo: rebindando
    `strategy.STRATEGY_BY_NAME = {}`, um `from ... import STRATEGY_BY_NAME` no topo de
    `handlers.py` continuava devolvendo as 5 estratégias velhas (binding de import
    time) enquanto `all_search_specs()` já devolvia 0 — metade do sistema no perfil
    novo e metade no velho, no MESMO processo, sem erro. O handler caía no ramo ad-hoc
    e atribuía INDIRECT a tudo, apagando em silêncio o prior de directness de cada
    frente.
    """
    return {s.name: s for s in all_strategies(profile)}


def all_search_specs(profile: FocusProfile) -> list[tuple[Strategy, str]]:
    """Achata em (estratégia, query) — uma unidade de trabalho por par."""
    return [(s, q) for s in all_strategies(profile) for q in s.queries]
