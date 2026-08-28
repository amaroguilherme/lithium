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
from lithium.types import Directness, SourceKind


@dataclass(slots=True, frozen=True)
class Strategy:
    name: str
    rationale: str
    expected_directness: Directness
    queries: tuple[str, ...]
    sources: tuple[SourceKind, ...] = (SourceKind.PUBMED,)
    """**NÃO É LIDO em lugar nenhum.** Botão de configuração que não configura nada.

    `harvest_query` resolve a fonte sozinho — a string `'pubmed'` aparece fixa quatro
    vezes nele (a busca da fonte, o filtro de novidade, o `kind` do payload e a chave de
    dedup). Ajustar este campo não muda que fonte é consultada, e um botão inerte é pior
    que ausência: alguém o ajusta e conclui que ajustou.

    Fica declarado porque removê-lo é uma mudança de API sem ganho, e porque o item 9
    mediu que **nenhuma segunda fonte deve entrar em `claims`** (ver o PLAN.md): o
    Europe PMC não adiciona um único artigo revisado que o PubMed já não traga (0 de 813),
    e uma bula ou um registro de ensaio não são desenho de estudo. Enquanto isso valer, a
    dívida de fiação não precisa ser paga — só declarada."""
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
