"""Enums e pesos do domínio — fonte única de verdade.

Estes valores aparecem em três lugares: nos `CHECK` do schema SQL, nos schemas
Pydantic que restringem a decodificação do LLM, e na lógica de pontuação. Definir
aqui e importar nos outros dois garante que uma mudança não deixe os três em
desacordo silencioso. `tests/test_types.py` trava esse contrato.
"""

from __future__ import annotations

from enum import StrEnum


class Grade(StrEnum):
    """Desenho do estudo, do mais forte ao mais fraco."""

    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "rct"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    PRECLINICAL = "preclinical"
    OPINION = "opinion"


class Directness(StrEnum):
    """Aderência da população estudada ao ALVO DO FOCO ATIVO.

    A prosa de cada nível é dado de PERFIL (`focus.toml`, bloco `[directness]`) e chega
    ao modelo pelo prompt. Aqui ficam só os NOMES, que são vocabulário global: eles são
    FK em `claim_directness.directness` e viram um StrEnum fechado na gramática.

    O alvo concreto NÃO pode voltar para este docstring, e este docstring é ele mesmo
    um exemplo do problema: ele viaja para dentro de `response_format` em SETE schemas
    (medido), inclusive nas extrações que rodam sob outro foco, e chega junto da
    restrição de decodificação — que é o que o modelo é OBRIGADO a seguir. Nenhum dos
    14 goldens vê uma linha disto, e nenhum teste de prompt olha para `schemas.py`.
    Por isso nem a *descrição* do defeito pode nomear o alvo antigo aqui.
    """

    DIRECT = "direct"              # o alvo, exatamente
    PARTIAL = "partial"            # uma das duas metades do alvo
    INDIRECT = "indirect"          # população vizinha
    EXTRAPOLATED = "extrapolated"  # mecanismo, pré-clínico, inferência


class Direction(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NULL = "null"
    MIXED = "mixed"


class QuestionKind(StrEnum):
    """Define o roteamento. PREFERENCE e CONTEXT nunca vão ao loop de pesquisa —
    são estruturalmente inauto-respondíveis e gastariam ciclo produzindo invenção."""

    FACTUAL = "FACTUAL"                  # auto
    SYNTHESIS = "SYNTHESIS"              # auto + crítica adversarial
    PREFERENCE = "PREFERENCE"            # sempre humano — trade-off de valores
    CONTEXT = "CONTEXT"                  # sempre humano — o modelo não tem como saber
    METHODOLOGICAL = "METHODOLOGICAL"    # humano, prioridade baixa


AUTO_ANSWERABLE: frozenset[QuestionKind] = frozenset(
    {QuestionKind.FACTUAL, QuestionKind.SYNTHESIS}
)


class QuestionStatus(StrEnum):
    OPEN = "OPEN"
    RESEARCHING = "RESEARCHING"
    ANSWERED_AUTO = "ANSWERED_AUTO"
    ESCALATED = "ESCALATED"
    ANSWERED_HUMAN = "ANSWERED_HUMAN"
    CLOSED = "CLOSED"


class StuckReason(StrEnum):
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    NEEDS_VALUE_JUDGMENT = "NEEDS_VALUE_JUDGMENT"
    IRRECONCILABLE_CONFLICT = "IRRECONCILABLE_CONFLICT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# `SourceKind` foi REMOVIDO na Fase D. O vocabulário de fontes vive em
# `sources_registry`, e quem o aplica é a FK de `sources.kind` — na escrita, onde um
# slug inválido levanta em vez de virar uma linha calada.
#
# Ele era um `StrEnum` de quatro valores espelhado num `CHECK (kind IN (...))`. Duas
# consequências: uma quinta fonte não conseguia ser NOMEADA, e três dos quatro valores
# (`epmc`, `ctgov`, `fda`) nunca tiveram adapter — um enum onde 3 de 4 membros não
# correspondem a nada é o mesmo "botão de configuração que não configura" que este repo
# recusa em `Strategy.tags`.
#
# Onde a decodificação restrita precisa de conjunto fechado (`SourceQuery.source`), ele é
# construído em RUNTIME a partir das fontes aprovadas — o mesmo padrão que a Fase B usou
# para os níveis de grade e directness.


# ─────────────────────────────────────────────────────────────── pesos de evidência
#
# Calibrados para uma propriedade específica do domínio: quando a evidência direta
# quase não existe, o desenho do estudo não pode dominar a aderência da população.
#
#     cohort + direct          = 0.55 × 1.00 = 0.550   ← vence
#     meta_analysis + indirect = 1.00 × 0.30 = 0.300
#     rct + indirect           = 0.85 × 0.30 = 0.255
#
# Um coorte em bipolar I com TAG vale mais que uma meta-análise em unipolar. É o
# julgamento que um revisor humano faria, e é por isso que `directness` multiplica
# em vez de ser uma nota lateral.

GRADE_WEIGHT: dict[Grade, float] = {
    Grade.META_ANALYSIS: 1.00,
    Grade.SYSTEMATIC_REVIEW: 0.95,
    Grade.RCT: 0.85,
    Grade.COHORT: 0.55,
    Grade.CASE_CONTROL: 0.45,
    Grade.CASE_SERIES: 0.25,
    Grade.CASE_REPORT: 0.15,
    Grade.PRECLINICAL: 0.10,
    Grade.OPINION: 0.05,
}

DIRECTNESS_WEIGHT: dict[Directness, float] = {
    Directness.DIRECT: 1.00,
    Directness.PARTIAL: 0.60,
    Directness.INDIRECT: 0.30,
    Directness.EXTRAPOLATED: 0.12,
}


def evidence_weight(grade: Grade, directness: Directness, confidence: float = 1.0) -> float:
    return GRADE_WEIGHT[grade] * DIRECTNESS_WEIGHT[directness] * confidence
