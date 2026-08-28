"""Contrato entre os enums Python e os CHECK do schema SQL.

Os mesmos valores aparecem em três camadas: CHECK do SQLite, gramática do LLM
(via schemas Pydantic) e lógica de pontuação. Se saírem de sincronia, o sintoma é
uma task morrendo com IntegrityError em produção — bem longe da causa. Estes testes
falham no lugar certo.
"""

from __future__ import annotations

import re

from lithium.db.store import SCHEMA_PATH
from lithium.llm import schemas
from lithium.types import (
    AUTO_ANSWERABLE,
    DIRECTNESS_WEIGHT,
    GRADE_WEIGHT,
    Direction,
    Directness,
    Grade,
    QuestionKind,
    QuestionStatus,
    StuckReason,
    evidence_weight,
)

SCHEMA = SCHEMA_PATH.read_text(encoding="utf-8")


def _check_values(column: str, table: str) -> set[str]:
    """Literais de um `CHECK (<column> IN (...))`, escopado a UMA tabela.

    A `table` é obrigatória, e não é zelo: `kind` existe como coluna em `sources`,
    `questions`, `memories`, `llm_calls` e `discoveries`. Enquanto a versão sem escopo
    varria o schema inteiro, ela devolvia o PRIMEIRO CHECK que casasse — e quando a Fase D
    trocou o CHECK de `sources.kind` por uma FK, o teste passou a comparar `SourceKind`
    contra o vocabulário de `questions.kind` ('FACTUAL', 'SYNTHESIS', ...). Ele falhou por
    sorte: a comparação era de igualdade. Um `>=` teria ficado verde medindo outra tabela.
    """
    ddl = re.search(rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\);",
                    SCHEMA, re.IGNORECASE | re.DOTALL)
    assert ddl, f"não achei o CREATE TABLE de {table!r}"
    match = re.search(rf"{column}\s+IN\s*\(([^)]*)\)", ddl.group(1),
                      re.IGNORECASE | re.DOTALL)
    assert match, f"nenhum CHECK ... IN para {table}.{column}"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def test_the_source_vocabulary_is_the_registry_not_a_check():
    """A Fase D trocou `CHECK (kind IN (4 literais))` por FK para `sources_registry`.

    O CHECK era o que impedia uma fonte nova de ser NOMEADA: o INSERT era rejeitado antes
    de qualquer portão de política. Este teste trava as duas metades — que o CHECK não
    voltou, e que a FK existe — porque só a segunda deixaria o vocabulário sem aplicação
    nenhuma.

    MUTAÇÃO: restaurar o CHECK em `sources.kind`, ou trocar a FK por `TEXT NOT NULL` puro.
    """
    ddl = re.search(r"CREATE TABLE IF NOT EXISTS sources\s*\((.*?)\n\);",
                    SCHEMA, re.IGNORECASE | re.DOTALL)
    assert ddl, "não achei o CREATE TABLE de sources"
    body = ddl.group(1)
    assert not re.search(r"kind\s+TEXT[^,]*CHECK", body, re.I), (
        "`sources.kind` voltou a ter CHECK: uma fonte nova deixa de poder ser nomeada"
    )
    assert re.search(r"kind\s+TEXT NOT NULL REFERENCES sources_registry\(slug\)", body), (
        "a FK para o registro sumiu — o vocabulário de fontes ficaria sem aplicação"
    )


def test_direction_matches_schema():
    assert _check_values("direction", "claims") == {d.value for d in Direction}


def test_question_status_matches_schema():
    assert _check_values("status", "questions") >= {s.value for s in QuestionStatus}


def test_stuck_reason_matches_schema():
    assert _check_values("stuck_reason", "questions") == {s.value for s in StuckReason}


def test_grade_enum_matches_seeded_weights():
    assert set(GRADE_WEIGHT) == set(Grade)


def test_directness_enum_matches_seeded_weights():
    assert set(DIRECTNESS_WEIGHT) == set(Directness)


def test_weights_are_monotonic_in_declared_order():
    """A ordem de declaração é a ordem de força — o `rank` em store.py depende disso."""
    assert list(GRADE_WEIGHT.values()) == sorted(GRADE_WEIGHT.values(), reverse=True)
    assert list(DIRECTNESS_WEIGHT.values()) == sorted(DIRECTNESS_WEIGHT.values(), reverse=True)


def test_right_population_beats_stronger_design():
    """A invariante central, agora em Python (a versão SQL está em test_store.py)."""
    assert evidence_weight(Grade.COHORT, Directness.DIRECT) > evidence_weight(
        Grade.META_ANALYSIS, Directness.INDIRECT
    )


def test_only_factual_and_synthesis_are_auto_answerable():
    """PREFERENCE e CONTEXT precisam ir direto ao humano. Mandá-las ao loop de
    pesquisa gasta rodada e produz resposta inventada."""
    assert AUTO_ANSWERABLE == {QuestionKind.FACTUAL, QuestionKind.SYNTHESIS}
    assert QuestionKind.PREFERENCE not in AUTO_ANSWERABLE
    assert QuestionKind.CONTEXT not in AUTO_ANSWERABLE


# ────────────────────────────────────────────── schemas Pydantic → gramática GBNF


ALL_SCHEMAS = [
    schemas.QueryPlan,
    schemas.ClaimExtraction,
    schemas.ExtractedClaim,
    schemas.CitationVerdict,
    schemas.Finding,
    schemas.SufficiencyVerdict,
    schemas.Critique,
    schemas.QuestionBatch,
    schemas.QuestionClassification,
]


def test_all_schemas_forbid_extra_fields():
    """Sem `additionalProperties: false` a gramática aceita chave inventada."""
    for model in ALL_SCHEMAS:
        js = model.model_json_schema()
        targets = [js, *js.get("$defs", {}).values()]
        for target in targets:
            if target.get("type") == "object":
                assert target.get("additionalProperties") is False, model.__name__


def test_enum_fields_are_closed_sets_in_json_schema():
    """Campos de enum precisam virar `enum` no JSON Schema, não string livre —
    é o que impede o modelo de inventar um grade novo."""
    defs = schemas.ExtractedClaim.model_json_schema()["$defs"]
    assert set(defs["Grade"]["enum"]) == {g.value for g in Grade}
    assert set(defs["Directness"]["enum"]) == {d.value for d in Directness}
    assert set(defs["Direction"]["enum"]) == {d.value for d in Direction}


def test_schemas_stay_shallow():
    """Aninhamento profundo degrada aderência de um 12B mesmo com gramática.
    Dois níveis de objeto é o teto que nos demos."""

    def depth(node: dict, defs: dict, seen: frozenset[str] = frozenset()) -> int:
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name in seen:
                return 0
            return depth(defs.get(name, {}), defs, seen | {name})
        if node.get("type") == "array":
            return depth(node.get("items", {}), defs, seen)
        if node.get("type") == "object" or "properties" in node:
            children = node.get("properties", {}).values()
            return 1 + max((depth(c, defs, seen) for c in children), default=0)
        for key in ("anyOf", "oneOf"):
            if key in node:
                return max(depth(o, defs, seen) for o in node[key])
        return 0

    for model in ALL_SCHEMAS:
        js = model.model_json_schema()
        assert depth(js, js.get("$defs", {})) <= 2, model.__name__


def test_optional_blocked_reason_allows_null():
    """`blocked_reason` só é preenchido quando mais busca não resolveria — null
    precisa ser um valor legal na gramática."""
    js = schemas.SufficiencyVerdict.model_json_schema()
    variants = js["properties"]["blocked_reason"]["anyOf"]
    assert {"type": "null"} in variants
