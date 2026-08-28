"""Geração, taxonomia, dedup e priorização de perguntas.

Três propriedades são critério de aceite do plano e estão travadas aqui:

* **Roteamento.** `PREFERENCE` e `CONTEXT` nunca entram no loop de pesquisa. Mandá-las
  para lá gasta três rodadas de busca e produz resposta inventada com citações de
  aparência legítima.
* **Sem duplicatas.** Sem dedup o gerador reformula a mesma lacuna indefinidamente.
* **Teto da fila humana.** Cinco escaladas simultâneas; o excedente fica represado
  com o diagnóstico salvo, e sobe quando abre vaga.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence

import pytest

from lithium.db import Store
from lithium.llm.schemas import (
    GeneratedQuestion,
    QuestionBatch,
    QuestionClassification,
)
from lithium.pipeline.question import COST, QuestionEngine, score_priority
from lithium.pipeline.state import Coverage, build_state
from lithium.types import Directness, Grade, QuestionKind, QuestionStatus, StuckReason

from conftest import onco_profile, prod_profile

PROFILE = onco_profile()


DIM = 32


def _bucket(word: str) -> int:
    """`hash()` de str é salinizado por processo: os vetores mudariam a cada
    execução e os testes ficariam flaky. crc32 é estável entre processos."""
    return zlib.crc32(word.encode()) % DIM



class FakeEmbedder:
    """Bag-of-words hasheada: perguntas com as mesmas palavras ficam próximas."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().replace("?", "").split():
                vec[_bucket(word)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class ScriptedLLM:
    def __init__(self, batches=(), classifications=()) -> None:
        self._batches = list(batches)
        self._classifications = list(classifications)
        self.state_seen: list[str] = []

    async def structured(self, messages, schema, **kw):
        self.state_seen.append(messages[0]["content"])
        if schema is QuestionBatch:
            return self._batches.pop(0)
        if schema is QuestionClassification:
            return self._classifications.pop(0)
        raise AssertionError(f"schema inesperado: {schema}")



# Perguntas de preferência lexicalmente distintas. Variar só um número faria o
# FakeEmbedder (bag-of-words em 32 buckets) colidir e deduplicar por acidente.
PREFERENCE_QUESTIONS = [
    "Priorizar remissão da ansiedade ou estabilidade do humor?",
    "Vale aceitar ganho metabólico em troca de menor risco de virada?",
    "Sedação diurna é custo tolerável para dormir melhor à noite?",
    "Preferir monoterapia simples ou combinação mais eficaz porém complexa?",
    "Quanto risco de rash cutâneo justifica o benefício antidepressivo?",
    "Terapia semanal presencial compete com adesão medicamentosa?",
    "Desfecho funcional importa mais que redução de escore sintomático?",
    "Evitar lítio por causa de monitoramento sérico frequente?",
]


def _q(text: str, kind=QuestionKind.FACTUAL, targets="quetiapine") -> dict:
    return {"text": text, "kind": kind, "rationale": "fecha uma lacuna", "targets": targets}


def _batch(*questions: dict) -> QuestionBatch:
    return QuestionBatch.model_validate({"questions": list(questions)})


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "q.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _engine(store, llm, **kw) -> QuestionEngine:
    return QuestionEngine(store, llm, FakeEmbedder(), **kw, profile=PROFILE)


def _seed_claim(store, *, intervention: str, grade: Grade, directness: Directness,
                direction: str = "positive", external_id: str | None = None) -> None:
    source_id = store.upsert_source(
        kind="pubmed", external_id=external_id or f"{intervention}-{direction}-{grade.value}",
        raw={}, title="Estudo", year=2020,
    )
    chunk_id = store.add_chunk(source_id=source_id, ord=0,
                               text="Um trecho de apoio com tamanho suficiente para indexar.")
    _cl = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "grade, scale_id, confidence, verified) VALUES(?,?,?,?,?,?,1,1.0,1) "
        "RETURNING id",
        (source_id, json.dumps([chunk_id]), f"achado sobre {intervention}",
         intervention, direction, grade.value),
    )
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (int(_cl.fetchone()["id"]), directness.value),
    )


# ──────────────────────────────────────────────────────────────── priorização


def test_zero_evidence_outranks_well_covered():
    """A lacuna mais barata de fechar e a mais informativa por unidade de esforço."""
    empty = Coverage(intervention="ipsrt")
    covered = Coverage(intervention="quetiapine", n_claims=20, total_weight=8.0,
                       best_grade=Grade.RCT, best_directness=Directness.DIRECT,
                       positive=20)
    assert score_priority(QuestionKind.FACTUAL, empty) > score_priority(
        QuestionKind.FACTUAL, covered
    )


def test_conflict_raises_priority():
    """Fontes que discordam é o único sinal que mais-do-mesmo não resolve."""
    consensus = Coverage(intervention="a", n_claims=6, total_weight=3.0,
                         best_grade=Grade.RCT, best_directness=Directness.PARTIAL,
                         positive=6)
    conflicted = Coverage(intervention="b", n_claims=6, total_weight=3.0,
                          best_grade=Grade.RCT, best_directness=Directness.PARTIAL,
                          positive=3, negative=3)
    assert conflicted.conflict == pytest.approx(1.0)
    assert consensus.conflict == 0.0
    assert score_priority(QuestionKind.FACTUAL, conflicted) > score_priority(
        QuestionKind.FACTUAL, consensus
    )


def test_single_claim_is_not_a_conflict():
    """Uma claim isolada não é discordância — precisa de duas direções opostas."""
    assert Coverage(intervention="a", n_claims=1, positive=1).conflict == 0.0


def test_indirect_only_evidence_raises_priority():
    """Sinal na população errada é justamente o que vale perguntar como melhorar."""
    wrong_pop = Coverage(intervention="a", n_claims=4, total_weight=2.0,
                         best_grade=Grade.RCT, best_directness=Directness.INDIRECT,
                         positive=4)
    right_pop = Coverage(intervention="b", n_claims=4, total_weight=2.0,
                         best_grade=Grade.RCT, best_directness=Directness.DIRECT,
                         positive=4)
    assert wrong_pop.directness_gap > right_pop.directness_gap
    assert score_priority(QuestionKind.FACTUAL, wrong_pop) > score_priority(
        QuestionKind.FACTUAL, right_pop
    )


def test_methodological_is_deprioritised():
    """O plano manda deixá-las em prioridade baixa: custam atenção humana e raramente
    desbloqueiam algo."""
    gap = Coverage(intervention="x")
    assert score_priority(QuestionKind.METHODOLOGICAL, gap) < score_priority(
        QuestionKind.FACTUAL, gap
    )
    assert COST[QuestionKind.METHODOLOGICAL] > COST[QuestionKind.SYNTHESIS]


def test_priority_stays_in_range():
    extremes = [
        Coverage(intervention="x"),
        Coverage(intervention="y", n_claims=999, total_weight=500.0,
                 best_grade=Grade.META_ANALYSIS, best_directness=Directness.DIRECT,
                 positive=999),
    ]
    for kind in QuestionKind:
        for cov in extremes:
            assert 0.0 < score_priority(kind, cov) <= 1.0


# ──────────────────────────────────────────────────────────────── roteamento


@pytest.mark.parametrize("kind", [QuestionKind.FACTUAL, QuestionKind.SYNTHESIS])
async def test_answerable_kinds_go_to_the_research_queue(store, kind):
    llm = ScriptedLLM([_batch(_q("Quetiapina tem RCT em TAG?", kind))])
    [record] = await _engine(store, llm).generate()

    assert record.status is QuestionStatus.OPEN
    assert _engine(store, llm).next_for_research().id == record.id


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        (QuestionKind.PREFERENCE, StuckReason.NEEDS_VALUE_JUDGMENT),
        (QuestionKind.CONTEXT, StuckReason.NEEDS_CONTEXT),
    ],
)
async def test_unanswerable_kinds_escalate_immediately(store, kind, reason):
    """Nenhuma busca resolve trade-off de valores. Ir ao loop gastaria três rodadas
    e produziria resposta inventada."""
    llm = ScriptedLLM([_batch(_q("Priorizar remissão ou estabilidade?", kind))])
    [record] = await _engine(store, llm).generate()

    assert record.status is QuestionStatus.ESCALATED
    row = store.conn.execute("SELECT * FROM questions WHERE id = ?", (record.id,)).fetchone()
    assert row["stuck_reason"] == reason.value
    assert row["escalated_at"] is not None
    assert row["rounds"] == 0, "não pode ter gasto rodada de pesquisa"


async def test_escalated_questions_never_enter_the_research_queue(store):
    llm = ScriptedLLM([_batch(
        _q("Trade-off de valores", QuestionKind.PREFERENCE),
        _q("Contexto do caso", QuestionKind.CONTEXT, targets="lamotrigine"),
    )])
    engine = _engine(store, llm)
    await engine.generate()
    assert engine.next_for_research() is None


async def test_partial_work_is_saved_on_escalation(store):
    """A pergunta que chega ao humano não pode vir nua — o motivo do impasse é o que
    transforma um pedido de pesquisa numa escolha de 30 segundos."""
    llm = ScriptedLLM([_batch(_q("Priorizar o quê?", QuestionKind.PREFERENCE))])
    [record] = await _engine(store, llm).generate()
    row = store.conn.execute("SELECT partial_work FROM questions WHERE id = ?",
                             (record.id,)).fetchone()
    assert row["partial_work"] == "fecha uma lacuna"


# ────────────────────────────────────────────────────────────────────── dedup


async def test_identical_question_is_not_inserted_twice(store):
    llm = ScriptedLLM([
        _batch(_q("Quetiapina reduz ansiedade em bipolar tipo I?")),
        _batch(_q("Quetiapina reduz ansiedade em bipolar tipo I?")),
    ])
    engine = _engine(store, llm)
    assert len(await engine.generate()) == 1
    assert await engine.generate() == []
    assert store.conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"] == 1


async def test_paraphrase_is_deduped(store):
    """"quetiapina funciona em TAG?" e "há evidência de quetiapina para TAG?" são a
    mesma pergunta; sem dedup o banco enche de paráfrases."""
    llm = ScriptedLLM([
        _batch(_q("quetiapina reduz ansiedade generalizada")),
        _batch(_q("quetiapina reduz ansiedade generalizada mesmo")),
    ])
    engine = _engine(store, llm, dedup_threshold=0.85)
    await engine.generate()
    assert await engine.generate() == []


async def test_distinct_questions_both_survive(store):
    llm = ScriptedLLM([_batch(
        _q("Quetiapina tem RCT em transtorno de ansiedade generalizada?"),
        _q("Lamotrigina exige titulação lenta por conta de rash cutâneo?",
           targets="lamotrigine"),
    )])
    assert len(await _engine(store, llm).generate()) == 2


async def test_dedup_compares_against_closed_questions_too(store):
    """Comparar só com as abertas faria o gerador ressuscitar perguntas respondidas
    assim que saíssem da fila — o laço mais fácil de criar aqui."""
    llm = ScriptedLLM([
        _batch(_q("Quetiapina reduz ansiedade em bipolar?")),
        _batch(_q("Quetiapina reduz ansiedade em bipolar?")),
    ])
    engine = _engine(store, llm)
    [record] = await engine.generate()
    store.conn.execute("UPDATE questions SET status = 'ANSWERED_AUTO' WHERE id = ?",
                       (record.id,))
    assert await engine.generate() == []


# ────────────────────────────────────────────────────────── teto da fila humana


async def test_human_queue_is_capped_and_surplus_is_held(store):
    """Um sistema que entrega quarenta perguntas por dia não é usado duas vezes."""
    llm = ScriptedLLM([_batch(*[
        _q(text, QuestionKind.PREFERENCE, targets=f"alvo{i}")
        for i, text in enumerate(PREFERENCE_QUESTIONS)
    ])])
    engine = _engine(store, llm, human_queue_limit=5)
    await engine.generate(max_questions=8)

    assert engine.escalated_count() == 5
    assert len(engine.human_queue()) == 5
    held = store.conn.execute(
        "SELECT COUNT(*) AS n FROM questions "
        " WHERE status = 'OPEN' AND stuck_reason IS NOT NULL"
    ).fetchone()["n"]
    assert held == 3


async def test_held_questions_keep_their_diagnosis(store):
    """Represada não pode perder o motivo do impasse — sem ele, quando subir, chega
    nua para o humano."""
    llm = ScriptedLLM([_batch(*[
        _q(text, QuestionKind.PREFERENCE, targets=f"alvo{i}")
        for i, text in enumerate(PREFERENCE_QUESTIONS[:7])
    ])])
    await _engine(store, llm, human_queue_limit=5).generate(max_questions=7)

    held = store.conn.execute(
        "SELECT stuck_reason, partial_work FROM questions "
        " WHERE status = 'OPEN' AND stuck_reason IS NOT NULL"
    ).fetchall()
    assert all(r["stuck_reason"] and r["partial_work"] for r in held)


async def test_answering_frees_a_slot_and_promotes_the_next(store):
    llm = ScriptedLLM([_batch(*[
        _q(text, QuestionKind.PREFERENCE, targets=f"alvo{i}")
        for i, text in enumerate(PREFERENCE_QUESTIONS[:7])
    ])])
    engine = _engine(store, llm, human_queue_limit=5)
    await engine.generate(max_questions=7)

    first = engine.human_queue()[0]
    engine.answer_from_human(first.id, "priorizar estabilidade do humor")

    assert engine.escalated_count() == 5, "a vaga liberada deve ser reocupada"
    row = store.conn.execute("SELECT * FROM questions WHERE id = ?", (first.id,)).fetchone()
    assert row["status"] == "ANSWERED_HUMAN"
    assert row["answer_origin"] == "human"


async def test_promotion_respects_priority_order(store):
    """Quem sobe é a mais valiosa represada, não a mais antiga."""
    llm = ScriptedLLM([_batch(*[
        _q(text, QuestionKind.PREFERENCE, targets=f"alvo{i}")
        for i, text in enumerate(PREFERENCE_QUESTIONS[:6])
    ])])
    engine = _engine(store, llm, human_queue_limit=5)
    await engine.generate(max_questions=6)

    held_id = store.conn.execute(
        "SELECT id FROM questions WHERE status = 'OPEN' AND stuck_reason IS NOT NULL"
    ).fetchone()["id"]
    store.conn.execute("UPDATE questions SET priority = 1.0 WHERE id = ?", (held_id,))

    engine.answer_from_human(engine.human_queue()[-1].id, "resposta")
    assert held_id in {q.id for q in engine.human_queue()}


# ─────────────────────────────────────────────────────────── pergunta manual


async def test_ask_classifies_and_maximises_priority(store):
    """Se você parou para digitar, é porque quer a resposta."""
    llm = ScriptedLLM(classifications=[
        QuestionClassification(kind=QuestionKind.FACTUAL, targets="pregabalin",
                               reason="respondível pela literatura")
    ])
    record = await _engine(store, llm).ask("Pregabalina tem RCT em TAG?")

    assert record is not None
    assert record.kind is QuestionKind.FACTUAL
    assert record.priority == 1.0
    assert store.conn.execute("SELECT origin FROM questions").fetchone()["origin"] == "human"


async def test_ask_routes_preference_to_the_human_queue(store):
    llm = ScriptedLLM(classifications=[
        QuestionClassification(kind=QuestionKind.PREFERENCE, targets="",
                               reason="julgamento de valores")
    ])
    record = await _engine(store, llm).ask("Vale aceitar ansiedade residual?")
    assert record.status is QuestionStatus.ESCALATED


async def test_ask_deduplicates_against_generated_questions(store):
    llm = ScriptedLLM(
        batches=[_batch(_q("Quetiapina reduz ansiedade em bipolar tipo I?"))],
        classifications=[
            QuestionClassification(kind=QuestionKind.FACTUAL, targets="quetiapine",
                                   reason="respondível")
        ],
    )
    engine = _engine(store, llm)
    await engine.generate()
    assert await engine.ask("Quetiapina reduz ansiedade em bipolar tipo I?") is None


# ───────────────────────────────────────────────────── estado do conhecimento


def test_state_exposes_untouched_intervention_classes(store):
    """A cobertura é por CLASSE, não por fármaco.

    A lista fixa de fármacos que existia antes era uma gaiola: só tornava visível a
    lacuna dentro dela própria. Com classes, o gerador vê "ninguém olhou
    neuromodulação" e fica livre para nomear qualquer coisa dentro daquele espaço —
    inclusive algo que não existe como tratamento psiquiátrico hoje.
    """
    _seed_claim(store, intervention="doxorubicin", grade=Grade.RCT,
                directness=Directness.PARTIAL)
    state = build_state(store, PROFILE)

    assert any(c.intervention == "doxorubicin" for c in state.coverage)
    # doxorrubicina é antraciclina: a CLASSE deixa de estar intocada
    assert "anthracycline" not in state.untouched
    # e as classes longe da prática padrão seguem visíveis como lacuna
    assert "oncolytic virotherapy" in state.untouched
    assert "metronomic maintenance" in state.untouched
    # Contra o PERFIL DE TESTE: nenhuma destas strings existe em `lithium/`, então o
    # caminho classe -> untouched -> $state -> prompt só pode tê-las lido do perfil.
    # A armadilha aqui seria "consertar" o teste trocando o literal por
    # `next(iter(profile.taxonomy.intervention_classes))` — aí a expectativa passaria
    # a vir da mesma fonte que ela policia, que é o defeito tautológico do repo.
    assert not any("neuromodulation" in c for c in state.untouched), (
        "vazou a taxonomia do perfil de produção"
    )


def test_untouched_classes_are_mostly_not_drugs(store):
    """Contrato de FORMA do perfil de PRODUÇÃO, e ele SOBREVIVE à parametrização.

    Metade das classes rastreadas precisa ser não-farmacológica, senão o sistema volta
    a ser um catálogo de medicamentos com outro nome. Roda contra produção porque é
    uma afirmação sobre a calibração que ESTE projeto escolheu.
    """
    labels = [c.label for c in prod_profile().taxonomy.intervention_classes]
    non_drug = [
        k for k in labels
        if any(t in k for t in ("psychotherapy", "neuromodulation", "chronotherapy",
                                "metabolic", "exercise", "device", "sequencing",
                                "deprescribing", "combination"))
    ]
    assert len(non_drug) >= len(labels) / 2


def test_state_reports_direction_split_and_conflict(store):
    _seed_claim(store, intervention="quetiapine", grade=Grade.RCT,
                directness=Directness.PARTIAL, direction="positive", external_id="a")
    _seed_claim(store, intervention="quetiapine", grade=Grade.COHORT,
                directness=Directness.DIRECT, direction="negative", external_id="b")

    state = build_state(store, PROFILE)
    coverage = state.by_intervention("quetiapine")
    assert (coverage.positive, coverage.negative) == (1, 1)
    assert coverage.conflict == pytest.approx(1.0)
    assert coverage in state.conflicts


def test_state_keeps_the_strongest_grade_and_directness(store):
    _seed_claim(store, intervention="lithium", grade=Grade.CASE_REPORT,
                directness=Directness.EXTRAPOLATED, external_id="a")
    _seed_claim(store, intervention="lithium", grade=Grade.META_ANALYSIS,
                directness=Directness.DIRECT, external_id="b")

    coverage = build_state(store, PROFILE).by_intervention("lithium")
    assert coverage.best_grade is Grade.META_ANALYSIS
    assert coverage.best_directness is Directness.DIRECT
    assert coverage.directness_gap == 0.0


def test_state_ignores_unverified_claims(store):
    source_id = store.upsert_source(kind="pubmed", external_id="x", raw={})
    store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "grade, scale_id, verified) VALUES(?,'[]','s','quetiapine','positive','rct',1,0)",
        (source_id,),
    )
    assert build_state(store, PROFILE).coverage == []


def test_rendered_state_lists_prior_questions(store):
    """O gerador precisa ver o que já foi perguntado, senão reformula sem parar."""
    store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status) "
        "VALUES(1, 'Quetiapina tem RCT em TAG?', 'FACTUAL', 'ANSWERED_AUTO')"
    )
    rendered = build_state(store, PROFILE).render()
    assert "do NOT repeat" in rendered
    assert "Quetiapina tem RCT em TAG?" in rendered


def test_rendered_state_survives_an_empty_corpus(store):
    rendered = build_state(store, PROFILE).render()
    assert "nothing extracted yet" in rendered
    assert "zero evidence gathered" in rendered
    assert "oncolytic virotherapy" in rendered


async def test_generator_receives_the_state_not_the_corpus(store):
    """O gerador vê o resumo agregado, não os abstracts. Com 30 abstracts na janela
    ele produz perguntas sobre o que estava no texto, não sobre o que falta."""
    _seed_claim(store, intervention="quetiapine", grade=Grade.RCT,
                directness=Directness.INDIRECT)
    llm = ScriptedLLM([_batch(_q("Pergunta qualquer sobre a lacuna encontrada"))])
    await _engine(store, llm).generate()

    prompt = llm.state_seen[0]
    assert "Evidence coverage by intervention" in prompt
    assert "quetiapine" in prompt
    assert "Um trecho de apoio" not in prompt, "texto bruto do corpus vazou no prompt"


# ────────────────────────────────────── dedup por alvo (regressão de caso real)


async def test_different_targets_are_never_duplicates(store):
    """Regressão medida num lote real do 12B.

    Cinco perguntas sobre cinco intervenções distintas — buspirona, pregabalina,
    antidepressivo, TCC, quetiapina — ficaram em 0.767–0.874 de similaridade entre si,
    porque compartilham o vocabulário do domínio ("bipolar I", "GAD", "efficacy") que
    domina o embedding. Essa faixa sobrepõe inteiramente a das paráfrases genuínas
    (0.788–0.952), então NENHUM limiar resolve. Só o escopo por alvo.

    Sem isto, um plan tick de 5 propostas rendia 1 pergunta.
    """
    template = "Qual a eficácia e segurança de {} para ansiedade comórbida no bipolar tipo I?"
    llm = ScriptedLLM([_batch(*[
        _q(template.format(drug), targets=drug)
        for drug in ("buspirona", "pregabalina", "quetiapina", "lamotrigina", "lítio")
    ])])
    # Limiar baixo o bastante para que o texto sozinho deduplicaria tudo.
    added = await _engine(store, llm, dedup_threshold=0.5).generate(max_questions=5)
    assert len(added) == 5, "perguntas sobre intervenções diferentes foram fundidas"


async def test_same_target_still_dedupes_paraphrases(store):
    """O escopo por alvo não pode desligar o dedup dentro do alvo."""
    llm = ScriptedLLM([
        _batch(_q("Quetiapina reduz ansiedade generalizada?", targets="quetiapine")),
        _batch(_q("Quetiapina reduz ansiedade generalizada mesmo?", targets="quetiapine")),
    ])
    engine = _engine(store, llm, dedup_threshold=0.85)
    assert len(await engine.generate()) == 1
    assert await engine.generate() == []


async def test_same_target_keeps_genuinely_distinct_questions(store):
    """Dentro de um alvo o texto volta a discriminar: eficácia e efeito adverso são
    perguntas diferentes sobre o mesmo fármaco."""
    llm = ScriptedLLM([_batch(
        _q("Quetiapina tem ensaio randomizado em ansiedade generalizada?",
           targets="quetiapine"),
        _q("Quetiapina provoca ganho de peso clinicamente relevante?",
           targets="quetiapine"),
    )])
    assert len(await _engine(store, llm, dedup_threshold=0.78).generate()) == 2


async def test_target_is_persisted_lowercased(store):
    llm = ScriptedLLM([_batch(_q("Pergunta sobre o fármaco", targets="  Quetiapine  "))])
    await _engine(store, llm).generate()
    assert store.conn.execute("SELECT targets FROM questions").fetchone()["targets"] == "quetiapine"


async def test_questions_without_target_fall_back_to_global_dedup(store):
    """Sem alvo declarado não há escopo — comparar globalmente é o único recurso."""
    llm = ScriptedLLM([
        _batch(_q("Como devemos enquadrar a busca?", QuestionKind.METHODOLOGICAL, targets="")),
        _batch(_q("Como devemos enquadrar a busca?", QuestionKind.METHODOLOGICAL, targets="")),
    ])
    engine = _engine(store, llm)
    assert len(await engine.generate()) == 1
    assert await engine.generate() == []
