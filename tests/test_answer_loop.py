"""O loop que consome a fila de perguntas — item 7.

A lacuna que ele fecha era a mais visível do sistema: perguntas eram geradas,
classificadas e priorizadas, e ficavam em `OPEN` para sempre.

**Metade destes testes existe por causa de um padrão de falha do próprio projeto.** Três
desenhos independentes deste loop foram auditados, e os três entregaram a peça **sem
produtor**: depois de aplicar qualquer um deles, `HANDLERS` não ganhava entrada, o
scheduler não ganhava `Job`, e o buraco continuava aberto byte por byte — com os testes
novos passando, porque chamavam o seam direto. Foi a sexta ocorrência dessa classe aqui.
Por isso a seção final testa a **fiação**, e o ramo que aprova uma resposta é exercitado
positivamente: nos três desenhos ele nunca chegava ao páreo, e em todos os três ele
explodia.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence

import pytest

from lithium.db import Store
from lithium.llm.schemas import SufficiencyVerdict
from lithium.pipeline.answer import (
    AUTO_QUEUE_CAP,
    Action,
    Answerer,
    evidence_fingerprint,
)
from lithium.types import Directness, Grade, QuestionKind, QuestionStatus

from conftest import prod_profile
from lithium.safety import ruleset_from_profile

# Perfil de PRODUÇÃO: o assunto deste arquivo é o vocabulário de segurança real
# (lítio, valproato, benzodiazepínico). Rodá-lo contra o perfil de teste — que declara
# `safety = false` — o deixaria verde afirmando sobre um ruleset que não existe.
PROFILE = prod_profile()
RULESET = ruleset_from_profile(PROFILE)


DIM = 16
QUESTION = "quetiapina reduz ansiedade no bipolar I com TAG?"


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def _verdict(**kw) -> SufficiencyVerdict:
    base = dict(sufficient=True, n_independent_sources=3,
                addresses_question_directly=True, sources_agree=True,
                missing="", blocked_reason=None)
    return SufficiencyVerdict(**{**base, **kw})


class ScriptedLLM:
    """Conta as chamadas por rótulo — "não gastou" é uma asserção tão importante
    quanto "respondeu"."""

    def __init__(self, verdicts=None, answer: str = "Resposta. [PMID:111]") -> None:
        self._verdicts = list(verdicts or [_verdict()])
        self.answer = answer
        self.calls: list[str] = []

    async def structured(self, messages, schema, **kw):
        self.calls.append(kw.get("label", "?"))
        assert schema is SufficiencyVerdict, f"schema inesperado: {schema}"
        return self._verdicts.pop(0) if self._verdicts else _verdict()

    async def complete(self, messages, **kw):
        self.calls.append(kw.get("label", "?"))
        return self.answer


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "a.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _answerer(store, llm=None) -> Answerer:
    return Answerer(store, llm or ScriptedLLM(), FakeEmbedder(), profile=PROFILE)


async def _claim(store, pmid: str, statement: str, *, doi: str | None = None) -> None:
    sid = store.upsert_source(kind="pubmed", external_id=pmid, raw={}, title="t",
                              year=2020, doi=doi)
    cid = store.add_chunk(source_id=sid, ord=0,
                          text=f"{statement} quetiapina ansiedade bipolar TAG contexto.")
    [vector] = await FakeEmbedder().embed([statement])
    store.set_chunk_embedding(cid, vector)
    _c = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "  grade, scale_id, confidence, verified) "
        "VALUES(?,?,?,'quetiapine','positive',?,1,0.9,1) RETURNING id",
        (sid, json.dumps([cid]), statement, Grade.RCT.value),
    )
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (int(_c.fetchone()["id"]), Directness.DIRECT.value),
    )


def _question(store, text=QUESTION, *, kind=QuestionKind.FACTUAL, priority=0.5,
              status="OPEN") -> int:
    cur = store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status, priority) "
        "VALUES(1,?,?,?,?) RETURNING id",
        (text, kind.value, status, priority),
    )
    return int(cur.fetchone()["id"])


def _row(store, question_id: int):
    return store.conn.execute(
        "SELECT * FROM questions WHERE id = ?", (question_id,)
    ).fetchone()


# ═══════════════════════════════════ 1. o ramo que aprova — o motivo do item


async def test_a_question_with_enough_evidence_is_answered(store):
    """O ramo ANSWER. Nos três desenhos auditados ele nunca era exercitado por teste
    e sempre explodia em produção."""
    await _claim(store, "111", "Quetiapina reduziu ansiedade em bipolar I.")
    await _claim(store, "222", "Quetiapina reduziu HAM-A versus placebo.")
    qid = _question(store)

    result = await _answerer(store).round(qid)

    assert result.action is Action.ANSWER
    row = _row(store, qid)
    assert row["status"] == QuestionStatus.ANSWERED_AUTO.value
    assert row["answer_origin"] == "auto"
    assert row["answer"] and row["answered_at"]


async def test_the_answer_is_persisted_as_a_finding_with_its_citations(store):
    """`findings` tinha zero escritores. E as citações vêm dos claim ids que estavam
    NO PROMPT, nunca do que o modelo escreveu."""
    await _claim(store, "111", "Quetiapina reduziu ansiedade em bipolar I.")
    await _claim(store, "222", "Quetiapina reduziu HAM-A versus placebo.")
    qid = _question(store)

    result = await _answerer(store).round(qid)

    finding = store.conn.execute(
        "SELECT * FROM findings WHERE id = ?", (result.finding_id,)
    ).fetchone()
    assert finding["question_id"] == qid
    pmids = {c["external_id"] for c in json.loads(finding["citations_json"])}
    assert pmids == {"111", "222"}


async def test_the_loop_never_stamps_an_answer_as_human(store):
    """`training_examples` dá peso 3.0 a material humano, e o rótulo é irreversível."""
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    await _claim(store, "222", "Quetiapina reduziu HAM-A.")
    qid = _question(store)
    await _answerer(store).round(qid)

    assert _row(store, qid)["answer_origin"] == "auto"


async def test_nothing_from_this_loop_is_exported_for_training(store):
    """Decisão de escopo, e ela é o que fecha três vazamentos por não criá-los.

    `training_examples` tem zero escritores **e zero leitores**, e o LoRA é o item 12.
    Três desenhos independentes deste loop vazaram material não-verificado para lá por
    caminhos diferentes: divisor de sentenças truncando a nona proposição, `caveats`
    nunca julgado, e prosa de dosagem fundindo duas proposições num veredito só.
    """
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    await _claim(store, "222", "Quetiapina reduziu HAM-A.")
    await _answerer(store).round(_question(store))

    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM training_examples"
    ).fetchone()["n"] == 0


# ═════════════════════════════ 2. o juiz: quatro campos, não só o booleano


@pytest.mark.parametrize("flipped", [
    {"sufficient": False},
    {"addresses_question_directly": False},
    {"n_independent_sources": 1},
    {"blocked_reason": "NEEDS_CONTEXT"},
])
async def test_each_field_of_the_verdict_can_block_the_answer(store, flipped):
    """Um portão que lê só `sufficient` torna os outros decorativos — e foi exatamente
    o defeito que a auditoria construiu no portão de derivação da Fase 2, com a função
    pura perfeita e o call site checando uma resposta de quatro.

    Comportamental: cada campo é virado e a asserção é sobre o ESTADO gravado.
    """
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    await _claim(store, "222", "Quetiapina reduziu HAM-A.")
    qid = _question(store)

    result = await _answerer(store, ScriptedLLM([_verdict(**flipped)])).round(qid)

    assert result.action is not Action.ANSWER, f"{list(flipped)[0]} não bloqueou"
    assert _row(store, qid)["status"] != QuestionStatus.ANSWERED_AUTO.value


async def test_a_blocked_reason_escalates_instead_of_burning_rounds(store):
    """`blocked_reason` significa "mais busca não resolve". Ignorá-lo gastaria o teto
    inteiro para chegar à mesma conclusão."""
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    qid = _question(store)
    llm = ScriptedLLM([_verdict(sufficient=False, blocked_reason="NEEDS_CONTEXT",
                                missing="o que já foi tentado")])

    result = await _answerer(store, llm).round(qid)

    assert result.action is Action.ESCALATE
    row = _row(store, qid)
    assert row["status"] == QuestionStatus.ESCALATED.value
    assert row["stuck_reason"] == "NEEDS_CONTEXT"
    assert row["rounds"] == 1, "escalou na primeira rodada, sem gastar o teto"


async def test_the_judge_never_sees_a_draft(store):
    """Se visse, julgaria a redação. E o repo registra duas vezes que modelo pequeno
    auto-avaliando o que escreveu diz "suficiente" quase sempre."""
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    qid = _question(store)
    llm = ScriptedLLM([_verdict(sufficient=False)])

    await _answerer(store, llm).round(qid)

    assert llm.calls == ["judge_sufficiency"], (
        f"a síntese rodou antes do juiz: {llm.calls}"
    )


async def test_an_unanswerable_question_costs_one_call_per_round(store):
    """O custo medido: julgar primeiro gasta 23 s numa pergunta irrespondível;
    sintetizar primeiro gasta 44 s a mais por rodada, jogados fora."""
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    qid = _question(store)
    llm = ScriptedLLM([_verdict(sufficient=False)])

    await _answerer(store, llm).round(qid)
    assert llm.calls.count("answer_question") == 0


# ══════════════════════════ 3. rodada sem material novo não gasta orçamento


async def test_an_unchanged_corpus_does_not_consume_a_round(store):
    """`search_claims` é função pura de (consulta, corpus): a rodada N+1 sobre um
    corpus imutável devolve a mesma lista, byte a byte.

    Se essa rodada consumisse orçamento, o teto viraria relógio de parede — e a
    simulação de 30 dias dessa variante fechou 132 de 150 perguntas em silêncio, sem
    ninguém ter lido nenhuma.
    """
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    qid = _question(store)
    llm = ScriptedLLM([_verdict(sufficient=False), _verdict(sufficient=False)])
    answerer = _answerer(store, llm)

    await answerer.round(qid)
    after_first = _row(store, qid)["rounds"]

    result = await answerer.round(qid)      # corpus inalterado

    assert result.action is Action.SEARCH_AGAIN
    assert result.reason == "corpus inalterado"
    assert _row(store, qid)["rounds"] == after_first, "a rodada vazia gastou orçamento"
    assert llm.calls.count("judge_sufficiency") == 1, "pagou o juiz de novo"


async def test_a_grown_corpus_does_consume_a_round(store):
    """A contrapartida: sem ela, um portão que nunca deixa passar seria indistinguível
    de um que sempre deixa."""
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    qid = _question(store)
    llm = ScriptedLLM([_verdict(sufficient=False), _verdict(sufficient=False)])
    answerer = _answerer(store, llm)

    await answerer.round(qid)
    before = _row(store, qid)["rounds"]
    await _claim(store, "222", "Quetiapina reduziu HAM-A em bipolar I.")
    await answerer.round(qid)

    assert _row(store, qid)["rounds"] == before + 1
    assert llm.calls.count("judge_sufficiency") == 2


async def test_the_fingerprint_is_order_independent():
    assert evidence_fingerprint([3, 1, 2]) == evidence_fingerprint([1, 2, 3])
    assert evidence_fingerprint([]) != evidence_fingerprint([1])


async def test_the_round_budget_ends_in_escalation(store):
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    qid = _question(store)
    answerer = Answerer(store, ScriptedLLM([_verdict(sufficient=False)] * 5),
                        FakeEmbedder(), profile=PROFILE, max_rounds=2)

    await answerer.round(qid)
    await _claim(store, "222", "Outra claim sobre quetiapina e ansiedade.")
    result = await answerer.round(qid)

    assert result.action is Action.ESCALATE
    row = _row(store, qid)
    assert row["status"] == QuestionStatus.ESCALATED.value
    assert row["stuck_reason"] == "INSUFFICIENT_EVIDENCE"


# ═══════════════════════════════ 4. o despachante e a reconciliação


async def test_a_question_already_in_flight_is_not_dispatched_twice(store):
    qid = _question(store)
    store.conn.execute(
        "INSERT INTO tasks(kind, payload_json, status, priority) "
        "VALUES('answer_question', ?, 'pending', 0.6)",
        (json.dumps({"question_id": qid}),),
    )
    assert _answerer(store).dispatchable() == []


async def test_a_question_at_the_round_cap_is_not_dispatched(store):
    qid = _question(store)
    store.conn.execute("UPDATE questions SET rounds = 3 WHERE id = ?", (qid,))
    assert _answerer(store).dispatchable() == []


async def test_only_auto_answerable_kinds_are_dispatched(store):
    _question(store, "preferência?", kind=QuestionKind.PREFERENCE)
    factual = _question(store, "fato?", kind=QuestionKind.FACTUAL)
    assert _answerer(store).dispatchable() == [factual]


async def test_a_dead_lettered_task_does_not_strand_the_question(store):
    """`TaskQueue.recover_orphans()` não cobre isto: ele mexe em `tasks`, e a pergunta é
    um SEGUNDO estado. Sem reconciliação ela fica em RESEARCHING para sempre."""
    qid = _question(store, status="RESEARCHING")
    store.conn.execute(
        "INSERT INTO tasks(kind, payload_json, status, priority) "
        "VALUES('answer_question', ?, 'dead', 0.6)",
        (json.dumps({"question_id": qid}),),
    )

    assert _answerer(store).release_stranded() == 1
    assert _row(store, qid)["status"] == QuestionStatus.OPEN.value


async def test_a_payload_without_question_id_does_not_disable_the_reconciler(store):
    """`NOT IN` com um único NULL no conjunto devolve NULL para TODA linha: uma tarefa
    sem `question_id` desligaria a reconciliação inteira, em silêncio."""
    qid = _question(store, status="RESEARCHING")
    store.conn.execute(
        "INSERT INTO tasks(kind, payload_json, status, priority) "
        "VALUES('answer_question', '{}', 'pending', 0.6)"
    )
    assert _answerer(store).release_stranded() == 1
    assert _row(store, qid)["status"] == QuestionStatus.OPEN.value


async def test_a_live_task_keeps_its_question_in_researching(store):
    qid = _question(store, status="RESEARCHING")
    store.conn.execute(
        "INSERT INTO tasks(kind, payload_json, status, priority) "
        "VALUES('answer_question', ?, 'running', 0.6)",
        (json.dumps({"question_id": qid}),),
    )
    assert _answerer(store).release_stranded() == 0


async def test_the_auto_queue_is_capped(store):
    """O teto é 15, não 40: o bloco "não repita" degrada muito antes, e na ordem
    inversa — pergunta aberta expulsa primeiro a já respondida."""
    for i in range(AUTO_QUEUE_CAP + 5):
        _question(store, f"pergunta {i}", priority=1.0 - i / 100)

    assert _answerer(store).park_overflow() == 5
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM questions WHERE status = 'OPEN'"
    ).fetchone()["n"] == AUTO_QUEUE_CAP


async def test_the_cap_closes_the_weakest_first(store):
    forte = _question(store, "forte", priority=0.9)
    for i in range(AUTO_QUEUE_CAP):
        _question(store, f"fraca {i}", priority=0.1)

    _answerer(store).park_overflow()
    assert _row(store, forte)["status"] == QuestionStatus.OPEN.value


# ═════════════════════════════════════ 5. A FIAÇÃO — onde três desenhos falharam


def test_the_handler_is_registered():
    """Sem entrada em HANDLERS o loop não existe, e a suíte inteira fica verde."""
    from lithium.worker.handlers import HANDLERS

    assert "answer_tick" in HANDLERS
    assert "answer_question" in HANDLERS


def test_the_scheduler_dispatches_the_loop():
    """Sem Job, o handler nunca roda sozinho — e a fila continua em OPEN para sempre,
    que é exatamente o buraco que o item 7 existe para fechar."""
    from lithium.worker.scheduler import DEFAULT_JOBS

    kinds = {job.task_kind for job in DEFAULT_JOBS}
    assert "answer_tick" in kinds


def test_every_scheduled_job_has_a_handler():
    """Um `Job` apontando para um handler inexistente vira dead-letter silencioso."""
    from lithium.worker.handlers import HANDLERS
    from lithium.worker.scheduler import DEFAULT_JOBS

    missing = [job.task_kind for job in DEFAULT_JOBS if job.task_kind not in HANDLERS]
    assert not missing, f"Jobs sem handler: {missing}"


async def test_the_tick_enqueues_a_task_per_dispatchable_question(store, tmp_path):
    """Ponta a ponta pelo handler real: a fila precisa SAIR de OPEN."""
    from lithium.config import Config
    from lithium.worker.handlers import HANDLERS
    from lithium.worker.queue import TaskQueue

    for i in range(3):
        _question(store, f"pergunta {i}")

    queue = TaskQueue(store)

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.queue = queue
            self.llm = ScriptedLLM()
            self.embedder = FakeEmbedder()
            self.config = Config(data_dir=tmp_path)

    await HANDLERS["answer_tick"]({}, Ctx())

    tasks = store.conn.execute(
        "SELECT payload_json FROM tasks WHERE kind = 'answer_question'"
    ).fetchall()
    assert len(tasks) == 3


async def test_the_round_handler_answers_end_to_end(store, tmp_path):
    """O handler `answer_question` real, com o ramo que aprova. Reverter o corpo dele
    para `return None` deixava a suíte verde nos três desenhos auditados."""
    from lithium.config import Config
    from lithium.worker.handlers import HANDLERS

    await _claim(store, "111", "Quetiapina reduziu ansiedade em bipolar I.")
    await _claim(store, "222", "Quetiapina reduziu HAM-A versus placebo.")
    qid = _question(store)

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.queue = None
            self.llm = ScriptedLLM()
            self.embedder = FakeEmbedder()
            self.config = Config(data_dir=tmp_path)

    await HANDLERS["answer_question"]({"question_id": qid}, Ctx())

    assert _row(store, qid)["status"] == QuestionStatus.ANSWERED_AUTO.value
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM findings"
    ).fetchone()["n"] == 1


async def test_escalation_respects_the_human_queue_cap(store):
    """Bug encontrado pela auditoria do item 7.5.

    `_escalate` gravava `ESCALATED` sem consultar vaga, enquanto `QuestionEngine`
    respeita o teto de 5. O loop **triplica** a taxa de escalação, então sem isto a fila
    humana enche — e fila cheia esconde as que importam, que é a razão do teto.
    """
    from lithium.pipeline.answer import HUMAN_QUEUE_LIMIT

    for i in range(HUMAN_QUEUE_LIMIT):
        store.conn.execute(
            "INSERT INTO questions(focus_id, text, kind, status) "
                "VALUES(1, ?, 'CONTEXT', 'ESCALATED')",
            (f"já escalada {i}",),
        )
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    qid = _question(store)
    llm = ScriptedLLM([_verdict(sufficient=False, blocked_reason="NEEDS_CONTEXT")])

    await _answerer(store, llm).round(qid)

    row = _row(store, qid)
    assert row["status"] == QuestionStatus.OPEN.value, "estourou o teto da fila humana"
    assert row["stuck_reason"] == "NEEDS_CONTEXT", "represada, não perdida"
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM questions WHERE status = 'ESCALATED'"
    ).fetchone()["n"] == HUMAN_QUEUE_LIMIT


# ══════════════════ a calibração do juiz, nos DOIS lados


def test_the_gating_field_is_documented_in_both_places():
    """O bug que fez o loop gravar zero achados em produção.

    `sufficient` era o **único** campo de `SufficiencyVerdict` sem descrição e o único
    ausente da seção de campos do prompt — e é o campo que gateia o loop inteiro. Medido
    contra o modelo real: **29 de 29** chamadas devolveram `sufficient: false`, sempre
    com `addresses_question_directly=true` e 2 a 8 fontes independentes. O `findings`
    recebia zero linhas.

    É a mesma falha que este repo já pagou e documentou em `critique_speculation.md`:
    "se você exige prova de segurança antes de uma hipótese sobreviver, só tratamento
    estabelecido passa".
    """
    from lithium.llm.prompts import PROMPTS_DIR
    from lithium.llm.schemas import SufficiencyVerdict

    for name, field in SufficiencyVerdict.model_fields.items():
        assert field.description, f"campo {name!r} sem descrição no schema"

    prompt = (PROMPTS_DIR / "judge_sufficiency.md").read_text(encoding="utf-8")
    for name in SufficiencyVerdict.model_fields:
        assert f"`{name}`" in prompt, f"campo {name!r} não aparece no prompt do juiz"


def test_the_prompt_separates_absence_from_weakness():
    """A calibração que destrava o juiz sem virar carimbo: recusa é por AUSÊNCIA."""
    from lithium.llm.prompts import PROMPTS_DIR

    prompt = (PROMPTS_DIR / "judge_sufficiency.md").read_text(encoding="utf-8")
    assert "Refuse for **absence**, never for weakness" in prompt
    assert "head-to-head" in prompt, (
        "a escassez estrutural precisa estar nomeada: RCT de TAG exclui bipolar, RCT de "
        "bipolar trata ansiedade como desfecho secundário, e nenhuma busca conserta isso"
    )


def test_the_prompt_gives_the_judge_no_prior_about_the_base_rate():
    """A auditoria propôs "espere que a maioria seja suficiente" — e isso está errado.

    Um juiz **por instância** não deve receber prior sobre a distribuição agregada: com
    ele, a peça deixa de julgar a evidência e passa a cumprir uma cota. Foi exatamente o
    defeito que o mutante de carimbo explorou.
    """
    from lithium.llm.prompts import PROMPTS_DIR

    prompt = (PROMPTS_DIR / "judge_sufficiency.md").read_text(encoding="utf-8").lower()
    for banned in ("expect most", "refusal rate", "a maioria das perguntas"):
        assert banned not in prompt, f"prior de taxa-base no prompt: {banned!r}"


REFUSAL_REGIMES = [
    ("uma fonte só", dict(sufficient=True, n_independent_sources=1)),
    ("intervenção errada", dict(sufficient=True, addresses_question_directly=False)),
    ("bloqueado", dict(sufficient=True, blocked_reason="IRRECONCILABLE_CONFLICT")),
]


@pytest.mark.parametrize("label,verdict_kw", REFUSAL_REGIMES)
async def test_the_gate_still_refuses_after_the_calibration_fix(store, label, verdict_kw):
    """O lado da RECUSA, que é o que a correção põe em risco.

    Um probe que só verifica "deixou de recusar tudo" não distingue "funciona" de
    "carimba tudo" — e carimbar é o modo de falha que afrouxar o prompt cria. Estes
    casos são o contrapeso: cada um vem com `sufficient=True` e ainda assim tem de ser
    reprovado por outro campo.
    """
    await _claim(store, "111", "Quetiapina reduziu ansiedade.")
    await _claim(store, "222", "Quetiapina reduziu HAM-A.")
    qid = _question(store)

    result = await _answerer(store, ScriptedLLM([_verdict(**verdict_kw)])).round(qid)

    assert result.action is not Action.ANSWER, f"{label} foi carimbado"
    assert _row(store, qid)["status"] != QuestionStatus.ANSWERED_AUTO.value


async def test_an_empty_corpus_is_never_answered(store):
    """Sem claim nenhuma não há o que citar, e o portão não pode depender do juiz para
    isso — um juiz carimbador produziria uma resposta sem fonte."""
    qid = _question(store)
    result = await _answerer(store, ScriptedLLM([_verdict()])).round(qid)

    assert result.action is not Action.ANSWER
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM findings"
    ).fetchone()["n"] == 0


# ═══════════════ o screen determinístico roda sobre o achado


async def test_the_finding_carries_the_safety_alerts(store):
    """O achado é o que o psiquiatra lê. Se o lembrete aparece no chat e não aqui, o
    canal que importa é justamente o desprotegido — e `findings.safety_json` tinha zero
    escritores."""
    await _claim(store, "111", "Sertralina reduziu ansiedade em bipolar I.")
    await _claim(store, "222", "Sertralina reduziu HAM-A.")
    qid = _question(store)
    llm = ScriptedLLM(answer="Sertralina em monoterapia reduziu ansiedade. [PMID:111]")

    result = await _answerer(store, llm).round(qid)

    alerts = json.loads(store.conn.execute(
        "SELECT safety_json FROM findings WHERE id = ?", (result.finding_id,)
    ).fetchone()["safety_json"])
    assert any(a["key"] == "antidepressant_monotherapy" for a in alerts), (
        f"o alerta de virada não chegou ao achado: {[a['key'] for a in alerts]}"
    )


async def test_the_negated_register_of_synthesis_prose_still_alerts(store):
    """O caso auto-anulante, e é o mais importante da tabela.

    `_NEGATION` foi calibrada na Fase 1 para o registro de restrição do usuário ("o
    paciente não tolera valproato"). Rodada sobre prosa de síntese ela perde 9 de 14
    alertas — e o pior caso é `abrupt_discontinuation` sendo desligado por
    `interromp\\w*`, ou seja pelo vocabulário que ele existe para pegar.
    """
    from lithium.safety import Segment, screen

    prosa = ("Pacientes que interromperam o lítio abruptamente tiveram mania de "
             "rebote em duas coortes.")

    assert "abrupt_discontinuation" in {
        a.key for a in screen([Segment("reply", prosa)], RULESET)
    }, "o alerta de mania de rebote foi silenciado pelo próprio vocabulário dele"

    # E o registro do chat continua com supressão — a mudança é confinada ao canal novo.
    assert screen([Segment("user", "nunca tomei lítio na vida")]) == []
    assert screen([Segment("evidence", "The patient denies lithium use.")]) == []


async def test_a_clean_finding_gets_an_empty_list_not_a_clean_bill(store):
    """Lista vazia é lista vazia. Um atestado afirmativo de limpeza seria pior que
    silêncio: o screen é um casamento de termo pequeno, não checagem de interação."""
    await _claim(store, "111", "Exercício aeróbico reduziu ansiedade.")
    await _claim(store, "222", "Exercício reduziu HAM-A em coorte.")
    qid = _question(store)
    llm = ScriptedLLM(answer="Exercício aeróbico reduziu ansiedade. [PMID:111]")

    result = await _answerer(store, llm).round(qid)
    row = store.conn.execute(
        "SELECT safety_json FROM findings WHERE id = ?", (result.finding_id,)
    ).fetchone()

    assert json.loads(row["safety_json"]) == []


async def test_the_screen_cannot_change_the_finding_text(store):
    """A propriedade portante do módulo: ANEXA, nunca altera nem suprime."""
    await _claim(store, "111", "Sertralina reduziu ansiedade.")
    await _claim(store, "222", "Sertralina reduziu HAM-A.")
    qid = _question(store)
    texto = "Sertralina em monoterapia reduziu ansiedade. [PMID:111]"
    llm = ScriptedLLM(answer=texto)

    result = await _answerer(store, llm).round(qid)
    row = store.conn.execute(
        "SELECT text FROM findings WHERE id = ?", (result.finding_id,)
    ).fetchone()

    assert row["text"] == texto
    assert _row(store, qid)["answer"] == texto


async def test_the_evidence_reaches_the_screen_so_co_terms_can_fire(store):
    """Co-termo é avaliado sobre a UNIÃO do turno, então a evidência tem de chegar lá.

    O texto do achado aqui carrega o co-termo ("interrupção abrupta") e **nenhum
    fármaco**; a claim citada carrega o fármaco. Nenhum dos dois sozinho dispara — só a
    união. Sem os segmentos de evidência, o alerta mais importante da tabela nunca sai
    num achado que fala de descontinuação.
    """
    await _claim(store, "111", "Lítio reduziu recaída em bipolar I.")
    await _claim(store, "222", "Lítio reduziu recaída em coorte de manutenção.")
    qid = _question(store)
    llm = ScriptedLLM(
        answer="A interrupção abrupta do estabilizador foi associada a recaída. [PMID:111]"
    )

    result = await _answerer(store, llm).round(qid)
    alerts = {a["key"] for a in json.loads(store.conn.execute(
        "SELECT safety_json FROM findings WHERE id = ?", (result.finding_id,)
    ).fetchone()["safety_json"])}

    assert "abrupt_discontinuation" in alerts, (
        f"a evidência não chegou ao screen: só {sorted(alerts)}"
    )


# ═══════════════════════════════════ o piso de citações conta ARTIGOS


async def test_two_claims_from_the_same_article_do_not_satisfy_the_floor(store):
    """O piso é de FONTES INDEPENDENTES, e duas linhas do mesmo DOI não são duas fontes.

    Este teste existe porque a mutação provou que faltava: reverter `n_articles` para
    `len(hits)` em `answer.py` deixava as 996 passando. O teste que havia exercitava
    `distinct_articles` ISOLADA — provava que a função conta certo, não que o loop a usa.
    É a classe "fiação-não-testada", a sexta ocorrência dela neste repo.

    Medido no item 9: 100% dos PMIDs que o PubMed colhe neste domínio também estão no
    Europe PMC. Com duas fontes, o mesmo paper entra duas vezes e chega ao juiz como duas
    fontes independentes concordando.

    MUTAÇÃO: `if _is_sufficient(verdict) and len(hits) >= MIN_CITATIONS:`.
    """
    doi = "10.1016/j.jad.2023.11.001"
    await _claim(store, "111", "Quetiapina reduziu ansiedade em bipolar I.", doi=doi)
    await _claim(store, "222", "Quetiapina reduziu HAM-A versus placebo.", doi=doi)
    qid = _question(store)

    result = await _answerer(store).round(qid)

    assert result.action is not Action.ANSWER, (
        "duas claims do MESMO artigo satisfizeram o piso de citações independentes"
    )
    assert _row(store, qid)["answer"] is None


async def test_two_claims_from_different_articles_still_answer(store):
    """A contrapartida. Sem ela o piso poderia virar "nunca responde" e o teste acima
    ficaria verde medindo o bug oposto."""
    await _claim(store, "111", "Quetiapina reduziu ansiedade em bipolar I.",
                 doi="10.1/um")
    await _claim(store, "222", "Quetiapina reduziu HAM-A versus placebo.",
                 doi="10.1/dois")
    qid = _question(store)

    assert (await _answerer(store).round(qid)).action is Action.ANSWER
