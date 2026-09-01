"""O loop que finalmente consome a fila de perguntas.

Sem ele, `plan_tick` gera perguntas, `classify` as tipa, `score_priority` as ordena — e
elas ficam em `OPEN` para sempre. Todo o andaime já existia morto: os estados
`RESEARCHING` e `ANSWERED_AUTO`, o `SufficiencyVerdict`, a tabela `findings`.

Quatro decisões, e cada uma vem de uma medição.

**O juiz vem PRIMEIRO, e não vê rascunho.** Julgar a evidência antes de sintetizar
custa 23 s numa pergunta irrespondível; sintetizar primeiro e julgar depois custa 44 s a
mais por rodada, jogados fora. Medido nos prompts reais: a versão ingênua gasta 198 s
numa pergunta que nunca fecha, esta gasta 91 s. E o repo já registra duas vezes que
modelo pequeno auto-avaliando o que acabou de escrever diz "suficiente" quase sempre —
por isso o juiz julga a **evidência**, não a resposta.

**Rodada sem material novo não é rodada.** `search_claims` é função pura de (consulta,
corpus): medido, a rodada N+1 sobre um corpus imutável devolve a mesma lista de claims,
byte a byte, com os mesmos scores. Rodadas consecutivas não são "mais busca" — são a
mesma busca paga de novo. Então a impressão digital da evidência aborta a rodada **sem
gastar orçamento**: gastar aqui converteria o teto de rodadas num relógio de parede, e a
simulação de 30 dias dessa variante fechou 132 de 150 perguntas em silêncio, sem
ninguém ter lido nenhuma.

**O gargalo é o despachante, não o LLM.** A vazão é `em_voo × varreduras_por_dia /
rodadas`, aritmética pura. Com `em_voo = 2` dá 2,7 perguntas/dia contra as ~5/dia que o
`plan_tick` produz — a fila satura e o excedente é triturado. Com 4, a fila esvazia, e
custa 16 rodadas/dia: 38 minutos, 2,6% do dia.

**Nada daqui é exportado para treino, de propósito.** `training_examples` tem zero
escritores *e zero leitores*, e o LoRA é o item 12. Três desenhos independentes deste
loop vazaram material não-verificado para lá por três caminhos diferentes — divisor de
sentenças truncando a nona proposição, `caveats` nunca julgado, e prosa de dosagem
fundindo duas proposições num veredito só. Uma lição tem `--forget`; uma atualização de
pesos não. A exportação entra quando tiver consumidor e um portão medido.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum

from lithium.db import Store
from lithium.llm import LLMClient, LLMError
from lithium.llm.prompts import budget_guard, render
from lithium.llm.schemas import SufficiencyVerdict
from lithium.pipeline.retrieval import ClaimHit, Retriever
from lithium.focus import FocusProfile
from lithium.safety.rules import ruleset_from_profile
from lithium.safety.screen import Segment, screen
from lithium.types import QuestionStatus, StuckReason
from lithium.worker.queue import iso, utcnow

log = logging.getLogger(__name__)

MAX_ROUNDS = 3
"""Rodadas de busca antes de escalar.

Só conta rodada que viu material novo — ver `Answerer.round`. Sem essa condição o teto
vira relógio: as rodadas 2 e 3 caem na mesma colheita e reprovam a pergunta por tempo,
não por insuficiência."""

IN_FLIGHT = 4
"""Quantas perguntas o despachante mantém em voo por varredura.

`vazão = em_voo × varreduras_por_dia / rodadas`. Com varredura de 6 h e 3 rodadas:
em_voo=2 → 2,7 perguntas/dia, abaixo das ~5/dia do `plan_tick`, e a fila satura;
em_voo=4 → 5,3/dia e ela esvazia. Custa 16 rodadas/dia ≈ 38 min de parede."""

HUMAN_QUEUE_LIMIT = 5
"""Teto da fila humana, o mesmo de `QuestionEngine`. Um sistema que entrega quarenta
perguntas por dia não é usado duas vezes."""

AUTO_QUEUE_CAP = 15
"""Teto da fila automática.

Não é 40. O bloco "não repita" de `build_state` degrada muito antes disso e na ordem
inversa da intuição: o `ORDER BY` põe `OPEN` primeiro, então pergunta aberta expulsa
primeiro a **já respondida**. Medido num histórico realista: com 18 abertas o histórico
já está sendo cortado; com 35, o gerador não vê nenhuma pergunta respondida e passa a
repropor o que já foi resolvido."""

MIN_CITATIONS = 2
"""Citações distintas mínimas para responder — e é DETERMINÍSTICO, não delegado ao juiz.

Um teste pegou o furo: com corpus vazio e um juiz carimbador, `_answer` gravava um achado
com `citations_json = []` e marcava a pergunta como respondida. Uma resposta sem fonte
chegando ao psiquiatra é o pior resultado que este sistema pode produzir, e não pode
depender de o modelo ter respondido bem a uma pergunta sobre si mesmo.

O número casa com `n_independent_sources >= 2` do juiz de propósito: os dois dizem a mesma
coisa, um por julgamento e um por contagem, e o determinístico é o que vale."""

EVIDENCE_K = 10
JUDGE_MAX_TOKENS = 320
ANSWER_MAX_TOKENS = 384


def distinct_articles(hits: list[ClaimHit], store: Store) -> int:
    """Quantos ARTIGOS distintos as citações representam.

    `len(hits)` conta CLAIMS, e duas claims podem vir do mesmo paper — ou, desde que existe
    mais de uma fonte, de duas LINHAS de `sources` que são o mesmo artigo. Medido no item
    9: 100% dos PMIDs que o PubMed colhe neste domínio também estão no Europe PMC, então o
    mesmo paper entraria duas vezes e chegaria ao juiz como duas fontes independentes
    concordando — satisfazendo o piso de citações com um artigo só.

    `article_key` é a coluna GENERATED que colapsa isso: DOI normalizado quando existe,
    `kind:external_id` quando não. Num banco legado ela não existe (o rebuild de `sources`
    é adiado quando há linhas), e aí o fallback conta `source_id` distinto — que é o
    comportamento antigo, e é honesto: sem a coluna não há como saber que dois ids são o
    mesmo artigo.
    """
    ids = {h.source_id for h in hits}
    if not ids:
        return 0
    # `table_xinfo` e NÃO `table_info`: uma coluna GENERATED VIRTUAL não aparece em
    # `table_info`, e `article_key` é VIRTUAL. Com `table_info` o guard sempre cai no
    # fallback, o piso volta a contar `source_id` distinto, e o teste que policia isso
    # falha — foi assim que este bug foi pego. O repo já documentava a armadilha em
    # `Store._ADDED_COLUMNS`, por outro motivo.
    cols = {r["name"] for r in store.conn.execute("PRAGMA table_xinfo(sources)")}
    if "article_key" not in cols:
        return len(ids)
    marks = ",".join("?" * len(ids))
    row = store.conn.execute(
        f"SELECT COUNT(DISTINCT article_key) AS n FROM sources WHERE id IN ({marks})",
        tuple(ids),
    ).fetchone()
    return int(row["n"])


class Action(StrEnum):
    ANSWER = "ANSWER"
    SEARCH_AGAIN = "SEARCH_AGAIN"
    ESCALATE = "ESCALATE"


@dataclass(slots=True)
class RoundResult:
    action: Action
    reason: str = ""
    finding_id: int | None = None
    claim_ids: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.claim_ids is None:
            self.claim_ids = []


def evidence_fingerprint(claim_ids: list[int]) -> str:
    """Identidade do conjunto recuperado. Mudou o corpus, muda a impressão."""
    return ",".join(str(i) for i in sorted(claim_ids))


class Answerer:
    def __init__(self, store: Store, llm: LLMClient, embedder, *,
                 profile: FocusProfile, n_ctx: int = 8192,
                 max_rounds: int = MAX_ROUNDS) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.profile = profile
        self.ruleset = ruleset_from_profile(profile)
        self.n_ctx = n_ctx
        self.max_rounds = max_rounds

    # ───────────────────────────────────────────────────────────── despachante

    def dispatchable(self, limit: int = IN_FLIGHT) -> list[int]:
        """Perguntas auto-respondíveis prontas para uma rodada.

        Exclui as que já têm tarefa viva: sem isso, uma varredura a cada 6 h empilha
        rodadas concorrentes sobre a mesma pergunta e o `rounds` estoura sem que
        nenhuma delas tenha visto colheita nova.

        AUTO_QUEUE_CAP e HUMAN_QUEUE_LIMIT são recursos DO FOCO, não do banco.
        MEDIDO e EXECUTADO sem o escopo: com 15 perguntas OPEN no foco antigo, as CINCO
        perguntas recém-geradas do foco novo eram FECHADAS por `park_overflow`, e
        `dispatchable` devolvia só as do foco antigo — o foco novo ficava
        estruturalmente incapaz de acumular fila de pesquisa. Na fila humana era pior:
        `escalated_count()` já era escopado e `_escalate` não, então as duas metades da
        mesma trava divergiam (uma via 0 vagas, a outra via 5).
        """
        from lithium.types import AUTO_ANSWERABLE

        kinds = ",".join(f"'{k.value}'" for k in AUTO_ANSWERABLE)
        rows = self.store.conn.execute(
            f"SELECT q.id FROM questions q "
            f" WHERE q.status = 'OPEN' AND q.kind IN ({kinds}) "
            f"   AND q.focus_id = (SELECT id FROM active_focus) "
            f"   AND q.stuck_reason IS NULL AND q.rounds < ? "
            f"   AND NOT EXISTS (SELECT 1 FROM tasks t "
            f"                    WHERE t.kind = 'answer_question' "
            f"                      AND t.status IN ('pending', 'running') "
            f"                      AND json_extract(t.payload_json, '$.question_id') = q.id) "
            f" ORDER BY q.priority DESC, q.id ASC LIMIT ?",
            (self.max_rounds, limit),
        ).fetchall()
        return [int(r["id"]) for r in rows]

    def release_stranded(self) -> int:
        """Devolve a `OPEN` pergunta presa em `RESEARCHING` sem tarefa viva.

        `TaskQueue.recover_orphans()` **não** cobre isto: ele mexe em `tasks`, e a
        pergunta é um segundo estado. Uma tarefa que chega ao dead-letter deixa a
        pergunta em `RESEARCHING` para sempre e nada mais olha para ela.

        A reconciliação é derivada da tabela `tasks` — sem relógio, sem TTL, sem lease.
        O `IS NOT NULL` na subconsulta é obrigatório: `NOT IN` com um único NULL no
        conjunto devolve NULL para **toda** linha, então uma tarefa sem `question_id`
        no payload desligaria a reconciliação inteira, em silêncio.
        """
        cur = self.store.conn.execute(
            "UPDATE questions SET status = 'OPEN' "
            " WHERE status = 'RESEARCHING' "
            "   AND focus_id = (SELECT id FROM active_focus) "
            "   AND id NOT IN (SELECT json_extract(payload_json, '$.question_id') "
            "                    FROM tasks "
            "                   WHERE kind = 'answer_question' "
            "                     AND status IN ('pending', 'running') "
            "                     AND json_extract(payload_json, '$.question_id') IS NOT NULL)"
        )
        if cur.rowcount:
            log.info("%d pergunta(s) devolvida(s) de RESEARCHING para OPEN", cur.rowcount)
        return cur.rowcount

    def park_overflow(self, cap: int = AUTO_QUEUE_CAP) -> int:
        """Fecha o excedente da fila automática, mais fraco primeiro.

        A fila cresce ~5/dia e o loop consome ~5/dia — o equilíbrio é apertado, e sem
        teto uma semana de daemon parado enche o bloco "não repita" e o gerador começa
        a repropor o que já foi resolvido.
        """
        from lithium.types import AUTO_ANSWERABLE

        kinds = ",".join(f"'{k.value}'" for k in AUTO_ANSWERABLE)
        cur = self.store.conn.execute(
            f"UPDATE questions SET status = 'CLOSED' "
            f" WHERE id IN (SELECT id FROM questions "
            f"               WHERE status = 'OPEN' AND kind IN ({kinds}) "
            f"                 AND focus_id = (SELECT id FROM active_focus) "
            f"                 AND stuck_reason IS NULL "
            f"               ORDER BY priority DESC, id ASC "
            f"               LIMIT -1 OFFSET ?)",
            (cap,),
        )
        if cur.rowcount:
            log.info("%d pergunta(s) fechada(s) pelo teto da fila automática", cur.rowcount)
        return cur.rowcount

    # ─────────────────────────────────────────────────────────── uma rodada

    async def round(self, question_id: int) -> RoundResult:
        row = self.store.conn.execute(
            "SELECT id, text, kind, rounds, partial_work FROM questions WHERE id = ?",
            (question_id,),
        ).fetchone()
        if row is None:
            return RoundResult(Action.ESCALATE, reason="pergunta não existe")

        self.store.conn.execute(
            "UPDATE questions SET status = 'RESEARCHING' WHERE id = ?", (question_id,)
        )
        try:
            return await self._round(row)
        except LLMError as exc:
            log.warning("rodada da pergunta #%d falhou: %s", question_id, exc)
            self.store.conn.execute(
                "UPDATE questions SET status = 'OPEN' WHERE id = ?", (question_id,)
            )
            raise

    def _record_round(self, row, hits, n_articles: int, verdict) -> None:
        """Uma linha por rodada, append-only. Nunca levanta.

        Instrumentação que derruba o loop que ela mede é pior que instrumentação nenhuma:
        a resposta é o produto, a medição é sobre o produto. Por isso o `except` largo —
        e é a mesma escolha que `UsageSink.record` já fazia.
        """
        try:
            self.store.conn.execute(
                "INSERT INTO answer_rounds(question_id, round, n_hits, n_articles, "
                "  judge_sufficient, floor_ok, blocked_reason) VALUES(?,?,?,?,?,?,?)",
                (int(row["id"]), int(row["rounds"]) + 1, len(hits), n_articles,
                 int(verdict is not None and _is_sufficient(verdict)),
                 int(n_articles >= MIN_CITATIONS),
                 str(verdict.blocked_reason) if verdict is not None
                 and verdict.blocked_reason is not None else None),
            )
        except Exception:  # noqa: BLE001
            log.debug("registro da rodada falhou", exc_info=True)

    async def _round(self, row) -> RoundResult:
        question_id, question = int(row["id"]), row["text"]
        hits = await Retriever(self.store, self.embedder).search_claims(
            question, k=EVIDENCE_K
        )
        claim_ids = [h.claim_id for h in hits]
        fingerprint = evidence_fingerprint(claim_ids)

        seen = json.loads(row["partial_work"] or "{}") if _is_json(row["partial_work"]) else {}
        if seen.get("fingerprint") == fingerprint and claim_ids:
            # Mesma evidência da rodada anterior: `search_claims` é função pura de
            # (consulta, corpus), então repetir aqui é pagar a mesma busca de novo.
            # NÃO consome rodada — se consumisse, o teto viraria relógio de parede e a
            # pergunta seria reprovada por tempo, não por insuficiência.
            self.store.conn.execute(
                "UPDATE questions SET status = 'OPEN' WHERE id = ?", (question_id,)
            )
            return RoundResult(Action.SEARCH_AGAIN, reason="corpus inalterado",
                               claim_ids=claim_ids)

        verdict = await self._judge(question, hits)
        self._bump(question_id, fingerprint, verdict)

        # A RODADA VIRA LINHA AQUI, antes de QUALQUER desvio — inclusive o de juiz
        # indisponível. Na primeira versão eu a registrava depois, e o ramo `verdict is
        # None` retornava antes: uma semana em que o juiz falhou em 30% das rodadas
        # apareceria como uma semana com 30% menos rodadas, que é a leitura errada.
        n_articles = distinct_articles(hits, self.store)
        self._record_round(row, hits, n_articles, verdict)

        if verdict is None:
            self.store.conn.execute(
                "UPDATE questions SET status = 'OPEN' WHERE id = ?", (question_id,)
            )
            return RoundResult(Action.SEARCH_AGAIN, reason="juiz indisponível")

        # ARTIGOS distintos, não claims. Duas claims do mesmo paper — ou duas linhas de
        # `sources` que são o mesmo artigo em fontes diferentes — satisfariam o piso com
        # uma fonte só, que é exatamente o que "duas citações independentes" nega.
        if _is_sufficient(verdict) and n_articles >= MIN_CITATIONS:
            return await self._answer(question_id, question, hits, verdict)
        if _is_sufficient(verdict):
            # O juiz aprovou e o corpus não sustenta. Não é falha dele: `_evidence_block`
            # diz "(no verified claim matched)" e um modelo complacente aprova mesmo
            # assim. O portão determinístico é o que impede uma resposta sem fonte.
            log.info(
                "pergunta #%d: juiz aprovou com %d claim(s) em %d artigo(s) distinto(s); "
                "mínimo é %d artigo(s)",
                question_id, len(hits), n_articles, MIN_CITATIONS,
            )

        rounds = int(row["rounds"]) + 1
        if verdict.blocked_reason is not None or rounds >= self.max_rounds:
            reason = verdict.blocked_reason or StuckReason.INSUFFICIENT_EVIDENCE
            self._escalate(question_id, reason, verdict.missing)
            return RoundResult(Action.ESCALATE, reason=str(reason))

        self.store.conn.execute(
            "UPDATE questions SET status = 'OPEN' WHERE id = ?", (question_id,)
        )
        return RoundResult(Action.SEARCH_AGAIN, reason=verdict.missing,
                           claim_ids=claim_ids)

    # ───────────────────────────────────────────────────────────────── portões

    async def _judge(self, question: str, hits) -> SufficiencyVerdict | None:
        """O juiz vê a EVIDÊNCIA, nunca um rascunho.

        Se visse o rascunho julgaria a redação; vendo só a evidência, julga se ela
        responde. E vem antes da síntese porque uma pergunta irrespondível não deve
        pagar por uma resposta que será descartada.
        """
        prompt = render(
            "judge_sufficiency", question=question, evidence=_evidence_block(hits),
            **self.profile.prompt_blocks("judge_sufficiency"),
        )
        try:
            budget_guard(prompt, label="judge_sufficiency",
                         max_tokens=JUDGE_MAX_TOKENS, n_ctx=self.n_ctx)
            return await self.llm.structured(
                [{"role": "user", "content": prompt}], SufficiencyVerdict,
                max_tokens=JUDGE_MAX_TOKENS, label="judge_sufficiency",
            )
        except LLMError as exc:
            log.warning("juiz de suficiência falhou: %s", exc)
            return None

    async def _answer(self, question_id: int, question: str, hits,
                      verdict: SufficiencyVerdict) -> RoundResult:
        prompt = render("answer_question", question=question,
                        evidence=_evidence_block(hits),
                        **self.profile.prompt_blocks("answer_question"))
        budget_guard(prompt, label="answer_question",
                     max_tokens=ANSWER_MAX_TOKENS, n_ctx=self.n_ctx)
        text = await self.llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=ANSWER_MAX_TOKENS, temperature=0.3, label="answer_question",
        )

        # As citações vêm dos claim ids que ESTAVAM NO PROMPT, nunca do que o modelo
        # escreveu. É a constraint (a) da fronteira de treino, e ela vale mesmo sem
        # exportação: o revisor lê `citations_json` como o que sustenta o achado.
        citations = [
            {"claim_id": h.claim_id, "source_id": h.source_id,
             "external_id": h.external_id}
            for h in hits
        ]
        # O screen determinístico roda sobre o achado, não só sobre o chat. O achado é o
        # que o psiquiatra LÊ — se o lembrete de risco de virada aparece na conversa e
        # não aqui, o canal que importa é justamente o desprotegido.
        #
        # O texto vai como `reply`, e o kind muda o resultado: em prosa de síntese a
        # supressão por negação se auto-anula (`interromperam ... abruptamente` desligava
        # o alerta de mania de rebote, que é o vocabulário que ele existe para pegar).
        alerts = screen(
            [Segment("reply", text.strip())]
            + [Segment("evidence", h.statement, h.external_id) for h in hits],
            self.ruleset,
        )
        cur = self.store.conn.execute(
            "INSERT INTO findings(question_id, text, confidence, citations_json, "
            "                     safety_json) "
            "VALUES(?, ?, ?, ?, ?) RETURNING id",
            (question_id, text.strip(), _confidence(verdict), json.dumps(citations),
             json.dumps([{"key": a.key, "severity": a.severity, "text": a.text}
                         for a in alerts])),
        )
        finding_id = int(cur.fetchone()["id"])

        self.store.conn.execute(
            "UPDATE questions SET status = 'ANSWERED_AUTO', answer = ?, "
            "  answer_origin = 'auto', answered_at = ? WHERE id = ?",
            (text.strip(), iso(utcnow()), question_id),
        )
        log.info("pergunta #%d respondida com %d claim(s)", question_id, len(hits))
        return RoundResult(Action.ANSWER, finding_id=finding_id,
                           claim_ids=[h.claim_id for h in hits])

    # ──────────────────────────────────────────────────────────────── estado

    def _bump(self, question_id: int, fingerprint: str,
              verdict: SufficiencyVerdict | None) -> None:
        note = {"fingerprint": fingerprint}
        if verdict is not None and verdict.missing:
            note["missing"] = verdict.missing
        self.store.conn.execute(
            "UPDATE questions SET rounds = rounds + 1, partial_work = ? WHERE id = ?",
            (json.dumps(note), question_id),
        )

    def _escalate(self, question_id: int, reason, missing: str) -> None:
        """Escala respeitando o teto da fila humana.

        Bug encontrado pela auditoria do item 7.5: esta função gravava `ESCALATED` sem
        consultar vaga, enquanto `QuestionEngine.escalate` respeita o limite de 5. O loop
        de resposta **triplica** a taxa de escalação, então sem o teto a fila humana
        enchia — e uma fila cheia esconde as que importam, que é a razão de o teto
        existir. Sem vaga a pergunta fica represada (`OPEN` com `stuck_reason`), e
        `promote_escalations` a sobe quando abrir espaço.
        """
        escalated = self.store.conn.execute(
            "SELECT COUNT(*) AS n FROM questions WHERE status = 'ESCALATED' "
            "   AND focus_id = (SELECT id FROM active_focus)"
        ).fetchone()["n"]
        status = (QuestionStatus.ESCALATED.value if escalated < HUMAN_QUEUE_LIMIT
                  else QuestionStatus.OPEN.value)
        self.store.conn.execute(
            "UPDATE questions SET status = ?, stuck_reason = ?, partial_work = ?, "
            "  escalated_at = ? WHERE id = ?",
            (status, str(reason),
             f"o que falta: {missing}" if missing else None,
             iso(utcnow()), question_id),
        )


def _is_sufficient(verdict: SufficiencyVerdict) -> bool:
    """Os quatro campos são consumidos, não só `sufficient`.

    Um portão que lê só o booleano torna os outros decorativos — foi exatamente o
    defeito que a auditoria construiu no portão de derivação da Fase 2, com a função
    pura perfeita e o call site checando uma resposta de quatro.
    """
    return (
        verdict.sufficient
        and verdict.addresses_question_directly
        and verdict.n_independent_sources >= 2
        and verdict.blocked_reason is None
    )


def _confidence(verdict: SufficiencyVerdict) -> float:
    """Discordância entre fontes não impede responder — rebaixa a confiança."""
    return 0.8 if verdict.sources_agree else 0.5


def _evidence_block(hits) -> str:
    if not hits:
        return "(no verified claim matched this question)"
    return "\n".join(
        f"  [PMID:{h.external_id}] ({h.grade.value}/{h.directness.value}) {h.statement}"
        for h in hits
    )


def _is_json(value: str | None) -> bool:
    return bool(value) and value.lstrip().startswith("{")
