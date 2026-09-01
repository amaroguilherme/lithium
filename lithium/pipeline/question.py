"""Geração, classificação, deduplicação e priorização de perguntas.

O ciclo assíncrono que o usuário pediu começa aqui: o sistema olha o próprio estado de
conhecimento, decide o que não sabe, e formula perguntas — respondendo sozinho as que
consegue e escalando as que não.

Três mecanismos carregam o peso:

**Roteamento por taxonomia.** `PREFERENCE` e `CONTEXT` são estruturalmente
inauto-respondíveis: nenhuma quantidade de busca resolve "priorizar remissão da
ansiedade ou estabilidade do humor?". Mandá-las ao loop de pesquisa gasta três rodadas
e produz uma resposta inventada com citações de aparência legítima. Elas vão direto ao
humano.

**Dedup por embedding.** Sem isto o gerador reformula a mesma lacuna indefinidamente —
"quetiapina funciona em TAG?" e "há evidência de quetiapina para ansiedade
generalizada?" são a mesma pergunta e o banco encheria de paráfrases.

**Teto na fila humana.** Cinco perguntas escaladas simultâneas, no máximo. O excedente
fica represado e sobe quando abre vaga. Um sistema que entrega quarenta perguntas por
dia não é usado duas vezes — e a fila cheia esconde as que importam.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lithium.db import Store
from lithium.focus import FocusProfile, NoActiveFocus
from lithium.llm import LLMClient
from lithium.llm.prompts import budget_guard, render
from lithium.llm.schemas import GeneratedQuestion, QuestionBatch, QuestionClassification
from lithium.pipeline.state import KnowledgeState, build_state
from lithium.types import AUTO_ANSWERABLE, QuestionKind, QuestionStatus, StuckReason
from lithium.worker.queue import iso, utcnow

log = logging.getLogger(__name__)

HUMAN_ONLY_KINDS = frozenset({QuestionKind.CONTEXT, QuestionKind.PREFERENCE})
"""Perguntas que só você pode responder — e para as quais a memória ajuda."""

MAX_RECALLED = 8
"""Teto do bloco de memórias anexado. Ordenado por tipo (restrição primeiro), com nota
de truncamento: um quadro parcial que se apresenta como completo é pior que um bloco
menor que declara o que escondeu."""

# Pesos da pontuação de ganho de informação. Somam 1.0.
W_NOVELTY = 0.45
"""Quanto pesa a ausência de evidência. É o maior porque uma intervenção sem nenhuma
claim é a lacuna mais barata de fechar e a mais informativa por unidade de esforço."""

W_CONFLICT = 0.35
"""Quanto pesa a discordância entre fontes. Alto porque conflito é o único sinal que
mais-do-mesmo NÃO resolve — exige perguntar por que discordam."""

W_DIRECTNESS = 0.20
"""Quanto pesa ter evidência só em população adjacente."""

# Divisores de custo por tipo. Não é o custo em tokens — é o custo de oportunidade.
COST: dict[QuestionKind, float] = {
    QuestionKind.FACTUAL: 1.0,
    QuestionKind.SYNTHESIS: 1.5,       # várias rodadas, crítica adversarial
    QuestionKind.PREFERENCE: 1.0,      # barata para a máquina, cara para o humano
    QuestionKind.CONTEXT: 1.0,
    QuestionKind.METHODOLOGICAL: 3.0,  # o plano manda deixar em prioridade baixa
}


@dataclass(slots=True)
class QuestionRecord:
    id: int
    text: str
    kind: QuestionKind
    status: QuestionStatus
    priority: float
    targets: str | None = None
    stuck_reason: StuckReason | None = None
    partial_work: str | None = None


def score_priority(kind: QuestionKind, coverage) -> float:
    """Ganho de informação esperado, normalizado em (0, 1].

    Não é uma medida bayesiana de verdade — é um proxy explicável construído sobre os
    três sinais que o estado do conhecimento realmente expõe. Explicável importa aqui:
    a prioridade decide o que ocupa as cinco vagas da fila humana, e o Space mostra a
    justificativa junto com a pergunta.
    """
    novelty = 1.0 / (1.0 + coverage.total_weight)
    raw = (
        W_NOVELTY * novelty
        + W_CONFLICT * coverage.conflict
        + W_DIRECTNESS * coverage.directness_gap
    )
    return max(0.01, min(1.0, raw / COST[kind]))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denom if denom else 0.0


def _target(store) -> str:
    """O alvo do foco ativo, ou um marcador honesto quando não há foco.

    Sem foco a classificação ainda é possível — a distinção FACTUAL/PREFERENCE não depende
    do domínio —, então isto degrada em vez de levantar. O que não pode acontecer é o
    prompt afirmar um domínio que não é o ativo.
    """
    focus = store.active_focus()
    return (focus["target"] if focus else "an unspecified research target")


class QuestionEngine:
    def __init__(
        self,
        store: Store,
        llm: LLMClient,
        embedder,
        *,
        profile: FocusProfile,
        dedup_threshold: float = 0.90,
        human_queue_limit: int = 5,
        n_ctx: int = 8192,
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.profile = profile
        self.dedup_threshold = dedup_threshold
        self.human_queue_limit = human_queue_limit
        self.n_ctx = n_ctx

    # ───────────────────────────────────────────────────────────────── geração

    def _focus_id(self) -> int:
        """O foco ativo AGORA, resolvido antes de qualquer escrita.

        O `focus_id` é resolvido no INÍCIO da operação e passado EXPLICITAMENTE, nunca
        pelo subselect `(SELECT id FROM active_focus)` dentro do INSERT — que resolvia
        DEPOIS da chamada de LLM. As janelas estão medidas no PLAN: 43,8 s em
        `generate_questions` e 117,9 s em `generate_speculation`. Como "trocar de foco
        quando quiser" é o pedido literal desta fase, elas deixaram de ser teóricas: o
        item era gerado a partir do estado de conhecimento do foco A e arquivado sob o
        foco B, sumindo de onde foi gerado e aparecendo no outro — e nada gravado
        permitia reconstruir que a causa foi timing.
        """
        focus = self.store.active_focus()
        if focus is None:
            raise NoActiveFocus(
                "nenhum foco ativo: a pergunta seria arquivada sob nenhum foco. "
                "Escolha um com `lithium focus --use <slug>`."
            )
        return int(focus["id"])

    async def generate(self, *, max_questions: int = 5) -> list[QuestionRecord]:
        """Um tick de planejamento: lê o estado, propõe perguntas, persiste as novas."""
        # Resolvido ANTES do POST — a chamada abaixo leva ~44 s — e PROPAGADO até o
        # INSERT. Resolver de novo depois seria o mesmo defeito com um passo a mais.
        focus_id = self._focus_id()
        state = build_state(self.store, self.profile)
        prompt = render("generate_questions", state=state.render(),
                        max_questions=max_questions,
                        **self.profile.prompt_blocks("generate_questions"))
        budget_guard(prompt, label="generate_questions", max_tokens=2048,
                     n_ctx=self.n_ctx)

        batch = await self.llm.structured(
            [{"role": "user", "content": prompt}],
            QuestionBatch,
            max_tokens=2048,
            label="generate_questions",
        )

        added: list[QuestionRecord] = []
        for proposal in batch.questions[:max_questions]:
            record = await self.add(proposal, state=state, focus_id=focus_id)
            if record is not None:
                added.append(record)

        log.info(
            "plan tick: %d propostas, %d novas (%d auto, %d para humano)",
            len(batch.questions), len(added),
            sum(q.kind in AUTO_ANSWERABLE for q in added),
            sum(q.kind not in AUTO_ANSWERABLE for q in added),
        )
        self.promote_escalations()
        return added

    async def ask(self, text: str) -> QuestionRecord | None:
        """Pergunta manual do usuário. Classificada, mas com prioridade máxima —
        se você parou para digitar, é porque quer a resposta."""
        classification = await self.llm.structured(
            # `target` do BANCO. O prompt afirmava o domínio numa frase fixa
            # ("bipolar I disorder with comorbid GAD"), então um foco de outro assunto
            # teria as próprias perguntas classificadas por um prompt que anuncia
            # psiquiatria — a mesma mentira que `scaffold_pending` impede no perfil TOML,
            # sobrevivendo no prompt.
            [{"role": "user", "content": render(
                "classify_question", question=text, target=_target(self.store))}],
            QuestionClassification,
            max_tokens=512,
            label="classify_question",
        )
        proposal = GeneratedQuestion(
            text=text,
            kind=classification.kind,
            rationale=classification.reason,
            targets=classification.targets,
        )
        return await self.add(proposal, origin="human", priority=1.0)

    async def add(
        self,
        proposal: GeneratedQuestion,
        *,
        state: KnowledgeState | None = None,
        origin: str = "auto",
        priority: float | None = None,
        focus_id: int | None = None,
    ) -> QuestionRecord | None:
        """Persiste, se não for duplicata. Devolve None quando já existe equivalente."""
        # Embedder fora do ar não pode custar a pergunta.
        #
        # Sem esta guarda, `embed()` levanta ANTES de qualquer INSERT: a pergunta nunca é
        # gravada, nunca é escalada, e nada aparece na fila humana. O sintoma é ausência,
        # que é a coisa que ninguém nota. Sem dedup o pior caso é uma paráfrase repetida
        # na fila — visível, e reversível.
        vector: list[float] | None
        try:
            [vector] = await self.embedder.embed([proposal.text])
        except Exception as exc:  # noqa: BLE001
            log.warning("embedder indisponível; pergunta gravada sem dedup: %s", exc)
            vector = None

        if vector is not None:
            duplicate = self.find_duplicate(vector, proposal.targets)
            if duplicate is not None:
                log.debug("pergunta duplicada de #%d, ignorada: %s",
                          duplicate, proposal.text[:70])
                return None

        state = state or build_state(self.store, self.profile)
        coverage = state.by_intervention(proposal.targets or proposal.text)
        computed = score_priority(proposal.kind, coverage) if priority is None else priority

        cur = self.store.conn.execute(
            "INSERT INTO questions(focus_id, text, kind, status, priority, origin, targets, embedding) "
            "VALUES(?, ?, ?, 'OPEN', ?, ?, ?, ?) "
            "RETURNING id",
            (
                self._focus_id() if focus_id is None else focus_id,
                proposal.text,
                proposal.kind.value,
                computed,
                origin,
                (proposal.targets or "").strip().lower() or None,
                Store.pack_embedding(vector) if vector is not None else None,
            ),
        )
        question_id = int(cur.fetchone()["id"])

        record = QuestionRecord(
            id=question_id,
            text=proposal.text,
            kind=proposal.kind,
            status=QuestionStatus.OPEN,
            priority=computed,
            targets=proposal.targets,
        )

        # PREFERENCE / CONTEXT / METHODOLOGICAL nunca passam pelo loop de pesquisa.
        if proposal.kind not in AUTO_ANSWERABLE:
            self.escalate(
                question_id,
                reason=_stuck_reason_for(proposal.kind),
                partial_work=self._with_recalled(proposal),
            )
            record.status = QuestionStatus.ESCALATED
        return record

    def _with_recalled(self, proposal: GeneratedQuestion) -> str:
        """Anexa ao `partial_work` o que você já contou — e **escala mesmo assim**.

        Fecha a promessa que a docstring de `chat.py` fazia e o código não cumpria:
        "quando o loop trava numa pergunta CONTEXT, ele consulta as memórias antes de
        escalar". A parte que a promessa não dizia, e que é a que importa, é que ele
        **não responde sozinho**.

        Por que nunca auto-responder: `training_examples` dá peso 3.0 a material humano,
        e os dois caminhos que carimbam `answer_origin` como humano recebem texto que
        veio de uma pessoa por um caminho síncrono. Uma síntese de LLM entrando ali é
        **irreversível** — uma memória pode receber `--forget`, uma atualização de pesos
        não. O conjunto de escritores é travado em `test_escalation_memory.py`.

        Então o que entra aqui é material para **você** ler, explicitamente rotulado
        como memória e não como fato verificado. Se responder a pergunta, ótimo: você
        responde em dois segundos em vez de dois minutos.
        """
        base = proposal.rationale or ""
        if proposal.kind not in HUMAN_ONLY_KINDS:
            return base

        rows = self.store.conn.execute(
            "SELECT id, kind, text FROM user_memories "
            " ORDER BY CASE kind WHEN 'constraint' THEN 0 WHEN 'preference' THEN 1 "
            "                    WHEN 'context' THEN 2 ELSE 3 END, id "
            " LIMIT ?",
            (MAX_RECALLED,),
        ).fetchall()
        if not rows:
            return base

        lines = [base, "", "o que você já me contou (memória, não evidência verificada):"]
        lines += [f"  [{r['kind']}] {r['text']}  (memória #{r['id']})" for r in rows]
        total = self.store.conn.execute(
            "SELECT COUNT(*) AS n FROM user_memories"
        ).fetchone()["n"]
        if total > len(rows):
            lines.append(f"  (+{total - len(rows)} outras — `lithium memories`)")
        return "\n".join(lines).strip()

    # ────────────────────────────────────────────────────────────────── dedup

    def find_duplicate(self, vector, targets: str | None = None) -> int | None:
        """Duplicata é mesma lacuna, não texto parecido.

        **O escopo por alvo não é otimização, é correção.** Similaridade de texto
        sozinha não funciona aqui, e o problema não se resolve mexendo no limiar.
        Medido num lote real gerado pelo 12B, cinco perguntas sobre cinco
        intervenções distintas (buspirona, pregabalina, antidepressivo, TCC,
        quetiapina) ficaram em 0.767–0.874 de similaridade entre si — porque
        compartilham o vocabulário do domínio ("bipolar I disorder", "GAD",
        "efficacy", "anxiety"), que domina o embedding. Essa faixa **sobrepõe
        inteiramente** a das paráfrases genuínas (0.788–0.952). Nove dos dez pares
        distintos seriam descartados por engano.

        Duas perguntas sobre intervenções diferentes não são duplicatas, por mais
        parecidas que soem. Dentro de um mesmo alvo o texto volta a discriminar:
        "quetiapina tem RCT em TAG?" vs "quetiapina causa ganho de peso?" medem
        0.711, abaixo do limiar.

        Compara contra TODAS as perguntas DO FOCO ATIVO, inclusive fechadas — comparar
        só com as abertas faria o gerador ressuscitar perguntas respondidas assim que
        saíssem da fila.

        **O limiar de 0.78 foi medido dentro de UM vocabulário.** A faixa 0.767–0.874
        citada acima vale entre intervenções do mesmo alvo; entre focos ela não foi
        medida e não há razão para supor que valha. O que a Fase A faz é escopar o
        CONJUNTO DE CANDIDATOS por foco, o que remove a comparação entre focos sem
        mexer no número. Recalibrar é da Fase B e precisa de medição nova, não de
        palpite — mesma disciplina de `strategy.py`.
        """
        normalized = (targets or "").strip().lower()
        if normalized:
            rows = self.store.conn.execute(
                "SELECT id, embedding FROM questions "
                " WHERE embedding IS NOT NULL AND focus_id = (SELECT id FROM active_focus)"
                "   AND LOWER(TRIM(COALESCE(targets, ''))) = ?",
                (normalized,),
            ).fetchall()
        else:
            # Sem alvo declarado não há escopo DENTRO do foco; cai para comparação
            # global — global dentro do foco, nunca entre focos. Sem o filtro, uma
            # pergunta do foco B que casa com uma do foco A recebe `return None` e um
            # `log.debug`: a pergunta nunca é criada e nada no sistema reporta pergunta
            # que não foi gerada.
            rows = self.store.conn.execute(
                "SELECT id, embedding FROM questions WHERE embedding IS NOT NULL "
                "  AND focus_id = (SELECT id FROM active_focus)"
            ).fetchall()

        if not rows:
            return None
        query = np.asarray(vector, dtype=np.float32)
        for row in rows:
            if cosine(Store.unpack_embedding(row["embedding"]), query) >= self.dedup_threshold:
                return int(row["id"])
        return None

    # ───────────────────────────────────────────────────────── fila do humano

    def escalate(
        self, question_id: int, *, reason: StuckReason, partial_work: str | None = None
    ) -> bool:
        """Marca a pergunta como travada e a escala, se houver vaga.

        Devolve False quando ficou represada: o trabalho parcial e o motivo já ficam
        gravados, e `promote_escalations` a promove quando abrir espaço. Sem isso, ou
        a fila estoura, ou perdemos o diagnóstico de por que ela travou.
        """
        has_room = self.escalated_count() < self.human_queue_limit
        self.store.conn.execute(
            "UPDATE questions SET status = ?, stuck_reason = ?, partial_work = ?, "
            "                     escalated_at = ? WHERE id = ?",
            (
                QuestionStatus.ESCALATED.value if has_room else QuestionStatus.OPEN.value,
                reason.value,
                partial_work,
                iso(utcnow()) if has_room else None,
                question_id,
            ),
        )
        return has_room

    def escalated_count(self) -> int:
        return int(
            self.store.conn.execute(
                "SELECT COUNT(*) AS n FROM questions WHERE status = 'ESCALATED' "
                "  AND focus_id = (SELECT id FROM active_focus)"
            ).fetchone()["n"]
        )

    def promote_escalations(self) -> int:
        """Preenche as vagas livres com as represadas de maior prioridade."""
        free = self.human_queue_limit - self.escalated_count()
        if free <= 0:
            return 0
        rows = self.store.conn.execute(
            "SELECT id FROM questions "
            " WHERE status = 'OPEN' AND stuck_reason IS NOT NULL "
            "   AND focus_id = (SELECT id FROM active_focus) "
            " ORDER BY priority DESC, id ASC LIMIT ?",
            (free,),
        ).fetchall()
        for row in rows:
            self.store.conn.execute(
                "UPDATE questions SET status = 'ESCALATED', escalated_at = ? WHERE id = ?",
                (iso(utcnow()), row["id"]),
            )
        if rows:
            log.info("promovidas %d perguntas represadas para a fila humana", len(rows))
        return len(rows)

    def human_queue(self) -> list[QuestionRecord]:
        rows = self.store.conn.execute(
            "SELECT id, text, kind, priority, stuck_reason, partial_work "
            "  FROM escalated_queue LIMIT ?",
            (self.human_queue_limit,),
        ).fetchall()
        return [
            QuestionRecord(
                id=r["id"],
                text=r["text"],
                kind=QuestionKind(r["kind"]),
                status=QuestionStatus.ESCALATED,
                priority=r["priority"],
                stuck_reason=StuckReason(r["stuck_reason"]) if r["stuck_reason"] else None,
                partial_work=r["partial_work"],
            )
            for r in rows
        ]

    def answer_from_human(self, question_id: int, answer: str) -> None:
        """Registra a resposta e libera a vaga na fila."""
        self.store.conn.execute(
            "UPDATE questions SET status = 'ANSWERED_HUMAN', answer = ?, "
            "                     answer_origin = 'human', answered_at = ? WHERE id = ?",
            (answer, iso(utcnow()), question_id),
        )
        self.promote_escalations()

    # ─────────────────────────────────────────────────────── fila automática

    def next_for_research(self) -> QuestionRecord | None:
        kinds = ",".join(f"'{k.value}'" for k in AUTO_ANSWERABLE)
        row = self.store.conn.execute(
            f"SELECT id, text, kind, priority FROM questions "
            f" WHERE status = 'OPEN' AND kind IN ({kinds}) AND stuck_reason IS NULL "
            f"   AND focus_id = (SELECT id FROM active_focus) "
            f" ORDER BY priority DESC, id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return QuestionRecord(
            id=row["id"],
            text=row["text"],
            kind=QuestionKind(row["kind"]),
            status=QuestionStatus.OPEN,
            priority=row["priority"],
        )


def _stuck_reason_for(kind: QuestionKind) -> StuckReason:
    if kind is QuestionKind.PREFERENCE:
        return StuckReason.NEEDS_VALUE_JUDGMENT
    if kind is QuestionKind.CONTEXT:
        return StuckReason.NEEDS_CONTEXT
    return StuckReason.NEEDS_CONTEXT
