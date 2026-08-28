"""Conversa com memória, ao lado do trabalho autônomo de pesquisa.

O daemon pesquisa sozinho; este é o canal para falar com ele quando você quiser. Duas
coisas o distinguem de um chat qualquer:

**Ele enxerga o corpus.** Cada turno recupera as claims verificadas relevantes e as
injeta no contexto, com PMID. O prompt exige separar "a literatura diz" de "eu acho" —
é a única coisa que este sistema tem de valioso, e uma frase confiante sem fonte
destrói isso.

**Ele propõe memórias, nunca grava sozinho.** Ao detectar algo durável — preferência,
restrição, contexto do caso — ele pergunta antes de guardar. Num domínio de saúde,
acumular fatos sobre a pessoa sem autorização explícita é diferente de acumular papers.

A memória fecha um ciclo que já estava projetado: quando o loop de pesquisa trava numa
pergunta `CONTEXT` ("o que já foi tentado?"), ele consulta as memórias antes de escalar.
O que você contou uma vez não precisa ser perguntado de novo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lithium.db import Store
from lithium.db.store import normalize_memory_text
from lithium.llm import LLMClient, LLMError
from lithium.llm.prompts import (
    BUDGET_MARGIN,
    CHARS_PER_TOKEN,
    budget_guard,
    estimate_tokens,
    render,
)
from lithium.llm.schemas import MemoryProposal
from lithium.pipeline.retrieval import CONTRA_DIRECTIONS, Retriever
from lithium.focus import FocusProfile
from lithium.safety.rules import (
    RuleSet,
    concepts_implicated_by,
    concepts_in,
    ruleset_from_profile,
)
from lithium.safety.screen import Alert, Segment
from lithium.safety.screen import render as render_alerts
from lithium.safety.screen import screen
from lithium.worker.queue import iso, utcnow

log = logging.getLogger(__name__)

VALID_KINDS = frozenset({"preference", "context", "constraint", "fact"})


def _pmid_of(line: str) -> str:
    return line.split("PMID:")[1].split("]")[0]


def _host_of(provenance: str | None) -> str:
    """O domínio de onde uma nota de recon veio, para a linha carregar proveniência."""
    import json
    from urllib.parse import urlsplit

    try:
        url = (json.loads(provenance or "{}") or {}).get("url") or ""
    except (TypeError, ValueError):
        return "the web"
    return (urlsplit(url).hostname or "the web").lower()

MEMORY_CHAR_BUDGET = 4_000
"""Teto do bloco de memórias, em caracteres (~1 k tokens).

A query não tinha WHERE, LIMIT nem ORDER BY: crescia sem teto dentro do prompt. Em
caracteres e não em contagem de linhas porque o texto de uma memória não tem limite de
tamanho — `add_manual` aceita qualquer coisa."""

CONSTRAINT_NOTES_CHAR_BUDGET = 2_500
"""Teto do bloco de colisões, também em caracteres.

Este é o bloco DOMINANTE, não o secundário: cada linha embute o texto inteiro da
restrição, repetido uma vez por claim que colide. Com dez restrições que alcançam
lítio/valproato/carbamazepina e textos de 600 letras, o turno de pior caso mede 10.137
tokens contra uma janela de 8.192 — e com 2.000 letras o `budget_guard` levanta em
**todo** `send()`, deixando o chat permanentemente morto."""

HISTORY_CHAR_BUDGET = 12_000
"""TETO do histórico replayado, em caracteres (~3 k tokens). Agora é um teto, não a
alocação: o valor efetivo é DERIVADO do prompt de sistema já renderizado, e este número
só impede o histórico de crescer quando sobra janela.

`history_turns=12` significavam 24 mensagens, e cada resposta do assistente pode ter
`max_tokens=1536` — ou seja **até ~18 400 tokens de histórico** contra uma janela de
8192. A sessão morria com traceback no primeiro turno longo. Contar mensagens é a
unidade errada; o que importa é o tamanho."""

RECON_NOTES_CHAR_BUDGET = 800
"""Teto do bloco de notas de recon.

Bloco PRÓPRIO, nunca dentro de `$evidence` e nunca dentro de `$memories`. Fora de
`$evidence` pela mesma razão que `_constraint_notes` é separado: para o modelo, texto
grudado na linha da claim é indistinguível de um rebaixamento, e seria a Invariante 1
violada com outra sintaxe. Fora de `$memories` porque aquele bloco tem o cabeçalho
literal "What you know about this user" — prosa de uma página web ali é afirmação falsa
de proveniência."""

MAX_REPLY_TOKENS = 1536

DEFAULT_N_CTX = 8192


@dataclass(slots=True)
class Turn:
    reply: str
    proposal: MemoryProposal | None = None
    cited: list[str] = None  # type: ignore[assignment]
    alerts: list[Alert] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cited is None:
            self.cited = []
        if self.alerts is None:
            self.alerts = []


class ChatEngine:
    def __init__(
        self,
        store: Store,
        llm: LLMClient,
        embedder,
        *,
        profile: FocusProfile,
        history_turns: int = 6,
        evidence_k: int = 6,
        n_ctx: int = DEFAULT_N_CTX,
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.profile = profile
        self.n_ctx = n_ctx
        self.ruleset: RuleSet = ruleset_from_profile(profile)
        """Construído no __init__ a partir do PERFIL, nunca importado do módulo.
        `CONCEPT_BY_KEY` no topo do arquivo era um binding de import time: trocar de
        foco deixava a anotação de colisão do chat falando dos fármacos do foco
        antigo."""
        self.history_turns = history_turns
        self.evidence_k = evidence_k

    # ─────────────────────────────────────────────────────────────────── turno

    async def send(self, message: str, *, detect_memory: bool = True) -> Turn:
        evidence, cited = await self._evidence_for(message)
        alerts = screen(self._segments(message, evidence), self.ruleset)
        system = render(
            "chat",
            **self.profile.prompt_blocks("chat"),
            memories=self._memory_block(),
            evidence=evidence,
            constraint_notes=self._constraint_notes(evidence),
            recon_notes=self._recon_notes(),
            safety=render_alerts(alerts, ruleset_declared=self.ruleset.declared),
        )

        self._append("user", message)
        # O orçamento do histórico é DERIVADO do sistema JÁ RENDERIZADO, não constante.
        # MEDIDO com o `chat.md` real e o perfil real ANTES desta fase: sistema vazio =
        # 3.097 chars (774 tok), pior caso com os tetos vigentes = 16.397 chars (4.099
        # tok); somando 12.000 chars de histórico e `max_tokens=1536` dá **8.635 tokens
        # contra os 7.936 utilizáveis** de `n_ctx=8192`. O chat já estourava, e o
        # `except PromptTooLarge` de cli.py era INALCANÇÁVEL porque `send()` nunca
        # chamava `budget_guard` — o llama-server devolvia 400 e o usuário lia "falha na
        # chamada". Acrescentar dois blocos sem isto seria matar o chat sem diagnóstico.
        messages = [
            {"role": "system", "content": system},
            *self._history(char_budget=self._history_budget(system)),
        ]
        budget_guard("".join(m["content"] for m in messages), label="chat",
                     max_tokens=MAX_REPLY_TOKENS, n_ctx=self.n_ctx)
        reply = await self.llm.complete(messages, temperature=0.4,
                                        max_tokens=MAX_REPLY_TOKENS, label="chat")
        self._append("assistant", reply)

        proposal = await self._detect_memory(message) if detect_memory else None
        return Turn(reply=reply, proposal=proposal, cited=cited, alerts=alerts)

    def _segments(self, message: str, evidence: str) -> list[Segment]:
        """O material do turno, para o screen. **Sem memórias** — ver o docstring de
        `safety.screen`: uma restrição gravada gerava alerta permanente em todo turno,
        e o sistema fabricava os próprios falsos positivos."""
        segments = [Segment("user", message)]
        for line in evidence.splitlines():
            if line.startswith("  [PMID:"):
                segments.append(Segment("evidence", line, _pmid_of(line)))
        return segments

    def _constraint_notes(self, evidence: str) -> str:
        """Colisões entre restrição declarada e evidência recuperada.

        Bloco **próprio**, nunca dentro de `$evidence`, e **derivado** dele. A
        assinatura é a peça de design que faz a trava valer para sempre: o método recebe
        o bloco já pronto e devolve texto novo, então não tem como filtrá-lo nem
        reordená-lo — não devolve evidência.

        Colar `[conflita com restrição]` na linha da claim seria a outra opção, e é
        pior: para o modelo, uma etiqueta grudada na linha é indistinguível de um
        rebaixamento. Seria o gesto que a invariante proíbe, cometido com outra sintaxe.
        """
        rows = self.store.conn.execute(
            "SELECT text FROM user_memories WHERE kind IN ('constraint', 'preference')"
        ).fetchall()
        if not rows:
            return "(nenhuma restrição declarada)"

        notes: list[str] = []
        for row in rows:
            implicated = concepts_implicated_by(row["text"], self.ruleset)
            if not implicated:
                continue
            for line in evidence.splitlines():
                if not line.startswith("  [PMID:"):
                    continue
                hit = concepts_in(line, self.ruleset) & implicated
                if hit:
                    label = ", ".join(sorted(self.ruleset.by_key[k].label for k in hit))
                    notes.append(
                        f"  [PMID:{_pmid_of(line)}] menciona {label} — colide com a "
                        f"restrição declarada: {row['text']}"
                    )
        if not notes:
            return "(nenhuma colisão neste turno)"
        return "\n".join(
            self._bounded(notes, CONSTRAINT_NOTES_CHAR_BUDGET, "colisões")
        )

    async def _evidence_for(self, query: str) -> tuple[str, list[str]]:
        try:
            hits = await Retriever(self.store, self.embedder).search_claims(
                query, k=self.evidence_k
            )
        except Exception as exc:  # noqa: BLE001
            # Sem servidor de embeddings a conversa continua, sem corpus. Melhor que
            # derrubar o chat — e o prompt já obriga a admitir quando não há fonte.
            log.warning("recuperação de evidência falhou: %s", exc)
            return "## Evidence from the corpus\n(retrieval unavailable)", []

        if not hits:
            # A contagem real importa. Dizer só "nothing matches" fez o modelo
            # afirmar "o corpus está vazio" com 1 claim indexada — uma afirmação
            # diferente, e falsa. O número torna a distinção impossível de perder.
            # Sobre `claim_weight`: com o fail-closed, "o corpus tem 400 claims
            # verificadas, nenhuma casou" é uma afirmação sobre RELEVÂNCIA quando o fato
            # é ausência de JULGAMENTO. A mesma linha que impedia uma mentira passaria a
            # produzir a mentira oposta.
            total = self.store.conn.execute(
                "SELECT COUNT(*) AS n FROM claim_weight"
            ).fetchone()["n"]
            unjudged = self.store.counts()["claims_unweighted"]
            if total == 0 and unjudged == 0:
                note = ("The corpus is empty — nothing has been harvested yet. Say so, "
                        "and offer to start the research.")
            else:
                note = (f"The corpus holds {total} verified claim(s) judged against the "
                        "active focus, but none matched this query. Do NOT say the "
                        "corpus is empty — say nothing relevant to *this* question "
                        "was found.")
                if unjudged:
                    note += (
                        f" A further {unjudged} verified claim(s) carry no directness "
                        "judgment for the active focus and are invisible to retrieval — "
                        "that is missing judgment, not missing evidence.")
            return f"## Evidence from the corpus\n{note}", []

        lines = ["## Evidence from the corpus (cite the PMID when you use these)"]
        for h in hits:
            mark = " [counter-evidence]" if h.direction in CONTRA_DIRECTIONS else ""
            lines.append(
                f"  [PMID:{h.external_id}] ({h.grade.value} / {h.directness.value}, "
                f"{h.year or '?'}){mark} {h.statement}"
            )
        # "Nada contra" e "não procuramos" são afirmações diferentes, e o modelo preenche
        # o silêncio com a primeira. A linha de ausência é explícita por isso — e a
        # contagem do corpus a torna verificável em vez de retórica.
        if not any(h.direction in CONTRA_DIRECTIONS for h in hits):
            total = self.store.conn.execute(
                "SELECT COUNT(*) AS n FROM claim_weight WHERE direction IN "
                f"({','.join('?' * len(CONTRA_DIRECTIONS))})",
                sorted(CONTRA_DIRECTIONS),
            ).fetchone()["n"]
            lines.append(
                f"  (no counter-evidence in this retrieval. The corpus holds {total} "
                "negative or null claim(s) overall — none of them matched this query. "
                "Do NOT read this as 'nothing contradicts it'.)"
            )
        return "\n".join(lines), [h.external_id for h in hits]

    # ─────────────────────────────────────────────────────────────── histórico

    def _append(self, role: str, content: str) -> None:
        self.store.conn.execute(
            "INSERT INTO messages(role, content) VALUES(?, ?)", (role, content)
        )

    def _history(self, *, char_budget: int = HISTORY_CHAR_BUDGET) -> list[dict[str, str]]:
        """Mais recente primeiro, cortado por tamanho e não só por contagem."""
        rows = self.store.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (self.history_turns * 2,),
        ).fetchall()

        kept: list[dict[str, str]] = []
        used = 0
        for row in rows:                      # já vem do mais novo para o mais antigo
            size = len(row["content"])
            if kept and used + size > char_budget:
                break                          # nunca descartar o turno atual
            kept.append({"role": row["role"], "content": row["content"]})
            used += size
        return list(reversed(kept))

    def _history_budget(self, system: str) -> int:
        """Quantos caracteres de histórico ainda cabem, dado o sistema já renderizado.

        Zero (ou negativo) significa "nem o prompt de sistema cabe" — e aí `/novo` não
        adianta. `budget_guard` levanta logo depois com os números, e o CLI usa este
        mesmo cálculo para distinguir as duas mensagens.
        """
        usable = self.n_ctx - BUDGET_MARGIN - MAX_REPLY_TOKENS - estimate_tokens(system)
        return max(0, min(HISTORY_CHAR_BUDGET, usable * CHARS_PER_TOKEN))

    def system_alone_overflows(self) -> bool:
        """True quando limpar o histórico não resolveria. Lido pela mensagem do CLI."""
        system = render(
            "chat",
            **self.profile.prompt_blocks("chat"),
            memories=self._memory_block(),
            evidence="## Evidence from the corpus\n(—)",
            constraint_notes="(—)",
            recon_notes="(—)",
            safety="(—)",
        )
        return estimate_tokens(system) + MAX_REPLY_TOKENS > self.n_ctx - BUDGET_MARGIN

    def clear_history(self) -> int:
        cur = self.store.conn.execute("DELETE FROM messages")
        return cur.rowcount

    # ───────────────────────────────────────────────────────────────── memória

    def _bounded(self, lines: list[str], budget: int, what: str) -> list[str]:
        """Corta por TAMANHO e declara o que escondeu.

        Teto por contagem é incalibrável quando a linha tem comprimento livre — foi o
        erro que `HISTORY_CHAR_BUDGET` já corrigiu uma vez (`history_turns=12` contava a
        unidade errada). E aqui é pior: `MemoryProposal.text` não tem limite, então vinte
        linhas de duas mil letras somam oito mil tokens e o `budget_guard` passa a
        levantar em **todo** `send()` — com o chat permanentemente morto e `/novo` sem
        poder ajudar, porque o prompt de sistema sozinho já não cabe.
        """
        kept, used = [], 0
        for line in lines:
            if kept and used + len(line) > budget:
                break
            kept.append(line)
            used += len(line)
        hidden = len(lines) - len(kept)
        if hidden:
            kept.append(f"  (+{hidden} {what} não mostradas — este quadro é parcial)")
        return kept

    def _memory_block(self) -> str:
        # `user_memories`, não `live_memories`: as lições de pesquisa são sobre o
        # trabalho do sistema, não sobre você, e entram nos prompts que decidem
        # pesquisa — não no que descreve quem você é.
        # Ordenado por tipo, não por relevância. Top-k por similaridade aqui está
        # CORTADO por duas razões medidas: (a) `context` carrega histórico de reação
        # adversa — o exemplo canônico do próprio prompt é "já tentou lamotrigina e teve
        # rash" — e o que cai fora de um top-k desaparece sem aviso, que é risco clínico;
        # (b) ranking por similaridade sobre memória é ruído neste corpus: consulta
        # conversacional no assunto certo deu cosseno 0,705, pergunta de cardiologia
        # 0,710. As faixas se sobrepõem por completo.
        rows = self.store.conn.execute(
            "SELECT text, kind FROM user_memories "
            " ORDER BY CASE kind WHEN 'constraint' THEN 0 WHEN 'preference' THEN 1 "
            "                    WHEN 'context' THEN 2 ELSE 3 END, id"
        ).fetchall()
        if not rows:
            return "## What you know about this user\n(nothing yet)"
        lines = self._bounded(
            [f"  [{r['kind']}] {r['text']}" for r in rows],
            MEMORY_CHAR_BUDGET, "memórias",
        )
        return "\n".join(["## What you know about this user", *lines])

    def _recon_notes(self) -> str:
        """O que você AUTORIZOU o batedor a guardar, no foco ATIVO.

        Lê `recon_memories`, que é fail-closed por construção: sem foco ativo o
        subselect é NULL e a view devolve zero linhas.

        **Só o que foi APROVADO entra aqui.** Uma descoberta PENDENTE não tem bloco
        nenhum no prompt de sistema, e isso é a decisão central desta superfície: a
        memória só importa PORQUE é injetada em prompt, então injetar o texto pendente
        durante os 14 dias da janela de expiração tornaria o regime de consentimento
        decorativo — o sistema gasta um rebuild de tabela, dois CHECKs e uma view nova
        para gatear a injeção e depois abriria um segundo caminho sem portão para o
        MESMO prompt. As pendentes têm duas superfícies sob seu controle: `lithium
        discoveries` e `/descobertas`, que imprimem no TERMINAL, fora do contexto do
        modelo.

        A proveniência vai em cada linha, não só no cabeçalho: o cabeçalho é uma linha e
        o bloco pode ter oito.
        """
        rows = self.store.conn.execute(
            "SELECT text, provenance FROM recon_memories ORDER BY id"
        ).fetchall()
        if not rows:
            return "(nothing yet — nothing has been read on the web and approved)"
        lines = [
            f"  [read on {_host_of(r['provenance'])}] {r['text']}" for r in rows
        ]
        return "\n".join(self._bounded(lines, RECON_NOTES_CHAR_BUDGET, "notas"))

    async def _detect_memory(self, message: str) -> MemoryProposal | None:
        existing = self.store.conn.execute("SELECT text FROM user_memories").fetchall()
        block = "\n".join(f"  - {r['text']}" for r in existing) or "  (none)"
        try:
            proposal = await self.llm.structured(
                [
                    {
                        "role": "user",
                        "content": render("detect_memory", existing=block, message=message),
                    }
                ],
                MemoryProposal,
                max_tokens=512,
                label="detect_memory",
            )
        except LLMError as exc:
            log.debug("detecção de memória falhou: %s", exc)
            return None

        if not proposal.worth_remembering or not proposal.text.strip():
            return None

        # Supressão do que você já recusou — em Python, DEPOIS da resposta do LLM.
        #
        # As duas alternativas óbvias são piores. Alargar `user_memories` para incluir
        # recusadas poria texto REJEITADO dentro de "o que você sabe sobre este
        # usuário", que é injetado no prompt de chat: uma recusa viraria um fato sobre
        # você. E listar as recusas no prompt de detecção custa +3.2 k tokens com 150
        # recusas — 9x o prompt estático, 39% da janela, todo turno. Medido, não
        # estimado. Este filtro custa ~0,2 ms e zero token.
        if normalize_memory_text(proposal.text) in self.declined_keys():
            log.debug("proposta suprimida por recusa anterior: %s", proposal.text[:70])
            return None

        if proposal.kind not in VALID_KINDS:
            # **Rejeita, não coage.** Coagir para `'fact'` parecia conservador e era o
            # contrário: `'fact'` é o último tier do `ORDER BY` do bloco de memórias,
            # então uma restrição que o modelo rotulou errado ia para o fim da fila **por
            # construção** — a coerção e a ordenação se combinavam justamente contra a
            # categoria que mais importa, e sem deixar rastro.
            #
            # Perder uma proposta é recuperável e visível: aparece no log, e você pode
            # gravar com o tipo certo por `lithium memories --add --kind`. Enterrar uma
            # restrição é invisível. E um modelo que emite `kind` fora de um enum de
            # quatro valores provavelmente entendeu a extração inteira errado.
            log.info("proposta descartada por kind inválido (%r): %s",
                     proposal.kind, proposal.text[:70])
            return None
        return proposal

    async def remember(self, proposal: MemoryProposal, *, source: str = "chat") -> int:
        """Grava — só deve ser chamado depois da confirmação do usuário.

        **O embedder pode falhar e o "sim" NÃO se perde.** Antes desta fase o
        `embed()` vinha ANTES de qualquer INSERT, que é literalmente o bug registrado em
        `test_escalation_memory.py::test_a_dead_embedder_still_records_and_escalates`
        ("a pergunta sumia inteira"): com o llama-server fora do ar — o estado normal
        depois de `lithium mode off` — a confirmação explícita do usuário virava
        traceback. Sem vetor a memória fica fora do dedup semântico; é degradação
        visível e recuperável, ao contrário de perder o consentimento.
        """
        vector = None
        try:
            [vector] = await self.embedder.embed([proposal.text])
        except Exception as exc:  # noqa: BLE001
            log.warning("embedder indisponível; memória gravada sem vetor: %s", exc)
            vector = None
        cur = self.store.conn.execute(
            "INSERT INTO memories(text, kind, rationale, source, confirmed, "
            "                     embedding, confirmed_at, text_key) "
            "VALUES(?, ?, ?, ?, 1, ?, ?, ?) RETURNING id",
            (
                proposal.text,
                proposal.kind,
                proposal.rationale,
                source,
                Store.pack_embedding(vector) if vector is not None else None,
                iso(utcnow()),
                normalize_memory_text(proposal.text),
            ),
        )
        memory_id = int(cur.fetchone()["id"])
        log.info("memória #%d gravada: %s", memory_id, proposal.text[:70])
        return memory_id

    def decline(self, proposal: MemoryProposal) -> int:
        """Registra a recusa sem ativar, e devolve o id — para o CLI poder oferecer
        o caminho de volta."""
        cur = self.store.conn.execute(
            "INSERT INTO memories(text, kind, rationale, source, confirmed, active, "
            "                     text_key) "
            "VALUES(?, ?, ?, 'chat', 0, 0, ?) RETURNING id",
            (
                proposal.text,
                proposal.kind,
                proposal.rationale,
                normalize_memory_text(proposal.text),
            ),
        )
        return int(cur.fetchone()["id"])

    def declined_keys(self) -> set[str]:
        """As chaves normalizadas do que você já recusou.

        Lida do banco a cada turno, não cacheada: `allow()` precisa ter efeito
        imediato, e o custo medido é sub-milissegundo — a query é servida pelo
        `idx_memories_live` que já existe.
        """
        rows = self.store.conn.execute(
            "SELECT text, text_key FROM declined_memories"
        ).fetchall()
        return {r["text_key"] or normalize_memory_text(r["text"]) for r in rows}

    def allow(self, memory_id: int) -> bool:
        """Desfaz uma recusa: a memória volta a poder ser proposta.

        Existe porque supressão permanente é esquecimento sem aviso — o mesmo modo de
        falha que a lista "o que eu não construiria" proíbe para memórias do usuário.
        Recusar hoje não pode significar nunca mais poder dizer sim.
        """
        cur = self.store.conn.execute(
            "DELETE FROM memories WHERE id = ? AND confirmed = 0 AND active = 0",
            (memory_id,),
        )
        return cur.rowcount > 0

    def memories(self, *, include_inactive: bool = False) -> list[dict]:
        sql = (
            "SELECT id, text, kind, rationale, source, confirmed, active, created_at "
            "  FROM memories "
            f"{'' if include_inactive else 'WHERE confirmed = 1 AND active = 1 '}"
            " ORDER BY id"
        )
        return [dict(r) for r in self.store.conn.execute(sql)]

    def forget(self, memory_id: int) -> bool:
        """Desativa em vez de apagar — o histórico de o que foi lembrado importa."""
        cur = self.store.conn.execute(
            "UPDATE memories SET active = 0 WHERE id = ? AND active = 1", (memory_id,)
        )
        return cur.rowcount > 0

    async def add_manual(self, text: str, kind: str = "fact") -> int:
        return await self.remember(
            MemoryProposal(
                worth_remembering=True,
                text=text,
                kind=kind if kind in VALID_KINDS else "fact",
                rationale="adicionada manualmente",
            ),
            source="manual",
        )
