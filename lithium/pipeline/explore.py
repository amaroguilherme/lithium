"""Trilha exploratória: hipóteses mecanísticas, incluindo especulação sem dado humano.

Existe porque a trilha de evidência, por construção, **suprime exatamente o que
buscamos aqui**. Um candidato inédito tem evidência `preclinical` (0.10) ou
`extrapolated` (0.12); multiplicados, ~0.01 — invisível ao lado de qualquer RCT. Isso é
correto lá e inútil aqui.

Então esta trilha usa outro critério de ordenação: **plausibilidade × ineditismo**.

* `plausibility` = fração dos elos da cadeia mecanística ancorados em citação real,
  não assumidos. É um número medível, não uma impressão.
* `novelty` = distância da prática padrão, declarada pelo gerador.

Ordenar só por plausibilidade devolve o óbvio; só por ineditismo, delírio. O produto
pede "o mais plausível que ninguém testou".

Três salvaguardas tornam a especulação auditável em vez de imprudente, e todas são
requisito de schema — não sugestão de prompt:

1. **Cadeia explícita** com cada elo marcado `supported` (com PMID) ou `assumed`.
2. **Falsificador obrigatório.** Hipótese que nada refuta é prosa e não passa.
3. **Crítica adversarial dedicada** que procura o elo mais fraco em vez de avaliar o
   conjunto — uma cadeia vale o seu pior elo, e a média esconde o elo quebrado.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from lithium.db import Store
from lithium.focus import FocusProfile, NoActiveFocus
from lithium.llm import LLMClient, LLMError
from lithium.llm.prompts import budget_guard, render
from lithium.llm.schemas import (
    RegroundedChain,
    Speculation,
    SpeculationBatch,
    SpeculationCritique,
    SpeculationQueries,
    SourceQuery,
)
from lithium.pipeline.mechanism import route_block, taxonomy_block
from lithium.pipeline.state import KnowledgeState, build_state
from lithium.types import SourceKind
from lithium.worker.queue import iso, utcnow

log = logging.getLogger(__name__)

AVAILABLE_SOURCES: frozenset[SourceKind] = frozenset({SourceKind.PUBMED})
"""Fontes com adapter implementado. Query para fonte inexistente vira tarefa morta,
então filtramos aqui. Cresce com o item 9 (ClinicalTrials.gov, openFDA, Europe PMC)."""

@dataclass(slots=True)
class SpeculationRecord:
    id: int
    statement: str
    intervention_class: str
    mechanism_target: str
    novelty: float
    plausibility: float
    survives: bool
    weakest_link: str = ""
    fatal_flaw: str = ""

    @property
    def score(self) -> float:
        """O mais plausível que ninguém testou."""
        return self.plausibility * self.novelty


_PMID = re.compile(r"(?<![\w.])(?:pmid[:\s]*)?(\d{1,8})(?![\w.])", re.I)
r"""1 a 8 dígitos, e a fronteira à ESQUERDA é o que impede uma colisão real.

Sem `(?<![\w.])`, um PMCID vira o PMID de **outro artigo**, deterministicamente:
`PMC7738613` → `7738613`, que é um paper de 1995 sobre linfoma não-Hodgkin no JCO. Se esse
número existir no corpus, `gate_citations` **aprova** — e o portão existe justamente para
pegar "identificador plausível e real", que é pior que inventado. O modelo escreve PMCID
com frequência: o Europe PMC os usa como id primário.

O `.` na classe negada cobre versão e DOI (`v2.7738613`, `10.1016/j.7738613`). Limite
inferior de 1 porque PMIDs antigos são curtos (o PMID 1 existe); um piso de 5 recusaria
citação legítima, e a segurança não vem do formato — vem da existência no corpus."""


def canonical_pmid(evidence: str | None) -> str | None:
    r"""Extrai o PMID de uma string de evidência, tolerando o que o modelo escreve.

    **Tolerante de propósito, e o número justifica.** O modelo escreve `PMID: 28544150`
    — com espaço depois do dois-pontos, sempre. Um regex estrito (`^PMID:\d+$`) rejeita
    **9 de 9** citações reais: não seria um filtro, seria um apagão da trilha inteira.
    """
    if not evidence:
        return None
    match = _PMID.search(str(evidence))
    return match.group(1) if match else None


def gate_citations(chain: list[dict], known: set[str]) -> tuple[list[dict], int]:
    """Rebaixa elo cuja citação não existe no corpus. Devolve (cadeia, recusados).

    **É o portão que faltava, e a falta era grave.** Em operação real, com gemma-4-12b,
    9 de 12 elos vieram `supported` — cada um com um PMID **verdadeiro do PubMed** que
    não tem relação nenhuma com a alegação. Os títulos reais dos nove: matriz dérmica
    acelular para feridas crônicas, microarray de retrovírus suíno, peixes-cachimbo da
    costa sul-africana, doenças priônicas, poeira doméstica, osteopenia em litíase
    renal, amiloidose cardíaca, displasia arritmogênica de VD, composto antimalárico.
    Nenhum sobre sigma-1, cetose, ansiedade ou bipolar. E `plausibility` reportava
    **0,75** para as três hipóteses, número que o quadro mostra ao psiquiatra.

    Alucinar um identificador *plausível e real* é pior que inventar um: ele sobrevive a
    qualquer checagem de formato e só cai contra o corpus.

    **Rebaixa o elo, não rejeita a hipótese.** Medido: rejeitar mataria 3 de 3 das
    hipóteses reais. E os portões que julgam mérito já existem (falsificador obrigatório,
    crítica adversarial) — este corrige um NÚMERO, não julga a ideia.
    """
    out, refused = [], 0
    for step in chain:
        step = dict(step)
        if not step.get("supported"):
            out.append(step)
            continue
        pmid = canonical_pmid(step.get("evidence"))
        if pmid and pmid in known:
            step["evidence"] = f"PMID:{pmid}"
        else:
            step["supported"] = False
            step["citation_refused"] = str(step.get("evidence") or "")[:80]
            refused += 1
        out.append(step)
    return out, refused


def plausibility(chain: list[dict] | list) -> float:
    """Fração dos elos ancorados em citação real.

    Um elo só conta como ancorado se `supported` for verdadeiro **e** houver
    identificador. Marcar `supported: true` sem citação é o atalho óbvio para inflar a
    pontuação, e é exatamente o que este `and` fecha.
    """
    if not chain:
        return 0.0
    anchored = sum(
        1
        for step in chain
        if (step.get("supported") if isinstance(step, dict) else step.supported)
        and (step.get("evidence") if isinstance(step, dict) else step.evidence)
    )
    return anchored / len(chain)


# O modelo escreve "none"/"N/A" quando o campo deveria ficar vazio. Sem normalizar,
# a interface exibe "FALHA FATAL: none" — que lê como se houvesse falha.
_EMPTY_MARKERS = frozenset({"", "none", "n/a", "na", "-", "nenhum", "nenhuma", "null"})


MAX_GAP_QUERIES = 8
"""Uma consulta por lacuna, com teto. Cada uma é um POST de embedding só (o
`batch_size` do cliente é 16), então o custo é prefill, não round-trip."""


def _lessons_queries(state: KnowledgeState, profile: FocusProfile) -> list[str]:
    """Uma consulta POR LACUNA, derivada do estado — e exógena à saída anterior.

    Duas restrições que parecem detalhe e não são:

    **`untouched` é ordenado antes de fatiar.** Ele chega na ordem do dict literal que o
    construiu, então uma fatia sem ordenar é arbitrária *e instável*: a mesma lacuna
    entra ou sai conforme uma chave nova apareça no meio do dict.

    **A consulta nunca vem de um digest sintetizado por LLM.** Isso poria a conclusão em
    cache do sistema no comando de selecionar quais das próprias crenças ele enxerga —
    realimentação fechada, e é o modo de falha que esta fase inteira existe para evitar.
    Lacuna é fato do banco: "nada sobre X" e "as fontes discordam sobre Y".

    **Conflito vem antes de lacuna vazia**, e isso não é ordem arbitrária. Escrevi a
    lista na ordem inversa primeiro e um teste pegou: com muitos rótulos `untouched` —
    o caso normal num corpus jovem — eles enchiam os oito slots e **nenhum conflito
    entrava nunca**. Conflito é o sinal mais informativo dos dois: significa que há dado
    e ele discorda, o que é uma pergunta respondível; "nada sobre X" pode ser uma classe
    que ninguém estuda por bons motivos.
    """
    prefix = profile.retrieval_prefix
    queries = [
        f"{prefix}: sources disagree on the direction of {c.intervention}"
        for c in state.conflicts
    ]
    queries += [
        f"{prefix}: no evidence at all on {label}"
        for label in sorted(state.untouched)
    ]
    return queries[:MAX_GAP_QUERIES] or [f"{prefix}: mechanistic hypotheses"]


def _blank(value: str | None) -> str:
    return "" if (value or "").strip().lower() in _EMPTY_MARKERS else (value or "").strip()


def _render_chain(chain: list[dict]) -> str:
    lines = []
    for i, step in enumerate(chain, 1):
        mark = f"[supported: {step['evidence']}]" if step.get("supported") else "[ASSUMED]"
        lines.append(f"{i}. {step['claim']}  {mark}")
    return "\n".join(lines)


class Explorer:
    def __init__(self, store: Store, llm: LLMClient, reflector=None,
                 *, profile: FocusProfile, n_ctx: int = 8192) -> None:
        self.store = store
        self.llm = llm
        self.profile = profile
        self.n_ctx = n_ctx
        self.reflector = reflector
        """Opcional. Quando presente, as lições de pesquisa relevantes entram no
        prompt de geração — é assim que becos sem saída deixam de ser repropostos."""

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
                "nenhum foco ativo: a hipótese seria arquivada sob nenhum foco. "
                "Escolha um com `lithium focus --use <slug>`."
            )
        return int(focus["id"])

    async def generate(self, *, max_items: int = 3) -> list[SpeculationRecord]:
        """Um tick exploratório: propõe hipóteses, critica cada uma, persiste."""
        # Resolvido ANTES do POST, e a chamada abaixo leva ~118 s.
        focus_id = self._focus_id()
        state = build_state(self.store, self.profile)
        prompt = render(
            "generate_speculation",
            **self.profile.prompt_blocks("generate_speculation"),
            taxonomy=taxonomy_block(self.profile),
            routes=route_block(self.profile),
            existing=self._existing_block(),
            lessons=await self._lessons_block(state),
            state=state.render(),
            max_items=max_items,
        )
        # Aborta antes do POST. Sem isto, o llama-server devolve 400, o cliente
        # corretamente não repete 4xx, e a task vai para dead-letter — a trilha
        # especulativa morrendo em silêncio conforme o corpus cresce.
        budget_guard(prompt, label="generate_speculation", max_tokens=3072,
                     n_ctx=self.n_ctx)

        batch = await self.llm.structured(
            [{"role": "user", "content": prompt}],
            SpeculationBatch,
            max_tokens=3072,
            label="generate_speculation",
        )

        records: list[SpeculationRecord] = []
        for item in batch.speculations[:max_items]:
            record = await self.evaluate_and_store(item, focus_id=focus_id)
            if record is not None:
                records.append(record)

        survived = sum(r.survives for r in records)
        log.info(
            "tick exploratório: %d propostas, %d persistidas, %d sobreviveram à crítica",
            len(batch.speculations), len(records), survived,
        )
        return records

    def corpus_pmids(self, chain: list[dict]) -> set[str]:
        """Quais dos PMIDs citados existem de fato no corpus.

        Uma query indexada por hipótese, não por elo. `kind` na cláusula não é
        decoração: o índice é `UNIQUE(kind, external_id)`, e sem a primeira coluna o
        SQLite troca SEARCH por SCAN. Medido: p50 = 10,5 µs com 5.000 fontes — 1e-7 do
        custo de um tick de especulação.
        """
        wanted = {
            pmid for pmid in (canonical_pmid(s.get("evidence")) for s in chain) if pmid
        }
        if not wanted:
            return set()
        holes = ",".join("?" for _ in wanted)
        rows = self.store.conn.execute(
            f"SELECT external_id FROM sources "
            f" WHERE kind = 'pubmed' AND external_id IN ({holes})",
            sorted(wanted),
        ).fetchall()
        return {r["external_id"] for r in rows}

    async def evaluate_and_store(self, item: Speculation, *,
                                 focus_id: int | None = None) -> SpeculationRecord | None:
        chain = [step.model_dump() for step in item.chain]
        chain, refused = gate_citations(chain, self.corpus_pmids(chain))
        if refused:
            log.info(
                "%d elo(s) rebaixado(s) por citação fora do corpus: %s",
                refused, item.statement[:70],
            )

        # O falsificador é requisito de schema, mas string vazia passa pela gramática.
        # Sem ele a hipótese não é testável, e não testável não entra no quadro.
        if not item.falsifier.strip():
            log.info("descartada por falta de falsificador: %s", item.statement[:80])
            return None

        critique = await self._critique(item, chain)
        survives = bool(critique and critique.survives)

        # `focus_id=None` só acontece em chamada direta (testes): aí resolve agora, que
        # é o comportamento antigo. O caminho de produção SEMPRE passa o id resolvido
        # antes do POST — ver `_focus_id`.
        focus_id = self._focus_id() if focus_id is None else focus_id
        cur = self.store.conn.execute(
            "INSERT INTO hypotheses(focus_id, statement, tier, status, intervention_class, "
            "  mechanism_target, route, combination, chain_json, falsifier, "
            "  test_proposal, known_risks, novelty, critique_json, survives_critique, "
            "  updated_at) "
            "VALUES(?, ?, 'speculative', ?, ?, ?, ?, ?, "
            "       ?, ?, ?, ?, ?, ?, ?, ?) "
            # `ON CONFLICT(focus_id, statement)`, não `INSERT OR IGNORE`: o segundo
            # engoliria também violações de CHECK e NOT NULL como se fossem "já existia",
            # trocando um erro alto por um log mentiroso. Resolve nas duas formas de
            # tabela — verificado nas duas.
            "ON CONFLICT(focus_id, statement) DO NOTHING RETURNING id",
            (
                focus_id,
                item.statement,
                "active" if survives else "refuted",
                item.intervention_class,
                item.mechanism_target,
                item.route,
                item.combination,
                json.dumps(chain),
                item.falsifier,
                item.test_proposal,
                item.known_risks,
                item.novelty,
                critique.model_dump_json() if critique else None,
                int(survives),
                iso(utcnow()),
            ),
        )
        row = cur.fetchone()
        if row is None:
            return None  # já existia

        return SpeculationRecord(
            id=int(row["id"]),
            statement=item.statement,
            intervention_class=item.intervention_class,
            mechanism_target=item.mechanism_target,
            novelty=item.novelty,
            plausibility=plausibility(chain),
            survives=survives,
            weakest_link=_blank(critique.weakest_link) if critique else "",
            fatal_flaw=_blank(critique.fatal_flaw) if critique else "",
        )

    async def _critique(
        self, item: Speculation, chain: list[dict]
    ) -> SpeculationCritique | None:
        try:
            return await self.llm.structured(
                [
                    {
                        "role": "user",
                        "content": render(
                            "critique_speculation",
                            **self.profile.prompt_blocks("critique_speculation"),
                            statement=item.statement,
                            intervention_class=item.intervention_class,
                            mechanism_target=item.mechanism_target,
                            novelty=f"{item.novelty:.2f}",
                            chain=_render_chain(chain),
                            falsifier=item.falsifier,
                            known_risks=item.known_risks or "(none stated)",
                        ),
                    }
                ],
                SpeculationCritique,
                max_tokens=1024,
                label="critique_speculation",
            )
        except LLMError as exc:
            # Crítica indisponível reprova: sem o passe adversarial a hipótese não
            # foi verificada, e entrar como "ativa" mentiria sobre isso.
            log.warning("crítica de especulação falhou: %s", exc)
            return None

    # ────────────────────────────────────────── o laço: especular → buscar → ancorar

    async def plan_queries(self, hypothesis_id: int) -> list[SourceQuery]:
        """Converte uma hipótese em buscas dirigidas.

        É o passo que transforma a trilha exploratória de gerador de ideias em motor
        de pesquisa. As 19 queries fixas do `harvest_sweep` não mencionam sigma-1,
        orexina, via transdérmica nem dispositivo algum — sem isto, o sistema propõe
        um alvo e nunca pergunta a nenhuma base sobre ele.
        """
        row = self.store.conn.execute(
            "SELECT * FROM hypotheses WHERE id = ? AND tier = 'speculative'",
            (hypothesis_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"especulação {hypothesis_id} não existe")

        plan = await self.llm.structured(
            [
                {
                    "role": "user",
                    "content": render(
                        "speculation_queries",
                        statement=row["statement"],
                        mechanism_target=row["mechanism_target"] or "—",
                        intervention_class=row["intervention_class"] or "—",
                        route=row["route"] or "—",
                        combination=row["combination"] or "(single agent)",
                        chain=_render_chain(json.loads(row["chain_json"] or "[]")),
                        test_proposal=row["test_proposal"] or "—",
                        sources=", ".join(s.value for s in AVAILABLE_SOURCES),
                    ),
                }
            ],
            SpeculationQueries,
            max_tokens=1536,
            label="speculation_queries",
        )
        # Fontes ainda não implementadas viriam a virar tarefa morta na fila.
        return [q for q in plan.queries if q.source in AVAILABLE_SOURCES]

    def mark_pursued(self, hypothesis_id: int) -> None:
        self.store.conn.execute(
            "UPDATE hypotheses SET pursued_at = ? WHERE id = ?",
            (iso(utcnow()), hypothesis_id),
        )

    async def reground(self, hypothesis_id: int, retriever) -> int:
        """Reavalia a cadeia contra o corpus atual. Devolve quantos elos foram ancorados.

        É por aqui que a plausibilidade sobe com o tempo. Sem isto a cadeia congela no
        estado em que nasceu, e perseguir a hipótese não teria consequência mensurável.
        """
        row = self.store.conn.execute(
            "SELECT * FROM hypotheses WHERE id = ? AND tier = 'speculative'",
            (hypothesis_id,),
        ).fetchone()
        if row is None:
            return 0

        chain = json.loads(row["chain_json"] or "[]")
        assumed = [i for i, step in enumerate(chain) if not step.get("supported")]
        if not assumed:
            return 0

        # Busca com a hipótese inteira mais os elos assumidos: os termos dos elos são
        # o que discrimina, mas sozinhos perdem o contexto do alvo.
        query = " ".join(
            [row["statement"], row["mechanism_target"] or ""]
            + [chain[i]["claim"] for i in assumed]
        )
        hits = await retriever.search_claims(query, k=12)
        if not hits:
            self._mark_regrounded(hypothesis_id)
            return 0

        evidence = "\n".join(
            f"  [PMID:{h.external_id}] ({h.grade.value}/{h.directness.value}) {h.statement}"
            for h in hits
        )
        known_pmids = {h.external_id for h in hits}

        try:
            result = await self.llm.structured(
                [
                    {
                        "role": "user",
                        "content": render(
                            "reground_chain",
                            chain=_render_chain(chain),
                            evidence=evidence,
                        ),
                    }
                ],
                RegroundedChain,
                max_tokens=1024,
                label="reground_chain",
            )
        except LLMError as exc:
            log.warning("reancoragem falhou para #%d: %s", hypothesis_id, exc)
            return 0

        anchored = 0
        for link in result.links:
            index = link.index - 1
            if not (0 <= index < len(chain)) or index not in assumed:
                continue
            if not link.now_supported:
                continue
            pmid = link.evidence.replace("PMID:", "").strip()
            # Só aceita citação que existe na evidência mostrada. Sem esta checagem,
            # a reancoragem vira o caminho mais fácil para inflar plausibilidade com
            # um PMID inventado.
            if pmid not in known_pmids:
                log.info("reancoragem recusada: PMID %r não estava na evidência", pmid)
                continue
            chain[index]["supported"] = True
            chain[index]["evidence"] = f"PMID:{pmid}"
            anchored += 1

        if anchored:
            self.store.conn.execute(
                "UPDATE hypotheses SET chain_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(chain), iso(utcnow()), hypothesis_id),
            )
            log.info(
                "hipótese #%d: %d elo(s) ancorado(s), plausibilidade agora %.0f%%",
                hypothesis_id, anchored, plausibility(chain) * 100,
            )
        self._mark_regrounded(hypothesis_id)
        return anchored

    def _mark_regrounded(self, hypothesis_id: int) -> None:
        self.store.conn.execute(
            "UPDATE hypotheses SET regrounded_at = ? WHERE id = ?",
            (iso(utcnow()), hypothesis_id),
        )

    def pending_pursuit(self, limit: int = 3) -> list[int]:
        """Hipóteses vivas que ainda não geraram buscas, mais promissoras primeiro."""
        rows = self.store.conn.execute(
            "SELECT id FROM speculation_board "
            " WHERE survives_critique = 1 AND pursued_at IS NULL "
            "   AND focus_id = (SELECT id FROM active_focus) "
            " ORDER BY plausibility * COALESCE(novelty, 0) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [int(r["id"]) for r in rows]

    def pending_reground(self, limit: int = 5) -> list[int]:
        """Hipóteses já perseguidas cuja cadeia ainda tem elo assumido."""
        rows = self.store.conn.execute(
            "SELECT id FROM speculation_board "
            " WHERE survives_critique = 1 AND pursued_at IS NOT NULL "
            "   AND focus_id = (SELECT id FROM active_focus) "
            "   AND supported_links < chain_length "
            " ORDER BY COALESCE(regrounded_at, '') ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [int(r["id"]) for r in rows]

    async def _lessons_block(self, state: KnowledgeState | None = None) -> str:
        """Recuperadas por relevância a cada lacuna, com teto.

        A consulta é **derivada do estado de conhecimento**, não um literal. Antes era um
        literal fixo, então o ranking de lições era idêntico em toda rodada pela vida do
        sistema: as mesmas oito em todo prompt, para sempre.
        """
        if self.reflector is None:
            return "(none)"
        lessons = await self.reflector.relevant_lessons(
            _lessons_queries(state or build_state(self.store, self.profile),
                             self.profile)
        )
        if not lessons:
            return "(none)"
        # A renderização mora no Reflector: ela depende da proveniência de cada lição
        # (um `dead_end` é renderizado a partir da hipótese que ele exclui, nunca do
        # texto que o modelo escreveu) e precisa de acesso ao banco.
        return self.reflector.lessons_for_speculation(lessons)

    def _existing_block(self) -> str:
        rows = self.store.conn.execute(
            "SELECT statement, mechanism_target FROM hypotheses "
            " WHERE tier = 'speculative' AND focus_id = (SELECT id FROM active_focus) "
            " ORDER BY id DESC LIMIT 30"
        ).fetchall()
        if not rows:
            return "(nothing proposed yet)"
        return "\n".join(f"  - [{r['mechanism_target']}] {r['statement']}" for r in rows)

    # ────────────────────────────────────────────────────────────────── leitura

    def board(self, *, only_surviving: bool = True, limit: int = 20) -> list[dict]:
        """O quadro especulativo, ordenado por plausibilidade × ineditismo."""
        rows = self.store.conn.execute(
            "SELECT * FROM speculation_board "
            " WHERE focus_id = (SELECT id FROM active_focus) "
            f"{'AND survives_critique = 1' if only_surviving else ''} "
            " ORDER BY plausibility * COALESCE(novelty, 0) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
