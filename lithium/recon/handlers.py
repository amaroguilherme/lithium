"""Os handlers do batedor. O FATIAMENTO é a peça de engenharia, não o código.

    recon_sweep ──> recon_query ──> recon_triage ──> recon_read
    (expira +        (UMA busca      (UMA chamada     (robots + fetch +
     monta as         FATURADA)       de LLM sobre     UMA chamada de LLM,
     queries)                         os snippets)     grava a observation)

`recon_query` é o ÚNICO passo que gasta DINHEIRO. Por isso ele é uma tarefa sozinha: o
retry de uma triagem ou de uma leitura custa zero. Fundir os três num handler só faria
`queue.fail` — que repete 3 vezes com backoff — triplicar a fatura da busca por uma
falha de rede na leitura, invisivelmente, porque nada no sistema reporta "gastei 3x pelo
mesmo resultado".

`recon_lead` NÃO mora aqui. Ele é A PONTE para o canal de evidência e vive em
`lithium/worker/handlers.py`, onde o resto do canal vive — ver o docstring de
`lithium/recon/__init__.py`.

**Atingir o teto é `log.info` + `return`, NUNCA `raise`.** Cota atingida é operação
normal. Transformá-la em dead-letter gasta 3 tentativas pagas e dispara um toast "N
tarefas falharam", e operação normal reportada como falha é o que treina alguém a
ignorar o aviso seguinte.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from lithium.focus import profile_for_slug
from lithium.llm.prompts import budget_guard, render
from lithium.llm.schemas import ReconObservation, ReconTriage
from lithium.recon import budget
from lithium.recon.leads import is_indexed_article, lead_from_hit
from lithium.recon.read import PageRefused, read_max_chars
from lithium.recon.search import WebHit
from lithium.recon.verbs import expire

log = logging.getLogger(__name__)

TRIAGE_MAX_TOKENS = 512
"""E não 320. Comparação com a calibração medida do próprio repo: `reflect` usa 1024
para no máximo 3 lições (~340 tok/item) e `generate_questions` usa 2048 para 5
perguntas. Dez vereditos em 320 tokens seriam 32 tok/item — e truncar saída ESTRUTURADA
não degrada: `LLMClient.structured` só levanta `LLMTruncated` com `raw` VAZIO; com JSON
cortado no meio o reparo reenvia com o MESMO teto, o resultado é determinístico, e a
tarefa morre depois de queimar ~9 chamadas. Com `reason` fora do schema, 512 é folga
larga."""

OBSERVE_MAX_TOKENS = 256

SUMMARY_MAX_CHARS = 400
TITLE_MAX_CHARS = 200
QUERY_MAX_CHARS = 200
"""Tetos de gravação. `title` e `summary` vêm da web e não têm limite de tamanho;
`lithium discoveries` e o chat têm orçamento de caracteres."""

FALLBACK_QUERY_SUFFIX = "guideline OR review"


# ────────────────────────────────────────────────────────────── a varredura


async def recon_sweep(payload: dict[str, Any], ctx: Any) -> None:
    """Expira o vencido, monta as queries e faz o fan-out. Não busca nada em si.

    **A ORDEM importa e é testada.** Expirar vem ANTES de checar `enabled`: pôr a
    expiração depois congelaria a fila de pendentes para sempre quando o usuário
    desligasse o recon — e a marca do notify a suprime, então seriam pendências
    invisíveis E imortais.
    """
    focus = ctx.store.active_focus()
    if focus is None:
        log.info("recon_sweep: nenhum foco ativo, nada a fazer")
        return
    focus_id = int(focus["id"])

    gone = expire(ctx.store, days=ctx.config.recon.expire_after_days)
    if gone:
        log.info("%d descoberta(s) expiraram sem decisão", gone)

    if ctx.recon is None:
        log.info("recon_sweep: batedor desabilitado; só a expiração rodou")
        return

    cfg = ctx.config.recon
    room = cfg.pending_limit - _pending_count(ctx.store, focus_id)
    if room <= 0:
        # CONTRAPRESSÃO, idioma de `human_queue_limit`. A varredura seguinte reencontra
        # as mesmas páginas (`freshness` cobre 31 dias) e as propõe quando houver vaga.
        log.info("fila de descobertas cheia (%d pendentes): varredura suprimida",
                 cfg.pending_limit)
        return

    queries = queries_for(ctx.store, focus, ctx.config.focuses_dir,
                          limit=cfg.max_calls_per_sweep)
    if not queries:
        log.info("recon_sweep: nenhuma pergunta automática para consultar")
        return

    queued = 0
    for question_id, query in queries:
        task_id = ctx.queue.enqueue(
            "recon_query",
            # `focus_id` no PAYLOAD, resolvido UMA vez aqui. Sem isto a cadeia inteira
            # (que leva ~9 min de GPU serializada) grava as descobertas sob o foco que
            # estiver ativo na hora da execução — o dano exato que `FocusDrift` existe
            # para recusar.
            {"query": query, "focus_id": focus_id, "question_id": question_id},
            priority=0.32,
            # Balde DIÁRIO por pergunta: sem ele as mesmas 5-6 queries vão para a API
            # paga todo dia, o `UNIQUE(focus_id, url)` recusa todo INSERT, e o
            # rendimento é ZERO com a fatura correndo. Idioma do `hq:` de
            # `harvest_sweep`.
            dedup_key=f"rq:{focus_id}:{question_id}:{_day()}",
            # FORÇADO, e é o idioma de `relens_sweep`, não o de `harvest_sweep`.
            # MEDIDO com o Scheduler real: um tick `scheduled` de `harvest_sweep`
            # produz 19 filhos com `origin='on_demand'`, porque `Job.payload` é None e
            # o handler faz `payload.get('origin','on_demand')` — e `mode off` não
            # alcança nenhum deles. Não existe `lithium cancel`: um recon em fuga
            # queimaria cota PAGA sem forma de parar.
            origin="scheduled",
        )
        queued += task_id is not None
    log.info("recon_sweep: %d busca(s) enfileirada(s) (teto %d/varredura, %d vaga(s) "
             "na fila)", queued, cfg.max_calls_per_sweep, room)


def _day() -> str:
    return budget.today()


def _pending_count(store, focus_id: int) -> int:
    return int(store.conn.execute(
        "SELECT COUNT(*) AS n FROM discoveries "
        " WHERE focus_id = ? AND status = 'pending'", (focus_id,),
    ).fetchone()["n"])


def queries_for(store, focus, focuses_dir, *, limit: int) -> list[tuple[int, str]]:
    """As buscas da varredura, vindas das `questions` do foco.

    **`origin = 'auto'` é uma decisão de PRIVACIDADE.** A query vai para um TERCEIRO (a
    API de busca), e uma pergunta que VOCÊ digitou em `lithium ask` pode carregar
    contexto clínico do caso. Este é um projeto de saúde de uso pessoal.

    **`CONTEXT` e `PREFERENCE` ficam de fora**, e é o oposto de "escaladas primeiro".
    Elas são escaladas na CRIAÇÃO (`HUMAN_ONLY_KINDS`) e o próprio
    `generate_questions.md` as define como "if no amount of literature searching could
    settle it" — buscar por elas é insolúvel POR DEFINIÇÃO, e como só saem por resposta
    humana, as mesmas 5 queries iriam para a API paga todo dia para sempre.
    """
    rows = store.conn.execute(
        "SELECT id, text FROM questions "
        " WHERE focus_id = ? AND origin = 'auto' "
        "   AND status IN ('ESCALATED', 'OPEN') "
        "   AND kind NOT IN ('CONTEXT', 'PREFERENCE') "
        " ORDER BY CASE status WHEN 'ESCALATED' THEN 0 ELSE 1 END, "
        "          priority DESC, id "
        " LIMIT ?",
        (int(focus["id"]), limit),
    ).fetchall()
    queries = [(int(r["id"]), r["text"][:QUERY_MAX_CHARS]) for r in rows]
    if queries:
        return queries
    # Fallback: o alvo do foco. Sem ele um foco recém-criado nunca busca nada e o
    # usuário conclui que a chave não funcionou.
    return [(0, f"{focus['target']} {FALLBACK_QUERY_SUFFIX}"[:QUERY_MAX_CHARS])]


# ─────────────────────────────────────────────────────── a chamada faturada


async def recon_query(payload: dict[str, Any], ctx: Any) -> None:
    """UMA busca faturada. Debita ANTES do GET e nunca estorna."""
    if ctx.recon is None:
        log.info("recon_query: batedor desabilitado")
        return
    _guard(ctx, payload, "recon_query")
    cfg = ctx.config.recon

    if not budget.debit_search(ctx.store, cfg.max_calls_per_day):
        return                                   # operação normal, nunca exceção

    hits = await ctx.recon.searcher.web_search(
        payload["query"], limit=ctx.recon.results_per_query
    )
    if not hits:
        log.info("busca %r não devolveu resultados", payload["query"][:60])
        return

    ctx.queue.enqueue(
        "recon_triage",
        {
            "query": payload["query"],
            "focus_id": payload["focus_id"],
            # Os hits CRUS viajam no payload da tarefa filha e não em `discoveries`
            # porque eles ainda NÃO SÃO descobertas. E é isto que faz o retry da
            # triagem custar zero: a busca já foi paga, o resultado está durável.
            "hits": [
                {"title": h.title, "url": h.url, "description": h.description,
                 "extra_snippets": h.extra_snippets}
                for h in hits
            ],
        },
        priority=0.33,
        origin="scheduled",
    )
    log.info("busca %r: %d resultado(s) para triar", payload["query"][:60], len(hits))


# ────────────────────────────────────────────────────────────────── triagem


async def recon_triage(payload: dict[str, Any], ctx: Any) -> None:
    """UMA chamada de LLM sobre os snippets que a API já devolveu.

    Dois estágios e não "ler tudo": triar 10 snippets numa chamada custa ~42 s e
    substitui 10 leituras que custariam ~9,4 min — ~13x.

    **Nenhuma `observation` é gravada aqui.** Ela só nasce em `recon_read`, DEPOIS dos
    seis portões. Sem isso, uma observação cujo `summary` é o snippet de um motor de
    busca fica indistinguível de uma página lida, e o gesto de consentimento é gasto no
    que o piso de conteúdo existe para impedir.
    """
    _guard(ctx, payload, "recon_triage")
    focus_id = int(payload["focus_id"])
    hits = [WebHit(**h) for h in payload["hits"]]
    query = payload["query"]

    profile = profile_for_slug(ctx.config.focuses_dir, _slug(ctx.store, focus_id))
    block = "\n".join(
        f"  [{i}] {h.title}\n"
        f"      {_domain(h.url)} — {(h.description or '(sem resumo)')[:240]}"
        for i, h in enumerate(hits, 1)
    )
    prompt = render("recon_triage", query=query, results=block,
                    **profile.prompt_blocks("recon_triage"))
    budget_guard(prompt, label="recon_triage", max_tokens=TRIAGE_MAX_TOKENS,
                 n_ctx=ctx.config.llm.n_ctx)
    triage = await ctx.llm.structured(
        [{"role": "user", "content": prompt}], ReconTriage,
        max_tokens=TRIAGE_MAX_TOKENS, label="recon_triage",
    )

    cfg = ctx.config.recon
    room = cfg.pending_limit - _pending_count(ctx.store, focus_id)
    reads, written, dropped = 0, 0, 0
    for verdict in triage.verdicts:
        # Índice fora do conjunto MOSTRADO é descartado — mesmo tratamento que
        # `verify_claim_ids` dá a uma citação inventada.
        if not 1 <= verdict.index <= len(hits):
            dropped += 1
            continue
        hit = hits[verdict.index - 1]
        kind = verdict.kind.strip().lower()
        if kind == "skip":
            continue
        if room <= 0:
            dropped += 1
            continue

        lead = lead_from_hit(hit.url, haystack=hit.haystack)
        if kind == "lead" or (lead is not None and is_indexed_article(hit.url)):
            # Roteia para `lead` mesmo quando o modelo disse `observation`: um artigo
            # indexado é do canal de evidência e não se lê como HTML.
            if lead is None:
                # O modelo disse `lead` e não há identificador extraível do que a API
                # devolveu. DESCARTADO — pedir o número a ele é o defeito que
                # `leads.py` documenta.
                dropped += 1
                continue
            written += _insert(ctx.store, focus_id, "lead", query, hit,
                               summary=hit.description, lead=lead)
            room -= 1
        elif kind == "source":
            written += _insert(ctx.store, focus_id, "source", query, hit,
                               summary=hit.description)
            room -= 1
        elif kind == "observation":
            if reads >= cfg.max_reads_per_query:
                continue
            enqueued = ctx.queue.enqueue(
                "recon_read",
                {"focus_id": focus_id, "query": query,
                 "hit": {"title": hit.title, "url": hit.url,
                         "description": hit.description,
                         "extra_snippets": hit.extra_snippets}},
                priority=0.31,
                dedup_key=f"rr:{focus_id}:{hit.url}:{_day()}",
                origin="scheduled",
            )
            reads += enqueued is not None
        else:
            dropped += 1
    log.info("triagem de %r: %d descoberta(s), %d leitura(s) enfileirada(s), "
             "%d veredito(s) descartado(s)", query[:50], written, reads, dropped)


# ─────────────────────────────────────────────────────────────── a leitura


async def recon_read(payload: dict[str, Any], ctx: Any) -> None:
    """Lê UMA página e grava a `observation` — se ela sobreviver aos seis portões."""
    if ctx.recon is None:
        log.info("recon_read: batedor desabilitado")
        return
    _guard(ctx, payload, "recon_read")
    cfg = ctx.config.recon
    focus_id = int(payload["focus_id"])
    hit = WebHit(**payload["hit"])

    if not budget.debit_page(ctx.store, cfg.max_pages_per_day):
        return

    try:
        page = await ctx.recon.reader.read(
            hit.url, max_chars=read_max_chars(ctx.config.llm.n_ctx,
                                              max_tokens=OBSERVE_MAX_TOKENS)
        )
    except PageRefused as exc:
        # Recusa de portão é operação NORMAL, não falha de tarefa: nenhuma linha é
        # escrita e nenhum token é gasto.
        log.info("leitura recusada: %s", exc)
        return

    profile = profile_for_slug(ctx.config.focuses_dir, _slug(ctx.store, focus_id))
    note = ("(this page was truncated: you are seeing only the beginning)"
            if page.truncated else "")
    prompt = render("recon_observe", url=page.url, title=page.title or hit.title,
                    text=page.text, truncation_note=note,
                    **profile.prompt_blocks("recon_observe"))
    budget_guard(prompt, label="recon_observe", max_tokens=OBSERVE_MAX_TOKENS,
                 n_ctx=ctx.config.llm.n_ctx)
    observation = await ctx.llm.structured(
        [{"role": "user", "content": prompt}], ReconObservation,
        max_tokens=OBSERVE_MAX_TOKENS, label="recon_observe",
    )
    if not observation.worth_reporting or not observation.summary.strip():
        log.info("página %s lida e descartada pelo modelo", page.url)
        return

    summary = observation.summary.strip()
    if observation.why_it_matters.strip():
        summary = f"{summary} — {observation.why_it_matters.strip()}"
    written = _insert(
        ctx.store, focus_id, "observation", payload["query"],
        WebHit(title=page.title or hit.title, url=page.url,
               description=hit.description, extra_snippets=hit.extra_snippets),
        summary=summary,
        extra={"truncated": page.truncated, "http": page.http_status,
               "chars": page.extras.get("chars")},
    )
    log.info("página %s: %d observação gravada", page.url, written)


# ──────────────────────────────────────────────────────────────── auxiliares


def _insert(store, focus_id: int, kind: str, query: str, hit: WebHit, *,
            summary: str, lead: tuple[str, str] | None = None,
            extra: dict | None = None) -> int:
    """Grava uma descoberta. `ON CONFLICT DO NOTHING` sobre o índice PARCIAL.

    Devolve 1 se gravou, 0 se a URL já estava lá (numa linha não expirada).
    """
    payload = {"domain": _domain(hit.url), "snippet": hit.description[:400],
               **(extra or {})}
    row = store.conn.execute(
        "INSERT INTO discoveries(focus_id, kind, query, url, title, summary, "
        "                        lead_kind, lead_external_id, payload_json) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(focus_id, url) WHERE status <> 'expired' DO NOTHING "
        "RETURNING id",
        (focus_id, kind, query[:QUERY_MAX_CHARS], hit.url,
         (hit.title or hit.url)[:TITLE_MAX_CHARS], (summary or "")[:SUMMARY_MAX_CHARS],
         lead[0] if lead else None, lead[1] if lead else None,
         json.dumps(payload)),
    ).fetchone()
    return 1 if row else 0


def _domain(url: str) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").lower()


def _slug(store, focus_id: int) -> str:
    row = store.conn.execute(
        "SELECT slug FROM focuses WHERE id = ?", (focus_id,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"foco #{focus_id} não existe mais")
    return row["slug"]


def _guard(ctx, payload: dict[str, Any], kind: str) -> None:
    """A cadeia inteira executa sob o foco em que foi enfileirada, ou recusa alto.

    Reusa a exceção do canal de evidência de propósito: a janela aqui é MAIOR que a do
    harvest — 6 triagens de ~42 s mais as leituras somam ~9 min de GPU serializada com
    concorrência 1, e o pedido literal desta fase é que o usuário troque de foco quando
    quiser.
    """
    from lithium.worker.handlers import _guard_focus

    active = ctx.store.active_focus()
    _guard_focus(payload, int(active["id"]) if active else 0, kind)


__all__ = [
    "queries_for", "recon_query", "recon_read", "recon_sweep", "recon_triage",
]
