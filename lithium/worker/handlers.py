"""Handlers das tarefas — a cola entre a fila e o pipeline.

Cada handler é pequeno de propósito e enfileira o próximo passo em vez de fazer tudo
em linha. Um harvest que buscasse, baixasse, indexasse e extraísse numa única tarefa
perderia horas de trabalho a cada falha de rede; picado, cada pedaço tem seu próprio
retry e o progresso é durável.

    harvest_sweep ──> harvest_query ──> fetch_source ──> extract_source
    (todas as               (uma            (baixa +        (claims +
     estratégias)            query)          indexa)         verificação)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from lithium.notify import backend_for, deliver
from lithium.notify import watch
from lithium.pipeline.answer import IN_FLIGHT, Answerer
from lithium.pipeline.explore import Explorer
from lithium.pipeline.extract import Extractor
from lithium.pipeline.question import QuestionEngine
from lithium.pipeline.relens import SECONDS_PER_CLAIM, claims_to_judge, judge_one
from lithium.pipeline.ingest import Ingestor
from lithium.pipeline.reflect import Reflector
from lithium.pipeline.retrieval import Retriever
from lithium.focus import active_profile, profile_for_slug
from lithium.pipeline.strategy import DEFAULT_SOURCE, all_search_specs, strategy_by_name
from lithium.recon import handlers as recon_handlers
from lithium.sources.base import SearchSpec, UnsupportedSourceKind
from lithium.types import Directness
from lithium.worker.runner import Context

log = logging.getLogger(__name__)

SEEKING_MAX_CHARS = 90
"""Teto do rótulo de elo que vai ao prompt de reflexão.

`seeking` é string livre de um 12B: se ele escrever um parágrafo em vez de nomear um
elo, o bloco de buscas (24 linhas no teto) estoura a janela. Noventa caracteres cabem
em "sigma-1 agonism -> anxiolysis in humans" com folga."""


def _focus_id(store) -> int:
    focus = store.active_focus()
    return int(focus["id"]) if focus else 0


class FocusDrift(RuntimeError):
    """A tarefa foi enfileirada sob um foco e está executando sob outro.

    RECUSA ALTA em vez de degradar em silêncio. MEDIDO o que a degradação custa: com o
    `focus_id` fora do payload, uma cadeia `harvest -> fetch -> extract` enfileirada sob
    o foco A executa sob o foco B; `strategy_by_name(perfil_B)` não conhece o nome, o
    handler cai no ramo ad-hoc, a prioridade cai de 0.95 para 0.5, `expected_directness`
    vira INDIRECT — e a query executada continua sendo a do foco A. Com 19 queries x até
    20 ids, mais as extrações, a fila leva HORAS para drenar: a janela é rotineira, não
    teórica.
    """


def _stable_key(text: str) -> str:
    """`str.__hash__` é salgado por processo e o repo não fixa `PYTHONHASHSEED`.

    Com `hash()`, o `dedup_key` mudava a cada restart do daemon — então "uma mesma
    query não precisa rodar duas vezes no mesmo dia" era falso, e qualquer contador de
    supressão resetava de forma não-determinística.
    """
    return hashlib.blake2s(text.encode("utf-8"), digest_size=3).hexdigest()


async def harvest_sweep(payload: dict[str, Any], ctx: Context) -> None:
    """Enfileira uma tarefa por (estratégia, query). Não busca nada em si."""
    focus, profile = active_profile(ctx.store, ctx.config.focuses_dir)
    focus_id = int(focus["id"])
    only = set(payload.get("strategies") or [])
    queued = 0
    for strategy, query in all_search_specs(profile):
        if only and strategy.name not in only:
            continue
        enqueued = ctx.queue.enqueue(
            "harvest_query",
            # O `focus_id` viaja no PAYLOAD, resolvido UMA vez aqui. Sem isto a fila
            # durável carrega trabalho do foco A para dentro do foco B — ver `FocusDrift`.
            {"strategy": strategy.name, "query": query, "focus_id": focus_id},
            priority=strategy.priority,
            # Uma mesma query não precisa rodar duas vezes no mesmo dia. ESCOPADA POR
            # FOCO desde a Fase B: a premissa que dispensava o escopo ("STRATEGIES é
            # constante de módulo amarrada ao alvo") acabou de ser revogada, e sem ele
            # dois focos com a mesma string de query colidem no dedup e o segundo nunca
            # roda.
            dedup_key=f"hq:{focus_id}:{strategy.name}:{_stable_key(query)}",
            origin=payload.get("origin", "on_demand"),
        )
        queued += enqueued is not None
    log.info("harvest_sweep enfileirou %d consultas", queued)


async def harvest_query(payload: dict[str, Any], ctx: Context) -> None:
    """Roda uma busca e enfileira o fetch dos ids novos.

    Aceita duas formas: `strategy` (as frentes fixas do plano) ou uma query solta com
    `expected_directness` — que é como as buscas dirigidas por especulação chegam.
    Sem a segunda forma, a trilha exploratória não teria como alimentar a coleta.
    """
    focus, profile = active_profile(ctx.store, ctx.config.focuses_dir)
    _guard_focus(payload, int(focus["id"]), "harvest_query")
    strategy = strategy_by_name(profile).get(payload.get("strategy", ""))
    label = payload.get("label") or (strategy.name if strategy else "ad-hoc")
    directness = (
        strategy.expected_directness if strategy
        else Directness(payload.get("expected_directness", Directness.INDIRECT.value))
    )
    priority = payload.get("priority", strategy.priority if strategy else 0.5)

    # A FONTE VEM DO PAYLOAD. Era a string `"pubmed"` fixa quatro vezes nesta função (a
    # busca, o filtro de novidade, o `kind` do payload e a chave de dedup), e o efeito era
    # que `SourceQuery.source` — campo que o LLM preenche e que atravessa o schema —
    # morria em dois lugares antes de chegar aqui. Registrar um adapter em
    # `Context.sources` não mudava fonte nenhuma: era um botão inerte.
    slug = payload.get("source") or (strategy.sources[0] if strategy
                                     and strategy.sources else DEFAULT_SOURCE)
    source = (ctx.sources or {}).get(slug)
    if source is None:
        raise RuntimeError(
            f"fonte {slug!r} não está em Context.sources — ou não foi aprovada no "
            f"registro (`lithium sources`), ou a credencial não resolve"
        )

    ids = await source.search(
        SearchSpec(
            query=payload["query"],
            expected_directness=directness,
            limit=payload.get("limit", strategy.limit if strategy else 20),
        )
    )

    # Novidade dentro da MESMA fonte: o `kind` na cláusula é o que faz o SQLite usar o
    # índice de `UNIQUE (kind, external_id)` em vez de varrer. Com fontes múltiplas, o
    # mesmo artigo em duas fontes é legitimamente dois `external_id` — quem colapsa a
    # duplicata para efeito de CITAÇÃO é `article_key`, não este filtro.
    known = {
        r["external_id"]
        for r in ctx.store.conn.execute(
            "SELECT external_id FROM sources WHERE kind = ?", (slug,)
        )
    }
    fresh = [i for i in ids if i not in known]

    for external_id in fresh:
        ctx.queue.enqueue(
            "fetch_source",
            {
                "kind": slug,
                "external_id": external_id,
                "expected_directness": directness.value,
                "strategy": label,
                "focus_id": int(focus["id"]),
            },
            priority=priority,
            dedup_key=f"fetch:{slug}:{external_id}",
        )
    log.info("[%s/%s] %s: %d resultados, %d novos",
             label, slug, payload["query"][:55], len(ids), len(fresh))


async def fetch_source(payload: dict[str, Any], ctx: Context) -> None:
    """Baixa um registro, fatia em chunks e indexa. Depois pede a extração."""
    kind = payload["kind"]
    if not ctx.store.source_yields_evidence(kind):
        # Portão de TIPO, no lugar que ingere — e agora ele CONSULTA o registro em vez de
        # comparar contra uma frozenset compilada. A pergunta não mudou ("isto pode virar
        # claim?"); mudou quem responde. Uma bula não é desenho de estudo, e a prosa de um
        # registro de ensaio passa o portão de citação literal porque a citação É literal
        # — ela só não é resultado. `yields_evidence` é onde esse contrato mora agora.
        #
        # Fail-closed cobre três estados com a mesma resposta: fonte desconhecida, fonte
        # conhecida e não aprovada, e fonte aprovada que não produz evidência.
        permitidas = [r["slug"] for r in ctx.store.active_sources()
                      if r["yields_evidence"]]
        raise UnsupportedSourceKind(
            f"{kind!r} não produz evidência de estudo neste banco; "
            f"fontes de evidência aprovadas: {permitidas or '(nenhuma)'}"
        )
    _guard_focus(payload, _focus_id(ctx.store), "fetch_source")
    source = (ctx.sources or {}).get(kind)
    if source is None:
        raise RuntimeError(f"fonte '{payload['kind']}' não configurada")

    records = await source.fetch([payload["external_id"]])
    if not records:
        # Sem abstract, ou id inexistente. Repetir não conserta.
        log.info("%s sem conteúdo aproveitável, ignorando", payload["external_id"])
        return

    ingestor = Ingestor(ctx.store, ctx.embedder)
    for record in records:
        # O prior da estratégia entra como rótulo; a extração decide o valor final.
        record.population_tag = payload.get("expected_directness", Directness.INDIRECT.value)
        result = await ingestor.ingest(record)
        ctx.queue.enqueue(
            "extract_source",
            {"source_id": result.source_id, "focus_id": _focus_id(ctx.store)},
            priority=payload.get("priority", 0.5),
            # Escopada por foco. Com a chave global, uma fonte já extraída pelo foco A
            # nunca é reextraída, então o foco B nunca ganha `claim_directness` para
            # aquelas claims e todas elas valem zero nele, para sempre, sem log.
            dedup_key=f"extract:{_focus_id(ctx.store)}:{result.source_id}",
        )


async def extract_source(payload: dict[str, Any], ctx: Context) -> None:
    """Extrai claims, com os dois portões de verificação."""
    focus, profile = active_profile(ctx.store, ctx.config.focuses_dir)
    # A janela é de MINUTOS: `fetch_source` calcula o dedup no ENQUEUE e a extração
    # resolve o foco na EXECUÇÃO. Trocar de foco entre os dois gravava as claims sob o
    # foco novo e deixava o de origem permanentemente sem aresta, sem log e sem refetch
    # possível (a fonte já é `known`).
    _guard_focus(payload, int(focus["id"]), "extract_source")
    result = await Extractor(
        ctx.store, ctx.llm, profile=profile, n_ctx=ctx.config.llm.n_ctx
    ).extract_source(payload["source_id"])
    # PERSISTE, e só depois loga. O log continua para quem está olhando o terminal; o
    # que mudou é que ele deixou de ser o único registro. O `[:5]` de antes descartava a
    # sexta rejeição em diante sem dizer, e o handler de log não escreve em arquivo.
    ctx.store.record_extraction(result, focus_id=int(focus["id"]))
    log.info(
        "fonte %s: %d propostas, %d ancoradas (%.0f%%), %d verificadas "
        "[chunks: %d aniquilados, %d estéreis]",
        payload["source_id"], result.proposed, result.anchored,
        result.anchor_rate * 100, result.verified,
        result.chunks_annihilated, result.chunks_sterile,
    )


async def plan_tick(payload: dict[str, Any], ctx: Context) -> None:
    """Olha o estado do conhecimento e formula o que investigar em seguida."""
    engine = QuestionEngine(
        ctx.store, ctx.llm, ctx.embedder,
        profile=active_profile(ctx.store, ctx.config.focuses_dir)[1],
        dedup_threshold=ctx.config.question.dedup_threshold,
        human_queue_limit=ctx.config.question.human_queue_limit,
        n_ctx=ctx.config.llm.n_ctx,
    )
    added = await engine.generate(max_questions=payload.get("max_questions", 5))
    for q in added:
        log.info("  [%s prio=%.2f] %s", q.kind.value, q.priority, q.text)


async def ask_question(payload: dict[str, Any], ctx: Context) -> None:
    """Pergunta manual vinda do `lithium ask`, classificada de forma assíncrona."""
    engine = QuestionEngine(
        ctx.store, ctx.llm, ctx.embedder,
        profile=active_profile(ctx.store, ctx.config.focuses_dir)[1],
        dedup_threshold=ctx.config.question.dedup_threshold,
        human_queue_limit=ctx.config.question.human_queue_limit,
        n_ctx=ctx.config.llm.n_ctx,
    )
    record = await engine.ask(payload["text"])
    if record is None:
        log.info("pergunta já existia no banco, ignorada")
    else:
        log.info("pergunta #%d classificada como %s", record.id, record.kind.value)


async def explore_tick(payload: dict[str, Any], ctx: Context) -> None:
    """Trilha exploratória: hipóteses mecanísticas, criticadas antes de entrar."""
    explorer = Explorer(
        ctx.store, ctx.llm,
        Reflector(ctx.store, ctx.llm, ctx.embedder,
                  profile=active_profile(ctx.store, ctx.config.focuses_dir)[1],
                  n_ctx=ctx.config.llm.n_ctx),
        profile=active_profile(ctx.store, ctx.config.focuses_dir)[1],
        n_ctx=ctx.config.llm.n_ctx,
    )
    records = await explorer.generate(max_items=payload.get("max_items", 3))
    for r in records:
        mark = "ok" if r.survives else f"REPROVADA ({r.fatal_flaw[:60]})"
        log.info("  [%s] plaus=%.0f%% ined=%.0f%% %s",
                 mark, r.plausibility * 100, r.novelty * 100, r.statement[:80])


async def pursue_speculation(payload: dict[str, Any], ctx: Context) -> None:
    """Converte hipóteses da trilha exploratória em buscas dirigidas.

    É o passo que fecha o laço. As 19 queries fixas do `harvest_sweep` não mencionam
    sigma-1, orexina, via transdérmica nem dispositivo — sem isto o sistema propõe um
    alvo e nunca pergunta a nenhuma base sobre ele.
    """
    explorer = Explorer(ctx.store, ctx.llm,
                        profile=active_profile(ctx.store, ctx.config.focuses_dir)[1])
    ids = payload.get("ids") or explorer.pending_pursuit(payload.get("limit", 2))
    if not ids:
        log.info("nenhuma especulação pendente de busca")
        return

    for hypothesis_id in ids:
        queries = await explorer.plan_queries(hypothesis_id)
        for q in queries:
            ctx.queue.enqueue(
                "harvest_query",
                {
                    "query": q.query,
                    # A FONTE QUE O MODELO ESCOLHEU. Ela atravessava o schema
                    # (`SourceQuery.source`), sobrevivia ao filtro de `plan_queries`, e
                    # morria AQUI: o payload era montado sem ela e `harvest_query` fixava
                    # `'pubmed'`. Duas mortes silenciosas em sequência, e o efeito é que
                    # "o modelo escolhe onde buscar" era verdade como estrutura de dados e
                    # falso como comportamento.
                    "source": q.source,
                    # Especulação parte de mecanismo, não de população: o prior
                    # honesto é `extrapolated`, e a extração corrige lendo o texto.
                    "expected_directness": Directness.EXTRAPOLATED.value,
                    "label": f"spec:{hypothesis_id}",
                    # O elo da cadeia que esta busca tenta ancorar. O modelo já o
                    # nomeava e o sistema descartava aqui — e é o que transforma a
                    # lição de "esta query não retornou nada" em "queries visando
                    # sigma-1 → ansiólise em humanos não retornam nada". A primeira é
                    # ruído; a segunda diz onde a cadeia está sem chão.
                    "seeking": (q.seeking or "").strip()[:SEEKING_MAX_CHARS] or None,
                    "limit": 15,
                },
                priority=0.55,
                # Escopado por foco. `hypothesis_id` só escopa transitivamente
                # enquanto a hipótese não puder ser sustentada por dois focos — e a
                # Fase B revoga essa premissa.
                # A fonte entra na chave: a MESMA query em duas fontes são duas buscas
                # legítimas, e sem isto a segunda seria engolida em silêncio pelo UNIQUE.
                dedup_key=f"specq:{_focus_id(ctx.store)}:{hypothesis_id}:{q.source}:"
                          f"{_stable_key(q.query)}",
                origin=payload.get("origin", "on_demand"),
            )
        explorer.mark_pursued(hypothesis_id)
        log.info("hipótese #%d: %d busca(s) dirigida(s) enfileirada(s)",
                 hypothesis_id, len(queries))


async def reground_speculations(payload: dict[str, Any], ctx: Context) -> None:
    """Reavalia cadeias contra o corpus atual — é onde a plausibilidade sobe."""
    explorer = Explorer(ctx.store, ctx.llm,
                        profile=active_profile(ctx.store, ctx.config.focuses_dir)[1])
    retriever = Retriever(ctx.store, ctx.embedder)
    ids = payload.get("ids") or explorer.pending_reground(payload.get("limit", 3))
    total = 0
    for hypothesis_id in ids:
        total += await explorer.reground(hypothesis_id, retriever)
    log.info("reancoragem: %d elo(s) ancorado(s) em %d hipótese(s)", total, len(ids))


async def reflect_tick(payload: dict[str, Any], ctx: Context) -> None:
    """O sistema aprendendo sobre o próprio trabalho.

    Grava sem confirmação, por decisão do usuário — pedir permissão para "aprendi que
    termos MeSH funcionam melhor" seria fricção que treinaria o usuário a clicar sem
    ler, e aí a confirmação que importa (fato sobre a pessoa) perderia valor.
    """
    reflector = Reflector(ctx.store, ctx.llm, ctx.embedder,
                          profile=active_profile(ctx.store, ctx.config.focuses_dir)[1],
                          n_ctx=ctx.config.llm.n_ctx)
    lessons = await reflector.reflect(max_lessons=payload.get("max_lessons", 3))
    for lesson in lessons:
        log.info("  [%s] %s", lesson.kind, lesson.text)
    # AO FIM, não ao início: avançar a marca antes de as lições serem escritas fecharia
    # o portão de `pattern` para a própria passada que a colheita nova habilitou.
    reflector.mark_literature_seen()


async def answer_tick(payload: dict[str, Any], ctx: Context) -> None:
    """O despachante da fila de perguntas. É o produtor do item 7.

    Ele existe porque o loop sem despachante é a peça que este projeto recusou três
    vezes: `next_for_research` estava no código desde o dia 1 com zero chamadores, e a
    fila ficava em `OPEN` para sempre. A ordem aqui não é arbitrária — reconciliar antes
    de despachar devolve à fila o que ficou preso, e podar antes evita gastar rodada
    numa pergunta que o teto vai fechar.
    """
    answerer = Answerer(ctx.store, ctx.llm, ctx.embedder,
                        profile=active_profile(ctx.store, ctx.config.focuses_dir)[1],
                        n_ctx=ctx.config.llm.n_ctx)
    answerer.release_stranded()
    answerer.park_overflow()

    ids = answerer.dispatchable(payload.get("in_flight", IN_FLIGHT))
    for question_id in ids:
        ctx.queue.enqueue(
            "answer_question",
            {"question_id": question_id},
            priority=0.6,
            origin=payload.get("origin", "on_demand"),
        )
    log.info("answer_tick despachou %d pergunta(s)", len(ids))


async def answer_question(payload: dict[str, Any], ctx: Context) -> None:
    """Uma rodada de pesquisa sobre uma pergunta."""
    answerer = Answerer(ctx.store, ctx.llm, ctx.embedder,
                        profile=active_profile(ctx.store, ctx.config.focuses_dir)[1],
                        n_ctx=ctx.config.llm.n_ctx)
    result = await answerer.round(payload["question_id"])
    log.info("pergunta #%s: %s%s", payload["question_id"], result.action.value,
             f" ({result.reason})" if result.reason else "")


async def notify_tick(payload: dict[str, Any], ctx: Context) -> None:
    """Avisa o que mudou desde o último tick. Um aviso agrupado, nunca um por item.

    **Este handler é estruturalmente incapaz de morrer**, e não é zelo: o delta de
    dead-letter exclui `notify_tick` justamente para o aviso não se auto-alimentar, e
    essa exclusão só é segura se ele nunca virar dead-letter. Sem o `except`, qualquer
    erro de banco ou `meta` corrompido produziria a morte silenciosa do canal inteiro —
    o sistema pararia de avisar e nada avisaria sobre isso.
    """
    try:
        await _notify_tick(ctx)
    except Exception:  # noqa: BLE001
        log.exception("notify_tick falhou; o canal de aviso segue vivo")


async def _notify_tick(ctx: Context) -> None:
    """Percorre `watch.DELTAS`, nunca uma lista literal de categorias.

    Com as categorias escritas à mão aqui, acrescentar a terceira significa lembrar de
    editar quatro lugares — e MEDIDO que esquecer um deles mata ZERO testes. O mapa é a
    única fonte da iteração.
    """
    import sys

    watch.seed_global_mark(ctx.store)
    marks = {scope: watch._mark(ctx.store, scope) for scope in watch.scopes()}
    fresh: dict = {}
    currents: dict[str, dict] = {scope: {} for scope in marks}
    for key, (scope, delta) in watch.DELTAS.items():
        seen = marks[scope].get(key, watch.EMPTY[key])
        fresh[key], currents[scope][key] = delta(ctx.store, seen)

    notice = watch.compose(fresh)
    if notice is None:
        return

    backend = backend_for(sys.platform, ntfy_url=ctx.config.notify.ntfy_url)
    delivered = await deliver(backend, notice)
    for scope, current in currents.items():
        watch._save(
            ctx.store,
            watch.reconcile(marks[scope], current, delivered=delivered),
            scope,
        )
    log.info("aviso %s: %s", "entregue" if delivered else "não entregue", notice.body)


async def purge_tasks(payload: dict[str, Any], ctx: Context) -> None:
    removed = ctx.queue.purge_done(older_than_days=payload.get("days", 7))
    if removed:
        log.info("purga removeu %d tarefas concluídas", removed)


async def relens_sweep(payload: dict[str, Any], ctx: Context) -> None:
    """Fan-out: uma tarefa de julgamento POR CLAIM, para o foco resolvido AQUI.

    Quatro medições forçam cada escolha, e nenhuma delas é estilo.

    **`origin='scheduled'`, não o `on_demand` que todo comando de CLI usa por padrão.**
    MEDIDO: `lithium mode off` NÃO para tarefas `on_demand` — 5 de 5 continuam sendo
    reivindicadas — e `scheduled` é o ÚNICO cancelamento que existe, porque não há
    `lithium cancel`. Sem isso, um relens de 7,68 h vira ininterruptível: matar o daemon
    deixa as N-1 pendentes para voltarem no restart.

    **Fan-out por claim, não uma tarefa longa.** A PK `(claim_id, focus_id)` faz o
    retomar ser grátis; com UMA tarefa de 7,68 h, uma queda no claim 4.700 dispara retry
    do zero.

    **Prioridade 0,3, abaixo de 0,35.** Acima disso mataria de fome `reflect` (0,35),
    `harvest` (0,4) e `reground` (0,45) por horas, com concorrência real de 1 no LLM.

    **A `dedup_key` é escopada por foco E limpa antes do fan-out.** `tasks.dedup_key` é
    `TEXT UNIQUE` sobre a tabela INTEIRA e `enqueue` deduplica contra `done` e `dead`
    também; `purge_done()` só apaga `done`. Uma noite de llama-server fora do ar deixa
    centenas de tarefas em `dead` com a chave QUEIMADA, e o relens seguinte devolve
    None em todas — reportando "N enfileiradas" enquanto nada roda, para sempre.
    """
    focus, profile = active_profile(ctx.store, ctx.config.focuses_dir)
    focus_id, scale_id = int(focus["id"]), focus["scale_id"]

    # Chaves de tentativas ANTERIORES deste mesmo foco. Só `done`/`dead`: apagar uma
    # `pending` ou `running` duplicaria trabalho em voo.
    ctx.store.conn.execute(
        "DELETE FROM tasks WHERE kind = 'relens_claim' "
        "   AND status IN ('done', 'dead') AND dedup_key LIKE ?",
        (f"relens:{focus_id}:%",),
    )

    claim_ids = claims_to_judge(ctx.store, focus_id, scale_id)
    queued = 0
    for claim_id in claim_ids:
        task_id = ctx.queue.enqueue(
            "relens_claim",
            {"claim_id": claim_id, "focus_id": focus_id, "target": focus["target"]},
            priority=0.3,
            dedup_key=f"relens:{focus_id}:{claim_id}",
            origin="scheduled",
        )
        # Conta o RETORNO, nunca o SELECT: o número que o usuário lê tem de ser o que
        # foi de fato enfileirado, senão "505 enfileiradas / nada roda" fica invisível.
        queued += task_id is not None
    log.info(
        "relens do foco %s: %d claim(s) sem julgamento, %d enfileirada(s) "
        "(~%.1f h de GPU)",
        focus["slug"], len(claim_ids), queued, queued * SECONDS_PER_CLAIM / 3600,
    )


async def relens_claim(payload: dict[str, Any], ctx: Context) -> None:
    """Julga UMA claim contra o foco do PAYLOAD.

    TUDO vem do payload ou é derivado dele — `target` e o perfil inclusive. Ler
    `active_focus()` aqui faria metade do lote ser julgada contra um foco e metade
    contra outro, gravando as duas metades sob o mesmo `focus_id`, sem log e sem nada
    em `judged_at` que permitisse reconstruir onde foi o corte. As janelas são reais:
    um relens de 4.800 claims leva 7,68 h e o pedido literal desta fase é que o usuário
    troque de foco quando quiser.
    """
    focus_id = int(payload["focus_id"])
    row = ctx.store.conn.execute(
        "SELECT slug, target FROM focuses WHERE id = ?", (focus_id,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"foco #{focus_id} não existe mais; nada a julgar")
    profile = profile_for_slug(ctx.config.focuses_dir, row["slug"])

    verdict = await judge_one(
        ctx.store, ctx.llm, profile,
        claim_id=int(payload["claim_id"]),
        focus_id=focus_id,
        # `target` do BANCO, relido pelo id do payload: é contra ele que toda aresta
        # deste foco foi julgada, e é o banco que vence nessa fronteira.
        target=row["target"],
        n_ctx=ctx.config.llm.n_ctx,
    )
    if verdict is None:
        return
    if not verdict.judgeable:
        log.info("claim %s: não julgável, nenhuma aresta gravada", payload["claim_id"])
    elif not verdict.in_scope:
        log.info("claim %s: fora de escopo em %s", payload["claim_id"], row["slug"])


async def recon_lead(payload: dict[str, Any], ctx: Context) -> None:
    """A PONTE. Um identificador atravessa; texto nenhum atravessa.

    Ela mora AQUI, e não em `lithium/recon/`, de propósito: é código do canal de
    EVIDÊNCIA, e o teste de AST daquele pacote proíbe justamente os nomes que esta
    função precisa (`sources`, `fetch_source`). Pôr a ponte dentro do batedor obrigaria
    a afrouxar a varredura que protege o batedor inteiro.

    **Esta função nunca lê `title`, `summary`, `url` nem `payload_json`.** O SELECT
    abaixo enumera QUATRO colunas, e `test_the_bridge_cannot_name_a_prose_column`
    reprova por AST quem acrescentar uma quinta. É o que impede o atalho "já li a
    página, por que buscar de novo?" — que passaria pelos CHECKs do banco e pelo teste
    de AST do pacote, porque nenhum dos dois olha para cá.

    **DOI resolve com o campo QUALIFICADO, `<doi>[doi]`.** Sem ele, `esearch` DESCARTA
    os tokens que não encontra: `term='10.1001/jamapsychiatry.2019.0035'` (DOI bem
    formado e inexistente) devolve `querytranslation: "10.1001"[All Fields]` e DOIS
    PMIDs reais e sem relação nenhuma — que seriam ingeridos como se você os tivesse
    aprovado. Com `[doi]`, um DOI inexistente devolve zero. E exigimos EXATAMENTE um
    resultado: mais de um significa que a resolução não é identidade.
    """
    row = ctx.store.conn.execute(
        "SELECT id, focus_id, lead_kind, lead_external_id FROM discoveries "
        " WHERE id = ?", (payload["discovery_id"],),
    ).fetchone()
    if row is None:
        log.info("descoberta #%s sumiu; nada a colher", payload["discovery_id"])
        return
    _guard_focus(payload, _focus_id(ctx.store), "recon_lead")
    focus_id = int(row["focus_id"])
    external_id = row["lead_external_id"]

    if row["lead_kind"] == "doi":
        external_id = await _pmid_for_doi(ctx, external_id)
        if external_id is None:
            ctx.store.conn.execute(
                "UPDATE discoveries SET status = 'deferred' WHERE id = ?", (row["id"],)
            )
            log.info("DOI %s não resolveu para exatamente um PMID; descoberta #%s "
                     "marcada como deferred", row["lead_external_id"], row["id"])
            return

    known = ctx.store.conn.execute(
        "SELECT id FROM sources WHERE kind = 'pubmed' AND external_id = ?",
        (external_id,),
    ).fetchone()
    if known is not None:
        # O artigo já foi colhido por outro foco. `fetch:pubmed:{id}` é dedup GLOBAL e
        # `enqueue` deduplica contra `done` e `dead` também, então reenfileirar o fetch
        # devolveria None: o CLI diria "✓ enfileirado", nada rodaria, e o artigo nunca
        # ganharia `claim_directness` NESTE foco — valeria zero nele para sempre.
        ctx.queue.enqueue(
            "extract_source",
            {"source_id": int(known["id"]), "focus_id": focus_id},
            priority=0.75,
            dedup_key=f"extract:{focus_id}:{known['id']}",
        )
        log.info("artigo %s já estava no corpus; extração pedida para o foco #%d",
                 external_id, focus_id)
        return

    ctx.queue.enqueue(
        "fetch_source",
        # SÓ identificador e metadados do canal. Nenhum campo da descoberta.
        {"kind": "pubmed", "external_id": external_id,
         "expected_directness": Directness.INDIRECT.value,
         "strategy": "recon", "focus_id": focus_id, "priority": 0.75},
        priority=0.75,
        dedup_key=f"fetch:pubmed:{external_id}",
    )
    log.info("artigo %s enfileirado a partir da descoberta #%s",
             external_id, row["id"])


async def _pmid_for_doi(ctx: Context, doi: str) -> str | None:
    source = (ctx.sources or {}).get("pubmed")
    if source is None:
        raise RuntimeError("fonte 'pubmed' não configurada no contexto")
    ids = await source.search(SearchSpec(query=f"{doi}[doi]", limit=2))
    return ids[0] if len(ids) == 1 else None


HANDLERS = {
    "relens_sweep": relens_sweep,
    "relens_claim": relens_claim,
    "recon_lead": recon_lead,
    "harvest_sweep": harvest_sweep,
    "harvest_query": harvest_query,
    "fetch_source": fetch_source,
    "extract_source": extract_source,
    "plan_tick": plan_tick,
    "ask_question": ask_question,
    "explore_tick": explore_tick,
    "pursue_speculation": pursue_speculation,
    "reground_speculations": reground_speculations,
    "reflect_tick": reflect_tick,
    "answer_tick": answer_tick,
    "answer_question": answer_question,
    "notify_tick": notify_tick,
    "purge_tasks": purge_tasks,
    # O batedor. Os quatro moram em `lithium/recon/handlers.py` para que a varredura de
    # AST que impede a web de virar claim os ALCANCE — o bypass mais barato é
    # acrescentar dois nomes a um import existente NESTE arquivo, que nenhuma camada
    # do portão inspeciona.
    "recon_sweep": recon_handlers.recon_sweep,
    "recon_query": recon_handlers.recon_query,
    "recon_triage": recon_handlers.recon_triage,
    "recon_read": recon_handlers.recon_read,
}


def _guard_focus(payload: dict[str, Any], active_id: int, kind: str) -> None:
    """A tarefa só executa sob o foco em que foi enfileirada.

    Payload sem `focus_id` é tarefa ANTIGA (enfileirada antes desta fase) e passa: um
    upgrade não pode transformar a fila pendente em dead-letter. Payload COM `focus_id`
    divergente é recusa alta — degradar em silêncio é como os resultados da busca de um
    foco entravam no corpus julgados contra o alvo de outro.
    """
    declared = payload.get("focus_id")
    if declared is not None and int(declared) != active_id:
        raise FocusDrift(
            f"{kind} foi enfileirada sob o foco #{declared} e o foco ativo é "
            f"#{active_id}. Recusando: executá-la agora colheria com a query de um "
            f"foco e julgaria o resultado contra o alvo do outro. Reative o foco "
            f"#{declared} ou deixe a tarefa morrer."
        )
