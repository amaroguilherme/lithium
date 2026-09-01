"""O fan-out do batedor: quem enfileira o quê, sob qual foco, com qual origem.

Aqui mora a classe de defeito que este repo já contou seis vezes — fiação não testada.
Cada teste roda o handler REAL pelo `Runner` real e lê o BANCO.
"""

from __future__ import annotations

import json

import httpx
import pytest

from lithium.mode import ResearchMode, set_mode
from lithium.worker.handlers import FocusDrift, HANDLERS

from reconkit import (
    FakeSearcher,
    TriagingLLM,
    make_ctx,
    runner,
    seed_discovery,
)


def _question(store, text, *, origin="auto", kind="FACTUAL", status="OPEN",
              focus_id=1, priority=0.5) -> int:
    cur = store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status, origin, priority) "
        "VALUES(?, ?, ?, ?, ?, ?) RETURNING id",
        (focus_id, text, kind, status, origin, priority))
    return int(cur.fetchone()["id"])


def _tasks(store, kind: str) -> list[dict]:
    return [
        {"origin": r["origin"], "status": r["status"], **json.loads(r["payload_json"])}
        for r in store.conn.execute(
            "SELECT origin, status, payload_json FROM tasks WHERE kind = ? ORDER BY id",
            (kind,))
    ]


# ═══════════════════════════════ `mode off` para o batedor de verdade


async def test_the_recon_fan_out_is_scheduled_so_mode_off_actually_stops_it(tmp_path):
    """MUTAÇÃO: trocar o `origin='scheduled'` literal por
    `payload.get('origin', 'on_demand')` — o padrão de `harvest_sweep`.

    MEDIDO com o Scheduler real: um tick `scheduled` de `harvest_sweep` produz 19
    filhos com `origin='on_demand'`, porque `Job.payload` é None. `lithium mode off` não
    para nenhum deles. Não existe `lithium cancel`: um recon em fuga queimaria cota PAGA
    sem forma de parar, e `recover_orphans()` devolve as `running` para a fila no
    restart.
    """
    ctx = make_ctx(tmp_path)
    _question(ctx.store, "o que há de novo em manutenção?")
    ctx.queue.enqueue("recon_sweep", {}, origin="scheduled")
    await runner(ctx).drain(max_tasks=1)

    children = _tasks(ctx.store, "recon_query")
    assert children, "o sweep não enfileirou nada"
    assert {c["origin"] for c in children} == {"scheduled"}, children

    set_mode(ctx.store, ResearchMode.OFF)
    assert ctx.queue.claim(on_demand_only=True) is None, (
        "`mode off` não parou o fan-out do recon"
    )
    ctx.store.close()


async def test_recon_now_refuses_while_the_mode_is_off(tmp_path, monkeypatch):
    """`lithium recon --now` RECUSA em vez de forçar `on_demand`.

    `mode off` é o gesto de "devolva a máquina", e o recon é a única coisa deste sistema
    cuja retomada é IRREVERSÍVEL: custa dinheiro.
    """
    import typer
    from typer.testing import CliRunner

    from lithium import cli

    ctx = make_ctx(tmp_path)
    set_mode(ctx.store, ResearchMode.OFF)
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: ctx.config)
    monkeypatch.setattr(cli, "_store", lambda cfg: ctx.store)

    result = CliRunner().invoke(cli.app, ["recon", "--now"])
    assert result.exit_code == 1
    assert "GASTA DINHEIRO" in result.output
    assert ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE kind = 'recon_sweep'"
    ).fetchone()["n"] == 0
    ctx.store.close()
    _ = typer


# ══════════════════════════════════════════════ privacidade das queries


async def test_recon_never_searches_for_a_question_you_typed(tmp_path):
    """MUTAÇÃO: tirar o filtro `origin = 'auto'`.

    A query de recon vai para um TERCEIRO (a API de busca). Uma pergunta que VOCÊ
    digitou em `lithium ask` pode carregar contexto clínico do caso, e este é um projeto
    de saúde de uso pessoal. Decisão de privacidade acrescentada ao escopo.
    """
    ctx = make_ctx(tmp_path)
    _question(ctx.store, "meu paciente Zmyrfkq tomou 1200 mg e teve tremor",
              origin="human")
    _question(ctx.store, "lítio em manutenção de bipolar I", origin="auto")
    ctx.queue.enqueue("recon_sweep", {})
    await runner(ctx).drain(max_tasks=1)

    queries = " ".join(c["query"] for c in _tasks(ctx.store, "recon_query"))
    assert queries, "nenhuma query foi montada"
    assert "Zmyrfkq" not in queries
    ctx.store.close()


async def test_recon_skips_the_question_kinds_no_search_can_settle(tmp_path):
    """`CONTEXT` e `PREFERENCE` são `HUMAN_ONLY_KINDS`: escaladas na CRIAÇÃO, e o
    próprio `generate_questions.md` as define como "if no amount of literature searching
    could settle it".

    Sem este filtro, "escaladas primeiro" escolhe justamente as 5 perguntas que NUNCA
    saem da fila, as mesmas 5 queries vão para a API paga todo dia, o
    `UNIQUE(focus_id,url)` recusa todo INSERT com `freshness` de 31 dias, e o rendimento
    é ZERO com a fatura correndo.
    """
    ctx = make_ctx(tmp_path)
    _question(ctx.store, "o que você já tentou?", kind="CONTEXT", status="ESCALATED")
    _question(ctx.store, "você prefere evitar sedação?", kind="PREFERENCE",
              status="ESCALATED")
    _question(ctx.store, "quetiapina vs lítio em manutenção", kind="FACTUAL")
    ctx.queue.enqueue("recon_sweep", {})
    await runner(ctx).drain(max_tasks=1)

    queries = [c["query"] for c in _tasks(ctx.store, "recon_query")]
    assert queries == ["quetiapina vs lítio em manutenção"], queries
    ctx.store.close()


async def test_the_same_question_is_not_re_searched_on_the_same_day(tmp_path):
    """Balde DIÁRIO por pergunta, idioma do `hq:` de `harvest_sweep`. Sem ele, duas
    varreduras no mesmo dia dobram a fatura por zero descoberta nova."""
    searcher = FakeSearcher()
    ctx = make_ctx(tmp_path, searcher=searcher)
    _question(ctx.store, "lítio em manutenção")
    for _ in range(2):
        ctx.queue.enqueue("recon_sweep", {})
        await runner(ctx).drain()
    assert len(searcher.calls) == 1, searcher.calls
    ctx.store.close()


async def test_a_focus_with_no_automatic_question_falls_back_to_the_target(tmp_path):
    """Sem fallback, um foco recém-criado nunca busca nada e o usuário conclui que a
    chave não funcionou."""
    ctx = make_ctx(tmp_path)
    ctx.queue.enqueue("recon_sweep", {})
    await runner(ctx).drain(max_tasks=1)
    queries = [c["query"] for c in _tasks(ctx.store, "recon_query")]
    assert len(queries) == 1 and "bipolar" in queries[0].lower()
    ctx.store.close()


# ═════════════════════════════════════════════════ o foco viaja no payload


async def test_the_whole_chain_refuses_to_run_under_another_focus(tmp_path):
    """MUTAÇÃO: remover `focus_id` do payload do fan-out.

    A janela é MAIOR que a do harvest: 6 triagens de ~42 s mais as leituras somam ~9 min
    de GPU serializada com concorrência 1. Sem o `focus_id` no payload, `recon_triage`
    resolveria `active_focus()` na EXECUÇÃO e gravaria descobertas sobre bipolar com
    `focus_id = onco-x` — o dano exato que o comentário do DDL diz querer evitar.
    """
    ctx = make_ctx(tmp_path)
    scale_id = int(ctx.store.conn.execute(
        "SELECT id FROM evidence_scales LIMIT 1").fetchone()["id"])
    ctx.store.conn.execute(
        "INSERT INTO focuses(id, slug, target, scale_id) VALUES(2, 'outro', 'x', ?)",
        (scale_id,))
    _question(ctx.store, "lítio em manutenção")
    ctx.queue.enqueue("recon_sweep", {})
    await runner(ctx).drain(max_tasks=1)

    ctx.store.conn.execute("UPDATE meta SET value = '2' WHERE key = 'active_focus'")
    with pytest.raises(FocusDrift):
        await HANDLERS["recon_query"]({"query": "q", "focus_id": 1}, ctx)


async def test_the_discovery_is_written_under_the_focus_of_the_payload(tmp_path):
    ctx = make_ctx(tmp_path)
    _question(ctx.store, "lítio em manutenção")
    ctx.queue.enqueue("recon_sweep", {})
    await runner(ctx).drain()
    rows = ctx.store.conn.execute(
        "SELECT DISTINCT focus_id FROM discoveries").fetchall()
    assert [r["focus_id"] for r in rows] == [1]
    ctx.store.close()


# ═════════════════════════════════ a triagem não inventa e não enche a fila


async def test_the_url_of_a_discovery_comes_from_the_api_never_from_the_model(tmp_path):
    """MUTAÇÃO: acrescentar `url: str` ao schema `ReconVerdict` e usá-lo.

    Um 12B a quem se pede uma URL inventa URL, e uma descoberta com URL alucinada é uma
    página que você abre e não existe — ou pior, existe e é outra coisa. O modelo
    referencia por ÍNDICE LOCAL, e um índice fora do conjunto mostrado é DESCARTADO,
    como `verify_claim_ids` faz.
    """
    from lithium.llm.schemas import ReconVerdict

    assert set(ReconVerdict.model_fields) == {"index", "kind"}, (
        "o veredito ganhou um campo que o modelo pode inventar"
    )

    llm = TriagingLLM(verdicts=[
        {"index": 1, "kind": "source"},
        {"index": 99, "kind": "source"},      # fora do conjunto mostrado
        {"index": 0, "kind": "source"},       # idem
    ])
    ctx = make_ctx(tmp_path, llm=llm)
    ctx.queue.enqueue("recon_query", {"query": "q", "focus_id": 1})
    await runner(ctx).drain()

    urls = [r["url"] for r in ctx.store.conn.execute("SELECT url FROM discoveries")]
    assert urls == ["https://www.nice.org.uk/guidance/cg185"], urls
    ctx.store.close()


async def test_the_lead_identifier_is_derived_from_the_hit_not_emitted(tmp_path):
    """O defeito FATAL que a refutação achou, fechado por construção.

    O modelo diz o TIPO; o identificador sai de `re.search` sobre a URL da API. Um hit
    marcado `lead` sem identificador extraível é DESCARTADO — o mesmo tratamento de um
    índice inventado. Antes disso, um dígito trocado no PMID resolveria para um artigo
    real e sem relação, os DOIS portões de extração o aprovariam (a citação é literal, a
    implicação é verdadeira), e nada ligaria a claim de volta à descoberta que você leu.
    """
    from lithium.recon.search import WebHit

    hits = [
        WebHit(title="meta-análise", url="https://pubmed.ncbi.nlm.nih.gov/30712879/",
               description="PMID 99999999 escrito errado no snippet"),
        WebHit(title="blog sem identificador", url="https://blog.invalid/post",
               description="fala de um estudo"),
    ]
    llm = TriagingLLM(verdicts=[{"index": 1, "kind": "lead"},
                                {"index": 2, "kind": "lead"}])
    ctx = make_ctx(tmp_path, llm=llm, searcher=FakeSearcher(hits))
    ctx.queue.enqueue("recon_query", {"query": "q", "focus_id": 1})
    await runner(ctx).drain()

    rows = ctx.store.conn.execute(
        "SELECT kind, lead_kind, lead_external_id FROM discoveries").fetchall()
    assert len(rows) == 1, "o lead sem identificador extraível devia ter sido descartado"
    assert (rows[0]["lead_kind"], rows[0]["lead_external_id"]) == ("pubmed", "30712879")
    ctx.store.close()


async def test_an_indexed_article_is_never_routed_to_a_page_read(tmp_path):
    """MUTAÇÃO: confiar no `kind` do modelo sem rotear.

    Um artigo do PubMed marcado `observation` gastaria uma leitura num domínio que
    MEDIDAMENTE devolve HTTP 203 "Cookies must be enabled".
    """
    from lithium.recon.search import WebHit

    hits = [WebHit(title="artigo", url="https://pubmed.ncbi.nlm.nih.gov/30712879/",
                   description="d")]
    llm = TriagingLLM(verdicts=[{"index": 1, "kind": "observation"}])
    ctx = make_ctx(tmp_path, llm=llm, searcher=FakeSearcher(hits))
    ctx.queue.enqueue("recon_query", {"query": "q", "focus_id": 1})
    await runner(ctx).drain()

    kinds = [r["kind"] for r in ctx.store.conn.execute("SELECT kind FROM discoveries")]
    assert kinds == ["lead"]
    assert llm.observe_calls == 0
    ctx.store.close()


async def test_an_observation_is_never_written_before_the_page_is_read(tmp_path):
    """MUTAÇÃO: `recon_triage` gravar a `observation` com o snippet como `summary`.

    Todos os seis portões de leitura atuam DEPOIS, então a linha ficaria `pending` com
    um resumo que nunca passou por leitura nenhuma, indistinguível das lidas nas duas
    superfícies. O usuário digita `/aprovar 12` e o texto de um snippet de motor de
    busca vira memória injetada em todo prompt do foco.
    """
    from lithium.recon.search import WebHit

    hits = [WebHit(title="guia", url="https://mudo.invalid/g", description="snippet")]
    llm = TriagingLLM(verdicts=[{"index": 1, "kind": "observation"}])

    def boom(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("robots.txt"):
            return httpx.Response(404, text="")
        raise httpx.ConnectError("rede caiu", request=request)

    from lithium.recon.read import PageReader

    reader = PageReader(httpx.AsyncClient(transport=httpx.MockTransport(boom)),
                        contact="c")
    ctx = make_ctx(tmp_path, llm=llm, searcher=FakeSearcher(hits), reader=reader)
    ctx.queue.enqueue("recon_query", {"query": "q", "focus_id": 1})
    await runner(ctx).drain()

    assert ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM discoveries").fetchone()["n"] == 0, (
        "uma observação apareceu sem a página ter sido lida"
    )
    ctx.store.close()


async def test_the_pending_queue_has_back_pressure(tmp_path):
    """MUTAÇÃO: remover `pending_limit`.

    `count=10` × `max_calls_per_sweep=6` = 60 resultados triados/dia. Com 30% de
    não-`skip` são 18 descobertas na primeira noite; com 60%, 36. O usuário pediu
    "comente comigo as suas descobertas", não uma fila de centenas que se aprova em lote
    sem ler. Contrapressão, não expiração: a varredura seguinte reencontra as mesmas
    páginas (`freshness` cobre 31 dias) e as propõe quando houver vaga.

    Idioma de `QuestionConfig.human_queue_limit` ("sistema que enche a fila não é
    usado"), aplicado por contrapressão real como em `question.py`.
    """
    searcher = FakeSearcher()
    ctx = make_ctx(tmp_path, searcher=searcher, pending_limit=2)
    for i in range(2):
        seed_discovery(ctx.store, url=f"https://ex.invalid/{i}")
    _question(ctx.store, "lítio em manutenção")

    ctx.queue.enqueue("recon_sweep", {})
    await runner(ctx).drain()

    assert searcher.calls == [], "a varredura gastou dinheiro com a fila cheia"
    assert ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM discoveries").fetchone()["n"] == 2
    ctx.store.close()


async def test_triage_stops_writing_when_the_queue_fills_mid_sweep(tmp_path):
    """A contrapressão também vale DENTRO da triagem: 10 vereditos não podem entrar numa
    fila com 1 vaga."""
    ctx = make_ctx(tmp_path, pending_limit=3,
                   llm=TriagingLLM(verdicts=[
                       {"index": i, "kind": "source"} for i in range(1, 6)]))
    ctx.queue.enqueue("recon_query", {"query": "q", "focus_id": 1})
    await runner(ctx).drain()
    assert ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM discoveries").fetchone()["n"] == 3
    ctx.store.close()


async def test_only_one_page_read_is_enqueued_per_query(tmp_path):
    """`max_reads_per_query` e não `max_pages_per_sweep`: o teto por VARREDURA não tem
    onde ser aplicado, porque são N triagens independentes — 6 × 5 = 30 leituras numa
    varredura anunciada como 5. Com 1 por query e 6 queries/dia, o custo bate com os
    ~9,9 min/dia declarados."""
    ctx = make_ctx(tmp_path, llm=TriagingLLM(verdicts=[
        {"index": i, "kind": "observation"} for i in range(1, 6)]))
    ctx.queue.enqueue("recon_query", {"query": "q", "focus_id": 1})
    await runner(ctx).drain(max_tasks=2)   # query + triage
    assert len(_tasks(ctx.store, "recon_read")) == 1
    ctx.store.close()


# ═══════════════════════════════════════════════════════ a ponte, na prática


def _pubmed(idlist, *, calls=None):
    from lithium.sources.pubmed import PubMedSource

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(dict(request.url.params).get("term", ""))
        return httpx.Response(200, json={"esearchresult": {"idlist": idlist}})

    return PubMedSource(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        rate_per_s=1000)


async def test_a_doi_lead_resolves_with_the_qualified_field(tmp_path):
    """MUTAÇÃO: mandar o DOI cru como `term`.

    EXECUTADO contra o NCBI real: `esearch(term='10.1001/jamapsychiatry.2019.0035')` —
    DOI bem formado e INEXISTENTE — devolve `querytranslation: "10.1001"[All Fields]`,
    `count=2`, e DOIS PMIDs reais e sem relação nenhuma ('Maternal caffeine intake…' e
    'hyponatremia in hip fracture'). O esearch DESCARTA os tokens que não encontra em vez
    de devolver vazio. Com `[doi]`, um DOI inexistente devolve zero.
    """
    calls: list[str] = []
    ctx = make_ctx(tmp_path, pubmed=_pubmed(["30658245"], calls=calls))
    did = seed_discovery(ctx.store, kind="lead", lead_kind="doi",
                         lead_external_id="10.1016/j.jad.2019.01.001")
    await HANDLERS["recon_lead"]({"discovery_id": did, "focus_id": 1}, ctx)

    assert calls == ["10.1016/j.jad.2019.01.001[doi]"], calls
    fetched = _tasks(ctx.store, "fetch_source")
    assert [f["external_id"] for f in fetched] == ["30658245"]
    ctx.store.close()


async def test_an_ambiguous_doi_resolution_enqueues_nothing(tmp_path):
    """Mais de um PMID significa que a resolução não é IDENTIDADE. `deferred`, e zero
    tarefas — em vez de ingerir dois artigos que você não aprovou."""
    ctx = make_ctx(tmp_path, pubmed=_pubmed(["26329421", "33631893"]))
    did = seed_discovery(ctx.store, kind="lead", lead_kind="doi",
                         lead_external_id="10.1001/jamapsychiatry.2019.0035")
    await HANDLERS["recon_lead"]({"discovery_id": did, "focus_id": 1}, ctx)

    assert _tasks(ctx.store, "fetch_source") == []
    assert ctx.store.conn.execute(
        "SELECT status FROM discoveries WHERE id = ?", (did,)
    ).fetchone()["status"] == "deferred"
    ctx.store.close()


async def test_approving_a_lead_another_focus_already_fetched_still_lands(tmp_path):
    """MUTAÇÃO: sempre enfileirar `fetch_source`.

    A `dedup_key` é `fetch:pubmed:{id}` — GLOBAL —, e `enqueue` deduplica contra `done` e
    `dead` também, enquanto `purge_done` só apaga `done`. O CLI imprimiria "✓
    enfileirado", `enqueue` devolveria None, nada rodaria, e o artigo nunca ganharia
    `claim_directness` neste foco: valeria zero nele para sempre. É o modo de falha que
    o comentário do `extract:` em handlers.py já descreve por escrito.
    """
    ctx = make_ctx(tmp_path, pubmed=_pubmed(["30712879"]))
    source_id = ctx.store.upsert_source(kind="pubmed", external_id="30712879",
                                        raw={}, title="já colhido")
    did = seed_discovery(ctx.store, kind="lead", lead_kind="pubmed",
                         lead_external_id="30712879")
    await HANDLERS["recon_lead"]({"discovery_id": did, "focus_id": 1}, ctx)

    assert _tasks(ctx.store, "fetch_source") == []
    extract = _tasks(ctx.store, "extract_source")
    assert [e["source_id"] for e in extract] == [source_id]
    assert [e["focus_id"] for e in extract] == [1]
    ctx.store.close()


async def test_the_bridge_refuses_to_run_under_another_focus(tmp_path):
    ctx = make_ctx(tmp_path, pubmed=_pubmed(["30712879"]))
    scale_id = int(ctx.store.conn.execute(
        "SELECT id FROM evidence_scales LIMIT 1").fetchone()["id"])
    ctx.store.conn.execute(
        "INSERT INTO focuses(id, slug, target, scale_id) VALUES(2, 'outro', 'x', ?)",
        (scale_id,))
    did = seed_discovery(ctx.store, kind="lead", lead_kind="pubmed",
                         lead_external_id="30712879")
    ctx.store.conn.execute("UPDATE meta SET value = '2' WHERE key = 'active_focus'")
    with pytest.raises(FocusDrift):
        await HANDLERS["recon_lead"]({"discovery_id": did, "focus_id": 1}, ctx)
    ctx.store.close()


# ══════════════════════════════════════════════════ o Job e o Scheduler


def test_the_scheduler_actually_fires_every_default_job():
    """O DEFEITO TOTAL E MUDO que era pré-requisito da fase.

    `due()` fazia `if last is None: return job.run_on_start`, e `_mark_run` só era
    chamado DEPOIS do disparo: para um job com `run_on_start=False`, `sched:<nome>`
    nunca existia e `due()` devolvia False PARA SEMPRE.

    EXECUTADO com o `Scheduler` e o `DEFAULT_JOBS` reais, 60 dias simulados: `harvest`,
    `plan`, `answer`, `notify` e `explore` disparavam; `pursue`, `reground`, `reflect` e
    `purge` disparavam **ZERO** vezes. Quatro laços do sistema estavam mortos, sem log e
    sem dead-letter — e o `Job('recon', …, run_on_start=False)` nasceria morto junto.

    MUTAÇÃO: voltar `due()` para `return job.run_on_start`. Este teste cai; antes dele,
    a mesma reversão matava 0 de 813.
    """
    from datetime import UTC, datetime, timedelta

    from lithium.db import Store
    from lithium.worker.queue import TaskQueue
    from lithium.worker.scheduler import DEFAULT_JOBS, Scheduler

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "s.db", embedding_dim=8)
        store.init_schema()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        clock = {"t": now}
        sched = Scheduler(store, TaskQueue(store), DEFAULT_JOBS,
                          clock=lambda: clock["t"])
        fired: dict[str, int] = {}
        for _ in range(60):
            for name in sched.tick():
                fired[name] = fired.get(name, 0) + 1
            clock["t"] += timedelta(days=1)

        missing = [j.name for j in DEFAULT_JOBS if not fired.get(j.name)]
        assert not missing, f"jobs que nunca dispararam em 60 dias: {missing}"
        assert fired.get("recon"), "o batedor nunca rodaria sozinho"
        store.close()


def test_the_recon_job_runs_once_a_day_and_yields_to_every_older_loop():
    """`interval_s = 86400` porque a `dedup_key` do scheduler é um balde de HORA:
    qualquer intervalo abaixo de 3600 s é estrangulado em silêncio para 1x/h enquanto o
    relógio avança — calibrar o teto POR VARREDURA contra uma cadência que não é a real
    é como o teto estoura pela metade.

    E a prioridade fica abaixo de todo laço que já funciona: recon é o consumidor mais
    novo e menos provado de GPU.
    """
    from lithium.worker.scheduler import DEFAULT_JOBS

    by_name = {j.name: j for j in DEFAULT_JOBS}
    recon = by_name["recon"]
    assert recon.interval_s >= 86400
    assert recon.run_on_start is False
    for older in ("reflect", "harvest", "answer", "plan", "notify"):
        assert recon.priority < by_name[older].priority, older


def test_the_default_caps_are_a_number_a_person_can_act_on():
    """MUTAÇÃO: `pending_limit = 10_000`.

    MEDIDO: com os testes de contrapressão passando o limite EXPLICITAMENTE, essa
    mutação matava 0 de 960 — o MECANISMO estava travado e o VALOR não. Um teto de
    10.000 é o mecanismo decorativo: `count=10 × 6 queries` nunca o alcança, e a fila
    humana volta a crescer sem freio. O idioma é o de `QuestionConfig.human_queue_limit`
    ("sistema que enche a fila não é usado"), e o número precisa ser da ordem do que uma
    pessoa decide numa sentada.

    Os tetos de CUSTO ficam aqui pela mesma razão: eles são a única defesa contra fatura
    em uso pessoal, e um default frouxo não tem quem o note.
    """
    from lithium.config import Config, QuestionConfig

    cfg = Config().recon
    assert cfg.enabled is False
    assert 3 <= cfg.pending_limit <= 2 * QuestionConfig().human_queue_limit
    # Brave: US$ 5/1.000 com US$ 5 de crédito mensal ≈ 1.000 grátis/mês ≈ 33/dia.
    assert 1 <= cfg.max_calls_per_sweep <= cfg.max_calls_per_day <= 32
    assert cfg.max_calls_per_day * 31 < 1_000, "o teto diário sai do tier grátis no mês"
    assert 1 <= cfg.max_reads_per_query <= 3
    assert cfg.max_reads_per_query * cfg.max_calls_per_sweep <= cfg.max_pages_per_day
    assert 7 <= cfg.expire_after_days <= 30
    assert cfg.respect_robots is True


def test_the_declared_gpu_cost_matches_the_actual_fan_out():
    """O número que justifica `priority=0.32` tem de ser o do fan-out REAL.

    Pelo modelo do PLAN (`t ≈ 2,31 ms × in + 161 ms × out`): uma triagem ≈ 42 s, uma
    leitura ≈ 57 s. Com 6 buscas/varredura e 1 leitura por query, o pior caso é
    6×42 + 6×57 ≈ 9,9 min/dia = 0,69% do dia, contra os ~38 min/dia (2,6%) do
    `answer_tick`.

    MUTAÇÃO: aplicar um teto de 5 leituras DENTRO da triagem, como o desenho pedia. São
    N triagens independentes, então viram 6 × 5 = 30 leituras numa varredura anunciada
    como 5 — 28,6 min/dia, a mesma ordem de grandeza do laço de auto-resposta, com
    prioridade MENOR: recon mataria de fome o que já funciona.
    """
    from lithium.config import Config

    cfg = Config().recon
    triage_s, read_s = 42.2, 56.6
    worst = cfg.max_calls_per_sweep * triage_s + \
        min(cfg.max_calls_per_sweep * cfg.max_reads_per_query,
            cfg.max_pages_per_day) * read_s
    assert worst / 86400 < 0.012, f"{worst / 60:.1f} min/dia é caro demais para o batedor"


def test_the_query_always_comes_from_the_active_focus(tmp_path):
    """Nenhuma query pode ser literal — nem no caminho de produção nem nos atalhos.

    O comando de gravação de fixture tinha `"bipolar maintenance lithium guideline"`
    fixo: buscava psiquiatria em qualquer foco e gravava a fixture com o domínio errado.
    Hoje ele chama `queries_for`, o mesmo construtor da varredura, então esta trava cobre
    os dois.

    MUTAÇÃO: em `queries_for`, devolver uma string literal em vez de derivar do alvo.
    """
    from lithium.recon.handlers import queries_for

    from lithium.db import Store

    store = Store(tmp_path / "q.db", embedding_dim=8)
    store.init_schema()

    # Um alvo que NENHUM literal plausível conteria. A primeira versão deste teste usava
    # o alvo de produção, que começa com "bipolar" — e a mutação que eu tentei matar
    # ("bipolar maintenance lithium guideline") também diz "bipolar". O teste passava
    # contra o próprio defeito. É a classe tautológica que este repo caça, e eu a
    # cometi aqui.
    alvo = "liga Ti-6Al-4V sob fadiga criogênica"
    store.conn.execute("UPDATE focuses SET target = ? WHERE id = ?",
                       (alvo, store.active_focus()["id"]))
    focus = store.active_focus()
    [(qid, q)] = queries_for(store, focus, tmp_path, limit=1)

    assert "Ti-6Al-4V" in q, (
        f"a query {q!r} não deriva do alvo {alvo!r} — um literal voltou"
    )
    assert qid == 0, "sem perguntas no banco, a query é o fallback derivado do alvo"
    store.close()
