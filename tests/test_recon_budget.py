"""O teto de chamadas: DURÁVEL, debitado antes da requisição, nunca estornado.

É a única parte deste sistema que gasta dinheiro, e em uso pessoal não há ninguém
olhando um dashboard.
"""

from __future__ import annotations

import httpx
import pytest

from lithium.recon import budget
from lithium.recon.search import BraveSearch
from lithium.config import ReconConfig
from lithium.worker.handlers import HANDLERS

from reconkit import FakeSearcher, make_ctx, make_store, runner


# ══════════════════════════════════════════════════ durabilidade e atomicidade


def test_the_daily_cap_is_durable_across_a_restart(tmp_path):
    """MUTAÇÃO: guardar o contador em atributo de instância (o que o `RateLimiter` faz).

    Sob launchd, um daemon que reinicia em laço zera a cota a cada subida — exatamente o
    cenário em que ela é a única proteção: uso pessoal, ninguém olhando dashboard, chave
    com fatura.
    """
    store = make_store(tmp_path)
    for _ in range(3):
        assert budget.debit_search(store, cap=3) is True
    assert budget.debit_search(store, cap=3) is False
    store.close()

    from lithium.db import Store

    reborn = Store(tmp_path / "recon.db", embedding_dim=16)
    reborn.init_schema()
    assert budget.debit_search(reborn, cap=3) is False, (
        "a cota voltou do zero depois do restart"
    )
    reborn.close()


def test_the_cap_counts_attempts_not_successes(tmp_path):
    """Debitado ANTES da requisição e NUNCA estornado.

    Uma falha DEPOIS de o provedor responder — parse, timeout de leitura, 500 — gasta
    cota que um contador de SUCESSOS registraria como zero. E `queue.fail` repete.
    """
    store = make_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="isto não é json")

    searcher = BraveSearch(
        ReconConfig(enabled=True, api_key="k", contact="c"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        rate_per_s=1000,
    )
    assert budget.debit_search(store, cap=5) is True
    import asyncio

    with pytest.raises(Exception):
        asyncio.run(searcher.web_search("q", limit=10))
    assert budget.spent(store)["search_calls"] == 1, (
        "uma chamada que morreu no parse não pode contar como zero"
    )
    store.close()


def test_a_cap_of_zero_stops_everything_including_the_first_call(tmp_path):
    """MEDIDO em sqlite 3.51.2: o ramo de INSERT do UPSERT não tem WHERE, então com
    `cap = 0` a PRIMEIRA chamada do dia é debitada e EXECUTADA assim mesmo, e a linha
    nasce com 1. "Teto 0" significando "uma por dia" é a diferença entre uma torneira e
    um pingo — e `max_calls_per_day = 0` é o gesto óbvio de "pare de gastar agora, sem
    editar código"."""
    store = make_store(tmp_path)
    assert budget.debit_search(store, cap=0) is False
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM recon_budget").fetchone()["n"] == 0
    store.close()


def test_pages_searches_and_robots_are_counted_separately(tmp_path):
    """Conflatá-las faria o número que o usuário lê como "quanto isto me custou" ser
    mentira: com o cache de robots frio (o estado a cada subida sob launchd) e 8 hosts
    distintos, "leituras hoje: 15" significaria 7 páginas lidas."""
    store = make_store(tmp_path)
    budget.debit_search(store, cap=9)
    budget.debit_page(store, cap=9)
    budget.debit_page(store, cap=9)
    budget.debit_robots(store, cap=9)
    spent = budget.spent(store)
    assert (spent["search_calls"], spent["page_fetches"], spent["robots_fetches"]) \
        == (1, 2, 1)
    store.close()


# ═════════════════════════════════ o teto é operação normal, não falha


async def test_hitting_the_cap_is_normal_operation_not_a_failure(tmp_path):
    """MUTAÇÃO: `raise` no teto.

    Todo dia ao bater a cota, N tarefas viram dead-letter gastando 3 tentativas cada, e
    você recebe um toast "N tarefas falharam (recon_query)". Operação normal reportada
    como falha é o que treina alguém a ignorar o aviso seguinte — e as tentativas são
    PAGAS.
    """
    searcher = FakeSearcher()
    ctx = make_ctx(tmp_path, searcher=searcher, max_calls_per_day=0)
    ctx.queue.enqueue("recon_query", {"query": "q", "focus_id": 1})
    await runner(ctx).drain()

    assert searcher.calls == [], "requisitou mesmo com a cota esgotada"
    statuses = {r["kind"]: r["status"] for r in ctx.store.conn.execute(
        "SELECT kind, status FROM tasks")}
    assert statuses["recon_query"] == "done", statuses
    assert "dead" not in statuses.values()

    from lithium.notify import watch

    fresh, _ = watch.dead_delta(ctx.store, seen={})
    assert fresh == {}, "o teto virou aviso de falha"
    ctx.store.close()


async def test_a_retry_of_the_page_read_never_re_bills_the_search(tmp_path):
    """O FATIAMENTO, e é a peça de engenharia da fase.

    MUTAÇÃO: fundir busca + triagem + leitura num handler só. `queue.fail` reagenda com
    backoff até `max_attempts=3`, então uma falha de rede na leitura TRIPLICARIA a
    fatura da busca — invisivelmente, porque nada no sistema reporta "gastei 3x pelo
    mesmo resultado".
    """
    import asyncio

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("rede caiu", request=request)

    from lithium.recon.read import PageReader

    reader = PageReader(httpx.AsyncClient(transport=httpx.MockTransport(boom)),
                        contact="c", respect_robots=False)
    searcher = FakeSearcher()
    ctx = make_ctx(tmp_path, searcher=searcher, reader=reader)

    ctx.queue.enqueue("recon_query", {"query": "q", "focus_id": 1})
    await runner(ctx).drain()
    billed_after_search = budget.spent(ctx.store)["search_calls"]
    assert billed_after_search == 1

    # A leitura falha e é retentada até morrer. A busca NÃO é refaturada.
    ctx.queue.enqueue("recon_read", {
        "focus_id": 1, "query": "q",
        "hit": {"title": "t", "url": "https://mudo.invalid/p", "description": "d",
                "extra_snippets": []}})
    for _ in range(4):
        await runner(ctx).drain()
        await asyncio.sleep(0)
        ctx.store.conn.execute(
            "UPDATE tasks SET scheduled_at = '2000-01-01T00:00:00.000Z' "
            " WHERE kind = 'recon_read' AND status = 'pending'")

    assert budget.spent(ctx.store)["search_calls"] == billed_after_search
    assert len(searcher.calls) == 1
    ctx.store.close()


def test_the_budget_is_not_the_rate_limiter():
    """O `RateLimiter` guarda `_tokens` em atributo de INSTÂNCIA. Ele é cortesia por
    host; ele NÃO conta dinheiro, e este teste é o que impede alguém de reusá-lo para
    isso."""
    import ast
    import inspect

    from lithium.rate_limit import RateLimiter

    assert "_tokens" in inspect.getsource(RateLimiter), (
        "o RateLimiter deixou de ser estado de instância; reveja este argumento"
    )
    # Sobre o CÓDIGO, não sobre a prosa: o módulo EXPLICA por que não usa o
    # RateLimiter, e explicar não é usar.
    tree = ast.parse(inspect.getsource(budget))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            node.value = ast.Constant(value="")
    code = ast.unparse(tree)
    assert "recon_budget" in code, "o contador deixou de ser durável"
    assert "RateLimiter" not in code, "a cota voltou a depender de estado de processo"


def test_the_handlers_debit_before_the_request_not_after():
    """AST: em `recon_query` e `recon_read`, o débito vem ANTES da chamada de rede.

    MUTAÇÃO: mover o `debit_*` para depois do `await`. Um erro na requisição passa a
    custar cota que ninguém contou.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "lithium" / "recon"
              / "handlers.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name, network in (("recon_query", "web_search"), ("recon_read", "reader")):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
        lines = ast.unparse(fn).splitlines()
        debit = next(i for i, line in enumerate(lines) if "debit_" in line)
        call = next(i for i, line in enumerate(lines) if network in line)
        assert debit < call, f"{name}: débito depois da rede"


def test_the_handler_registry_knows_every_recon_step():
    for kind in ("recon_sweep", "recon_query", "recon_triage", "recon_read",
                 "recon_lead"):
        assert kind in HANDLERS, f"{kind} não está registrado: nada o executa"
