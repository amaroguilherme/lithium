"""Modo pesquisa: ligado (24/7) versus desligado (sob demanda).

A propriedade que define o recurso: desligado **não** é inerte. O daemon continua
atendendo o que você pediu explicitamente e continua aprendendo com a conversa — só
para de pesquisar por conta própria. Um `stop` global congelaria também o pedido que
você acabou de fazer, e deixar tudo rodando não devolveria a máquina.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lithium.config import Config
from lithium.db import Store
from lithium.mode import ResearchMode, backlog, get_mode, is_researching, set_mode
from lithium.worker.queue import TaskQueue
from lithium.worker.runner import Context, Runner
from lithium.worker.scheduler import Job, Scheduler


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", embedding_dim=8)
    s.init_schema()
    yield s
    s.close()


@pytest.fixture
def queue(store):
    return TaskQueue(store)


def _context(store, queue, tmp_path) -> Context:
    return Context(config=Config(data_dir=tmp_path), store=store, queue=queue)


# ────────────────────────────────────────────────────────────────── estado


def test_default_is_on(store):
    """O padrão é o comportamento que o projeto assume; desligar é ato deliberado."""
    assert get_mode(store) is ResearchMode.ON
    assert is_researching(store)


def test_mode_persists_across_reopen(tmp_path):
    """Se o estado vivesse em memória, reiniciar o daemon religaria a pesquisa
    sozinho — e o ponto de desligar é que ela fique desligada."""
    path = tmp_path / "p.db"
    first = Store(path, embedding_dim=8)
    first.init_schema()
    set_mode(first, ResearchMode.OFF)
    first.close()

    second = Store(path, embedding_dim=8)
    second.init_schema()
    assert get_mode(second) is ResearchMode.OFF
    second.close()


def test_corrupt_value_falls_back_to_on(store):
    store.conn.execute("INSERT INTO meta(key, value) VALUES('research_mode', 'talvez')")
    assert get_mode(store) is ResearchMode.ON


# ──────────────────────────────────────────────────────────────── scheduler


def test_scheduler_fires_when_on(store, queue):
    jobs = (Job("h", 3600, "harvest_sweep", run_on_start=True),)
    assert Scheduler(store, queue, jobs).tick() == ["h"]


def test_scheduler_is_silent_when_off(store, queue):
    set_mode(store, ResearchMode.OFF)
    jobs = (Job("h", 3600, "harvest_sweep", run_on_start=True),)
    assert Scheduler(store, queue, jobs).tick() == []
    assert queue.stats() == {}


def test_off_does_not_advance_the_job_clock(store, queue):
    """Religar não pode provocar enxurrada de disparos atrasados."""
    now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    jobs = (Job("h", 3600, "harvest_sweep", run_on_start=True),)

    set_mode(store, ResearchMode.OFF)
    Scheduler(store, queue, jobs, clock=lambda: now).tick()
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM meta WHERE key LIKE 'sched:%'"
    ).fetchone()["n"] == 0

    set_mode(store, ResearchMode.ON)
    assert Scheduler(store, queue, jobs, clock=lambda: now).tick() == ["h"]


def test_scheduled_work_is_tagged_as_such(store, queue):
    Scheduler(store, queue, (Job("h", 3600, "harvest_sweep", run_on_start=True),)).tick()
    assert store.conn.execute("SELECT origin FROM tasks").fetchone()["origin"] == "scheduled"


# ─────────────────────────────────────────────────── reivindicação por origem


def test_off_mode_still_serves_what_you_asked_for(store, queue):
    """O núcleo do recurso: desligado atende você, ignora o scheduler."""
    queue.enqueue("harvest_sweep", origin="scheduled")
    queue.enqueue("ask_question", {"text": "?"}, origin="on_demand")

    set_mode(store, ResearchMode.OFF)
    claimed = queue.claim(on_demand_only=True)
    assert claimed is not None and claimed.kind == "ask_question"

    assert queue.claim(on_demand_only=True) is None, "trabalho agendado não pode sair"


def test_on_mode_claims_everything(store, queue):
    queue.enqueue("harvest_sweep", origin="scheduled")
    queue.enqueue("ask_question", origin="on_demand")
    kinds = {queue.claim().kind, queue.claim().kind}
    assert kinds == {"harvest_sweep", "ask_question"}


def test_enqueue_defaults_to_on_demand(store, queue):
    """Só o scheduler marca `scheduled`. Qualquer outro caminho é pedido seu, e
    precisa continuar funcionando com a pesquisa desligada."""
    queue.enqueue("ask_question")
    assert store.conn.execute("SELECT origin FROM tasks").fetchone()["origin"] == "on_demand"


def test_backlog_counts_only_held_scheduled_work(store, queue):
    queue.enqueue("harvest_sweep", origin="scheduled")
    queue.enqueue("harvest_query", origin="scheduled")
    queue.enqueue("ask_question", origin="on_demand")
    assert backlog(store) == 2


# ──────────────────────────────────────────────────────────────── runner


async def test_runner_honours_off_mode(store, queue, tmp_path):
    seen: list[str] = []

    async def handler(payload, ctx):
        seen.append(payload["tag"])

    queue.enqueue("t", {"tag": "agendada"}, origin="scheduled")
    queue.enqueue("t", {"tag": "pedida"}, origin="on_demand")
    set_mode(store, ResearchMode.OFF)

    runner = Runner(_context(store, queue, tmp_path), {"t": handler})
    assert await runner.drain() == 1
    assert seen == ["pedida"]
    assert backlog(store) == 1


async def test_switching_on_releases_the_backlog(store, queue, tmp_path):
    async def handler(payload, ctx):
        pass

    queue.enqueue("t", origin="scheduled")
    set_mode(store, ResearchMode.OFF)
    runner = Runner(_context(store, queue, tmp_path), {"t": handler})
    assert await runner.drain() == 0

    set_mode(store, ResearchMode.ON)
    assert await runner.drain() == 1


async def test_mode_is_read_per_claim_not_at_startup(store, queue, tmp_path):
    """`lithium mode off` precisa valer agora, não no próximo reinício — devolver a
    máquina é o ponto."""
    processed: list[int] = []

    async def handler(payload, ctx):
        processed.append(payload["i"])
        if payload["i"] == 0:
            set_mode(ctx.store, ResearchMode.OFF)

    for i in range(3):
        queue.enqueue("t", {"i": i}, origin="scheduled")

    runner = Runner(_context(store, queue, tmp_path), {"t": handler})
    await runner.drain()
    assert processed == [0], "os workers deveriam parar assim que o modo mudou"
