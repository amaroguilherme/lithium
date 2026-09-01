"""Fila, pool de workers e scheduler.

Dois testes aqui são critério de aceite do MVP, não detalhe:

* `test_concurrent_claims_never_double_assign` — 10 workers reais em threads reais.
  Sem `BEGIN IMMEDIATE` no claim, dois workers pegam a mesma linha e o trabalho roda
  duplicado.
* `test_orphans_return_to_the_queue_after_crash` — `kill -9` no meio de um harvest
  não pode perder nem duplicar tarefa.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta

import pytest

from lithium.config import Config
from lithium.db import Store
from lithium.llm.server import LlamaServer, ServerError
from lithium.worker.queue import TaskQueue, iso, utcnow
from lithium.worker.runner import Context, Runner
from lithium.worker.scheduler import Job, Scheduler


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "q.db", embedding_dim=8)
    s.init_schema()
    yield s
    s.close()


@pytest.fixture
def queue(store):
    return TaskQueue(store, max_attempts=3, backoff_base_s=1.0)


def _context(store, queue, tmp_path) -> Context:
    return Context(config=Config(data_dir=tmp_path), store=store, queue=queue)


# ─────────────────────────────────────────────────────────────── ciclo básico


def test_enqueue_claim_complete(queue):
    task_id = queue.enqueue("harvest", {"query": "quetiapina"})
    task = queue.claim()

    assert task is not None
    assert task.id == task_id
    assert task.kind == "harvest"
    assert task.payload == {"query": "quetiapina"}
    assert task.attempts == 1

    queue.complete(task.id)
    assert queue.claim() is None
    assert queue.stats() == {"done": 1}


def test_claim_on_empty_queue_returns_none(queue):
    assert queue.claim() is None


def test_higher_priority_is_claimed_first(queue):
    queue.enqueue("baixa", priority=0.1)
    queue.enqueue("alta", priority=0.9)
    assert queue.claim().kind == "alta"


def test_equal_priority_is_fifo(queue):
    first = queue.enqueue("a", priority=0.5)
    queue.enqueue("b", priority=0.5)
    assert queue.claim().id == first


def test_delayed_task_is_not_claimable_yet(queue):
    queue.enqueue("depois", delay_s=3600)
    assert queue.claim() is None


def test_dedup_key_blocks_duplicates(queue):
    """O plan tick reenfileira as mesmas buscas todo dia; sem chave o backlog
    cresce sem limite."""
    first = queue.enqueue("harvest", {"q": "x"}, dedup_key="harvest:x")
    second = queue.enqueue("harvest", {"q": "x"}, dedup_key="harvest:x")

    assert first is not None and second is None
    assert queue.stats() == {"pending": 1}


def test_tasks_without_dedup_key_are_never_deduped(queue):
    """UNIQUE em SQLite trata NULL como sempre distinto — comportamento desejado."""
    assert queue.enqueue("a") is not None
    assert queue.enqueue("a") is not None
    assert queue.stats() == {"pending": 2}


# ──────────────────────────────────────────────────────────── retry e dead-letter


def test_failure_reschedules_with_backoff(queue):
    queue.enqueue("frágil")
    task = queue.claim()
    assert queue.fail(task.id, "erro transitório") == "pending"

    # Reagendada no futuro, então não sai de novo agora.
    assert queue.claim() is None
    assert queue.stats() == {"pending": 1}


def test_backoff_grows_between_attempts(queue, store):
    queue.enqueue("frágil")
    delays = []
    for _ in range(2):
        task = queue.claim()
        queue.fail(task.id, "erro")
        scheduled = store.conn.execute(
            "SELECT scheduled_at FROM tasks WHERE id = ?", (task.id,)
        ).fetchone()["scheduled_at"]
        delays.append(scheduled)
        # devolve à fila para a próxima tentativa
        store.conn.execute(
            "UPDATE tasks SET scheduled_at = ? WHERE id = ?", (iso(utcnow()), task.id)
        )
    assert delays[1] > delays[0]


def test_dead_letter_after_max_attempts(queue):
    queue.enqueue("condenada")
    for _ in range(3):
        task = queue.claim()
        assert task is not None
        status = queue.fail(task.id, "sempre falha")
        queue.store.conn.execute(
            "UPDATE tasks SET scheduled_at = ? WHERE id = ?", (iso(utcnow()), task.id)
        )
    assert status == "dead"
    assert queue.claim() is None
    assert queue.stats() == {"dead": 1}

    [letter] = queue.dead_letters()
    assert letter["attempts"] == 3 and "sempre falha" in letter["error"]


def test_fail_on_missing_task_is_safe(queue):
    assert queue.fail(9999, "erro") == "missing"


# ─────────────────────────────────────────────────────── atomicidade e recuperação


def test_concurrent_claims_never_double_assign(store):
    """Critério de aceite: 10 workers, zero atribuição dupla.

    Sem `BEGIN IMMEDIATE`, dois workers fazem o mesmo SELECT e recebem a mesma
    linha — o harvest roda duas vezes, a extração duplica claims no banco.
    """
    queue = TaskQueue(store)
    n_tasks = 60
    for i in range(n_tasks):
        queue.enqueue("trabalho", {"i": i})

    claimed: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def worker() -> None:
        local = TaskQueue(store)
        barrier.wait()  # maximiza a chance de colisão
        while True:
            task = local.claim()
            if task is None:
                return
            with lock:
                claimed.append(task.id)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == n_tasks
    assert len(set(claimed)) == n_tasks, "a mesma tarefa foi entregue a dois workers"


def test_orphans_return_to_the_queue_after_crash(queue):
    """`kill -9` deixa tarefas em `running`; ninguém as pegaria de novo."""
    queue.enqueue("interrompida")
    task = queue.claim()
    assert queue.stats() == {"running": 1}

    assert queue.recover_orphans() == 1
    recovered = queue.claim()
    assert recovered is not None and recovered.id == task.id


def test_recovery_preserves_attempts_so_poison_tasks_still_die(queue):
    """Tarefa que derruba o processo toda vez precisa chegar ao dead-letter, não
    reiniciar em laço para sempre."""
    queue.enqueue("veneno")
    queue.claim()
    queue.recover_orphans()
    assert queue.claim().attempts == 2


def test_purge_done_respects_cutoff(queue, store):
    queue.enqueue("velha")
    task = queue.claim()
    queue.complete(task.id)
    store.conn.execute(
        "UPDATE tasks SET finished_at = ? WHERE id = ?",
        (iso(utcnow() - timedelta(days=30)), task.id),
    )
    assert queue.purge_done(older_than_days=7) == 1
    assert queue.stats() == {}


# ───────────────────────────────────────────────────────────────────── runner


async def test_runner_processes_queue_and_completes(store, queue, tmp_path):
    seen: list[int] = []

    async def handler(payload, ctx):
        seen.append(payload["i"])

    for i in range(5):
        queue.enqueue("trabalho", {"i": i})

    runner = Runner(_context(store, queue, tmp_path), {"trabalho": handler})
    assert await runner.drain() == 5
    assert sorted(seen) == [0, 1, 2, 3, 4]
    assert runner.processed == 5 and runner.failed == 0
    assert queue.stats() == {"done": 5}


async def test_handler_exception_becomes_retry(store, queue, tmp_path):
    async def boom(payload, ctx):
        raise ValueError("estourou")

    queue.enqueue("ruim")
    runner = Runner(_context(store, queue, tmp_path), {"ruim": boom})
    await runner.drain()

    assert runner.failed == 1 and runner.processed == 0
    assert queue.stats() == {"pending": 1}
    error = store.conn.execute("SELECT error FROM tasks").fetchone()["error"]
    assert "ValueError: estourou" in error


async def test_unknown_kind_goes_straight_to_dead_letter(store, queue, tmp_path):
    """Handler ausente não é falha transitória — gastar 3 tentativas é desperdício."""
    queue.enqueue("inexistente")
    runner = Runner(_context(store, queue, tmp_path), {})
    await runner.drain()
    assert queue.stats() == {"dead": 1}


async def test_drain_respects_max_tasks(store, queue, tmp_path):
    async def noop(payload, ctx):
        pass

    for _ in range(5):
        queue.enqueue("t")
    runner = Runner(_context(store, queue, tmp_path), {"t": noop})
    assert await runner.drain(max_tasks=2) == 2
    assert queue.stats()["pending"] == 3


async def test_workers_stop_promptly_on_shutdown(store, queue, tmp_path):
    """Poll de 30s não pode virar 30s de espera no Ctrl-C."""
    runner = Runner(
        _context(store, queue, tmp_path), {}, concurrency=2, poll_interval_s=30.0
    )
    stop = asyncio.Event()

    async def shutdown():
        await asyncio.sleep(0.05)
        stop.set()

    async with asyncio.timeout(3):
        await asyncio.gather(runner.run(stop), shutdown())


async def test_runner_recovers_orphans_before_starting(store, queue, tmp_path):
    queue.enqueue("órfã")
    queue.claim()

    runner = Runner(_context(store, queue, tmp_path), {}, concurrency=1)
    stop = asyncio.Event()
    stop.set()
    await runner.run(stop)

    assert queue.stats() == {"pending": 1}


# ──────────────────────────────────────────────────────────────────── scheduler


def _fixed_clock(moment: datetime):
    return lambda: moment


def test_run_on_start_jobs_fire_immediately(store, queue):
    jobs = (Job("agora", 3600, "tarefa_a", run_on_start=True),
            Job("depois", 3600, "tarefa_b", run_on_start=False))
    fired = Scheduler(store, queue, jobs).tick()
    assert fired == ["agora"]


def test_job_does_not_refire_within_its_interval(store, queue):
    now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    jobs = (Job("diária", 86400, "t", run_on_start=True),)

    assert Scheduler(store, queue, jobs, clock=_fixed_clock(now)).tick() == ["diária"]
    later = now + timedelta(hours=6)
    assert Scheduler(store, queue, jobs, clock=_fixed_clock(later)).tick() == []


def test_job_refires_after_interval(store, queue):
    now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    jobs = (Job("diária", 86400, "t", run_on_start=True),)

    Scheduler(store, queue, jobs, clock=_fixed_clock(now)).tick()
    tomorrow = now + timedelta(days=1, minutes=1)
    assert Scheduler(store, queue, jobs, clock=_fixed_clock(tomorrow)).tick() == ["diária"]


def test_schedule_state_survives_restart(store, queue):
    """O último disparo vive no banco. Se fosse em memória, reiniciar o daemon
    dispararia tudo de novo — e um harvest completo não é barato."""
    now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    jobs = (Job("diária", 86400, "t", run_on_start=True),)

    Scheduler(store, queue, jobs, clock=_fixed_clock(now)).tick()
    # "reinício": Scheduler novo, mesmo banco
    fresh = Scheduler(store, queue, jobs, clock=_fixed_clock(now + timedelta(minutes=5)))
    assert fresh.tick() == []


def test_restarts_within_the_hour_do_not_stack_work(store, queue):
    """A chave de dedup carrega a hora; reiniciar várias vezes na mesma janela
    não empilha o mesmo harvest."""
    now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    jobs = (Job("h", 1, "harvest_sweep", run_on_start=True),)

    for minute in range(4):
        clock = _fixed_clock(now + timedelta(minutes=minute))
        store.conn.execute("DELETE FROM meta WHERE key LIKE 'sched:%'")  # simula restart
        Scheduler(store, queue, jobs, clock=clock).tick()

    assert queue.stats() == {"pending": 1}


# ──────────────────────────────────────────────────────────── llama-server (args)


def test_command_is_argv_list_not_shell_string(tmp_path):
    """Contrato de portabilidade: sem shell, sem string a ser parseada."""
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    command = LlamaServer(model_path=model, port=8080).command()

    assert isinstance(command, list)
    assert all(isinstance(a, str) for a in command)
    assert not any(" " in a for a in command[1:] if not a.startswith("/"))


def test_generation_server_disables_reasoning_by_default(tmp_path):
    """Sem isto o Gemma-4 gasta o orçamento pensando e devolve `content` vazio."""
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    command = LlamaServer(model_path=model, port=8080).command()

    assert "--reasoning-budget" in command
    assert command[command.index("--reasoning-budget") + 1] == "0"
    assert "--jinja" in command


def test_embedding_server_omits_chat_only_flags(tmp_path):
    model = tmp_path / "e.gguf"
    model.write_bytes(b"x")
    command = LlamaServer(model_path=model, port=8081, embedding=True).command()

    assert "--embedding" in command
    assert "--reasoning-budget" not in command
    assert "--jinja" not in command


def test_lora_adapter_is_hot_loaded(tmp_path):
    """A LoRA do Kaggle entra como adapter GGUF em runtime — champion/challenger
    vira A/B de uma flag, sem merge nem requantização."""
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    lora = tmp_path / "v3.gguf"
    lora.write_bytes(b"y")
    command = LlamaServer(model_path=model, port=8080, lora_path=lora).command()

    assert "--lora" in command
    assert command[command.index("--lora") + 1] == str(lora)


def test_missing_model_fails_with_actionable_message(tmp_path):
    with pytest.raises(ServerError, match="modelo não encontrado"):
        LlamaServer(model_path=tmp_path / "ausente.gguf", port=8080).command()


def test_missing_binary_names_the_install_command(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    with pytest.raises(ServerError, match="brew install llama.cpp"):
        LlamaServer(model_path=model, port=8080, binary="binario-que-nao-existe").command()


def test_base_url_matches_client_expectation(tmp_path):
    assert LlamaServer(model_path=tmp_path / "m", port=9999).base_url == (
        "http://127.0.0.1:9999/v1"
    )


# ═══════════════════════════════ `lithium run` drena de verdade


def test_the_run_command_actually_drains_the_queue(tmp_path, monkeypatch):
    """`lithium run` prometia "drena a fila uma vez e sai" e não drenava nada.

    ENCONTRADO no primeiro contato real, não por leitura: enfileirei uma varredura, rodei
    `lithium run --max-tasks 1`, e a tarefa continuou `pending`. O comando imprimia
    "daemon no ar: 1 workers" e saía, sem erro e sem executar.

    A causa: ele montava `stop = asyncio.Event(); stop.set()` e passava para
    `Daemon.run(stop=...)`. Como `Runner._worker` é `while not stop.is_set()`, nenhum
    worker reivindicava uma única tarefa. `Runner.drain(max_tasks=...)` já existia — com
    um docstring afirmando ser usada aqui — e nunca era chamada, o que também fazia de
    `--max-tasks` um botão que não fazia nada.

    Este teste afirma sobre o COMPORTAMENTO (a fila esvazia), não sobre o fonte. Sem ele,
    reverter a fiação deixa a suíte inteira verde — medido.

    MUTAÇÃO: restaurar `stop.set()` e `Daemon.run(stop=stop)`.
    """
    from typer.testing import CliRunner

    from lithium import cli
    from lithium.config import load_config
    from lithium.db import Store
    from lithium.worker.queue import TaskQueue

    # `Daemon.run` checa os pesos antes de qualquer coisa, e `--max-tasks` não muda isso.
    # Arquivos vazios bastam: a checagem é `is_file()`, e `manage_servers=False` impede
    # que alguém tente carregá-los.
    gen = tmp_path / "gen.gguf"; gen.touch()
    emb = tmp_path / "emb.gguf"; emb.touch()
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text(
        f'data_dir = "{tmp_path / "d"}"\n'
        f'[llm]\nmodel_path = "{gen}"\n'
        f'[embedding]\nmodel_path = "{emb}"\n',
        encoding="utf-8")
    runner = CliRunner()
    assert runner.invoke(cli.app, ["init", "-c", str(cfgfile)]).exit_code == 0

    cfg = load_config(cfgfile)
    store = Store(cfg.db_path, embedding_dim=cfg.embedding.dim)
    queue = TaskQueue(store)
    for i in range(3):
        queue.enqueue("noop_probe", {"i": i}, dedup_key=f"probe:{i}")
    assert store.counts()["tasks_pending"] == 3
    store.close()

    seen: list[int] = []

    async def _probe(payload, ctx):
        seen.append(payload["i"])

    from lithium.worker import handlers as H
    monkeypatch.setitem(H.HANDLERS, "noop_probe", _probe)

    # O daemon espera o servidor de embeddings ficar pronto ANTES de drenar, com timeout
    # de 180 s. Sem este stub o teste passa quando há um llama-server no ar e trava por
    # três minutos quando não há — dependência de estado ambiente, que é o defeito que
    # este repo chama de teste não-hermético. Foi como eu o escrevi da primeira vez: verde
    # na minha máquina porque o servidor estava carregado, e a suíte inteira saltou de
    # 20 s para 200 s no primeiro `pytest` sem ele.
    import lithium.daemon as D
    from lithium.llm import LLMClient

    async def _pronto_embed(embedder, timeout_s=180.0):
        return None

    async def _pronto_gen(self, *a, **k):
        return None

    # OS DOIS. `daemon.run` espera o servidor de geração (`llm.wait_healthy`, que levanta
    # `LLMUnavailable`) E o de embeddings (`_wait_embedder`). Stubar só um deixa o teste
    # travando nos 180 s do outro — foi o que aconteceu na primeira correção.
    monkeypatch.setattr(D, "_wait_embedder", _pronto_embed)
    monkeypatch.setattr(LLMClient, "wait_healthy", _pronto_gen)

    result = runner.invoke(cli.app, ["run", "-c", str(cfgfile), "--max-tasks", "2"])
    assert result.exit_code == 0, result.output

    assert len(seen) == 2, (
        f"`--max-tasks 2` executou {len(seen)} tarefa(s): o comando não drena, ou o "
        f"limite não chega ao Runner"
    )
    store = Store(cfg.db_path, embedding_dim=cfg.embedding.dim)
    assert store.counts()["tasks_pending"] == 1, "a fila não foi drenada"
    store.close()


def test_external_servers_does_not_require_local_weights(tmp_path, monkeypatch):
    """Com `--external-servers`, os GGUF podem estar em OUTRA máquina.

    `_check_models` era chamado incondicionalmente, então o daemon exigia 9 GB de peso
    local para conversar com um llama-server remoto que já os tem carregados — o que
    tornava a opção inútil para o caso que ela mais serve: rodar a inferência numa máquina
    com GPU dedicada e deixar só o daemon aqui.

    O cliente sempre foi agnóstico a host (`base_url` num httpx); quem bloqueava era esta
    checagem.

    MUTAÇÃO: voltar `self._check_models()` para fora do `if self.manage_servers`.
    """
    from lithium.config import load_config
    from lithium.daemon import Daemon
    from lithium.llm import LLMClient

    cfgfile = tmp_path / "c.toml"
    # `model_path` aponta para arquivos que NÃO existem: é o cenário remoto.
    cfgfile.write_text(
        f'data_dir = "{tmp_path / "d"}"\n'
        f'[llm]\nmodel_path = "{tmp_path / "nao-existe.gguf"}"\n'
        f'base_url = "http://10.0.0.7:8080/v1"\n'
        f'[embedding]\nmodel_path = "{tmp_path / "tambem-nao.gguf"}"\n'
        f'base_url = "http://10.0.0.7:8081/v1"\n',
        encoding="utf-8")
    cfg = load_config(cfgfile)

    import lithium.daemon as D

    async def _pronto_embed(embedder, timeout_s=180.0):
        return None

    async def _pronto_gen(self, *a, **k):
        return None

    monkeypatch.setattr(D, "_wait_embedder", _pronto_embed)
    monkeypatch.setattr(LLMClient, "wait_healthy", _pronto_gen)

    import asyncio
    asyncio.run(Daemon(cfg, manage_servers=False).run(drain=True, drain_max=0))

    # gerenciar os servidores AQUI continua exigindo os pesos: a contrapartida, sem a
    # qual o guard poderia sumir de vez e este teste ficaria verde medindo o bug oposto.
    with pytest.raises(D.MissingModel):
        asyncio.run(Daemon(cfg, manage_servers=True).run(drain=True, drain_max=0))
