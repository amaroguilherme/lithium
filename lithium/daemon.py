"""Montagem do daemon: sobe os servidores, monta o contexto, roda workers e scheduler.

Tudo em context managers aninhados para que Ctrl-C — que chega como `CancelledError`
nas tarefas — ainda derrube os llama-server. Um GGUF de 8 GB deixado para trás segura
memória que ninguém devolve até o próximo reboot.

Sem tratamento de sinal explícito: `loop.add_signal_handler` não existe no Windows.
`KeyboardInterrupt` propagando por `asyncio.run` funciona nos dois, e é o que o CLI usa.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from lithium.config import Config
from lithium.db import Store
from lithium.llm import EmbeddingClient, LLMClient
from lithium.llm.server import LlamaServer
from lithium.llm.usage import UsageSink
from lithium.sources.factory import build_sources
from lithium.worker.handlers import HANDLERS
from lithium.worker.queue import TaskQueue
from lithium.worker.runner import Context, ReconBundle, Runner
from lithium.worker.scheduler import DEFAULT_JOBS, Scheduler

log = logging.getLogger(__name__)


class MissingModel(RuntimeError):
    pass


@dataclass
class Daemon:
    config: Config
    manage_servers: bool = True
    """False quando você já subiu os llama-server à mão (útil no desenvolvimento,
    porque recarregar 8 GB a cada reinício do daemon custa um minuto)."""

    def _check_models(self) -> None:
        missing: list[str] = []
        if not self.config.llm.model_path.is_file():
            missing.append(f"modelo de geração: {self.config.llm.model_path}")
        embed_path = self.config.embedding.model_path
        if embed_path is None or not embed_path.is_file():
            missing.append(
                "modelo de embedding não configurado. Baixe o GGUF do bge-m3 e aponte "
                "`embedding.model_path` no config.local.toml:\n"
                "    curl -L -o base_models/bge-m3-Q8_0.gguf https://huggingface.co/"
                "gpustack/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf"
            )
        if missing:
            raise MissingModel("\n  ".join(["pré-requisitos ausentes:", *missing]))

    async def run(self, *, stop: asyncio.Event | None = None,
                  drain: bool = False, drain_max: int | None = None) -> None:
        self._check_models()
        stop = stop or asyncio.Event()
        cfg = self.config
        cfg.ensure_dirs()

        async with contextlib.AsyncExitStack() as stack:
            if self.manage_servers:
                gen = LlamaServer(
                    model_path=cfg.llm.model_path,
                    port=_port(cfg.llm.base_url),
                    n_ctx=cfg.llm.n_ctx,
                    n_parallel=cfg.llm.n_parallel,
                    n_gpu_layers=cfg.llm.n_gpu_layers,
                    kv_cache_type=cfg.llm.kv_cache_type,
                    reasoning_budget=cfg.llm.reasoning_budget,
                    lora_path=cfg.llm.lora_path,
                    log_path=cfg.data_dir / "llama-gen.log",
                )
                embed = LlamaServer(
                    model_path=cfg.embedding.model_path,  # type: ignore[arg-type]
                    port=_port(cfg.embedding.base_url),
                    n_ctx=8192,
                    embedding=True,
                    log_path=cfg.data_dir / "llama-embed.log",
                )
                await stack.enter_async_context(gen)
                await stack.enter_async_context(embed)

            # O store precisa existir antes do cliente: o sink escreve nele.
            store = Store(cfg.db_path, embedding_dim=cfg.embedding.dim)
            store.init_schema()
            sink = UsageSink(store)
            stack.push_async_callback(sink.flush)

            llm = await stack.enter_async_context(
                LLMClient(
                    cfg.llm.base_url, cfg.llm.model,
                    timeout_s=cfg.llm.timeout_s,
                    max_retries=cfg.llm.max_retries,
                    temperature=cfg.llm.temperature,
                    on_usage=sink.record,
                )
            )
            embedder = await stack.enter_async_context(
                EmbeddingClient(
                    cfg.embedding.base_url, cfg.embedding.model,
                    dim=cfg.embedding.dim, batch_size=cfg.embedding.batch_size,
                )
            )

            log.info("aguardando os llama-server ficarem prontos…")
            await llm.wait_healthy()
            await _wait_embedder(embedder)
            log.info("servidores prontos")

            queue = TaskQueue(
                store,
                max_attempts=cfg.worker.max_attempts,
                backoff_base_s=cfg.worker.backoff_base_s,
            )
            # AS FONTES VÊM DO REGISTRO. Era um `PubMedSource` construído à mão e um
            # dict literal — o que fazia da aprovação no registro uma decoração, porque a
            # escolha real morava aqui. `build_sources` instancia uma fonte por linha
            # aprovada e deixa de fora aquelas cuja credencial não resolve.
            sources = build_sources(store, cfg, stack)

            # O batedor da web, só quando a chave está lá. Em CAMPO PRÓPRIO do
            # Context — nunca em `sources`, que é indexado por `kind` e é lido por
            # `fetch_source` DEPOIS do portão de tipo.
            recon = await _build_recon(cfg, stack, store)

            context = Context(
                config=cfg, store=store, queue=queue,
                llm=llm, embedder=embedder, sources=sources,
                recon=recon,
            )
            runner = Runner(
                context, HANDLERS,
                concurrency=cfg.worker.concurrency,
                poll_interval_s=cfg.worker.poll_interval_s,
            )
            scheduler = Scheduler(store, queue, DEFAULT_JOBS)

            if drain_max is not None or drain:
                # `lithium run`: drena o que já está na fila e sai, sem scheduler.
                # Antes isto era `Daemon.run(stop=<já setado>)`, e como `_worker` é
                # `while not stop.is_set()` o runner saía sem reivindicar uma única
                # tarefa. O comando imprimia "daemon no ar" e não fazia nada — e
                # `--max-tasks` não chegava a lugar nenhum. `Runner.drain` já existia,
                # com um docstring afirmando ser usada aqui.
                queue.recover_orphans()
                n = await runner.drain(max_tasks=drain_max)
                log.info("drenadas %d tarefa(s)", n)
                return

            log.info("daemon no ar: %d workers", cfg.worker.concurrency)
            async with asyncio.TaskGroup() as group:
                group.create_task(runner.run(stop), name="workers")
                group.create_task(scheduler.run(stop), name="scheduler")


async def _build_recon(cfg: Config, stack: contextlib.AsyncExitStack,
                       store: Store | None = None):
    """Monta o `ReconBundle`, ou devolve None. Nunca derruba o daemon.

    `ReconDisabled` aqui é um estado NORMAL — o provedor é desabilitado por padrão. O
    daemon inteiro não pode morrer porque a chave da Brave não está configurada; o que
    não pode acontecer é o recon "rodar" e não descobrir nada em silêncio, e disso cuida
    o fail-loud de `BraveSearch.__init__` mais o `log.info` daqui.
    """
    import httpx

    from lithium.recon.budget import debit_robots
    from lithium.recon.read import PageReader
    from lithium.recon.search import BraveSearch, ReconDisabled, RESULTS_PER_QUERY

    if not cfg.recon.enabled:
        log.info("batedor da web desabilitado ([recon] enabled = false)")
        return None
    try:
        searcher = BraveSearch(cfg.recon)
    except ReconDisabled as exc:
        log.warning("batedor da web não subiu: %s", exc)
        return None
    stack.push_async_callback(searcher.aclose)

    page_client = httpx.AsyncClient(follow_redirects=True)
    stack.push_async_callback(page_client.aclose)
    # O MESMO `store` do daemon: a cota de robots.txt é durável e mora no banco. Uma
    # segunda conexão para o mesmo arquivo funcionaria e seria só desperdício — mas
    # também esconderia que este contador é o mesmo que `lithium status` mostra.
    budget_store = store if store is not None else Store(
        cfg.db_path, embedding_dim=cfg.embedding.dim)
    reader = PageReader(
        page_client, contact=cfg.recon.contact or "",
        respect_robots=cfg.recon.respect_robots,
        robots_budget=lambda: debit_robots(budget_store,
                                           cfg.recon.max_robots_per_day),
    )
    log.info("batedor da web no ar: %s, até %d busca(s)/dia e %d leitura(s)/dia",
             cfg.recon.provider, cfg.recon.max_calls_per_day,
             cfg.recon.max_pages_per_day)
    return ReconBundle(searcher=searcher, reader=reader,
                       results_per_query=RESULTS_PER_QUERY)


def _port(base_url: str) -> int:
    return int(base_url.rstrip("/").removesuffix("/v1").rsplit(":", 1)[-1])


async def _wait_embedder(embedder: EmbeddingClient, timeout_s: float = 180.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if await embedder.healthy():
            return
        await asyncio.sleep(2.0)
    raise RuntimeError(f"servidor de embeddings não respondeu: {embedder.base_url}")
