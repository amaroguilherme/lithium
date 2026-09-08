"""Pool de workers asyncio sobre a fila do SQLite.

`Store` é síncrono (sqlite3 é síncrono), então toda chamada a banco sai por
`asyncio.to_thread`. Com conexões thread-local em WAL, isso dá leitura concorrente
de verdade em vez de serializar tudo atrás de um lock.

O handler recebe um `Context` em vez de globais: é o que permite os testes rodarem
o pool inteiro com LLM e fontes falsos, sem tocar em rede nem em GPU.
"""

from __future__ import annotations

import asyncio

from lithium.llm.client import LLMUnavailable
import logging
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from lithium.config import Config
from lithium.db import Store
from lithium.mode import is_researching
from lithium.worker.queue import Task, TaskQueue, redact_secrets

log = logging.getLogger(__name__)


@dataclass
class ReconBundle:
    """O batedor da web, montado só quando `[recon] enabled = true`.

    Existe como TIPO próprio para poder ser um CAMPO próprio do `Context`. Ver lá.
    """

    searcher: Any
    reader: Any
    results_per_query: int = 10


@dataclass
class Context:
    """Dependências entregues a cada handler."""

    config: Config
    store: Store
    queue: TaskQueue
    llm: Any = None
    embedder: Any = None
    sources: dict[str, Any] | None = None
    recon: ReconBundle | None = None
    """CAMPO PRÓPRIO, jamais uma entrada em `sources`.

    `Context.sources` é indexado por `kind` e `fetch_source` faz
    `(ctx.sources or {}).get(kind)` DEPOIS do portão de tipo — registrar o buscador da
    web ali é a linha ÚNICA que converte a web em fonte de evidência, e é o caminho de
    menor resistência em `daemon.py`. `test_the_daemon_registers_only_evidence_kinds`
    já pega essa reversão de graça, e `BraveSearch` nem tem atributo `kind`.
    """


Handler = Callable[[dict[str, Any], Context], Awaitable[None]]


class UnknownTaskKind(RuntimeError):
    pass


def is_pressure(exc: BaseException) -> bool:
    """A falha indica RECURSO indisponível, e não dado ruim?

    A distinção é o que separa contrapressão útil de estrangulamento por bug: um
    `ValueError` num payload malformado não melhora se o worker esperar, e reduzir a
    concorrência por causa dele só faz o sistema ficar lento sem consertar nada.

    Três famílias contam como pressão:

    - `LLMUnavailable` — o servidor não respondeu ou devolveu erro. MEDIDO em 4/9: com o
      swap a 97%, o servidor de embeddings passou a devolver `500` porque suas páginas
      foram paginadas e não voltavam. Não era bug dele; era a máquina sem memória.
    - **429** — teto de taxa de terceiro. O NCBI devolveu dois durante a operação.
    - **5xx** — servidor de terceiro em dificuldade.

    Qualquer outra coisa devolve False e não move a permissão.
    """
    if isinstance(exc, LLMUnavailable):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status is not None and (status == 429 or 500 <= status < 600)


class Throttle:
    """Quantas execuções simultâneas são permitidas AGORA. Encolhe na pressão.

    **Por que não medir memória.** O modo de falha real, medido em 4/9, foi swap a 97% com
    o servidor de embeddings devolvendo `500`. Medir memória diretamente exigiria código
    por plataforma — `vm_stat` no macOS, outra coisa no Windows — ou uma dependência nova
    (`psutil`), e o projeto tem CI nas duas. Reagir ao SINAL é portável, não custa
    dependência, e cobre mais casos: teto de taxa de terceiro e 5xx entram de graça.

    **Aumento aditivo, redução multiplicativa** — a disciplina do controle de congestão do
    TCP, pela mesma razão: subir devagar não reintroduz a pressão que acabou de ceder, e
    cair pela metade sai dela rápido. Piso de 1, para o sistema nunca parar de progredir.

    O ganho concreto: com a permissão em 1, as retentativas de uma tarefa acontecem
    ESPAÇADAS em vez de sob a mesma pressão que as derrubou. Foi o que matou dois
    `fetch_source` na primeira operação — as três tentativas caíram na mesma janela de
    swap saturado.
    """

    def __init__(self, maximum: int) -> None:
        self.maximum = max(1, int(maximum))
        self.allowed = self.maximum
        self.in_flight = 0
        self.shrinks = 0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cond:
            while self.in_flight >= self.allowed:
                await self._cond.wait()
            self.in_flight += 1

    async def release(self, *, pressure: bool = False, ok: bool = True) -> None:
        """Devolve a vaga e ajusta a permissão.

        `pressure` reduz pela metade. `ok` sobe UMA unidade — e só sobe quando houve
        sucesso: falha de dado devolve a vaga sem mexer na permissão, porque ela não é
        evidência de folga nem de aperto.
        """
        async with self._cond:
            self.in_flight -= 1
            if pressure:
                antes = self.allowed
                self.allowed = max(1, self.allowed // 2)
                if self.allowed != antes:
                    self.shrinks += 1
                    log.warning("contrapressão: concorrência %d -> %d",
                                antes, self.allowed)
            elif ok and self.allowed < self.maximum:
                self.allowed += 1
            self._cond.notify_all()


class Runner:
    def __init__(
        self,
        context: Context,
        handlers: dict[str, Handler],
        *,
        concurrency: int = 3,
        poll_interval_s: float = 2.0,
    ) -> None:
        self.ctx = context
        self.handlers = handlers
        self.concurrency = concurrency
        self.poll_interval_s = poll_interval_s
        # A permissão começa cheia e encolhe sob pressão. Ver `Throttle`.
        self.throttle = Throttle(concurrency)
        self.processed = 0
        self.failed = 0

    async def run(self, stop: asyncio.Event) -> None:
        """Sobe N workers e só volta quando todos drenarem após o `stop`."""
        self.ctx.queue.recover_orphans()
        async with asyncio.TaskGroup() as group:
            for i in range(self.concurrency):
                group.create_task(self._worker(i, stop), name=f"worker-{i}")

    async def drain(self, *, max_tasks: int | None = None) -> int:
        """Processa até a fila esvaziar e retorna. Usado em testes e no `lithium run`."""
        done = 0
        while max_tasks is None or done < max_tasks:
            task = await asyncio.to_thread(self._claim)
            if task is None:
                return done
            await self._execute(task)
            done += 1
        return done

    async def _worker(self, index: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                task = await asyncio.to_thread(self._claim)
            except Exception:
                log.exception("worker %d falhou ao reivindicar tarefa", index)
                await self._sleep_or_stop(stop, self.poll_interval_s)
                continue

            if task is None:
                await self._sleep_or_stop(stop, self.poll_interval_s)
                continue

            await self._execute(task)

    def _claim(self) -> Task | None:
        """Consulta o modo a cada reivindicação, não na subida.

        Ler uma vez no início faria `lithium mode off` só valer no próximo reinício —
        e o ponto de desligar é devolver a máquina agora.
        """
        return self.ctx.queue.claim(on_demand_only=not is_researching(self.ctx.store))

    async def _execute(self, task: Task) -> None:
        handler = self.handlers.get(task.kind)
        if handler is None:
            # Tipo desconhecido não é transitório: vai direto ao dead-letter em vez
            # de consumir as três tentativas.
            #
            # `queue.kill` e não `store.conn.execute`: passar um método já vinculado
            # à conexão para `to_thread` resolveria `conn` na thread chamadora e o
            # executaria em outra, que o sqlite3 proíbe. O acesso à conexão precisa
            # acontecer *dentro* da thread trabalhadora.
            await asyncio.to_thread(
                self.ctx.queue.kill, task.id, f"handler ausente para '{task.kind}'"
            )
            self.failed += 1
            log.error("tipo de tarefa desconhecido: %s", task.kind)
            return

        # A VAGA, não o número de workers. Os N workers continuam existindo; quantos
        # EXECUTAM ao mesmo tempo é o que a pressão controla.
        await self.throttle.acquire()
        pressure = False
        ok = False
        try:
            await handler(task.payload, self.ctx)
            ok = True
        except asyncio.CancelledError:
            # Shutdown: devolve à fila em vez de contar como falha.
            await asyncio.to_thread(self.ctx.queue.fail, task.id, "cancelada no shutdown")
            raise
        except Exception as exc:  # noqa: BLE001
            # REDIGIDO nas TRÊS saídas, não só na coluna. `queue.fail` cobre
            # `tasks.error`; estas duas linhas mandam o MESMO texto para o log do
            # daemon, que sob launchd é um arquivo. MEDIDO com o eutils real:
            # `'SEGREDO' in str(HTTPStatusError)` é True quando a chave vai por query
            # param, e a proteção da Fase A (`getLogger('httpx')`) não alcança o logger
            # `lithium.worker.runner`.
            pressure = is_pressure(exc)
            safe = redact_secrets(f"{type(exc).__name__}: {exc}")
            detail = f"{safe}\n{redact_secrets(traceback.format_exc()[-1500:])}"
            status = await asyncio.to_thread(self.ctx.queue.fail, task.id, detail)
            self.failed += 1
            if status == "dead":
                log.error("tarefa %s [%s] morreu após %d tentativas: %s",
                          task.id, task.kind, task.attempts, safe)
            else:
                log.warning("tarefa %s [%s] falhou, vai repetir: %s",
                            task.id, task.kind, safe)
        else:
            await asyncio.to_thread(self.ctx.queue.complete, task.id)
            self.processed += 1
        finally:
            # `finally` e não no `else`: sem isto, uma falha vaza a vaga para sempre e
            # depois de `allowed` falhas o pool inteiro travaria — pior que o problema
            # que a contrapressão existe para resolver.
            await self.throttle.release(pressure=pressure, ok=ok)

    @staticmethod
    async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
        """Dorme, mas acorda na hora se o shutdown chegar."""
        try:
            await asyncio.wait_for(stop.wait(), timeout=seconds)
        except TimeoutError:
            pass
