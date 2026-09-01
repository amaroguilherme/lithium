"""Cadência interna: enfileira trabalho periódico.

Não usa cron nem APScheduler. Cron é POSIX e o Task Scheduler do Windows tem outra
semântica — amarrar a cadência a qualquer um dos dois viola o contrato de
portabilidade. Aqui a cadência é um laço asyncio dentro do próprio daemon, e a
supervisão externa (launchd / Task Scheduler) só precisa manter *um* processo vivo.

O último disparo fica no banco, não em memória: reiniciar o daemon não deve
ressetar o relógio e disparar tudo de novo.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from lithium.db import Store
from lithium.mode import is_researching
from lithium.worker.queue import TaskQueue, iso, utcnow

log = logging.getLogger(__name__)

META_PREFIX = "sched:"


@dataclass(slots=True)
class Job:
    name: str
    interval_s: float
    task_kind: str
    priority: float = 0.5
    payload: dict | None = None
    run_on_start: bool = False
    """Se True, dispara já na subida quando nunca rodou antes."""


# Só entram aqui jobs cujo handler existe. Agendar um `task_kind` sem handler mandaria
# a tarefa direto ao dead-letter a cada ciclo — ruído constante mascarando falha real.
# `weekly_report` e `sync_answers` entram junto com os itens 8 e 10.
DEFAULT_JOBS: tuple[Job, ...] = (
    Job("harvest", 86400, "harvest_sweep", priority=0.4, run_on_start=True),
    # Depois do harvest, de propósito: planejar sobre um corpus que acabou de crescer
    # produz perguntas melhores que planejar sobre o estado da véspera.
    Job("plan", 86400, "plan_tick", priority=0.7, run_on_start=True),
    # Trilha exploratória em cadência própria e mais lenta: especulação boa depende
    # de ter corpus para se apoiar, e gerar todo dia rende repetição.
    Job("explore", 172800, "explore_tick", priority=0.5, run_on_start=True),
    # O laço da trilha exploratória: perseguir gera buscas dirigidas, reancorar
    # converte elos assumidos em citados conforme o corpus cresce. Defasados de
    # propósito — reancorar antes da coleta chegar não encontra nada.
    Job("pursue", 172800, "pursue_speculation", priority=0.55, run_on_start=False),
    Job("reground", 259200, "reground_speculations", priority=0.45, run_on_start=False),
    # Reflexão depois do ciclo exploratório: generalizar sobre pesquisa que ainda não
    # aconteceu não produz lição, produz ruído.
    Job("reflect", 259200, "reflect_tick", priority=0.35, run_on_start=False),
    # O consumidor da fila de perguntas. Quatro varreduras por dia: a vazão é
    # `em_voo x varreduras / rodadas`, e 4x4/3 = 5,3 perguntas/dia contra as ~5/dia que
    # o `plan_tick` produz. Com duas varreduras a fila satura e o excedente é fechado
    # pelo teto sem ninguém ter lido.
    Job("answer", 21600, "answer_tick", priority=0.6, run_on_start=True),
    # Uma hora, e o número não é livre: `dedup_key` usa um balde horário
    # (`iso(now)[:13]`), então qualquer Job com intervalo menor que 3600 s é
    # silenciosamente estrangulado para 1x/hora — o segundo disparo colide na chave,
    # `enqueue` devolve None, e o relógio avança como se tivesse rodado.
    Job("notify", 3600, "notify_tick", priority=0.8, run_on_start=True),
    # Semanal. `run_on_start=False`: subir o daemon não é motivo para relatório, e
    # o primeiro sairia sobre uma janela de zero segundo.
    Job("report", 604800, "write_report", priority=0.2, run_on_start=False),
    Job("purge", 86400, "purge_tasks", priority=0.1),
    # O batedor da web. UMA varredura por dia, e o número não é livre: `dedup_key` usa
    # um balde de HORA, então qualquer intervalo abaixo de 3600 s é estrangulado em
    # silêncio para 1x/h enquanto `_mark_run` avança o relógio assim mesmo — calibrar o
    # teto POR VARREDURA contra uma cadência que não é a real é como o teto estoura
    # pela metade.
    #
    # `priority=0.32` fica acima de `relens_claim` (0,3) e abaixo de `reflect` (0,35),
    # `harvest` (0,4) e `answer` (0,6): recon é o consumidor mais novo e menos provado
    # de GPU e não pode matar de fome nenhum laço que já funciona.
    #
    # `run_on_start=False` para habilitar a chave não disparar uma varredura na
    # primeira subida. (Só é honesto depois do conserto de `due()` acima — antes dele
    # isso significaria "nunca".)
    Job("recon", 86400, "recon_sweep", priority=0.32, run_on_start=False),
)


class Scheduler:
    def __init__(self, store: Store, queue: TaskQueue, jobs=DEFAULT_JOBS,
                 *, tick_s: float = 30.0,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self.store = store
        self.queue = queue
        self.jobs = list(jobs)
        self.tick_s = tick_s
        self.clock = clock

    def _last_run(self, name: str) -> datetime | None:
        row = self.store.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (META_PREFIX + name,)
        ).fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(row["value"].replace("Z", "+00:00"))

    def _mark_run(self, name: str, moment: datetime) -> None:
        self.store.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (META_PREFIX + name, iso(moment)),
        )

    def due(self, job: Job, now: datetime) -> bool:
        """Vence agora? E — se nunca rodou e não dispara na subida — SEMEIA o relógio.

        **O defeito que isto conserta é total e mudo.** `if last is None: return
        job.run_on_start` combinado com `_mark_run` sendo chamado só DEPOIS do disparo
        forma um laço fechado: para um job com `run_on_start=False`, `sched:<nome>`
        nunca existe, então `due()` devolve False PARA SEMPRE. EXECUTADO com o
        `Scheduler` e o `DEFAULT_JOBS` reais, relógio avançado dia a dia por 60 dias
        simulados: `harvest`, `plan`, `answer`, `notify` e `explore` disparam; `pursue`,
        `reground`, `reflect` e `purge` disparam **ZERO** vezes — os quatro com
        `run_on_start=False`. Quatro laços do sistema estão mortos hoje, sem log e sem
        dead-letter.

        A semeadura é o conserto mínimo: a primeira vez que o job é VISTO, o relógio
        começa a contar; a partir daí o intervalo vale. É pré-requisito da Fase C, não
        escopo extra — sem ele o `Job('recon', …, run_on_start=False)` nunca rodaria e
        o usuário habilitaria a chave da Brave para nada.
        """
        last = self._last_run(job.name)
        if last is None:
            if job.run_on_start:
                return True
            self._mark_run(job.name, now)
            return False
        return now - last >= timedelta(seconds=job.interval_s)

    def tick(self) -> list[str]:
        """Enfileira o que venceu. Devolve os nomes disparados.

        Com o modo pesquisa desligado não dispara nada: é o que "sob demanda"
        significa. O relógio de cada job também não avança, então religar não
        provoca uma enxurrada de disparos atrasados.
        """
        if not is_researching(self.store):
            return []

        now = self.clock()
        fired: list[str] = []
        for job in self.jobs:
            if not self.due(job, now):
                continue
            # A chave de dedup carrega o instante: reiniciar o daemon várias vezes
            # na mesma janela não empilha o mesmo trabalho.
            self.queue.enqueue(
                job.task_kind,
                job.payload or {},
                priority=job.priority,
                dedup_key=f"{job.name}:{iso(now)[:13]}",
                origin="scheduled",
            )
            self._mark_run(job.name, now)
            fired.append(job.name)
        if fired:
            log.info("scheduler disparou: %s", ", ".join(fired))
        return fired

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.tick)
            except Exception:
                log.exception("tick do scheduler falhou")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.tick_s)
            except TimeoutError:
                pass
