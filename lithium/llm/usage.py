"""Contabilidade de custo de inferência.

Uma linha por **POST HTTP**, não por chamada lógica: `structured()` permite dois
reparos, e cada reparo anexa a saída ruim mais o erro à conversa, então a terceira
tentativa carrega o prompt inteiro mais dois rounds falhos. Contar por chamada lógica
esconderia justamente o custo que mais dói.

**Por que o registro é assíncrono e engole exceção.** Os handlers rodam awaited na
thread do event loop (`worker/runner.py`), e o `Store` é síncrono. Um INSERT direto
ali pode bloquear o loop até o `busy_timeout` de 30 s. Pior: uma exceção no caminho
de medição converteria uma resposta de LLM **bem-sucedida** em falha de task e retry —
o medidor de tokens causando gasto de token. Então o registro só enfileira em memória,
o flush sai por `asyncio.to_thread`, e tudo é engolido com log em debug.

Perder uma linha de telemetria é irrelevante. Perder uma resposta de LLM de 60 s por
causa da telemetria não é.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from lithium.db import Store

log = logging.getLogger(__name__)

FLUSH_EVERY = 20


@dataclass(slots=True)
class CallRecord:
    """Um POST HTTP ao llama-server."""

    label: str
    attempt: int
    elapsed_ms: int
    ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class UsageSink:
    """Buffer em memória com flush assíncrono. Nunca levanta."""

    def __init__(self, store: Store, *, flush_every: int = FLUSH_EVERY) -> None:
        self.store = store
        self.flush_every = flush_every
        self._buffer: list[CallRecord] = []
        self._flushing = False

    def record(self, entry: CallRecord) -> None:
        """Ponto de entrada do cliente LLM. Síncrono, barato, e nunca levanta."""
        try:
            self._buffer.append(entry)
            if len(self._buffer) >= self.flush_every and not self._flushing:
                self._schedule_flush()
        except Exception:  # noqa: BLE001
            log.debug("registro de uso falhou", exc_info=True)

    def _schedule_flush(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # fora de loop (testes síncronos): fica no buffer até o flush manual
        self._flushing = True
        task = loop.create_task(self.flush())
        # Sem isso, uma exceção na task viraria "Task exception was never retrieved".
        task.add_done_callback(lambda t: t.exception())

    async def flush(self) -> int:
        """Drena o buffer para o banco. Devolve quantas linhas gravou."""
        pending, self._buffer = self._buffer, []
        if not pending:
            self._flushing = False
            return 0
        try:
            written = await asyncio.to_thread(self._write, pending)
        except Exception:  # noqa: BLE001
            log.debug("flush de uso falhou; %d linhas descartadas", len(pending),
                      exc_info=True)
            written = 0
        finally:
            self._flushing = False
        return written

    def _write(self, entries: list[CallRecord]) -> int:
        self.store.conn.executemany(
            "INSERT INTO llm_calls(label, attempt, prompt_tokens, completion_tokens, "
            "                      elapsed_ms, finish_reason, ok) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            [
                (e.label, e.attempt, e.prompt_tokens, e.completion_tokens,
                 e.elapsed_ms, e.finish_reason, int(e.ok))
                for e in entries
            ],
        )
        return len(entries)


def summarize(store: Store, *, since_days: int | None = None) -> list[dict]:
    """Agregado por label, para `lithium tokens`.

    p50/p95 via `NTILE` seria mais elegante, mas SQLite não tem função de percentil;
    calculamos com uma subquery de offset, que é exata e legível.
    """
    where = ""
    params: tuple = ()
    if since_days is not None:
        where = "WHERE created_at >= datetime('now', ?)"
        params = (f"-{since_days} days",)

    rows = store.conn.execute(
        f"""
        SELECT label,
               COUNT(*)                                   AS n,
               SUM(ok = 0)                                AS failures,
               SUM(attempt > 0)                           AS repairs,
               AVG(prompt_tokens)                         AS avg_prompt,
               MAX(prompt_tokens)                         AS max_prompt,
               AVG(completion_tokens)                     AS avg_completion,
               AVG(elapsed_ms)                            AS avg_ms,
               MAX(elapsed_ms)                            AS max_ms,
               -- SUM, não só AVG/MAX. "Quanto tempo a re-lente levou NO TOTAL" é O
               -- número desta fase, e ele não saía de `lithium tokens`: existia média
               -- e máximo de `elapsed_ms`, nunca a soma. Entregar a otimização sem o
               -- número que a justificava é o que a disciplina deste repo recusa — e
               -- a instrumentação não precisou de tabela nem coluna nova, só do label
               -- `judge_directness` e deste SUM.
               SUM(COALESCE(elapsed_ms, 0))               AS total_ms,
               SUM(COALESCE(prompt_tokens, 0)
                 + COALESCE(completion_tokens, 0))        AS total_tokens
          FROM llm_calls {where}
         GROUP BY label ORDER BY total_tokens DESC
        """,
        params,
    ).fetchall()

    out: list[dict] = []
    for row in rows:
        record = dict(row)
        record["p50_prompt"] = _percentile(store, row["label"], "prompt_tokens", 0.50)
        record["p95_prompt"] = _percentile(store, row["label"], "prompt_tokens", 0.95)
        record["p50_ms"] = _percentile(store, row["label"], "elapsed_ms", 0.50)
        out.append(record)
    return out


def _percentile(store: Store, label: str, column: str, q: float) -> int | None:
    row = store.conn.execute(
        f"SELECT {column} AS v FROM llm_calls "
        f" WHERE label = ? AND {column} IS NOT NULL "
        f" ORDER BY {column} "
        f" LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * ? AS INTEGER) FROM llm_calls "
        f"                  WHERE label = ? AND {column} IS NOT NULL)",
        (label, q, label),
    ).fetchone()
    return int(row["v"]) if row and row["v"] is not None else None
