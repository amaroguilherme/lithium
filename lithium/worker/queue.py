"""Fila de tarefas durável sobre o próprio SQLite.

Sem Redis, sem Celery. Um daemon de pesquisa que roda por horas precisa sobreviver a
`kill -9`, a reboot e a queda de rede no meio de um harvest — e isso é o que uma
tabela transacional já dá. O que ela não dá é broker distribuído, e não precisamos.

**O claim é atômico via `BEGIN IMMEDIATE`.** Sem ele, dois workers fazem o mesmo
SELECT e pegam a mesma linha. `BEGIN IMMEDIATE` toma o lock de escrita antes do
SELECT, então o segundo worker espera e enxerga o estado já atualizado.

**Premissa de daemon único.** `recover_orphans()` devolve para a fila tudo que ficou
em `running`, partindo do princípio de que quem estava executando morreu. Com dois
daemons no mesmo banco, um roubaria tarefas em andamento do outro. É premissa
consciente: a estação de trabalho é uma só.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from lithium.db import Store

log = logging.getLogger(__name__)

_SECRET_RX = re.compile(
    r"((?:api[_-]?key|apikey|key|token|access[_-]?token|subscription[_-]?token"
    r"|password|secret)[\"']?\s*[=:]\s*[\"']?)([^\s&'\"]+)",
    re.I,
)
"""As aspas OPCIONAIS nos dois lados do separador não são zelo: a credencial aparece em
query string (`&api_key=X`), em header (`X-Subscription-Token: X`) e em corpo JSON
(`{"token": "X"}`), e o traceback que vai para `tasks.error` pode carregar qualquer uma
das três."""

REDACTED = "«redigido»"


def redact_secrets(text: str) -> str:
    """Apaga credenciais de um texto antes de ele virar linha de banco ou de log.

    DEFEITO VIVO, reproduzido: `httpx.HTTPStatusError.__str__` carrega a URL COMPLETA
    com query string, `PubMedSource._params` injeta `api_key` como query param, e
    `Runner._execute` monta `f'{tipo}: {exc}\\n{traceback}'` e GRAVA em `tasks.error`.
    Um único 429 do eutils põe a chave da NCBI no banco — e `purge_done` só apaga
    `done`, nunca `dead`, então ela fica lá para sempre.

    A correção da Fase A cobriu o LOGGER do httpx e só ele. Este caminho é
    PERSISTENTE, não é log, e a Fase C traz uma credencial COM FATURA por ele.

    Não substitui a mitigação estrutural (chave em header, ver `BraveSearch`): é a
    segunda camada, para a credencial que já está no formato errado.
    """
    return _SECRET_RX.sub(lambda m: m.group(1) + REDACTED, text)


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    """Mesmo formato que o `strftime` dos defaults do schema, para as comparações
    de `scheduled_at <= ?` ordenarem como texto corretamente."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


@dataclass(slots=True)
class Task:
    id: int
    kind: str
    payload: dict[str, Any]
    attempts: int
    priority: float


class TaskQueue:
    def __init__(self, store: Store, *, max_attempts: int = 3,
                 backoff_base_s: float = 5.0) -> None:
        self.store = store
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s

    # ──────────────────────────────────────────────────────────────── produção

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: float = 0.5,
        dedup_key: str | None = None,
        delay_s: float = 0.0,
        origin: str = "on_demand",
    ) -> int | None:
        """Enfileira. Devolve None se `dedup_key` já estiver na fila.

        A deduplicação importa mais do que parece: o plan tick reenfileira as mesmas
        buscas todo dia, e sem chave o backlog cresce sem limite.
        """
        scheduled = iso(utcnow() + timedelta(seconds=delay_s))
        cur = self.store.conn.execute(
            "INSERT INTO tasks(kind, payload_json, priority, dedup_key, scheduled_at, "
            "                  origin) "
            "VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(dedup_key) DO NOTHING RETURNING id",
            (kind, json.dumps(payload or {}), priority, dedup_key, scheduled, origin),
        )
        row = cur.fetchone()
        return int(row["id"]) if row else None

    # ──────────────────────────────────────────────────────────────── consumo

    def claim(self, *, on_demand_only: bool = False) -> Task | None:
        """Toma a próxima tarefa elegível, atomicamente.

        `on_demand_only` é o modo pesquisa desligado: os workers seguem vivos e
        atendem o que VOCÊ pediu, mas ignoram o que o scheduler enfileirou. Um stop
        global congelaria também o pedido que você acabou de fazer.
        """
        conn = self.store.conn
        filter_sql = " AND origin = 'on_demand'" if on_demand_only else ""
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "UPDATE tasks SET status = 'running', claimed_at = ?, attempts = attempts + 1 "
                " WHERE id = (SELECT id FROM tasks "
                "              WHERE status = 'pending' AND scheduled_at <= ?"
                f"{filter_sql} "
                "              ORDER BY priority DESC, id ASC LIMIT 1) "
                "RETURNING id, kind, payload_json, attempts, priority",
                (iso(utcnow()), iso(utcnow())),
            ).fetchone()
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

        if row is None:
            return None
        return Task(
            id=row["id"],
            kind=row["kind"],
            payload=json.loads(row["payload_json"]),
            attempts=row["attempts"],
            priority=row["priority"],
        )

    def complete(self, task_id: int) -> None:
        self.store.conn.execute(
            "UPDATE tasks SET status = 'done', finished_at = ?, error = NULL WHERE id = ?",
            (iso(utcnow()), task_id),
        )

    def kill(self, task_id: int, error: str) -> None:
        """Manda direto ao dead-letter, sem gastar tentativas.

        Para falhas que repetir não conserta: handler inexistente, payload
        malformado, fonte removida.
        """
        self.store.conn.execute(
            "UPDATE tasks SET status = 'dead', finished_at = ?, error = ? WHERE id = ?",
            (iso(utcnow()), redact_secrets(error)[:2000], task_id),
        )

    def fail(self, task_id: int, error: str) -> str:
        """Reagenda com backoff exponencial, ou manda para dead-letter.

        Devolve o status final, para o chamador logar de forma diferenciada — falha
        transitória é ruído, dead-letter merece atenção.
        """
        row = self.store.conn.execute(
            "SELECT attempts, dedup_key FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return "missing"

        if row["attempts"] >= self.max_attempts:
            self.store.conn.execute(
                "UPDATE tasks SET status = 'dead', finished_at = ?, error = ? WHERE id = ?",
                (iso(utcnow()), redact_secrets(error)[:2000], task_id),
            )
            return "dead"

        delay = self.backoff_base_s * (2 ** (row["attempts"] - 1))
        self.store.conn.execute(
            "UPDATE tasks SET status = 'pending', scheduled_at = ?, error = ?, "
            "                 claimed_at = NULL WHERE id = ?",
            (iso(utcnow() + timedelta(seconds=delay)), redact_secrets(error)[:2000],
             task_id),
        )
        return "pending"

    # ────────────────────────────────────────────────────────────── manutenção

    def recover_orphans(self) -> int:
        """Devolve à fila o que ficou `running` — o daemon anterior morreu no meio.

        Não zera `attempts`: uma tarefa que derruba o processo toda vez precisa
        chegar ao dead-letter em vez de reiniciar em laço para sempre.
        """
        cur = self.store.conn.execute(
            "UPDATE tasks SET status = 'pending', claimed_at = NULL, "
            "                 error = 'órfã: daemon reiniciou' "
            " WHERE status = 'running'"
        )
        n = cur.rowcount
        if n:
            log.info("recuperadas %d tarefas órfãs", n)
        return n

    def purge_done(self, older_than_days: int = 7) -> int:
        cutoff = iso(utcnow() - timedelta(days=older_than_days))
        cur = self.store.conn.execute(
            "DELETE FROM tasks WHERE status = 'done' AND finished_at < ?", (cutoff,)
        )
        return cur.rowcount

    def stats(self) -> dict[str, int]:
        rows = self.store.conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def dead_letters(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.store.conn.execute(
            "SELECT id, kind, attempts, error, finished_at FROM tasks "
            " WHERE status = 'dead' ORDER BY finished_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
