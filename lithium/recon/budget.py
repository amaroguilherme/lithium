"""Teto de chamadas, DURÁVEL. É engenharia, não config opcional.

A chave de busca web tem FATURA. Um daemon 24/7 que enfileira as próprias buscas pode
queimar cota sozinho, e em uso pessoal NÃO HÁ NINGUÉM olhando um dashboard.

Três furos independentes que se somam, e só um contador durável fecha os três:

* `RateLimiter` guarda `_tokens` em atributo de INSTÂNCIA e a instância morre com o
  processo. Sob launchd, um daemon que reinicia em laço zera a cota a cada subida —
  exatamente o cenário em que ela é a única proteção.
* `mode off` só barra `origin='scheduled'`, e não existe `lithium cancel`. O teto não
  pode depender do modo.
* `queue.fail` repete até 3 vezes com backoff. O fatiamento em handlers separados evita
  que um retry de leitura refature a busca, mas o débito PRÉ-requisição é a defesa que
  sobra quando a falha é dentro da própria chamada faturada.

**UPDATE condicional com RETURNING, nunca SELECT-depois-UPDATE**: dois workers
concorrentes leriam o mesmo saldo, que é o mesmo erro que `RateLimiter._lock` existe
para evitar.

**Debitado ANTES da requisição e NUNCA estornado**: uma falha DEPOIS de o provedor
responder (parse, timeout de leitura) gasta cota que um contador de sucessos
registraria como zero.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

log = logging.getLogger(__name__)

COLUMNS = ("search_calls", "page_fetches", "robots_fetches")


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _debit(store, column: str, cap: int) -> bool:
    if cap <= 0:
        # O ramo de INSERT do UPSERT não tem WHERE, então com `cap = 0` a PRIMEIRA
        # chamada do dia seria debitada e EXECUTADA assim mesmo (MEDIDO em sqlite
        # 3.51.2: a linha nasce com 1). "Teto 0" significando "uma por dia" é a
        # diferença entre uma torneira e um pingo — e `max_calls_per_day = 0` é o gesto
        # óbvio de "pare de gastar agora, sem editar código".
        return False
    row = store.conn.execute(
        f"INSERT INTO recon_budget(day, {column}) VALUES(?, 1) "
        f"ON CONFLICT(day) DO UPDATE SET {column} = {column} + 1 "
        f"  WHERE {column} < ? "
        f"RETURNING {column}",
        (today(), cap),
    ).fetchone()
    return row is not None


def debit_search(store, cap: int) -> bool:
    """A única coluna com FATURA. Debita e devolve False quando o teto já foi atingido."""
    ok = _debit(store, "search_calls", cap)
    if not ok:
        log.info("teto diário de buscas web atingido (%d); nada foi requisitado", cap)
    return ok


def debit_page(store, cap: int) -> bool:
    ok = _debit(store, "page_fetches", cap)
    if not ok:
        log.info("teto diário de leituras atingido (%d)", cap)
    return ok


def debit_robots(store, cap: int) -> bool:
    """robots.txt não é faturado, mas é acesso automatizado a terceiros.

    Coluna PRÓPRIA. Conflatá-lo com `page_fetches` faria o teto de leitura ser
    consumido por requisições que não leram nada — com o cache frio (o estado a cada
    subida sob launchd) e 8 hosts distintos, `lithium status` mostraria "leituras hoje:
    15" com 7 páginas lidas.
    """
    return _debit(store, "robots_fetches", cap)


def spent(store) -> dict[str, int]:
    """Consumo de hoje e do mês, para `lithium status` e o rodapé de `discoveries`."""
    day = today()
    row = store.conn.execute(
        "SELECT search_calls, page_fetches, robots_fetches FROM recon_budget "
        " WHERE day = ?", (day,),
    ).fetchone()
    month = store.conn.execute(
        "SELECT COALESCE(SUM(search_calls), 0) AS s, "
        "       COALESCE(SUM(page_fetches), 0) AS p "
        "  FROM recon_budget WHERE day LIKE ?", (day[:7] + "%",),
    ).fetchone()
    return {
        "search_calls": int(row["search_calls"]) if row else 0,
        "page_fetches": int(row["page_fetches"]) if row else 0,
        "robots_fetches": int(row["robots_fetches"]) if row else 0,
        "month_search_calls": int(month["s"]),
        "month_page_fetches": int(month["p"]),
    }
