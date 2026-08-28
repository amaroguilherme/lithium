"""Token bucket assíncrono, um por fonte.

O NCBI bloqueia IP que estoura 3 req/s (10 com API key). Como o daemon é feito para
rodar sem supervisão por horas, levar um ban não é hipótese remota — é o resultado
esperado de não ter isso aqui.
"""

from __future__ import annotations

import asyncio


class RateLimiter:
    def __init__(self, rate_per_s: float, burst: int | None = None) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s precisa ser positivo")
        self.rate = rate_per_s
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_s)))
        self._tokens = self.capacity
        self._updated: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Bloqueia até haver um token. Serializado: sem isto, N corrotinas leem o
        mesmo saldo e passam todas juntas."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._updated is None:
                self._updated = now
            self._tokens = min(
                self.capacity, self._tokens + (now - self._updated) * self.rate
            )
            self._updated = now

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._updated = loop.time()
            else:
                self._tokens -= 1.0
