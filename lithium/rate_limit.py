"""Token bucket assíncrono, um por host.

O NCBI bloqueia IP que estoura 3 req/s (10 com API key). Como o daemon é feito para
rodar sem supervisão por horas, levar um ban não é hipótese remota — é o resultado
esperado de não ter isso aqui.

**Fora de `lithium/sources/` desde a Fase C**, e a mudança de casa é estrutural: o
batedor da web (`lithium/recon/`) também precisa de cortesia por host, e o teste de AST
que impede o batedor de tocar no canal de evidência proíbe a palavra `sources` em
qualquer arquivo daquele pacote — inclusive num `import`. Um limitador de vazão não é
uma fonte de evidência; ele só estava morando lá.

**Não serve de teto de cota.** `_tokens` é estado de INSTÂNCIA e morre com o processo:
sob launchd, um daemon que reinicia em laço zeraria a cota a cada subida. Quem conta
dinheiro é `recon_budget`, no banco.
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
