"""Busca na web aberta, atrás de um Protocol de um método só.

**Nada de scraping de motor de busca.** O provedor é uma API com ToS explícito para uso
programático (Brave Search API), e a credencial vai em HEADER.

Header e não query param é mitigação ESTRUTURAL, não estilo. MEDIDO com o eutils real:
com a chave em query param, `str(httpx.HTTPStatusError)` a carrega e `Runner._execute`
a grava em `tasks.error` — ou seja a chave grátis da NCBI já está no banco de qualquer
usuário que tenha levado um 429. Com `X-Subscription-Token`, não vaza. `redact_secrets`
é a segunda camada; esta é a primeira.

**O Protocol tem UM método, com nome DIFERENTE de `search`/`fetch`, e a classe NÃO tem
atributo `kind`.** Isso não é gosto: registrar o buscador em `Context.sources` — a linha
única de menor resistência em `daemon.py` — passa a produzir `AttributeError` no ato, em
vez de converter a web em fonte de evidência. A trava mora no módulo do batedor, não na
lembrança de quem escreve o daemon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from lithium.rate_limit import RateLimiter

log = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

SEARCH_TIMEOUT_S = 20.0
"""Busca é interativa; não é um efetch de 20 registros a 60 s."""

RESULTS_PER_QUERY = 10
"""E não 20: o custo de triagem é linear no número de resultados, e 10 já produz mais
leads do que o teto de leituras do dia consegue ler."""

FRESHNESS = "pm"
"""Últimos 31 dias. Uma varredura DIÁRIA sobre a web inteira reencontra as mesmas
páginas; uma semana é estreita demais para a cadência de publicação biomédica e
produziria varreduras vazias."""


class ReconDisabled(RuntimeError):
    """O provedor de busca não está configurado. Fail-loud na CONSTRUÇÃO.

    Nunca `return []`. MEDIDO antes desta fase: `grep -rn '\\.enabled' lithium tests`
    devolvia ZERO ocorrências — `SourceConfig.enabled` e `NotifyConfig.enabled` são
    config decorativa. Com `return []`, o recon "roda" e não descobre nada, para
    sempre, em silêncio. Mesmo molde de `MissingModel` e `NoActiveFocus`.
    """


@dataclass(slots=True)
class WebHit:
    """Um resultado de busca, como a API o devolveu.

    A URL vem SEMPRE daqui e NUNCA do modelo — o LLM referencia por índice local. Um
    12B a quem se pede uma URL inventa URL, e uma descoberta com URL alucinada é uma
    página que você abre e não existe, ou pior, existe e é outra coisa.
    """

    title: str
    url: str
    description: str = ""
    extra_snippets: list[str] = field(default_factory=list)

    @property
    def haystack(self) -> str:
        """Todo o texto que a API devolveu para este hit, junto.

        É contra ISTO que `leads.py` procura um identificador — nunca contra a saída do
        modelo.
        """
        return " ".join([self.url, self.title, self.description, *self.extra_snippets])


class WebSearcher(Protocol):
    """UM método. Trocar de provedor é trocar uma classe."""

    async def web_search(self, query: str, *, limit: int) -> list[WebHit]:
        ...


def enable_hint(provider: str = "brave") -> str:
    """A mensagem que diz EXATAMENTE o que fazer. Nomeia arquivo, seção e linhas."""
    return (
        "o batedor da web está DESABILITADO. Para ligar, acrescente ao "
        "config.local.toml (gitignored — a chave nunca vai para o banco nem para o "
        "git):\n"
        "\n"
        "    [recon]\n"
        "    enabled = true\n"
        f'    provider = "{provider}"\n'
        '    api_key = "<sua chave da Brave Search API>"\n'
        '    contact = "seu@email"   # vai no User-Agent; acesso automatizado se '
        "identifica\n"
        "\n"
        "a chave sai de https://api-dashboard.search.brave.com (tier grátis ≈ 1.000 "
        "buscas/mês). Depois: `lithium recon --now` faz uma varredura na hora."
    )


class BraveSearch:
    """Brave Search API. Trocável em uma linha — o contrato é `web_search`.

    Molde de `PubMedSource`: cliente injetável, `_owns_client`, `aclose()`. É o que
    permite o teste rodar com `httpx.MockTransport` e ZERO rede.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        client: httpx.AsyncClient | None = None,
        rate_per_s: float = 1.0,
    ) -> None:
        if not getattr(cfg, "enabled", False):
            raise ReconDisabled(enable_hint(getattr(cfg, "provider", "brave")))
        if not getattr(cfg, "api_key", None):
            raise ReconDisabled(
                "[recon] enabled = true, mas `api_key` não está definida. "
                + enable_hint(getattr(cfg, "provider", "brave"))
            )
        if not getattr(cfg, "contact", None):
            raise ReconDisabled(
                "[recon] enabled = true, mas `contact` não está definida. Acesso "
                "automatizado a sites de terceiros precisa se identificar — uso "
                "pessoal não relaxa ToS (PLAN.md §7). "
                + enable_hint(getattr(cfg, "provider", "brave"))
            )
        self._api_key = str(cfg.api_key)
        self.contact = str(cfg.contact)
        self.limiter = RateLimiter(rate_per_s)
        self._client = client or httpx.AsyncClient(timeout=SEARCH_TIMEOUT_S)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def web_search(self, query: str, *, limit: int = RESULTS_PER_QUERY
                         ) -> list[WebHit]:
        await self.limiter.acquire()
        response = await self._client.get(
            BRAVE_ENDPOINT,
            params={"q": query, "count": limit, "freshness": FRESHNESS},
            headers={
                "Accept": "application/json",
                # HEADER. Ver o docstring do módulo.
                "X-Subscription-Token": self._api_key,
            },
        )
        response.raise_for_status()
        return parse_brave(response.json(), limit=limit)


def parse_brave(payload: dict[str, Any], *, limit: int) -> list[WebHit]:
    """Extrai os hits do JSON da Brave. Tolerante a chave ausente, de propósito.

    Um provedor que muda a forma da resposta não pode derrubar a varredura com
    `KeyError` — o custo da busca JÁ FOI FATURADO quando este código roda.
    """
    results = ((payload or {}).get("web") or {}).get("results") or []
    hits: list[WebHit] = []
    for raw in results[:limit]:
        url = (raw.get("url") or "").strip()
        if not url:
            continue
        hits.append(
            WebHit(
                title=(raw.get("title") or "").strip(),
                url=url,
                description=(raw.get("description") or "").strip(),
                extra_snippets=[
                    s.strip() for s in (raw.get("extra_snippets") or []) if s
                ],
            )
        )
    return hits
