"""URL → texto, com `html.parser` da stdlib e SEIS portões antes de gastar token.

**Sem dependência nova.** O repo recusa dependência facilmente evitável (usa `tomllib`
em vez de PyYAML); `bs4`/`lxml`/`trafilatura` seriam três, para um extrator de 60
linhas.

Todos os números aqui são DERIVADOS de páginas reais baixadas para
`tests/fixtures/recon/`, não escolhidos. Escrever o código antes de gravar a fixture
inverte a ordem e produz constante chutada — e o docstring de `test_sources_pubmed.py`
já registra por quê: "testar contra XML inventado esconde exatamente os casos que
quebram na prática". Medido: 1 de 4 páginas reais era muro de robô, e ela era do NCBI,
o domínio mais relevante deste projeto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from lithium.rate_limit import RateLimiter

log = logging.getLogger(__name__)

DROP_TAGS = frozenset({
    "script", "style", "noscript", "template", "svg", "iframe", "form",
    "nav", "header", "footer", "aside", "button", "select",
})

BLOCK_TAGS = frozenset({
    "p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3",
    "h4", "h5", "h6", "blockquote", "td", "th", "dd", "dt", "pre",
})

MIN_LINE_CHARS = 60
"""Linhas mais curtas que isto são cromo de navegação, não conteúdo.

MEDIDO: sem este filtro a página do NCBI entrega "Skip to main content / Log in / NLM /
NIH / HHS / USA.gov / Back to Top" junto do conteúdo, e a fração de âncora de uma
página de wiki cai de ~58% para ~0% com ele."""

MIN_CONTENT_CHARS = 1_000
"""Piso de conteúdo. Abaixo disso a página não foi lida — foi barrada.

MEDIDO nas fixtures reais: o muro de robô do pubmed extrai 76 chars; a menor página
real extrai 4.779. Qualquer piso entre 300 e 800 já os separa; 1.000 dá folga sem
alcançar nenhuma página real medida.

Sem ele o sistema gera uma `observation` cujo conteúdo é "Enable cookies for
pubmed.ncbi.nlm.nih.gov", te pergunta se pode memorizar, e o regime de confirmação que
existe para proteger o que ele passa a acreditar é gasto num muro de robô."""

MAX_BODY_BYTES = 2_000_000
"""Teto de corpo, por streaming. `httpx.AsyncClient` NÃO limita tamanho de resposta, e
uma página de wiki de fármaco mede 1,18 MB."""

INTERSTITIAL_TITLE_MARKS = (
    "cookies must be enabled", "enable cookies", "are you a robot",
    "just a moment", "access denied", "attention required",
    "checking your browser", "verify you are human", "captcha",
)
"""Assinaturas de interstitial, procuradas no título E no começo do texto.

As DUAS superfícies, e a razão é medida na fixture real: o muro de robô do PubMed tem
`<title>pubmed.ncbi.nlm.nih.gov</title>` — inocente — e o corpo diz "Enable cookies for
pubmed.ncbi.nlm.nih.gov and reload this page to continue". Um checador só de título
deixaria passar exatamente a página que motivou o portão.

Complementa o piso de conteúdo em vez de duplicá-lo: uma página de verificação COM
bastante texto (um Cloudflare com rodapé grande) passa pelo piso e cai aqui."""

INTERSTITIAL_PROBE_CHARS = 300
"""Quanto do texto extraído entra na busca de assinatura. Curto de propósito: um artigo
que MENCIONA CAPTCHA no meio do corpo não pode ser recusado por isso."""

EVIDENCE_DOMAINS = frozenset({
    "pubmed.ncbi.nlm.nih.gov",
    "eutils.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
})
"""A denylist ESTRUTURAL: os domínios que o canal de evidência já possui.

Não é preferência de qualidade e não é configurável — é a FRONTEIRA entre os dois
canais. MEDIDO: `pubmed.ncbi.nlm.nih.gov/30712879/` devolve HTTP 203 com "Cookies must
be enabled", e mesmo que funcionasse, ler HTML de um artigo que o E-utilities entrega
ESTRUTURADO é errado. O batedor não lê o que o canal de evidência já possui — ele
APONTA para lá, como `lead`."""

DEFAULT_HOST_RATE = 0.5
"""req/s por host quando o robots.txt não declara `Crawl-delay`."""

ROBOTS_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 25.0

OBSERVE_PROMPT_OVERHEAD_CHARS = 1_500
"""Folga para o texto estático de `recon_observe.md` mais o título e a URL."""


def read_max_chars(n_ctx: int, *, max_tokens: int = 256) -> int:
    """Teto de texto de página, DERIVADO da janela em runtime — não constante.

    MEDIDO com páginas reais: a fixture do PMC extrai 156.958 chars (≈39 k tokens) e
    uma página de wiki de fármaco extrai 62 k chars, contra ~7 k tokens disponíveis com
    `n_ctx = 8192`. Ou seja **a maioria das páginas substantivas NÃO CABE** — não é caso
    de borda.

    Truncar em silêncio produziria um resumo confiante de metade de um artigo; recusar
    páginas longas descartaria justamente as boas. A escolha é truncar, gravar
    `truncated=True` no payload, e a superfície renderizar `[parcial]`. A visibilidade é
    o que torna a truncagem honesta.
    """
    from lithium.llm.prompts import BUDGET_MARGIN, CHARS_PER_TOKEN

    usable = max(0, n_ctx - BUDGET_MARGIN - max_tokens)
    return max(2_000, usable * CHARS_PER_TOKEN - OBSERVE_PROMPT_OVERHEAD_CHARS)


def user_agent(contact: str) -> str:
    """UA identificável, com contato. Acesso automatizado se identifica — PLAN.md §7."""
    return f"lithium/0.1 (+personal research assistant; contact: {contact})"


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def is_evidence_domain(url: str) -> bool:
    host = host_of(url)
    return any(host == d or host.endswith("." + d) for d in EVIDENCE_DOMAINS)


# ────────────────────────────────────────────────────────────────── extração


class _Extractor(HTMLParser):
    """Coletor de texto. `convert_charrefs=True` (o default) resolve entidades."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppress = 0
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in DROP_TAGS:
            self._suppress += 1
        elif tag == "title":
            self._in_title = True
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in DROP_TAGS:
            self._suppress = max(0, self._suppress - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._suppress:
            return
        self.parts.append(data)


def extract_text(html: str) -> tuple[str, str]:
    """Devolve `(titulo, texto)`. O texto já vem sem cromo de navegação."""
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - HTML real é malformado o tempo todo
        log.debug("parser de HTML engasgou; usando o que já foi coletado")
    title = " ".join("".join(parser.title_parts).split())
    raw = "".join(parser.parts)
    lines = [" ".join(line.split()) for line in raw.split("\n")]
    kept = [line for line in lines if len(line) >= MIN_LINE_CHARS]
    return title, "\n".join(kept)


# ───────────────────────────────────────────────────────────────────── robots


@dataclass(slots=True)
class HostPolicy:
    parser: RobotFileParser | None
    limiter: RateLimiter
    allow_all: bool = False
    deny_all: bool = False

    def can_fetch(self, agent: str, url: str) -> bool:
        if self.deny_all:
            return False
        if self.allow_all or self.parser is None:
            return True
        return bool(self.parser.can_fetch(agent, url))


class RobotsCache:
    """Um `RobotFileParser` e um `RateLimiter` por host.

    **`.parse(texto.splitlines())`, NUNCA `.read()`.** VERIFICADO: `.read()` faz
    `urlopen` BLOQUEANTE, e dentro do event loop do worker isso trava o loop inteiro
    por um RTT, por host. Mesma disciplina que `llm/usage.py` documenta.

    **O status HTTP é tratado ANTES do parse.** `RobotFileParser.parse()` não conhece
    código HTTP: alimentado com corpo vazio ou HTML de erro, ele devolve
    `can_fetch() == True` para TUDO. A RFC 9309 §2.3.1.4 exige que 5xx seja tratado
    como disallow completo — e um domínio sob carga que devolve 503 no /robots.txt
    passaria a ser lido inteiro, com um User-Agent que carrega o seu e-mail.
    """

    def __init__(self, client: httpx.AsyncClient, *, agent: str,
                 budget=None) -> None:
        self._client = client
        self._agent = agent
        self._budget = budget
        self._hosts: dict[str, HostPolicy] = {}

    async def policy(self, url: str) -> HostPolicy:
        host = host_of(url)
        cached = self._hosts.get(host)
        if cached is not None:
            return cached
        policy = await self._fetch_policy(url, host)
        self._hosts[host] = policy
        return policy

    async def _fetch_policy(self, url: str, host: str) -> HostPolicy:
        if self._budget is not None and not self._budget():
            # Sem cota para nem consultar o robots.txt: recusa. Ler sem consultar
            # seria trocar um teto por uma violação de ToS.
            log.info("cota de robots.txt esgotada; %s fica de fora hoje", host)
            return HostPolicy(None, RateLimiter(DEFAULT_HOST_RATE), deny_all=True)
        base = f"{urlsplit(url).scheme or 'https'}://{host}"
        try:
            response = await self._client.get(
                f"{base}/robots.txt",
                headers={"User-Agent": self._agent},
                timeout=ROBOTS_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            log.info("robots.txt de %s indisponível (%s): host recusado", host, exc)
            return HostPolicy(None, RateLimiter(DEFAULT_HOST_RATE), deny_all=True)

        if response.status_code in (404, 410):
            # Ausente = sem restrição. É o que a RFC diz.
            return HostPolicy(None, RateLimiter(DEFAULT_HOST_RATE), allow_all=True)
        if response.status_code != 200:
            log.info("robots.txt de %s devolveu HTTP %s: host recusado",
                     host, response.status_code)
            return HostPolicy(None, RateLimiter(DEFAULT_HOST_RATE), deny_all=True)

        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        delay = _crawl_delay(response.text)
        rate = (1.0 / delay) if delay and delay > 0 else DEFAULT_HOST_RATE
        return HostPolicy(parser, RateLimiter(min(rate, 2.0)))


def _crawl_delay(text: str) -> float | None:
    """`RobotFileParser.crawl_delay` exige `User-agent` casado e é chato de acertar
    offline; ler a diretiva direto do texto é honesto e testável contra a fixture."""
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name.strip().lower() == "crawl-delay":
            try:
                return float(value.strip())
            except ValueError:
                return None
    return None


# ───────────────────────────────────────────────────────────────── a leitura


class PageRefused(RuntimeError):
    """A página não foi lida, e o motivo é nomeado. NÃO é falha de tarefa."""


@dataclass(slots=True)
class PageText:
    url: str
    title: str
    text: str
    truncated: bool = False
    http_status: int = 200
    extras: dict = field(default_factory=dict)


class PageReader:
    """Os SEIS portões, na ordem em que custam menos.

    1. domínio na denylist do canal de evidência
    2. robots `can_fetch` (com o status HTTP tratado)
    3. `raise_for_status()`  — um 404 com corpo de 155 KB extrai 2.904 chars de página
       de erro, e SÓ o código pega isso
    4. teto de corpo por streaming
    5. piso de conteúdo extraído
    6. assinatura de interstitial no título
    """

    def __init__(self, client: httpx.AsyncClient, *, contact: str,
                 robots: RobotsCache | None = None, respect_robots: bool = True,
                 robots_budget=None) -> None:
        self.agent = user_agent(contact)
        self._client = client
        self.respect_robots = respect_robots
        self.robots = robots or RobotsCache(client, agent=self.agent,
                                            budget=robots_budget)

    async def read(self, url: str, *, max_chars: int) -> PageText:
        if is_evidence_domain(url):
            raise PageRefused(
                f"{host_of(url)} pertence ao canal de evidência: um artigo daí entra "
                "como `lead` e é recolhido pelo E-utilities, estruturado. O batedor "
                "não lê o que o canal de evidência já possui."
            )

        policy = await self.robots.policy(url)
        if self.respect_robots and not policy.can_fetch(self.agent, url):
            raise PageRefused(f"robots.txt de {host_of(url)} proíbe {url}")
        await policy.limiter.acquire()

        body, status = await self._download(url)
        title, text = extract_text(body)

        probe = (title + " " + text[:INTERSTITIAL_PROBE_CHARS]).lower()
        if any(mark in probe for mark in INTERSTITIAL_TITLE_MARKS):
            raise PageRefused(f"interstitial de robô em {url} (título: {title[:60]!r})")
        if len(text) < MIN_CONTENT_CHARS:
            raise PageRefused(
                f"{url} rendeu {len(text)} chars extraídos, abaixo do piso de "
                f"{MIN_CONTENT_CHARS} — muro de robô, paywall ou página vazia"
            )

        truncated = len(text) > max_chars
        return PageText(
            url=url,
            title=title,
            text=text[:max_chars],
            truncated=truncated,
            http_status=status,
            extras={"host": host_of(url), "chars": len(text)},
        )

    async def _download(self, url: str) -> tuple[str, int]:
        buf: list[bytes] = []
        size = 0
        async with self._client.stream(
            "GET", url,
            headers={"User-Agent": self.agent, "Accept": "text/html,*/*"},
            timeout=READ_TIMEOUT_S,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            async for piece in response.aiter_bytes():
                buf.append(piece)
                size += len(piece)
                if size >= MAX_BODY_BYTES:
                    log.info("corpo de %s passou de %d B; truncado no download",
                             url, MAX_BODY_BYTES)
                    break
            status = response.status_code
        return b"".join(buf).decode("utf-8", errors="replace"), status
