"""O leitor: os seis portões, contra páginas REAIS baixadas com UA identificável.

Todas as constantes de `read.py` são DERIVADAS destas fixtures, não escolhidas — e a
medição que decidiu o piso de conteúdo está no docstring de cada teste. 1 de 4 páginas
reais baixadas era muro de robô, e ela era do NCBI: o domínio mais relevante deste
projeto.

ZERO rede: `httpx.MockTransport` servindo os bytes gravados.
"""

from __future__ import annotations

import ast

from pathlib import Path

import pytest

from lithium.recon.read import (
    MIN_CONTENT_CHARS,
    PageRefused,
    extract_text,
    is_evidence_domain,
    read_max_chars,
)

from reconkit import COOKIEWALL_HTML, NICE_HTML, PMC_HTML, ROBOTS_PMC, reader_for

RECON_DIR = Path(__file__).resolve().parent.parent / "lithium" / "recon"


# ═══════════════════════════ o piso de conteúdo, calibrado nas páginas reais


def test_the_content_floor_separates_a_bot_wall_from_the_smallest_real_page():
    """A MEDIÇÃO que fixa `MIN_CONTENT_CHARS`, refeita a cada execução.

    Baixados hoje com `User-Agent: lithium/0.1 (+…; contact: …)`:

        pubmed_cookiewall.html   HTTP 203, 5.565 B  →      76 chars extraídos
        nice_cg185.html          HTTP 200, 25.923 B →   4.743 chars extraídos
        pmc_article.html         HTTP 200, 655.300 B → 156.958 chars extraídos

    O piso fica 13x acima do muro de robô e 4,7x abaixo da menor página real. Qualquer
    valor entre 300 e 800 já os separaria; 1.000 dá folga sem alcançar nenhuma página
    real medida.

    MUTAÇÃO: `MIN_CONTENT_CHARS = 50`. Este teste fica vermelho, e
    `test_the_reader_refuses_a_bot_wall` também.
    """
    _, wall = extract_text(COOKIEWALL_HTML)
    _, nice = extract_text(NICE_HTML)
    _, pmc = extract_text(PMC_HTML)

    assert len(wall) < MIN_CONTENT_CHARS < len(nice) <= len(pmc), (
        f"muro={len(wall)} piso={MIN_CONTENT_CHARS} nice={len(nice)} pmc={len(pmc)}"
    )
    assert len(wall) < 200, "o muro de robô deixou de ser reconhecivelmente vazio"


def test_navigation_chrome_does_not_survive_extraction():
    """Sem o filtro de 60 chars, a página do NCBI entrega "Skip to main content / Log
    in / NLM / NIH / HHS / USA.gov / Back to Top" junto do conteúdo, e o resumo do
    modelo passa a descrever o menu."""
    _, text = extract_text(NICE_HTML)
    for chrome in ("Skip to", "Back to Top", "USA.gov"):
        assert chrome not in text, f"cromo de navegação sobreviveu: {chrome!r}"
    assert "bipolar" in text.lower(), "o conteúdo real precisa sobreviver"


def test_script_and_style_never_reach_the_text():
    _, text = extract_text(
        "<html><head><style>body{color:red}</style>"
        "<script>var segredo = 'Zmyrfkq';</script></head>"
        "<body><p>" + "conteudo real com bastante texto para passar do filtro. " * 40
        + "</p></body></html>"
    )
    assert "Zmyrfkq" not in text and "color:red" not in text


# ══════════════════════════════════════════════════ os portões, ponta a ponta


async def test_the_reader_refuses_a_bot_wall(caplog):
    """O muro de robô real do PubMed, servido de fixture. Recusado por DOIS portões.

    Sem isto o sistema gera uma `observation` cujo conteúdo é "Enable cookies for
    pubmed.ncbi.nlm.nih.gov", te pergunta se pode memorizar, e o regime de confirmação
    que existe para proteger o que ele passa a acreditar é gasto num muro de robô.
    """
    url = "https://wall.invalid/x"
    reader = reader_for({url: (203, COOKIEWALL_HTML)},
                        robots={"https://wall.invalid/robots.txt": (404, "")})
    with pytest.raises(PageRefused):
        await reader.read(url, max_chars=20_000)

    # E as duas metades do portão pegam sozinhas: com o piso rebaixado a 10, a
    # assinatura de interstitial ainda recusa.
    import lithium.recon.read as read_mod

    original = read_mod.MIN_CONTENT_CHARS
    read_mod.MIN_CONTENT_CHARS = 10
    try:
        with pytest.raises(PageRefused, match="interstitial"):
            await reader.read(url, max_chars=20_000)
    finally:
        read_mod.MIN_CONTENT_CHARS = original


async def test_the_reader_refuses_a_soft_404_with_a_big_body():
    """Um 404 com corpo grande extrai milhares de chars de PÁGINA DE ERRO, e nenhum
    portão de conteúdo pega isso — só o CÓDIGO HTTP."""
    url = "https://erro.invalid/artigo"
    body = "<html><title>Page not found</title><body><p>" + (
        "esta pagina nao existe mas o rodape do site e enorme e tem muito texto. " * 60
    ) + "</p></body></html>"
    reader = reader_for({url: (404, body)},
                        robots={"https://erro.invalid/robots.txt": (404, "")})
    _, extracted = extract_text(body)
    assert len(extracted) > MIN_CONTENT_CHARS, (
        "a fixture precisa passar do piso, senão o teste não exercita o raise_for_status"
    )

    import httpx

    with pytest.raises(httpx.HTTPStatusError):
        await reader.read(url, max_chars=20_000)


async def test_a_page_that_does_not_fit_is_truncated_and_says_so():
    """MEDIDO: o artigo do PMC extrai 156.958 chars (≈39 k tokens) contra ~6,4 k
    disponíveis com `n_ctx = 8192`. A MAIORIA das páginas substantivas não cabe — não é
    caso de borda. Truncar em silêncio produziria um resumo confiante de metade de um
    artigo."""
    url = "https://pmc.invalid/articles/PMC1"
    reader = reader_for({url: (200, PMC_HTML)},
                        robots={"https://pmc.invalid/robots.txt": (404, "")})
    cap = read_max_chars(8192)
    page = await reader.read(url, max_chars=cap)

    assert page.truncated is True
    assert len(page.text) == cap
    assert page.extras["chars"] > cap * 3, "a fixture precisa ser bem maior que o teto"

    small = await reader_for(
        {"https://nice.invalid/g": (200, NICE_HTML)},
        robots={"https://nice.invalid/robots.txt": (404, "")},
    ).read("https://nice.invalid/g", max_chars=cap)
    assert small.truncated is False, "a página que cabe não pode ser marcada parcial"


def test_the_read_budget_is_derived_from_the_window_not_constant():
    """MUTAÇÃO: `read_max_chars` virar constante. Com `n_ctx=16384` o teto tem de
    crescer, senão metade da janela do usuário fica ociosa."""
    assert read_max_chars(16384) > read_max_chars(8192) > 2_000


# ══════════════════════════════════ a fronteira com o canal de evidência


@pytest.mark.parametrize("url", [
    "https://pubmed.ncbi.nlm.nih.gov/30712879/",
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id=1",
])
async def test_the_reader_never_touches_a_domain_the_evidence_channel_owns(url):
    """ZERO requisições, nem a de robots.txt.

    MEDIDO: `pubmed.ncbi.nlm.nih.gov/30712879/` devolve HTTP 203 "Cookies must be
    enabled" — mas mesmo que funcionasse, ler HTML de um artigo que o E-utilities
    entrega ESTRUTURADO é o batedor invadindo o canal que ele existe para alimentar.

    MUTAÇÃO: esvaziar `EVIDENCE_DOMAINS`. `requests` deixa de ser vazio e o teste cai.
    """
    requests: list[str] = []
    reader = reader_for({url: (200, NICE_HTML)}, log=requests)
    with pytest.raises(PageRefused, match="canal de evid"):
        await reader.read(url, max_chars=20_000)
    assert requests == [], f"o batedor bateu no canal de evidência: {requests}"


def test_the_denylist_is_about_domains_not_paths():
    assert is_evidence_domain("https://pubmed.ncbi.nlm.nih.gov/qualquer/coisa")
    assert not is_evidence_domain("https://www.nice.org.uk/guidance/cg185")


# ════════════════════════════════════════════════════════════════ robots.txt


async def test_robots_is_honoured_using_the_real_pmc_file():
    """`robots_pmc.txt` real: `Allow: /articles/`, `Disallow: /`, `Crawl-delay: 1`."""
    assert "Crawl-delay: 1" in ROBOTS_PMC and "Allow: /articles/" in ROBOTS_PMC
    allowed = "https://pmcmirror.invalid/articles/PMC1"
    denied = "https://pmcmirror.invalid/outro"
    requests: list[str] = []
    reader = reader_for(
        {allowed: (200, NICE_HTML), denied: (200, NICE_HTML)},
        robots={"https://pmcmirror.invalid/robots.txt": (200, ROBOTS_PMC)},
        log=requests,
    )

    page = await reader.read(allowed, max_chars=20_000)
    assert page.http_status == 200

    with pytest.raises(PageRefused, match="robots"):
        await reader.read(denied, max_chars=20_000)
    assert denied not in requests, "a página proibida foi requisitada mesmo assim"


async def test_the_crawl_delay_of_the_host_seeds_the_rate_limiter():
    """`Crawl-delay: 1` vira 1 req/s para AQUELE host, em vez do default de 0,5."""
    reader = reader_for(
        {"https://pmcmirror.invalid/articles/PMC1": (200, NICE_HTML)},
        robots={"https://pmcmirror.invalid/robots.txt": (200, ROBOTS_PMC)},
    )
    policy = await reader.robots.policy("https://pmcmirror.invalid/articles/PMC1")
    assert policy.limiter.rate == pytest.approx(1.0)

    other = reader_for({}, robots={"https://outro.invalid/robots.txt": (404, "")})
    default = await other.robots.policy("https://outro.invalid/x")
    assert default.limiter.rate == pytest.approx(0.5)


@pytest.mark.parametrize("status", [500, 503, 403])
async def test_a_robots_that_does_not_answer_200_denies_the_whole_host(status):
    """A RFC 9309 §2.3.1.4 exige 5xx = disallow completo, e `RobotFileParser.parse()`
    NÃO conhece código HTTP: alimentado com HTML de erro ou corpo vazio ele devolve
    `can_fetch() == True` para TUDO.

    Cenário concreto: um domínio sob carga devolve 503 no /robots.txt, o cache guarda um
    parser vazio para o resto do processo, e o lithium lê o host inteiro com um
    User-Agent que carrega o e-mail do usuário.

    MUTAÇÃO: chamar `parse()` incondicionalmente. A página passa a ser requisitada e
    `requests` deixa de ter só a linha do robots.
    """
    url = "https://sobcarga.invalid/pagina"
    requests: list[str] = []
    reader = reader_for(
        {url: (200, NICE_HTML)},
        robots={"https://sobcarga.invalid/robots.txt": (status, "<html>erro</html>")},
        log=requests,
    )
    with pytest.raises(PageRefused, match="robots"):
        await reader.read(url, max_chars=20_000)
    assert requests == ["https://sobcarga.invalid/robots.txt"]


async def test_a_robots_that_times_out_denies_the_whole_host():
    import httpx

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("estourou", request=request)

    from lithium.recon.read import PageReader

    reader = PageReader(httpx.AsyncClient(transport=httpx.MockTransport(boom)),
                        contact="c")
    with pytest.raises(PageRefused, match="robots"):
        await reader.read("https://mudo.invalid/x", max_chars=20_000)


async def test_a_missing_robots_allows_the_host():
    """404/410 = sem restrição. É o que a RFC diz, e recusar tudo tornaria o batedor
    inútil na maioria dos sites."""
    url = "https://semrobots.invalid/pagina"
    reader = reader_for({url: (200, NICE_HTML)},
                        robots={"https://semrobots.invalid/robots.txt": (404, "")})
    assert (await reader.read(url, max_chars=20_000)).http_status == 200


def test_no_file_under_recon_calls_the_blocking_robotfileparser_read():
    """`.read()` faz `urlopen` BLOQUEANTE: dentro do event loop do worker ele trava o
    loop inteiro por um RTT, por host. É a mesma disciplina que `llm/usage.py`
    documenta, e um teste sem rede o pega porque `.read()` tentaria sair para a
    internet.

    MUTAÇÃO: trocar `.parse(texto.splitlines())` por `.read()` em `read.py`.
    """
    def zero_arg_reads(source: str) -> list[str]:
        """`.read()` SEM argumento. É a assinatura de `RobotFileParser.read()`, e é o
        que distingue a forma proibida do `reader.read(url, max_chars=…)` legítimo
        deste mesmo pacote."""
        return [
            ast.unparse(n)
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "read" and not n.args and not n.keywords
        ]

    offenders = {
        path.name: found
        for path in RECON_DIR.rglob("*.py")
        if (found := zero_arg_reads(path.read_text(encoding="utf-8")))
    }
    assert not offenders, offenders
    # auto-verificação: o matcher pega a forma proibida e ignora a legítima
    assert zero_arg_reads("rp = RobotFileParser()\nrp.read()") == ["rp.read()"]
    assert zero_arg_reads("await reader.read(url, max_chars=10)") == []
    assert ".parse(" in (RECON_DIR / "read.py").read_text(encoding="utf-8")



async def test_the_robots_budget_is_wired_and_denies_the_host_when_exhausted():
    """FIAÇÃO do terceiro contador. `robots.txt` não é faturado, mas é acesso
    automatizado a terceiros e tem teto próprio.

    MUTAÇÃO: `RobotsCache` ignorar `budget`. Sem esta trava, `debit_robots` existiria
    com ZERO chamadores — a coluna encheria de zeros e `lithium status` reportaria um
    consumo que não é o real.

    E esgotar a cota RECUSA o host em vez de ler sem consultar: trocar um teto por uma
    violação de ToS não é degradação aceitável.
    """
    calls: list[bool] = []
    requests: list[str] = []
    reader = reader_for({"https://x.invalid/p": (200, NICE_HTML)},
                        robots={"https://x.invalid/robots.txt": (404, "")},
                        log=requests,
                        robots_budget=lambda: calls.append(True) or False)
    with pytest.raises(PageRefused, match="robots"):
        await reader.read("https://x.invalid/p", max_chars=20_000)
    assert calls == [True], "o contador de robots.txt não foi consultado"
    assert requests == [], "requisitou sem cota"


async def test_the_daemon_gives_the_reader_a_robots_budget():
    """Pelo construtor REAL do daemon: `debit_robots` precisa ter um chamador."""
    import ast
    from pathlib import Path as _P

    source = (_P(__file__).resolve().parent.parent / "lithium"
              / "daemon.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_build_recon")
    body = ast.unparse(fn)
    assert "robots_budget" in body and "debit_robots" in body
