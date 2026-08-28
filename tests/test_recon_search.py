"""O provedor de busca: desabilitado por padrão, e a chave nunca sai do header.

ZERO rede. `httpx.MockTransport` em tudo — o padrão de `test_sources_pubmed.py`.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from lithium.config import Config, ReconConfig
from lithium.recon.search import (
    BRAVE_ENDPOINT,
    BraveSearch,
    ReconDisabled,
    parse_brave,
)

from reconkit import BRAVE_JSON

SECRET = "SEGREDO-BRAVE-123"


def _cfg(**kw) -> ReconConfig:
    base = {"enabled": True, "api_key": SECRET, "contact": "eu@exemplo"}
    return ReconConfig(**{**base, **kw})


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ════════════════════════════ desabilitado por padrão, DE VERDADE


def test_the_provider_is_disabled_by_default_and_says_exactly_where_to_enable():
    """MUTAÇÃO: trocar a exceção por `return []`.

    MEDIDO antes desta fase: `grep -rn '\\.enabled' lithium tests` devolvia ZERO
    ocorrências. `SourceConfig.enabled` e `NotifyConfig.enabled` existem no modelo e
    NINGUÉM os lê — copiar esse padrão entregaria um "desabilitado por padrão" que não
    desabilita nada, e com `return []` o recon "roda" e não descobre nada, para sempre,
    em silêncio.
    """
    assert Config().recon.enabled is False, "o default não pode ligar a torneira"

    with pytest.raises(ReconDisabled) as exc:
        BraveSearch(ReconConfig())
    message = str(exc.value)
    for token in ("config.local.toml", "[recon]", "enabled", "api_key", "contact"):
        assert token in message, f"a mensagem não diz {token!r}"


@pytest.mark.parametrize("missing", ["api_key", "contact"])
def test_enabling_without_a_key_or_a_contact_still_refuses(missing):
    """`contact` é obrigatório porque o UA se identifica: acesso automatizado a
    terceiros sem identificação é o que o PLAN.md §7 recusa — uso pessoal não relaxa
    ToS."""
    with pytest.raises(ReconDisabled, match=missing):
        BraveSearch(_cfg(**{missing: None}))


def test_a_typo_in_the_recon_section_is_not_silently_a_default():
    """`extra='forbid'` só no modelo NOVO. Para uma feature cujo estado normal é
    "desabilitado, falta a chave", `api_kei` virando `api_key=None` em silêncio torna
    *errei o nome* indistinguível de *não configurei*."""
    with pytest.raises(Exception, match="api_kei|extra"):
        ReconConfig(enabled=True, api_kei=SECRET)


def test_an_unknown_top_level_section_is_named_in_a_warning(tmp_path, caplog):
    """`[reccon]` com a chave certa dentro não pode ser silêncio total. E a mensagem
    NOMEIA o arquivo: `load_config` devolve o PRIMEIRO que existe, sem merge, então
    editar o arquivo errado é o modo de falha comum."""
    from lithium.config import load_config

    path = tmp_path / "config.local.toml"
    path.write_text('[reccon]\nenabled = true\napi_key = "x"\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = load_config(path)

    assert cfg.recon.enabled is False
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "reccon" in blob and str(path) in blob


# ═════════════════════════════════ a chave vai por HEADER, nunca na URL


async def test_the_key_travels_in_a_header_and_never_in_the_url():
    """A mitigação ESTRUTURAL do vazamento medido.

    Com a chave em query param, `str(httpx.HTTPStatusError)` a carrega — e
    `Runner._execute` grava `str(exc)` em `tasks.error`, que `purge_done` nunca apaga.
    Com header, o `str` do erro não tem a chave.

    MUTAÇÃO: mandar `api_key` em `params`. As duas asserções abaixo caem.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["header"] = request.headers.get("x-subscription-token")
        return httpx.Response(200, json=BRAVE_JSON)

    searcher = BraveSearch(_cfg(), client=_client(handler), rate_per_s=1000)
    await searcher.web_search("bipolar manutenção", limit=10)

    assert seen["header"] == SECRET
    assert SECRET not in seen["url"], "a chave foi para a query string"
    assert seen["url"].startswith(BRAVE_ENDPOINT)


async def test_a_429_error_string_does_not_carry_the_key():
    """O irmão direto de `test_the_api_key_never_reaches_the_log`, mas sobre a
    EXCEÇÃO — que é o que chega ao banco."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    searcher = BraveSearch(_cfg(), client=_client(handler), rate_per_s=1000)
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await searcher.web_search("q", limit=10)
    assert SECRET not in str(exc.value)


async def test_the_search_is_scoped_to_recent_pages_and_ten_results():
    """`freshness` e `count` são parâmetros de CUSTO, não estilo: uma varredura diária
    sobre a web inteira reencontra as mesmas páginas, e o custo de triagem é linear no
    número de resultados."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=BRAVE_JSON)

    searcher = BraveSearch(_cfg(), client=_client(handler), rate_per_s=1000)
    await searcher.web_search("q", limit=10)
    assert seen["count"] == "10"
    assert seen["freshness"] == "pm"


# ═══════════════════════════════════════════════════════════ o parsing


@pytest.mark.provisional
def test_the_response_is_parsed_into_hits_with_the_api_url():
    """PROVISIONAL: a fixture da Brave é SINTÉTICA.

    O implementador não tem chave (a conta exige cartão) e a disciplina do repo é
    explícita — "testar contra XML inventado esconde exatamente os casos que quebram na
    prática". A FORMA aqui segue a documentação pública. `lithium recon
    --record-fixture` grava a primeira resposta REAL e este marcador sai.
    """
    hits = parse_brave(BRAVE_JSON, limit=10)
    assert len(hits) == 5
    assert hits[0].url == "https://www.nice.org.uk/guidance/cg185"
    assert hits[0].extra_snippets


@pytest.mark.provisional
def test_a_malformed_response_returns_nothing_instead_of_raising():
    """A busca JÁ FOI FATURADA quando este código roda: um `KeyError` aqui converteria
    dinheiro gasto em dead-letter e três novas tentativas."""
    assert parse_brave({}, limit=10) == []
    assert parse_brave({"web": {"results": [{"title": "sem url"}]}}, limit=10) == []


def test_the_searcher_has_no_kind_and_no_fetch():
    """Registrar o buscador em `Context.sources` passa a produzir `AttributeError` no
    ato, em vez de converter a web em fonte de evidência. A trava mora no módulo do
    batedor, não na lembrança de quem escreve o daemon."""
    searcher = BraveSearch(_cfg(), client=_client(lambda r: httpx.Response(200)))
    assert not hasattr(searcher, "kind")
    assert not hasattr(searcher, "fetch")
    assert not hasattr(searcher, "search")
    assert hasattr(searcher, "web_search")


def test_a_fully_configured_but_disabled_provider_still_refuses():
    """MUTAÇÃO: remover a checagem de `enabled` e deixar só a de `api_key`.

    Sem este caso, a mutação matava 0 de 962: o teste de default construía
    `ReconConfig()` SEM chave, então a exceção continuava vindo — só que pelo motivo
    errado. Uma chave presente com `enabled = false` é o estado real de quem configurou
    e desligou de propósito, e é exatamente aí que a torneira precisa fechar.
    """
    with pytest.raises(ReconDisabled) as exc:
        BraveSearch(_cfg(enabled=False))
    assert "DESABILITADO" in str(exc.value)
