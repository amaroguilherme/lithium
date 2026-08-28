"""As travas da Fase D: o registro de fontes e o adapter genérico.

Cada trava vem com a MUTAÇÃO que a mata escrita no docstring, e todas foram EXECUTADAS.
Zero rede: `httpx.MockTransport`, como em `test_sources_pubmed.py`.
"""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from lithium import cli
from lithium.db import Store
from lithium.sources.base import SearchSpec
from lithium.sources.http import HttpSource, SpecError, resolve_credential

runner = CliRunner()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "d.db", embedding_dim=4)
    s.init_schema()
    yield s
    s.close()


def _row(store, slug="openalex", *, search=None, fetch=None, rate=10.0):
    store.conn.execute(
        "INSERT INTO sources_registry(slug, description, base_url, search_spec_json, "
        "  fetch_spec_json, rate_per_s, adapter, yields_evidence, approved_at) "
        "VALUES(?, 'desc', 'https://api.example.invalid', ?, ?, ?, 'http', 1, "
        "  strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
        (slug, json.dumps(search or {}), json.dumps(fetch or {}), rate),
    )
    return store.conn.execute(
        "SELECT * FROM sources_registry WHERE slug = ?", (slug,)).fetchone()


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ═══════════════════════════════════ 1. o adapter genérico


async def test_a_new_source_needs_no_new_code(store):
    """A propriedade que a fase existe para entregar.

    Uma fonte descrita por JSON no registro busca e devolve `SourceRecord` sem uma linha
    de código novo. Antes disto, acrescentar fonte era escrever módulo, importar no
    `daemon.py` e editar um dict literal — e a decisão real morava no import, fora de
    qualquer política.

    MUTAÇÃO: `HttpSource.search` ignorar `id_path`/`id_field` e devolver `data` cru.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/works") and "search" in request.url.params:
            return httpx.Response(200, json={"results": [{"id": "W1"}, {"id": "W2"}]})
        if request.url.path.endswith("/works/W1"):
            return httpx.Response(200, json={
                "title": "Sigma-1 agonism and anxiety", "doi": "10.1/abc",
                "publication_year": 2021,
                "abstract": "Sigma-1 agonism reduced anxiety-like behaviour in rats.",
            })
        return httpx.Response(404)

    src = HttpSource(
        _row(store,
             search={"path": "works", "query_param": "search",
                     "id_path": "results", "id_field": "id", "limit_param": "per-page"},
             fetch={"path": "works/{id}",
                    "fields": {"title": "title", "doi": "doi",
                               "year": "publication_year", "passages": ["abstract"]}}),
        client=_client(handler),
    )
    assert await src.search(SearchSpec(query="sigma-1", limit=5)) == ["W1", "W2"]

    [rec] = await src.fetch(["W1"])
    assert rec.kind == "openalex"
    assert rec.doi == "10.1/abc" and rec.year == 2021
    assert "anxiety-like behaviour" in rec.full_text
    # `design` NÃO é chutado: a fonte genérica não sabe traduzir tipo de publicação em
    # `grade`, e `opinion` como default esmagaria evidência real.
    assert rec.design is None
    await src.aclose()


async def test_a_record_without_prose_is_dropped(store):
    """Registro sem prosa citável não é evidência: o portão de citação verbatim não teria
    onde ancorar. Mesma regra do PubMed, que descarta artigo sem abstract.

    MUTAÇÃO: deixar `texts` vazio produzir um `SourceRecord` com `passages=[]`.
    """
    def handler(request):
        return httpx.Response(200, json={"title": "só metadados", "abstract": None})

    src = HttpSource(_row(store, fetch={"path": "w/{id}",
                                        "fields": {"title": "title",
                                                   "passages": ["abstract"]}}),
                     client=_client(handler))
    assert await src.fetch(["W1"]) == []
    await src.aclose()


async def test_a_spec_without_a_query_param_fails_by_name(store):
    """Spec incompleta falha com o nome do problema, antes de qualquer requisição.

    Sem isto a busca sai sem o termo, a fonte devolve o que quiser, e a tarefa "funciona"
    colhendo material aleatório.

    MUTAÇÃO: remover a checagem de `query_param`.
    """
    src = HttpSource(_row(store, search={"path": "works"}), client=_client(
        lambda r: httpx.Response(200, json={})))
    with pytest.raises(SpecError, match="query_param"):
        await src.search(SearchSpec(query="x"))
    await src.aclose()


async def test_a_broken_path_yields_an_empty_field_not_a_dead_task(store):
    """A spec é escrita à mão: caminho errado tem de virar campo vazio com log, não
    exceção. `_dig` devolve None em qualquer passo que não resolva.

    MUTAÇÃO: `_dig` levantar KeyError/IndexError em vez de devolver None.
    """
    def handler(request):
        return httpx.Response(200, json={"results": [{"id": "W1"}]})

    src = HttpSource(_row(store, search={"query_param": "q", "id_path": "nao.existe.0",
                                         "id_field": "id"}),
                     client=_client(handler))
    assert await src.search(SearchSpec(query="x")) == []
    await src.aclose()


def test_the_user_agent_identifies_itself(store):
    """Acesso automatizado se identifica. Uso pessoal não relaxa ToS de terceiro — é
    decisão registrada no PLAN.md §7.

    MUTAÇÃO: devolver um User-Agent genérico ou vazio.
    """
    src = HttpSource(_row(store), contact="eu@exemplo.org")
    ua = src._client.headers["user-agent"]
    assert "lithium" in ua and "eu@exemplo.org" in ua


# ═══════════════════════════════════ 2. a credencial nunca vem do banco


def test_a_credential_is_referenced_by_name_never_stored(monkeypatch, store):
    """`credential_ref` guarda o NOME de onde a chave está.

    O movimento óbvio é uma coluna com a chave dentro; `config.local.toml` está no
    `.gitignore` e uma coluna do banco não está protegida por nada — e é o banco que o
    item 8 sincroniza para o Hugging Face.

    MUTAÇÃO: `resolve_credential` devolver `ref` quando não resolve (tratando a referência
    como se fosse o segredo) — o nome da variável iria como api_key para a fonte.
    """
    monkeypatch.setenv("MINHA_CHAVE", "s3gr3d0")
    assert resolve_credential("env:MINHA_CHAVE", None) == "s3gr3d0"
    assert resolve_credential("env:NAO_EXISTE", None) is None
    assert resolve_credential(None, None) is None

    class Cfg:
        class sources:
            class pubmed:
                api_key = "abc123"

    assert resolve_credential("sources.pubmed.api_key", Cfg) == "abc123"
    assert resolve_credential("sources.pubmed.inexistente", Cfg) is None

    # e o segredo não está no banco
    cols = {r["name"] for r in store.conn.execute(
        "PRAGMA table_xinfo(sources_registry)")}
    assert "api_key" not in cols and "secret" not in cols, (
        "o registro ganhou coluna de segredo: ele sai de casa no sync do item 8"
    )


def test_a_source_whose_credential_does_not_resolve_stays_out(store, tmp_path):
    """Fonte sem credencial fica FORA do dict, não tenta sem auth.

    Um 401 dentro do worker vira retry, backoff e dead-letter, e o diagnóstico aponta para
    a fila em vez da configuração.

    MUTAÇÃO: construir a fonte de qualquer forma quando `key is None`.
    """
    from lithium.config import Config
    from lithium.sources.factory import build_sources

    store.conn.execute(
        "INSERT INTO sources_registry(slug, description, base_url, credential_ref, "
        "  adapter, approved_at) VALUES('paga', 'exige chave', 'https://x', "
        "  'env:NAO_DEFINIDA', 'http', strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
    )
    built = build_sources(store, Config(data_dir=tmp_path), None)
    assert "paga" not in built, "fonte sem credencial foi construída e vai receber 401"
    assert "pubmed" in built, (
        "o PubMed saiu: a api_key da NCBI é OPCIONAL (3 req/s sem ela) e é assim que o "
        "projeto roda hoje"
    )


# ═══════════════════════════════════ 3. o comando, e a separação de decisões


def _cfg(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(f'data_dir = "{tmp_path / "d"}"\n', encoding="utf-8")
    return str(p)


def test_activating_a_source_and_calling_it_evidence_are_two_decisions(tmp_path):
    """`--approve` e `--evidence` são separados de propósito.

    "Consulte esta fonte" e "o que ela devolve pode virar evidência graduável" são
    afirmações diferentes: um registro de ensaios é útil para descobrir o que existe e não
    é desenho de estudo. Juntá-las faria a segunda pegar carona na primeira — que é como
    uma bula virou claim com `grade='rct'` e peso 0,408 na medição do item 9.

    MUTAÇÃO: `--approve` gravar `yields_evidence = 1` sempre.
    """
    cfg = _cfg(tmp_path)
    assert runner.invoke(cli.app, ["init", "-c", cfg]).exit_code == 0

    from lithium.config import load_config
    store = Store(load_config(tmp_path / "c.toml").db_path, embedding_dim=1024)
    store.init_schema()
    store.propose_source("ctgov", "registro de ensaios", "https://clinicaltrials.gov")
    store.conn.execute(
        "UPDATE sources_registry SET search_spec_json = '{\"query_param\": \"q\"}', "
        "  fetch_spec_json = '{\"path\": \"x\"}' WHERE slug = 'ctgov'")
    store.close()

    r = runner.invoke(cli.app, ["sources", "-c", cfg, "--approve", "ctgov"])
    assert r.exit_code == 0, r.output
    store = Store(load_config(tmp_path / "c.toml").db_path, embedding_dim=1024)
    assert store.source_state("ctgov") == "ativa"
    assert not store.source_yields_evidence("ctgov"), (
        "ativar a fonte a tornou fonte de EVIDÊNCIA sem ninguém afirmar isso"
    )
    store.close()


def test_approving_an_http_source_as_evidence_without_a_spec_is_refused(tmp_path):
    """Ativar como evidência sem spec enfileiraria tarefas que morrem no primeiro
    `search`. Recusa nomeada em vez de fila com dead-letter.

    MUTAÇÃO: remover a checagem de spec do ramo `--approve --evidence`.
    """
    cfg = _cfg(tmp_path)
    runner.invoke(cli.app, ["init", "-c", cfg])
    from lithium.config import load_config
    store = Store(load_config(tmp_path / "c.toml").db_path, embedding_dim=1024)
    store.init_schema()
    store.propose_source("sem-spec", "proposta crua", "https://x")
    store.close()

    r = runner.invoke(cli.app, ["sources", "-c", cfg, "--approve", "sem-spec",
                                "--evidence"])
    assert r.exit_code == 1
    assert "RECUSADO" in r.output and "spec" in r.output


def test_revoking_does_not_rewrite_what_was_already_harvested(tmp_path):
    """Revogar desativa a fonte; o corpus já colhido continua válido e continua pesando.

    O contrário seria reescrever julgamento passado — a mesma razão pela qual a Fase B
    recusou `--regrade`.

    MUTAÇÃO: `--revoke` apagar a linha do registro (a FK de `sources.kind` levanta) ou
    apagar as `sources` daquela fonte.
    """
    cfg = _cfg(tmp_path)
    runner.invoke(cli.app, ["init", "-c", cfg])
    from lithium.config import load_config
    store = Store(load_config(tmp_path / "c.toml").db_path, embedding_dim=1024)
    store.init_schema()
    store.upsert_source(kind="pubmed", external_id="1", raw={}, title="colhido")
    store.close()

    r = runner.invoke(cli.app, ["sources", "-c", cfg, "--revoke", "pubmed"])
    assert r.exit_code == 0, r.output

    store = Store(load_config(tmp_path / "c.toml").db_path, embedding_dim=1024)
    assert store.source_state("pubmed") == "proposta"
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM sources").fetchone()["n"] == 1, (
        "revogar apagou o que já havia sido colhido"
    )
    store.close()
