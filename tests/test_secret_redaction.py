"""A credencial não pode chegar ao log, à coluna `tasks.error` nem ao dead-letter.

DEFEITO VIVO, reproduzido: `httpx.HTTPStatusError.__str__` carrega a URL COMPLETA com
query string, `PubMedSource._params` injeta `api_key` como query param, e
`Runner._execute` monta `f'{tipo}: {exc}\\n{traceback}'` e GRAVA. Um único 429 do eutils
põe a chave da NCBI no banco de qualquer usuário — e `purge_done` só apaga `done`, nunca
`dead`, então ela fica lá para sempre. A correção da Fase A cobriu o LOGGER do httpx e
SÓ ELE; este caminho é persistente e não é log.

Parametrizado sobre as DUAS credenciais de propósito. Um teste escrito só com a Brave
passaria POR CONSTRUÇÃO — a chave dela vai em header e `str(exc)` não a carrega — e não
policiaria nada.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from lithium.config import Config, ReconConfig
from lithium.db import Store
from lithium.recon.search import BraveSearch
from lithium.sources.base import SearchSpec
from lithium.sources.pubmed import PubMedSource
from lithium.worker.queue import TaskQueue, redact_secrets
from lithium.worker.runner import Context, Runner

NCBI_SECRET = "SEGREDO-NCBI-123"
BRAVE_SECRET = "SEGREDO-BRAVE-456"


def _429(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, text="Too Many Requests")


def _ncbi_task(tmp_path):
    source = PubMedSource(
        api_key=NCBI_SECRET, rate_per_s=1000,
        client=httpx.AsyncClient(transport=httpx.MockTransport(_429)))

    async def handler(payload, ctx):
        await source.search(SearchSpec(query="lithium"))

    return handler, NCBI_SECRET


def _brave_task(tmp_path):
    searcher = BraveSearch(
        ReconConfig(enabled=True, api_key=BRAVE_SECRET, contact="c"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_429)),
        rate_per_s=1000)

    async def handler(payload, ctx):
        await searcher.web_search("lithium", limit=10)

    return handler, BRAVE_SECRET


@pytest.mark.parametrize("build", [_ncbi_task, _brave_task],
                         ids=["ncbi-query-param", "brave-header"])
async def test_the_key_never_reaches_the_log_the_task_error_or_the_dead_letter(
        tmp_path, caplog, build):
    """Pelo `Runner` REAL, até o dead-letter, afirmando sobre o BANCO.

    MUTAÇÃO: tirar `redact_secrets` de `queue.fail`/`kill`, ou dos dois `log.*` de
    `Runner._execute`. Com a NCBI (chave em query param) as três asserções caem; com a
    Brave, nenhuma — que é exatamente por que o teste é parametrizado.
    """
    store = Store(tmp_path / "s.db", embedding_dim=8)
    store.init_schema()
    handler, secret = build(tmp_path)
    queue = TaskQueue(store, max_attempts=1, backoff_base_s=0.0)
    ctx = Context(config=Config(data_dir=tmp_path), store=store, queue=queue)
    runner = Runner(ctx, {"leaky": handler}, concurrency=1)

    queue.enqueue("leaky", {})
    with caplog.at_level(logging.INFO):
        await runner.drain()

    stored = "\n".join(str(r[0]) for r in store.conn.execute(
        "SELECT COALESCE(error, '') FROM tasks"))
    assert secret not in stored, "a credencial foi gravada em tasks.error"

    dead = "\n".join(str(d) for d in queue.dead_letters())
    assert secret not in dead, "a credencial está no dead-letter"

    # Só os loggers do PRÓPRIO projeto. O `httpx` loga a URL completa em INFO e quem o
    # silencia é a proteção da Fase A (`getLogger('httpx').setLevel(WARNING)` em
    # `cli._setup_logging`) — coberta pelo teste abaixo. Misturar os dois aqui faria
    # esta asserção passar ou falhar por causa da OUTRA correção.
    logged = "\n".join(r.getMessage() for r in caplog.records
                       if r.name.startswith("lithium"))
    assert secret not in logged, "a credencial foi para o log do daemon"

    # Anti-vácuo: a tarefa REALMENTE morreu, e o motivo REALMENTE foi gravado.
    assert queue.dead_letters(), "a tarefa não chegou ao dead-letter"
    assert "429" in stored
    store.close()


def test_the_phase_a_protection_of_the_httpx_logger_is_still_in_place():
    """A OUTRA metade, e ela é a que o teste acima deliberadamente não mede.

    O `httpx` loga a URL COMPLETA em INFO, uma vez por query, 19 por `harvest_sweep`.
    `redact_secrets` não alcança esse logger — quem o alcança é `_setup_logging`.
    """
    import logging as _logging

    from lithium.cli import _setup_logging

    _setup_logging(verbose=False)
    assert _logging.getLogger("httpx").level >= _logging.WARNING


def test_the_ncbi_key_really_does_leak_without_redaction():
    """A pré-condição anti-vacuidade: o vazamento é REAL, não hipotético.

    Sem ela, o teste acima passaria mesmo que `str(exc)` nunca tivesse carregado a
    chave — e a correção estaria protegendo nada.
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&api_key={NCBI_SECRET}"
    request = httpx.Request("GET", url)
    # Levantado por `raise_for_status()`, como em produção: é ELE que monta a mensagem
    # com a URL inteira. Construir a exceção à mão testaria a mensagem do teste.
    try:
        httpx.Response(429, request=request).raise_for_status()
    except httpx.HTTPStatusError as caught:
        exc = caught
    raw = f"{type(exc).__name__}: {exc}"
    assert NCBI_SECRET in raw, "a premissa mudou: revise se a redação ainda é necessária"
    assert NCBI_SECRET not in redact_secrets(raw)


@pytest.mark.parametrize("text,leaks", [
    ("...&api_key=SEGREDO&retmode=json", "SEGREDO"),
    ("X-Subscription-Token: SEGREDO", "SEGREDO"),
    ('{"token": "SEGREDO"}', "SEGREDO"),
    ("password=SEGREDO", "SEGREDO"),
])
def test_redaction_covers_the_shapes_a_credential_takes(text, leaks):
    assert leaks not in redact_secrets(text)


def test_redaction_does_not_eat_the_diagnosis():
    """Redigir demais transforma o dead-letter em ruído: o traceback ainda precisa
    dizer o que aconteceu."""
    detail = ("HTTPStatusError: Client error '429 Too Many Requests' for url "
              "'https://eutils.ncbi.nlm.nih.gov/esearch.fcgi?db=pubmed&api_key=X1'")
    safe = redact_secrets(detail)
    assert "429" in safe and "eutils.ncbi.nlm.nih.gov" in safe and "db=pubmed" in safe


@pytest.mark.parametrize("verb", ["fail", "kill"])
def test_the_queue_redacts_even_when_the_caller_did_not(tmp_path, verb):
    """`queue.fail` e `queue.kill` são os DOIS únicos escritores de `tasks.error`.

    `Runner._execute` já redige antes de chamar — então a redação AQUI é a segunda
    camada, e sem este teste ela é indistinguível de código morto: MEDIDO, removê-la
    matava 0 de 962, porque todo caminho testado passava pelo runner. A segunda camada
    importa porque `kill()` também é chamado direto (handler ausente) e porque um
    handler novo pode chamar `fail` por conta própria.
    """
    store = Store(tmp_path / "q.db", embedding_dim=8)
    store.init_schema()
    queue = TaskQueue(store, max_attempts=1)
    task_id = queue.enqueue("x", {})
    queue.claim()
    getattr(queue, verb)(task_id, f"boom: ...&api_key={NCBI_SECRET}&retmode=json")

    stored = store.conn.execute(
        "SELECT error FROM tasks WHERE id = ?", (task_id,)).fetchone()["error"]
    assert NCBI_SECRET not in stored
    assert "boom" in stored
    store.close()
