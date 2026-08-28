"""Peças compartilhadas dos testes de recon. ZERO rede em todos eles.

Um só lugar para o contexto porque são seis arquivos: sem isto, seis fixtures quase
iguais divergem na primeira correção, e a que se perde é sempre a que importa.
"""

from __future__ import annotations

import json
import math
import shutil
import zlib
from collections.abc import Sequence
from pathlib import Path

import httpx

from lithium.config import Config, ReconConfig
from lithium.db import Store
from lithium.llm.schemas import ReconObservation, ReconTriage
from lithium.recon.read import PageReader
from lithium.recon.search import WebHit
from lithium.worker.queue import TaskQueue
from lithium.worker.runner import Context, ReconBundle, Runner
from lithium.worker.handlers import HANDLERS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "recon"
FIXTURE_FOCUSES = Path(__file__).resolve().parent / "fixtures" / "focuses"
DIM = 16

NICE_HTML = (FIXTURES / "nice_cg185.html").read_text(encoding="utf-8", errors="replace")
PMC_HTML = (FIXTURES / "pmc_article.html").read_text(encoding="utf-8", errors="replace")
COOKIEWALL_HTML = (FIXTURES / "pubmed_cookiewall.html").read_text(
    encoding="utf-8", errors="replace")
ROBOTS_PMC = (FIXTURES / "robots_pmc.txt").read_text(encoding="utf-8")
BRAVE_JSON = json.loads(
    (FIXTURES / "brave_web_search.SYNTHETIC.json").read_text(encoding="utf-8")
)


def brave_hits() -> list[WebHit]:
    from lithium.recon.search import parse_brave

    return parse_brave(BRAVE_JSON, limit=10)


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class DeadEmbedder:
    async def embed(self, texts):
        raise RuntimeError("llama-server de embedding fora do ar")


class FakeSearcher:
    """Devolve os hits da fixture e CONTA as chamadas. Sem HTTP nenhum."""

    def __init__(self, hits: list[WebHit] | None = None) -> None:
        self.hits = hits if hits is not None else brave_hits()
        self.calls: list[str] = []

    async def web_search(self, query: str, *, limit: int = 10) -> list[WebHit]:
        self.calls.append(query)
        return self.hits[:limit]


class TriagingLLM:
    """Classifica por regra fixa e conta as chamadas. Nada de rede, nada de GPU."""

    def __init__(self, verdicts: list[dict] | None = None,
                 observation: dict | None = None) -> None:
        self.verdicts = verdicts
        self.observation = observation
        self.triage_calls = 0
        self.observe_calls = 0
        self.prompts: list[str] = []

    async def structured(self, messages, schema, **kw):
        self.prompts.append(messages[0]["content"])
        if schema is ReconTriage:
            self.triage_calls += 1
            if self.verdicts is not None:
                return ReconTriage.model_validate({"verdicts": self.verdicts})
            return ReconTriage.model_validate({"verdicts": [
                {"index": 1, "kind": "observation"},
                {"index": 2, "kind": "lead"},
                {"index": 3, "kind": "lead"},
                {"index": 4, "kind": "source"},
                {"index": 5, "kind": "skip"},
            ]})
        if schema is ReconObservation:
            self.observe_calls += 1
            return ReconObservation.model_validate(
                self.observation
                or {"worth_reporting": True,
                    "summary": "a página descreve o manejo de manutenção",
                    "why_it_matters": "cobre o alvo do foco"}
            )
        raise AssertionError(f"schema inesperado: {schema}")

    async def complete(self, *a, **k):
        raise AssertionError("recon não usa complete()")


def page_transport(pages: dict[str, tuple[int, str]],
                   robots: dict[str, tuple[int, str]] | None = None,
                   log: list[str] | None = None) -> httpx.MockTransport:
    """MockTransport para leitura de página e robots.txt. Registra TODA requisição."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if log is not None:
            log.append(url)
        if url.endswith("/robots.txt"):
            status, body = (robots or {}).get(url, (404, ""))
            return httpx.Response(status, text=body)
        status, body = pages.get(url, (404, "<html><title>404</title></html>"))
        return httpx.Response(status, text=body,
                              headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


def make_store(tmp_path, name: str = "recon.db") -> Store:
    store = Store(tmp_path / name, embedding_dim=DIM)
    store.init_schema()
    return store


def make_ctx(tmp_path, *, store=None, llm=None, embedder=None, searcher=None,
             reader=None, recon_enabled: bool = True, pubmed=None,
             **recon_overrides) -> Context:
    """Contexto completo, com o foco de PRODUÇÃO (`bipolar-tag`) ativo.

    O perfil real e não o de teste: os prompts de recon injetam `$target_prose`, e o
    diretório de perfis precisa conter o slug do foco ativo.
    """
    focuses = tmp_path / "focuses"
    if not focuses.exists():
        shutil.copytree(Path(__file__).resolve().parent.parent / "focuses", focuses)
    store = store or make_store(tmp_path)
    recon_cfg = ReconConfig(enabled=recon_enabled, api_key="k", contact="c",
                            **recon_overrides)
    cfg = Config(data_dir=tmp_path, focuses_dir=focuses, recon=recon_cfg)
    bundle = None
    if recon_enabled:
        bundle = ReconBundle(searcher=searcher or FakeSearcher(),
                             reader=reader or _null_reader(),
                             results_per_query=10)
    return Context(config=cfg, store=store, queue=TaskQueue(store),
                   llm=llm or TriagingLLM(), embedder=embedder or FakeEmbedder(),
                   sources={"pubmed": pubmed} if pubmed else {}, recon=bundle)


def _null_reader() -> PageReader:
    transport = page_transport({})
    return PageReader(httpx.AsyncClient(transport=transport), contact="c")


def reader_for(pages, robots=None, log=None, *, respect_robots: bool = True,
               robots_budget=None) -> PageReader:
    client = httpx.AsyncClient(transport=page_transport(pages, robots, log),
                               follow_redirects=True)
    return PageReader(client, contact="teste@exemplo", respect_robots=respect_robots,
                      robots_budget=robots_budget)


def runner(ctx: Context) -> Runner:
    return Runner(ctx, HANDLERS, concurrency=1)


def seed_discovery(store, *, focus_id=1, kind="observation", status="pending",
                   query="q", url="https://ex.invalid/a", title="t", summary="s",
                   lead_kind=None, lead_external_id=None, payload=None,
                   created_at=None) -> int:
    cols = ["focus_id", "kind", "status", "query", "url", "title", "summary",
            "lead_kind", "lead_external_id", "payload_json"]
    vals = [focus_id, kind, status, query, url, title, summary, lead_kind,
            lead_external_id, json.dumps(payload or {})]
    if created_at is not None:
        cols.append("created_at")
        vals.append(created_at)
    cur = store.conn.execute(
        f"INSERT INTO discoveries({', '.join(cols)}) "
        f"VALUES({', '.join('?' * len(cols))}) RETURNING id", vals)
    return int(cur.fetchone()["id"])
