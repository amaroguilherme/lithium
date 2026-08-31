"""Encadeamento dos handlers, com PubMed real (fixture) e LLM/embedder falsos.

O que estes testes protegem é a propriedade que torna o daemon utilizável: cada
handler faz um pedaço pequeno e enfileira o próximo. Se um deles voltar a fazer tudo
em linha, uma falha de rede no meio do harvest passa a custar horas de trabalho já
feito em vez de um retry.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from lithium.config import Config
from lithium.db import Store
from lithium.llm.schemas import (
    CitationVerdict,
    ClaimExtraction,
    DirectnessVerdict,
)
from lithium.types import Directness
from lithium.sources.pubmed import PubMedSource
from lithium.worker.handlers import HANDLERS
from lithium.worker.queue import TaskQueue
from lithium.worker.runner import Context, Runner

import shutil

from conftest import FIXTURE_FOCUSES, onco_profile


FIXTURES = Path(__file__).parent / "fixtures"
EFETCH_XML = (FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8")
DIM = 16


def _bucket(word: str) -> int:
    """`hash()` de str é salinizado por processo: os vetores mudariam a cada
    execução e os testes ficariam flaky. crc32 é estável entre processos."""
    return zlib.crc32(word.encode()) % DIM



class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[_bucket(word)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FakeLLM:
    """Extrai uma claim ancorada por chunk, copiando um trecho literal do texto."""

    def __init__(self) -> None:
        self.extraction_calls = 0
        self.prompts: list[str] = []

    async def structured(self, messages, schema, **kw):
        self.prompts.append(messages[0]["content"])
        if schema is CitationVerdict:
            return CitationVerdict(supported=True, reason="ok")
        if schema is DirectnessVerdict:
            return DirectnessVerdict(in_scope=True, judgeable=True,
                                     directness=Directness.PARTIAL, rationale="r")
        self.extraction_calls += 1
        body = messages[0]["content"].rsplit("---", 2)[1].strip()
        quote = " ".join(body.split()[:8])
        return ClaimExtraction.model_validate(
            {
                "claims": [
                    {
                        "statement": f"Achado sobre {quote[:40]}",
                        "supporting_quote": quote,
                        "population": "adultos",
                        "intervention": "quetiapina",
                        "comparator": "",
                        "outcome": "ansiedade",
                        "direction": "positive",
                        "effect": "",
                        "grade": "rct",
                        "directness_judgeable": True,
                        "directness": "partial",
                        "confidence": 0.8,
                    }
                ]
            }
        )


@pytest.fixture
def ctx(tmp_path):
    """Contexto rodando sob o PERFIL DE TESTE, e isso é a metade que prova a fiação.

    MEDIDO: com a taxonomia e as estratégias vindas de constante de módulo, apontar o
    `focuses_dir` para outro domínio não muda nada — `handlers.STRATEGY_BY_NAME` era um
    binding de import time e continuava devolvendo as cinco frentes de produção mesmo
    com `strategy.STRATEGY_BY_NAME = {}`. Aqui o foco ativo no BANCO é `onco-vet`, e o
    único lugar de onde `linfoma_canino_direto` pode vir é o TOML do perfil.
    """
    import shutil as _shutil
    focuses = tmp_path / "focuses"
    _shutil.copytree(FIXTURE_FOCUSES, focuses)
    store = Store(tmp_path / "h.db", embedding_dim=DIM)
    store.init_schema()
    scale_id = int(store.conn.execute(
        "SELECT id FROM evidence_scales LIMIT 1").fetchone()["id"])
    store.conn.execute(
        "INSERT INTO focuses(slug, target, scale_id) VALUES('onco-vet', ?, ?)",
        ("canine multicentric lymphoma", scale_id))
    store.conn.execute(
        "UPDATE meta SET value = (SELECT id FROM focuses WHERE slug = 'onco-vet') "
        " WHERE key = 'active_focus'")

    def handler(request: httpx.Request) -> httpx.Response:
        if "esearch" in str(request.url):
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": ["30712879", "37956131",
                                                   "26834458", "21403524"]}},
            )
        return httpx.Response(200, text=EFETCH_XML)

    pubmed = PubMedSource(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), rate_per_s=1000
    )
    context = Context(
        config=Config(data_dir=tmp_path, focuses_dir=focuses),
        store=store,
        queue=TaskQueue(store),
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        sources={"pubmed": pubmed},
    )
    yield context
    store.close()


def _runner(ctx: Context) -> Runner:
    return Runner(ctx, HANDLERS, concurrency=1)


# ─────────────────────────────────────────────────────────── encadeamento


async def test_sweep_fans_out_into_one_task_per_query(ctx):
    """`harvest_sweep` não busca nada — só enfileira. É o que permite retry por
    consulta em vez de refazer a varredura inteira."""
    from lithium.pipeline.strategy import all_search_specs

    ctx.queue.enqueue("harvest_sweep", {})
    await _runner(ctx).drain(max_tasks=1)

    pending = ctx.store.conn.execute(
        "SELECT kind, COUNT(*) AS n FROM tasks WHERE status = 'pending' GROUP BY kind"
    ).fetchall()
    assert dict((r["kind"], r["n"]) for r in pending) == {
        "harvest_query": len(all_search_specs(onco_profile()))
    }


async def test_sweep_can_be_scoped_to_one_strategy(ctx):
    from lithium.pipeline.strategy import strategy_by_name

    ctx.queue.enqueue("harvest_sweep", {"strategies": ["linfoma_canino_direto"]})
    await _runner(ctx).drain(max_tasks=1)

    n = ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE kind = 'harvest_query'"
    ).fetchone()["n"]
    assert n == len(strategy_by_name(onco_profile())["linfoma_canino_direto"].queries)


async def test_query_enqueues_fetch_for_each_new_id(ctx):
    ctx.queue.enqueue("harvest_query", {"strategy": "linfoma_canino_direto", "query": "x"})
    await _runner(ctx).drain(max_tasks=1)

    rows = ctx.store.conn.execute(
        "SELECT payload_json FROM tasks WHERE kind = 'fetch_source'"
    ).fetchall()
    assert len(rows) == 4


async def test_query_skips_already_known_sources(ctx):
    """Sem isso, cada varredura diária rebaixaria o corpus inteiro."""
    ctx.store.upsert_source(kind="pubmed", external_id="30712879", raw={})
    ctx.store.upsert_source(kind="pubmed", external_id="26834458", raw={})

    ctx.queue.enqueue("harvest_query", {"strategy": "linfoma_canino_direto", "query": "x"})
    await _runner(ctx).drain(max_tasks=1)

    n = ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE kind = 'fetch_source'"
    ).fetchone()["n"]
    assert n == 2


async def test_fetch_ingests_and_requests_extraction(ctx):
    ctx.queue.enqueue(
        "fetch_source",
        {"kind": "pubmed", "external_id": "30712879", "expected_directness": "direct"},
    )
    await _runner(ctx).drain(max_tasks=1)

    assert ctx.store.conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"] == 4
    assert ctx.store.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] > 4
    assert ctx.store.conn.execute("SELECT COUNT(*) AS n FROM chunk_vec").fetchone()["n"] > 4

    kinds = {
        r["kind"]
        for r in ctx.store.conn.execute("SELECT kind FROM tasks WHERE status = 'pending'")
    }
    assert kinds == {"extract_source"}


async def test_fetch_of_unknown_id_is_not_an_error(ctx):
    """Sem abstract ou id inexistente: repetir não conserta, e não é falha."""

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<PubmedArticleSet/>")

    ctx.sources["pubmed"] = PubMedSource(
        client=httpx.AsyncClient(transport=httpx.MockTransport(empty)), rate_per_s=1000
    )
    ctx.queue.enqueue("fetch_source", {"kind": "pubmed", "external_id": "0"})
    runner = _runner(ctx)
    await runner.drain(max_tasks=1)

    assert runner.failed == 0
    assert ctx.queue.stats() == {"done": 1}


async def test_full_chain_produces_verified_claims(ctx):
    """Ponta a ponta: sweep → query → fetch → extract, com claims no banco."""
    ctx.queue.enqueue("harvest_sweep", {"strategies": ["linfoma_canino_direto"]})
    runner = _runner(ctx)
    await runner.drain()

    assert runner.failed == 0
    verified = ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM claims WHERE verified = 1"
    ).fetchone()["n"]
    assert verified > 0

    # E as claims já aparecem ponderadas na view que alimenta o placar.
    weights = ctx.store.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"]
    assert weights == verified


async def test_chain_is_idempotent_across_runs(ctx):
    """Rodar a varredura duas vezes não pode duplicar fonte, chunk nem tarefa."""
    for _ in range(2):
        ctx.queue.enqueue("harvest_sweep", {"strategies": ["linfoma_canino_direto"]})
        await _runner(ctx).drain()

    counts = ctx.store.counts()
    assert counts["sources"] == 4
    # dedup_key de `extract:<id>` impede reextração, então nada de claim duplicada
    per_source = ctx.store.conn.execute(
        "SELECT source_id, COUNT(*) AS n FROM claims GROUP BY source_id"
    ).fetchall()
    chunks_per_source = ctx.store.conn.execute(
        "SELECT source_id, COUNT(*) AS n FROM chunks GROUP BY source_id"
    ).fetchall()
    assert [r["n"] for r in per_source] == [r["n"] for r in chunks_per_source]


async def test_missing_source_adapter_fails_the_task(ctx):
    ctx.sources = {}
    ctx.queue.enqueue("harvest_query", {"strategy": "linfoma_canino_direto", "query": "x"})
    runner = _runner(ctx)
    await runner.drain(max_tasks=1)

    assert runner.failed == 1
    error = ctx.store.conn.execute("SELECT error FROM tasks").fetchone()["error"]
    assert "pubmed" in error


async def test_purge_handler_runs(ctx):
    ctx.queue.enqueue("purge_tasks", {"days": 0})
    runner = _runner(ctx)
    await runner.drain(max_tasks=1)
    assert runner.failed == 0


async def test_the_same_source_is_re_extracted_for_a_different_focus(ctx):
    """Passa pelo HANDLER, não por um `dedup_key` remontado no teste.

    MUTAÇÃO: reverter para `dedup_key=f"extract:{result.source_id}"`. O segundo
    enfileiramento é engolido por `ON CONFLICT(dedup_key) DO NOTHING`, o foco #2 nunca
    ganha `claim_directness` para aquelas claims, e todas elas valem zero nele para
    sempre, sem log. `test_chain_is_idempotent_across_runs` continua verde porque não
    troca de foco — é por isso que este teste precisa existir separado.
    """
    payload = {"kind": "pubmed", "external_id": "30712879"}
    ctx.queue.enqueue("fetch_source", dict(payload))
    await _runner(ctx).drain(max_tasks=1)
    n_first = ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE kind = 'extract_source'"
    ).fetchone()["n"]
    assert n_first > 0

    # tudo que ficou pendente vira 'done': o que se mede é o ENFILEIRAMENTO, não o
    # trabalho, e um `drain` seguinte poderia consumir a fila de extração no lugar do
    # fetch.
    ctx.store.conn.execute("UPDATE tasks SET status = 'done' WHERE status = 'pending'")

    scale = ctx.store.conn.execute("SELECT id FROM evidence_scales").fetchone()["id"]
    # O perfil em disco tem de existir: o handler resolve `active_profile` e o slug do
    # banco é o nome do diretório. `onco-vet-2` é um symlink conceitual — reusa o mesmo
    # diretório de perfil sob outro id de foco, que é o que o teste precisa (dois FOCOS,
    # não dois vocabulários).
    import shutil
    shutil.copytree(ctx.config.focuses_dir / "onco-vet",
                    ctx.config.focuses_dir / "onco-vet-2", dirs_exist_ok=True)
    (ctx.config.focuses_dir / "onco-vet-2" / "focus.toml").write_text(
        (FIXTURE_FOCUSES / "onco-vet" / "focus.toml").read_text(encoding="utf-8")
        .replace('slug   = "onco-vet"', 'slug   = "onco-vet-2"'), encoding="utf-8")
    ctx.store.conn.execute(
        "INSERT INTO focuses(slug, target, scale_id) VALUES('onco-vet-2', 'outro alvo', ?)",
        (scale,))
    ctx.store.conn.execute(
        "UPDATE meta SET value = (SELECT id FROM focuses WHERE slug = 'onco-vet-2') "
        " WHERE key = 'active_focus'")

    ctx.queue.enqueue("fetch_source", dict(payload),
                      dedup_key="fetch:pubmed:30712879:foco2")
    await _runner(ctx).drain(max_tasks=1)
    n_second = ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE kind = 'extract_source'"
    ).fetchone()["n"]
    assert n_second == 2 * n_first, (
        "a fonte não foi reenfileirada para extração sob o foco #2: as claims dela "
        "valem zero nesse foco, para sempre, sem log"
    )


# ══════════════════════════════════════════════════════════ relens (Fase B)


def _verified_claim(ctx, statement="alegação", judged=False):
    """Uma claim verificada na escala do foco ativo. SQL cru, como o conftest."""
    store = ctx.store
    sid = store.upsert_source(kind="pubmed", external_id=f"x{statement}", raw={})
    focus = store.conn.execute("SELECT id, scale_id FROM active_focus").fetchone()
    cur = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "  grade, scale_id, confidence, verified) "
        "VALUES(?, '[]', ?, 'x', 'positive', 'rct', ?, 1.0, 1) RETURNING id",
        (sid, statement, focus["scale_id"]))
    claim_id = int(cur.fetchone()["id"])
    if judged:
        store.conn.execute(
            "INSERT INTO claim_directness(claim_id, focus_id, directness) "
            "VALUES(?, ?, 'direct')", (claim_id, int(focus["id"])))
    return claim_id


async def test_relens_is_enqueued_as_scheduled_so_mode_off_can_stop_it(ctx):
    """As tarefas do fan-out têm `origin='scheduled'`, e `mode off` as represa.

    MUTAÇÃO: trocar para `origin='on_demand'`, que é o default de TODO comando de CLI
    hoje. MEDIDO E EXECUTADO: com `mode off`, 5 de 5 tarefas `on_demand` continuam
    sendo reivindicadas e 5 `scheduled` ficam represadas. Sem esta trava um relens de
    7,68 h vira ININTERRUPTÍVEL — não existe `lithium cancel`, e matar o daemon deixa
    as N-1 pendentes para voltarem no restart.
    """
    for i in range(3):
        _verified_claim(ctx, f"c{i}")
    ctx.queue.enqueue("relens_sweep", {})
    await _runner(ctx).drain(max_tasks=1)

    origins = {
        r["origin"] for r in ctx.store.conn.execute(
            "SELECT origin FROM tasks WHERE kind = 'relens_claim'")
    }
    assert origins == {"scheduled"}, origins
    # e é isto que o `mode off` de fato para: nenhuma é reivindicada
    assert ctx.queue.claim(on_demand_only=True) is None


async def test_relens_dedup_key_is_scoped_by_focus(ctx):
    """Re-lentear o foco B depois do A enfileira a MESMA claim de novo.

    MUTAÇÃO: usar `relens:{claim_id}` sem o `focus_id`. Repete literalmente o defeito
    que a Fase A consertou em `extract:` — o segundo foco recebe None em toda claim e o
    comando reporta "0 julgadas" como se estivesse tudo em dia, para sempre.
    """
    _verified_claim(ctx, "c0")
    ctx.queue.enqueue("relens_sweep", {})
    await _runner(ctx).drain(max_tasks=1)
    primeiro = {r["dedup_key"] for r in ctx.store.conn.execute(
        "SELECT dedup_key FROM tasks WHERE kind = 'relens_claim'")}
    assert len(primeiro) == 1

    scale = ctx.store.conn.execute(
        "SELECT scale_id FROM active_focus").fetchone()["scale_id"]
    shutil.copytree(ctx.config.focuses_dir / "onco-vet",
                    ctx.config.focuses_dir / "onco-vet-b", dirs_exist_ok=True)
    f = ctx.config.focuses_dir / "onco-vet-b" / "focus.toml"
    f.write_text(f.read_text(encoding="utf-8")
                 .replace('slug   = "onco-vet"', 'slug   = "onco-vet-b"'),
                 encoding="utf-8")
    ctx.store.conn.execute(
        "INSERT INTO focuses(slug, target, scale_id) VALUES('onco-vet-b', 'outro', ?)",
        (scale,))
    ctx.store.conn.execute(
        "UPDATE meta SET value = (SELECT id FROM focuses WHERE slug = 'onco-vet-b') "
        " WHERE key = 'active_focus'")

    ctx.queue.enqueue("relens_sweep", {}, dedup_key="sweep-b")
    await _runner(ctx).drain(max_tasks=1)
    todas = {r["dedup_key"] for r in ctx.store.conn.execute(
        "SELECT dedup_key FROM tasks WHERE kind = 'relens_claim'")}
    assert len(todas) == 2, (
        f"a mesma claim não foi reenfileirada para o segundo foco: {todas}"
    )


async def test_a_dead_relens_task_does_not_burn_its_dedup_key_forever(ctx):
    """Uma tarefa em `dead` NÃO pode impedir a próxima varredura de reenfileirar.

    `tasks.dedup_key` é `TEXT UNIQUE` sobre a tabela INTEIRA e `enqueue` deduplica
    contra `done` E `dead`; `purge_done()` só apaga `done`, e não existe
    `lithium requeue`. MEDIDO: uma noite de llama-server fora do ar produz ~505 tarefas
    em `dead` (57 s cada, concorrência 1), e o relens seguinte devolve None em TODAS —
    reportando "505 enfileiradas" enquanto nada roda, para sempre.

    MUTAÇÃO: remover o DELETE de `done`/`dead` no início de `relens_sweep`.
    """
    claim_id = _verified_claim(ctx, "c0")
    ctx.queue.enqueue("relens_sweep", {})
    await _runner(ctx).drain(max_tasks=1)
    ctx.store.conn.execute(
        "UPDATE tasks SET status = 'dead' WHERE kind = 'relens_claim'")

    ctx.queue.enqueue("relens_sweep", {}, dedup_key="sweep-2")
    await _runner(ctx).drain(max_tasks=1)
    pendentes = ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE kind = 'relens_claim' "
        "  AND status = 'pending'").fetchone()["n"]
    assert pendentes == 1, (
        "a chave ficou queimada pela tarefa morta: a claim nunca mais é julgada"
    )
    assert claim_id


async def test_relens_skips_claims_already_judged_in_this_focus(ctx):
    """O sweep não reenfileira quem já tem aresta — é o que torna o retomar grátis."""
    _verified_claim(ctx, "julgada", judged=True)
    _verified_claim(ctx, "pendente")
    ctx.queue.enqueue("relens_sweep", {})
    await _runner(ctx).drain(max_tasks=1)
    n = ctx.store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE kind = 'relens_claim'").fetchone()["n"]
    assert n == 1


async def test_a_harvest_task_from_another_focus_refuses_instead_of_degrading(ctx):
    """Tarefa enfileirada sob o foco A RECUSA ao executar sob o B.

    MEDIDO o que a degradação silenciosa custa: `strategy_by_name(perfil_B)` não conhece
    o nome, o handler cai no ramo ad-hoc, a prioridade cai de 0,95 para 0,5,
    `expected_directness` vira INDIRECT — e a query executada continua sendo a do foco
    A. Os resultados entram no corpus julgados contra o alvo do foco B.

    MUTAÇÃO: remover `_guard_focus` de `harvest_query` e deixar o ramo ad-hoc absorver.
    """
    ctx.queue.enqueue("harvest_query",
                      {"strategy": "linfoma_canino_direto", "query": "x",
                       "focus_id": 999})
    await _runner(ctx).drain(max_tasks=1)
    dead = ctx.store.conn.execute(
        "SELECT error FROM tasks WHERE kind = 'harvest_query'").fetchone()["error"]
    assert dead and "999" in dead, dead


async def test_relens_judges_against_the_focus_carried_in_the_payload(ctx):
    """O ALVO renderizado vem do foco do payload, não de `active_focus()`.

    Esta é a metade que a asserção óbvia não cobre: olhar o `focus_id` ESCRITO deixa o
    teste verde mesmo com o alvo vindo do foco errado, porque o id vem do payload nos
    dois casos. MUTAÇÃO EXECUTADA: `target=(ctx.store.active_focus() or row)["target"]`
    dentro de `relens_claim` — antes desta trava ela NÃO matava nada.

    Cenário real: um relens de 4.800 claims leva 7,68 h; na hora 3 o usuário troca de
    foco. As ~2.000 claims restantes recebem no prompt o alvo do foco NOVO e a linha é
    gravada sob o foco ANTIGO, sem log e sem nada em `judged_at` que permita
    reconstruir onde foi o corte.
    """
    claim_id = _verified_claim(ctx, "c0")
    scale = ctx.store.conn.execute(
        "SELECT scale_id FROM active_focus").fetchone()["scale_id"]
    shutil.copytree(ctx.config.focuses_dir / "onco-vet",
                    ctx.config.focuses_dir / "onco-vet-c", dirs_exist_ok=True)
    f = ctx.config.focuses_dir / "onco-vet-c" / "focus.toml"
    f.write_text(f.read_text(encoding="utf-8")
                 .replace('slug   = "onco-vet"', 'slug   = "onco-vet-c"'),
                 encoding="utf-8")
    cur = ctx.store.conn.execute(
        "INSERT INTO focuses(slug, target, scale_id) "
        "VALUES('onco-vet-c', 'feline injection-site sarcoma', ?) RETURNING id", (scale,))
    outro = int(cur.fetchone()["id"])

    ctx.queue.enqueue("relens_claim",
                      {"claim_id": claim_id, "focus_id": outro,
                       "target": "feline injection-site sarcoma"})
    await _runner(ctx).drain(max_tasks=1)

    prompt = next(p for p in ctx.llm.prompts if "judging ONE claim" in p)
    assert "feline injection-site sarcoma" in prompt
    assert "canine multicentric lymphoma" not in prompt, (
        "o julgamento usou o alvo do foco ATIVO, não o do payload"
    )
