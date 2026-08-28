"""Fase 0: instrumentação e os dois tetos que estavam estourados.

O teste que mais importa é `test_coverage_table_is_capped_so_the_prompt_fits`: com o
código anterior, o mesmo cenário produzia ≥10.000 tokens contra uma janela de 8192, o
llama-server devolvia 400, o cliente corretamente não repetia 4xx, e a task ia para
dead-letter. **A trilha especulativa morria em silêncio conforme o corpus amadurecia**, e
nenhum teste pegava porque o sintoma só aparece com corpus grande.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from lithium.db import Store
from lithium.llm.client import LLMClient
from lithium.llm.prompts import (
    BUDGET_MARGIN,
    PromptTooLarge,
    budget_guard,
    estimate_tokens,
    render,
)
from lithium.llm.usage import CallRecord, UsageSink, summarize
from lithium.pipeline.mechanism import route_block, taxonomy_block
from lithium.pipeline.state import MAX_COVERAGE_ROWS, build_state
from lithium.types import Directness, Grade

from conftest import onco_profile, prod_profile

PROFILE = onco_profile()


DIM = 8


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "p0.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _seed_claims(store: Store, n: int) -> None:
    """`n` intervenções distintas — o que um 12B produz de verdade: 'quetiapine',
    'quetiapine XR', 'adjunctive quetiapine' são três linhas."""
    for i in range(n):
        sid = store.upsert_source(kind="pubmed", external_id=f"p{i}", raw={},
                                  title=f"Estudo {i}", year=2020)
        cid = store.add_chunk(source_id=sid, ord=0,
                              text="Um trecho de apoio com tamanho suficiente para indexar.")
        _cl = store.conn.execute(
            "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
            "grade, scale_id, confidence, verified) VALUES(?,?,?,?,'positive',?,1,0.9,1) "
            "RETURNING id",
            (sid, json.dumps([cid]), f"achado {i}",
             f"intervenção experimental número {i} de nome longo", Grade.RCT.value),
        ).fetchone()["id"]
        store.conn.execute(
            "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
            (int(_cl), Directness.PARTIAL.value),
        )


# ───────────────────────────────────── o teto que matava a trilha especulativa


def test_coverage_table_is_capped_so_the_prompt_fits(store):
    """Regressão do modo de falha mais grave encontrado.

    `claims.intervention` é texto livre, então a cardinalidade cresce com o corpus e
    não converge. Sem `LIMIT`, 200 intervenções distintas põem o prompt de especulação
    acima da janela — e o caminho de erro termina em dead-letter, silencioso.
    """
    _seed_claims(store, 200)
    state = build_state(store, PROFILE)

    assert len(state.coverage) == MAX_COVERAGE_ROWS
    assert state.coverage_omitted > 0

    # Com o perfil de PRODUÇÃO: esta trava sobrevive à parametrização e GANHA valor —
    # ela passa a medir o custo real do perfil, e um taxonomy.toml mais longo consome a
    # folga. Sem ela, a trilha especulativa some em dead-letter em silêncio conforme o
    # perfil cresce.
    prod = prod_profile()
    prompt = render("generate_speculation",
                    **prod.prompt_blocks("generate_speculation"),
                    taxonomy=taxonomy_block(prod),
                    routes=route_block(prod), existing="", lessons="",
                    state=state.render(), max_items=3)

    # É esta asserção que falhava antes da mudança.
    assert estimate_tokens(prompt) + 3072 <= 8192 - BUDGET_MARGIN


def test_truncated_coverage_declares_what_it_hides(store):
    """Um bloco truncado que não se anuncia faz o modelo raciocinar sobre um quadro
    parcial acreditando que é completo — a falha de fragmento-incompleto do artigo."""
    _seed_claims(store, 60)
    rendered = build_state(store, PROFILE).render()
    assert "não mostradas" in rendered
    assert "quadro é parcial" in rendered


def test_small_corpus_has_no_truncation_note(store):
    _seed_claims(store, 5)
    state = build_state(store, PROFILE)
    assert state.coverage_omitted == 0
    assert "não mostradas" not in state.render()


def test_conflicts_are_capped_and_ranked_by_consequence(store):
    """Conflito também é lista sem teto. Ordenar por conflito × peso põe primeiro a
    discordância que mais importa, não a que foi inserida primeiro."""
    for i in range(30):
        for direction in ("positive", "negative"):
            sid = store.upsert_source(kind="pubmed", external_id=f"c{i}{direction}", raw={})
            cid = store.add_chunk(source_id=sid, ord=0,
                                  text="Trecho de apoio longo o suficiente para indexar.")
            _cl2 = store.conn.execute(
                "INSERT INTO claims(source_id, chunk_ids, statement, intervention, "
                "direction, grade, scale_id, confidence, verified) "
                "VALUES(?,?,'x',?,?,'rct',1,0.9,1) RETURNING id",
                (sid, json.dumps([cid]), f"fármaco {i}", direction),
            ).fetchone()["id"]
            store.conn.execute(
                "INSERT INTO claim_directness(claim_id, focus_id, directness) "
                "VALUES(?, 1, 'partial')", (int(_cl2),),
            )
    assert len(build_state(store, PROFILE).conflicts) <= 10


# ──────────────────────────────────────────────────────── o budget guard


def test_budget_guard_aborts_before_the_call():
    with pytest.raises(PromptTooLarge, match="excedem a janela"):
        budget_guard("x" * 40_000, label="teste", max_tokens=1024, n_ctx=8192)


def test_budget_guard_message_carries_the_numbers():
    """Diagnóstico imediato: sem os números, a mensagem obriga a reproduzir."""
    with pytest.raises(PromptTooLarge) as exc:
        budget_guard("x" * 40_000, label="generate_speculation", max_tokens=3072, n_ctx=8192)
    text = str(exc.value)
    assert "generate_speculation" in text and "3072" in text and "8192" in text


def test_budget_guard_passes_and_returns_the_estimate():
    assert budget_guard("x" * 400, label="t", max_tokens=512, n_ctx=8192) == 100


async def test_explorer_does_not_post_when_the_prompt_is_too_large(store):
    """O guard tem que abortar **antes** do POST, não depois do 400."""
    _seed_claims(store, 400)

    class ExplodingLLM:
        async def structured(self, *a, **kw):
            raise AssertionError("não deveria chegar ao POST")

    from lithium.pipeline.explore import Explorer

    # n_ctx apertado de propósito: força o guard mesmo com a cobertura já capada.
    with pytest.raises(PromptTooLarge):
        await Explorer(store, ExplodingLLM(), n_ctx=2048, profile=PROFILE).generate()


# ──────────────────────────────────────────────────── instrumentação de custo


def _reply(prompt_tokens=100, completion_tokens=20) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


async def test_one_row_per_http_post_not_per_logical_call(store):
    """`structured()` permite dois reparos, e cada reparo anexa a saída ruim ao
    contexto. Contar por chamada lógica esconderia justamente o custo que mais dói."""
    from pydantic import BaseModel

    class Toy(BaseModel):
        ok: bool

    replies = iter([
        httpx.Response(200, json={"choices": [{"message": {"content": "lixo"},
                                               "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 100, "completion_tokens": 5}}),
        httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'},
                                               "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 260, "completion_tokens": 8}}),
    ])
    sink = UsageSink(store, flush_every=1000)
    client = LLMClient("http://t/v1", "m", on_usage=sink.record)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: next(replies))
    )
    await client.structured([{"role": "user", "content": "x"}], Toy)
    await client.aclose()
    await sink.flush()

    rows = store.conn.execute(
        "SELECT label, attempt, prompt_tokens FROM llm_calls ORDER BY id"
    ).fetchall()
    assert len(rows) == 2, "dois POSTs, duas linhas"
    assert [r["attempt"] for r in rows] == [0, 1]
    # O reparo carrega mais prompt que a primeira tentativa — o custo que se esconderia.
    assert rows[1]["prompt_tokens"] > rows[0]["prompt_tokens"]


async def test_elapsed_ms_is_always_recorded(store):
    """Sem tempo por label, nenhum número de wall-clock deste plano é falsificável."""
    sink = UsageSink(store, flush_every=1000)
    client = LLMClient("http://t/v1", "m", on_usage=sink.record)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _reply()))
    await client.complete([{"role": "user", "content": "x"}], label="chat")
    await client.aclose()
    await sink.flush()

    row = store.conn.execute("SELECT label, elapsed_ms, ok FROM llm_calls").fetchone()
    assert row["label"] == "chat"
    assert row["elapsed_ms"] >= 0
    assert row["ok"] == 1


async def test_failed_posts_are_recorded_too(store):
    """Uma falha que não aparece na contabilidade é uma falha que ninguém investiga."""
    sink = UsageSink(store, flush_every=1000)
    client = LLMClient("http://t/v1", "m", on_usage=sink.record)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(400, text="prompt grande"))
    )
    from lithium.llm import LLMError

    with pytest.raises(LLMError):
        await client.complete([{"role": "user", "content": "x"}], label="extract_claims")
    await client.aclose()
    await sink.flush()

    row = store.conn.execute("SELECT ok, prompt_tokens FROM llm_calls").fetchone()
    assert row["ok"] == 0 and row["prompt_tokens"] is None


async def test_a_raising_usage_hook_never_breaks_a_successful_call(store):
    """O medidor de tokens não pode causar gasto de token.

    Handlers rodam awaited na thread do event loop; uma exceção no caminho de medição
    converteria uma resposta de LLM bem-sucedida em falha de task e retry.
    """
    def exploding(_entry: CallRecord) -> None:
        raise RuntimeError("o medidor quebrou")

    client = LLMClient("http://t/v1", "m", on_usage=exploding)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _reply()))
    assert await client.complete([{"role": "user", "content": "x"}]) == "ok"
    await client.aclose()


async def test_sink_swallows_a_broken_store(store, tmp_path):
    """Perder telemetria é irrelevante; perder uma resposta de 60 s por causa dela não."""
    broken = Store(tmp_path / "sem-schema.db", embedding_dim=DIM)  # sem init_schema
    sink = UsageSink(broken, flush_every=1000)
    sink.record(CallRecord(label="t", attempt=0, elapsed_ms=1, ok=True))
    assert await sink.flush() == 0
    broken.close()


def test_sink_buffers_outside_an_event_loop(store):
    """Chamado de código síncrono, o registro não pode explodir por não haver loop."""
    sink = UsageSink(store, flush_every=1)
    sink.record(CallRecord(label="t", attempt=0, elapsed_ms=1, ok=True))
    assert asyncio.run(sink.flush()) == 1


def test_summarize_reports_per_label(store):
    sink = UsageSink(store, flush_every=1000)
    for i in range(10):
        sink.record(CallRecord(label="extract_claims", attempt=0, elapsed_ms=50_000,
                               ok=True, prompt_tokens=700 + i, completion_tokens=300))
    for i in range(3):
        sink.record(CallRecord(label="verify_citation", attempt=0, elapsed_ms=5_000,
                               ok=True, prompt_tokens=330, completion_tokens=40))
    asyncio.run(sink.flush())

    rows = {r["label"]: r for r in summarize(store)}
    assert rows["extract_claims"]["n"] == 10
    assert rows["verify_citation"]["n"] == 3
    assert rows["extract_claims"]["p50_prompt"] is not None
    # Ordenado por custo total, então a etapa dominante vem primeiro.
    assert summarize(store)[0]["label"] == "extract_claims"


def test_summarize_counts_repairs_separately(store):
    sink = UsageSink(store, flush_every=1000)
    sink.record(CallRecord(label="t", attempt=0, elapsed_ms=1, ok=True, prompt_tokens=10))
    sink.record(CallRecord(label="t", attempt=1, elapsed_ms=1, ok=True, prompt_tokens=30))
    asyncio.run(sink.flush())
    assert summarize(store)[0]["repairs"] == 1


# ─────────────────────────────────────────────── o teto do histórico de chat


async def test_chat_history_is_capped_by_size_not_count(store):
    """24 mensagens × 1536 tokens de resposta chegavam a ~18.400 tokens contra uma
    janela de 8192. Contar mensagens é a unidade errada."""
    from lithium.chat import HISTORY_CHAR_BUDGET, ChatEngine

    for i in range(20):
        store.conn.execute("INSERT INTO messages(role, content) VALUES('user', ?)",
                           ("x" * 5000,))
        store.conn.execute("INSERT INTO messages(role, content) VALUES('assistant', ?)",
                           ("y" * 5000,))

    engine = ChatEngine(store, None, None, profile=PROFILE)
    history = engine._history()
    total = sum(len(m["content"]) for m in history)
    assert total <= HISTORY_CHAR_BUDGET + 5000, "só o turno mais recente pode estourar"
    assert len(history) < 12


async def test_a_single_oversized_turn_is_never_dropped(store):
    """Descartar o turno atual deixaria a conversa sem a mensagem que a motivou."""
    from lithium.chat import ChatEngine

    store.conn.execute("INSERT INTO messages(role, content) VALUES('user', ?)",
                       ("x" * 99_000,))
    assert len(ChatEngine(store, None, None, profile=PROFILE)._history()) == 1


def test_history_preserves_chronological_order(store):
    from lithium.chat import ChatEngine

    for i in range(4):
        store.conn.execute("INSERT INTO messages(role, content) VALUES('user', ?)", (f"m{i}",))
    history = ChatEngine(store, None, None, profile=PROFILE)._history()
    assert [m["content"] for m in history] == ["m0", "m1", "m2", "m3"]


# ───────────────────────────────────────────── dedup_key estável entre restarts


def test_dedup_key_is_stable_across_processes():
    """`str.__hash__` é salgado por processo e o repo não fixa `PYTHONHASHSEED`, então
    a chave mudava a cada restart do daemon — e "a mesma query não roda duas vezes no
    dia" era falso."""
    import subprocess
    import sys

    code = (
        "from lithium.worker.handlers import _stable_key; "
        "print(_stable_key('bipolar disorder AND generalized anxiety'))"
    )
    seen = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       check=True).stdout.strip()
        for _ in range(3)
    }
    assert len(seen) == 1, f"chave instável entre processos: {seen}"
