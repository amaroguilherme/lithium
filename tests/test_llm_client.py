"""Testes do cliente LLM com transporte mockado — sem rede, sem GPU.

O teste de integração contra um llama-server real está em test_llm_live.py e só roda
com `-m live`.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from lithium.llm.client import (
    EmbeddingClient,
    LLMClient,
    LLMError,
    LLMInvalidOutput,
    LLMTruncated,
    LLMUnavailable,
)


class Toy(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)


def _chat_reply(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content, "role": "assistant"}}]}
    )


def _client(handler) -> LLMClient:
    c = LLMClient("http://test/v1", "toy-model", max_retries=2)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


# ────────────────────────────────────────────────────────────────── geração básica


async def test_complete_returns_content():
    c = _client(lambda _: _chat_reply("olá"))
    assert await c.complete([{"role": "user", "content": "oi"}]) == "olá"
    await c.aclose()


async def test_null_content_becomes_empty_string():
    """llama-server pode devolver content: null; não pode virar TypeError adiante."""
    handler = lambda _: httpx.Response(  # noqa: E731
        200, json={"choices": [{"message": {"content": None}}]}
    )
    c = _client(handler)
    assert await c.complete([{"role": "user", "content": "x"}]) == ""
    await c.aclose()


# ────────────────────────────────────────────────────────── decodificação restrita


async def test_structured_sends_json_schema_and_parses():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _chat_reply('{"name": "quetiapina", "score": 0.8}')

    c = _client(handler)
    out = await c.structured([{"role": "user", "content": "x"}], Toy)
    await c.aclose()

    assert out == Toy(name="quetiapina", score=0.8)
    rf = seen["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Toy"
    assert rf["json_schema"]["strict"] is True
    assert "name" in rf["json_schema"]["schema"]["properties"]


async def test_structured_repairs_semantic_violation():
    """A gramática garante a forma, não a faixa numérica. score=5.0 é JSON válido
    mas viola o Pydantic — o reparo é o que fecha essa brecha."""
    replies = iter(['{"name": "a", "score": 5.0}', '{"name": "a", "score": 0.5}'])
    prompts: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["messages"])
        return _chat_reply(next(replies))

    c = _client(handler)
    out = await c.structured([{"role": "user", "content": "x"}], Toy)
    await c.aclose()

    assert out.score == 0.5
    # a segunda chamada precisa carregar a saída ruim + o erro de validação
    assert len(prompts[1]) == 3
    assert prompts[1][1]["role"] == "assistant"
    assert "validação" in prompts[1][2]["content"]


async def test_structured_gives_up_after_repair_attempts():
    c = _client(lambda _: _chat_reply('{"name": "a", "score": 99}'))
    with pytest.raises(LLMInvalidOutput, match="Toy"):
        await c.structured([{"role": "user", "content": "x"}], Toy, repair_attempts=1)
    await c.aclose()


async def test_truncated_empty_content_fails_fast():
    """Gemma-4 é modelo de raciocínio: o thinking pode consumir todo o orçamento e
    deixar `content` vazio. Isso é determinístico — repetir só queima minutos.

    Regressão de um caso real: três tentativas idênticas gastaram 14 minutos antes
    de desistir, quando a causa (raciocínio ligado) não muda entre tentativas.
    """
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "reasoning_content": "pensando " * 400},
                    }
                ]
            },
        )

    c = _client(handler)
    with pytest.raises(LLMTruncated, match="reasoning-budget"):
        await c.structured([{"role": "user", "content": "x"}], Toy)
    await c.aclose()
    assert calls["n"] == 1, "não pode gastar tentativas de reparo num erro determinístico"


async def test_truncation_with_partial_content_still_tries_repair():
    """Truncou mas veio algo: pode ser JSON recuperável, então vale a tentativa."""
    replies = iter(
        [
            ({"content": '{"name": "a", "sco'}, "length"),
            ({"content": '{"name": "a", "score": 0.3}'}, "stop"),
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        message, finish = next(replies)
        return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": message}]})

    c = _client(handler)
    assert (await c.structured([{"role": "user", "content": "x"}], Toy)).score == 0.3
    await c.aclose()


async def test_reasoning_content_is_ignored_for_parsing():
    """O JSON restrito vem em `content`; `reasoning_content` é ruído para nós."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"name": "paris", "score": 0.9}',
                            "reasoning_content": "hmm, deixa eu pensar...",
                        },
                    }
                ]
            },
        )

    c = _client(handler)
    assert (await c.structured([{"role": "user", "content": "x"}], Toy)).name == "paris"
    await c.aclose()


async def test_structured_handles_non_json_output():
    replies = iter(["desculpa, não posso", '{"name": "a", "score": 0.1}'])
    c = _client(lambda _: _chat_reply(next(replies)))
    assert (await c.structured([{"role": "user", "content": "x"}], Toy)).score == 0.1
    await c.aclose()


# ──────────────────────────────────────────────────────────────── erros e retry


async def test_retries_on_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else _chat_reply("ok")

    c = _client(handler)
    c.max_retries = 2
    # backoff real deixaria o teste lento; 2s é aceitável, mas cortamos
    import lithium.llm.client as mod

    original = mod.asyncio.sleep

    async def _fast(_):
        await original(0)

    mod.asyncio.sleep = _fast
    try:
        assert await c.complete([{"role": "user", "content": "x"}]) == "ok"
    finally:
        mod.asyncio.sleep = original
        await c.aclose()
    assert calls["n"] == 2


async def test_4xx_fails_immediately():
    """Payload malformado nosso: repetir só queima tempo."""
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad grammar")

    c = _client(handler)
    with pytest.raises(LLMError, match="400"):
        await c.complete([{"role": "user", "content": "x"}])
    await c.aclose()
    assert calls["n"] == 1


async def test_unreachable_server_raises_unavailable():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("recusado")

    c = _client(handler)
    import lithium.llm.client as mod

    original = mod.asyncio.sleep
    mod.asyncio.sleep = lambda _: original(0)
    try:
        with pytest.raises(LLMUnavailable):
            await c.complete([{"role": "user", "content": "x"}])
    finally:
        mod.asyncio.sleep = original
        await c.aclose()


async def test_health_endpoint_strips_v1_suffix():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    c = _client(handler)
    assert await c.healthy() is True
    await c.aclose()
    assert seen == ["http://test/health"]


# ──────────────────────────────────────────────────────────────────── embeddings


def _embed_client(handler, dim: int = 4) -> EmbeddingClient:
    c = EmbeddingClient("http://test/v1", "bge", dim=dim, batch_size=2)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


async def test_embed_preserves_input_order_across_batches():
    """batch_size=2 com 3 entradas: duas requisições, ordem precisa se manter."""

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        # devolve fora de ordem de propósito — `index` é quem manda
        data = [
            {"index": i, "embedding": [float(len(t))] * 4}
            for i, t in reversed(list(enumerate(texts)))
        ]
        return httpx.Response(200, json={"data": data})

    c = _embed_client(handler)
    out = await c.embed(["a", "bb", "ccc"])
    await c.aclose()
    assert [v[0] for v in out] == [1.0, 2.0, 3.0]


async def test_embed_empty_input_skips_request():
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("não deveria chamar o servidor")

    c = _embed_client(handler)
    assert await c.embed([]) == []
    await c.aclose()


async def test_embed_dim_mismatch_is_loud():
    """Trocar de modelo de embedding sem recriar o índice é um erro silencioso caro."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 768}]})

    c = _embed_client(handler, dim=1024)
    with pytest.raises(LLMError, match="768"):
        await c.embed(["x"])
    await c.aclose()
