"""Cliente do LLM — HTTP contra uma API OpenAI-compatible.

`llama-server` roda como processo separado, nunca como biblioteca in-process. Duas
razões: o binding in-process segura a GIL e mataria o worker pool asyncio; e manter
a fronteira em HTTP é o que torna a migração Mac→Windows (Metal→CUDA) uma troca de
build, não de código.

O método que importa é `structured()`: decodificação restrita por gramática com
auto-reparo. Um 12B sem gramática não produz JSON confiável em volume — e "em volume"
é o regime normal deste sistema.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from lithium.llm.usage import CallRecord

log = logging.getLogger(__name__)

UsageHook = Callable[[CallRecord], None]

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Servidor fora do ar ou inalcançável — vale retry no nível da task."""


class LLMInvalidOutput(LLMError):
    """O modelo não produziu saída válida nem após as tentativas de reparo."""


class LLMTruncated(LLMError):
    """Bateu o teto de tokens antes de fechar a saída.

    Gemma-4 é um modelo de raciocínio: emite um bloco de thinking que o llama-server
    separa em `reasoning_content`. Com o raciocínio ligado, esse bloco consome o
    orçamento inteiro e `content` volta vazio — o JSON nunca chega a ser emitido.

    Suba o llama-server com `--reasoning-budget 0` para as etapas mecânicas
    (extração, classificação, plano de busca). O parâmetro por requisição é
    ignorado; só a flag de servidor vale.

    Isto é erro determinístico: repetir com o mesmo orçamento dá o mesmo resultado.
    Por isso falha na hora em vez de gastar as tentativas de reparo.
    """


def _health_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/health"


def _content(data: dict[str, Any]) -> str:
    return data["choices"][0]["message"].get("content") or ""


def _was_truncated(data: dict[str, Any]) -> bool:
    return data["choices"][0].get("finish_reason") == "length"


def _reasoning_tokens(data: dict[str, Any]) -> int:
    """Quanto do orçamento foi gasto pensando em vez de respondendo."""
    reasoning = data["choices"][0]["message"].get("reasoning_content") or ""
    return len(reasoning) // 4  # estimativa grosseira, só para a mensagem de erro


class LLMClient:
    """Cliente de geração. Use como async context manager."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 300.0,
        max_retries: int = 3,
        temperature: float = 0.2,
        on_usage: UsageHook | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.on_usage = on_usage
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ─────────────────────────────────────────────────────────────────── saúde

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(_health_url(self.base_url), timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def wait_healthy(self, timeout_s: float = 180.0, interval_s: float = 2.0) -> None:
        """Carregar um GGUF de 8 GB leva dezenas de segundos — o daemon espera em vez
        de falhar na primeira task."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if await self.healthy():
                return
            await asyncio.sleep(interval_s)
        raise LLMUnavailable(f"llama-server não respondeu em {timeout_s}s: {self.base_url}")

    # ───────────────────────────────────────────────────────────────── geração

    def _emit(self, label: str, attempt: int, started: float, *, ok: bool,
              data: dict[str, Any] | None = None) -> None:
        """Uma linha por POST. Nunca levanta — ver `llm/usage.py`."""
        if self.on_usage is None:
            return
        usage = (data or {}).get("usage") or {}
        try:
            self.on_usage(
                CallRecord(
                    label=label,
                    attempt=attempt,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    ok=ok,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=(
                        (data or {}).get("choices", [{}])[0].get("finish_reason")
                        if data else None
                    ),
                )
            )
        except Exception:  # noqa: BLE001
            log.debug("hook de uso levantou; ignorando", exc_info=True)

    async def _post_chat(
        self, payload: dict[str, Any], *, label: str = "unlabeled", attempt: int = 0
    ) -> dict[str, Any]:
        last: Exception | None = None
        for retry in range(self.max_retries):
            started = time.monotonic()
            try:
                r = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as exc:
                self._emit(label, attempt, started, ok=False)
                # 4xx é payload malformado nosso: repetir não conserta.
                if 400 <= exc.response.status_code < 500:
                    raise LLMError(
                        f"{exc.response.status_code}: {exc.response.text[:500]}"
                    ) from exc
                last = exc
            except httpx.HTTPError as exc:
                self._emit(label, attempt, started, ok=False)
                last = exc
            else:
                self._emit(label, attempt, started, ok=True, data=data)
                return data
            await asyncio.sleep(2.0 * (2**retry))
        raise LLMUnavailable(f"falhou após {self.max_retries} tentativas: {last}")

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "complete",
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = await self._post_chat(payload, label=label)
        return _content(data)

    async def structured[T: BaseModel](
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        repair_attempts: int = 2,
        label: str | None = None,
    ) -> T:
        """Gera saída conforme `schema`, com gramática + auto-reparo.

        A gramática garante JSON sintaticamente válido e com as chaves certas, mas não
        garante restrições semânticas (faixa numérica, min_length de lista). Quando o
        Pydantic reprova, devolvemos o erro ao modelo e pedimos correção — barato, e
        resolve a grande maioria dos casos que sobram.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        convo = list(messages)
        last_error = ""
        tag = label or schema.__name__
        for attempt in range(repair_attempts + 1):
            payload["messages"] = convo
            data = await self._post_chat(payload, label=tag, attempt=attempt)
            raw = _content(data)

            if _was_truncated(data) and not raw:
                # Determinístico: repetir com o mesmo orçamento repete o resultado.
                # Falhar aqui em vez de queimar as tentativas de reparo.
                raise LLMTruncated(
                    f"{schema.__name__}: teto de tokens atingido com `content` vazio "
                    f"(~{_reasoning_tokens(data)} tokens gastos em raciocínio). "
                    "Suba o llama-server com `--reasoning-budget 0` ou aumente max_tokens."
                )
            try:
                return schema.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = str(exc)[:800]
                log.warning(
                    "saída inválida para %s (tentativa %d/%d): %s",
                    schema.__name__, attempt + 1, repair_attempts + 1, last_error,
                )
                convo = [
                    *convo,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Sua saída falhou na validação:\n"
                            f"{last_error}\n\n"
                            "Responda de novo com o JSON corrigido. Só o JSON."
                        ),
                    },
                ]
        raise LLMInvalidOutput(
            f"{schema.__name__} inválido após {repair_attempts + 1} tentativas: {last_error}"
        )


class EmbeddingClient:
    """Segunda instância de llama-server, em modo --embedding."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        dim: int = 1024,
        timeout_s: float = 120.0,
        batch_size: int = 16,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def __aenter__(self) -> EmbeddingClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(_health_url(self.base_url), timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeddings em lote, na ordem de entrada."""
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                r = await self._client.post(
                    f"{self.base_url}/embeddings", json={"model": self.model, "input": batch}
                )
                r.raise_for_status()
            except httpx.HTTPError as exc:
                raise LLMUnavailable(f"servidor de embeddings indisponível: {exc}") from exc

            # A resposta pode vir fora de ordem; `index` é a fonte de verdade.
            items = sorted(r.json()["data"], key=lambda d: d["index"])
            for item in items:
                vec = item["embedding"]
                if len(vec) != self.dim:
                    raise LLMError(
                        f"embedding com {len(vec)} dims, config diz {self.dim}. "
                        "Ajuste embedding.dim ou recrie o índice vetorial."
                    )
                out.append(vec)
        return out
