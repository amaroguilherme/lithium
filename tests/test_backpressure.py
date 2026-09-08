"""Contrapressão e log durável — item H.

A primeira operação real (1-8/9) perdeu DUAS tarefas `fetch_source` porque as três
tentativas caíram na mesma janela de swap saturado: 14.898 de 15.360 MB, memória livre
0,14 GB, e o servidor de embeddings devolvendo `500` porque suas páginas foram paginadas e
não voltavam. Não era bug dele — era a máquina sem memória, e o pipeline não tinha como
ceder.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from lithium.llm.client import LLMError, LLMUnavailable
from lithium.worker.runner import Throttle, is_pressure


# ═════════════════════ o classificador: pressão vs dado ruim


def test_resource_failures_count_as_pressure():
    """`LLMUnavailable`, 429 e 5xx são recurso indisponível — esperar ajuda.

    MUTAÇÃO: fazer `is_pressure` devolver True para tudo.
    """
    assert is_pressure(LLMUnavailable("servidor de embeddings indisponível"))
    for status in (429, 500, 503):
        resp = httpx.Response(status, request=httpx.Request("GET", "http://x"))
        assert is_pressure(httpx.HTTPStatusError("x", request=resp.request,
                                                 response=resp)), status


def test_data_failures_do_not_count_as_pressure():
    """A CONTRAPARTIDA, e é ela que separa contrapressão de estrangulamento por bug.

    Um `ValueError` num payload malformado não melhora se o worker esperar, e reduzir a
    concorrência por causa dele deixa o sistema lento sem consertar nada.

    MUTAÇÃO: tratar qualquer `Exception` como pressão.
    """
    assert not is_pressure(ValueError("payload sem chave"))
    assert not is_pressure(KeyError("source_id"))
    assert not is_pressure(LLMError("json inválido"))          # erro de forma, não de rede
    resp = httpx.Response(404, request=httpx.Request("GET", "http://x"))
    assert not is_pressure(httpx.HTTPStatusError("x", request=resp.request, response=resp))


# ═════════════════════ a permissão: cai pela metade, sobe de um


async def test_pressure_halves_the_allowance_and_success_climbs_back():
    """Aumento aditivo, redução multiplicativa — a disciplina do TCP, pela mesma razão:
    subir devagar não reintroduz a pressão que acabou de ceder, e cair pela metade sai
    dela rápido.

    MUTAÇÃO: subir a permissão de volta ao máximo num único sucesso, ou reduzir de 1 em 1.
    """
    t = Throttle(8)
    assert t.allowed == 8

    await t.acquire(); await t.release(pressure=True, ok=False)
    assert t.allowed == 4, "não caiu pela metade"
    await t.acquire(); await t.release(pressure=True, ok=False)
    assert t.allowed == 2

    for esperado in (3, 4, 5):
        await t.acquire(); await t.release(ok=True)
        assert t.allowed == esperado, "não sobe de UM em um"


async def test_the_allowance_never_reaches_zero():
    """Piso de 1: o sistema desacelera, nunca para de progredir.

    MUTAÇÃO: `self.allowed = self.allowed // 2` sem o `max(1, ...)`.
    """
    t = Throttle(4)
    for _ in range(10):
        # Checa ANTES do próximo `acquire`: com a permissão em zero, `acquire` esperaria
        # para sempre e o teste TRAVARIA em vez de falhar. Um teste que trava não diz o
        # que quebrou — medido, ao rodar esta mutação pela primeira vez.
        assert t.allowed >= 1, "a permissão zerou; o pool travaria para sempre"
        await t.acquire()
        await t.release(pressure=True, ok=False)
    assert t.allowed == 1


async def test_a_data_failure_leaves_the_allowance_untouched():
    """Falha de dado devolve a vaga sem mexer na permissão: não é evidência de folga nem
    de aperto.

    MUTAÇÃO: tratar `ok=False` como pressão.
    """
    t = Throttle(4)
    await t.acquire(); await t.release(pressure=False, ok=False)
    assert t.allowed == 4


async def test_the_allowance_actually_blocks_concurrent_work():
    """A permissão precisa BLOQUEAR, não só contar. Sem isto ela é um número decorativo.

    MUTAÇÃO: em `acquire`, remover o `while self.in_flight >= self.allowed`.
    """
    t = Throttle(4)
    for _ in range(3):
        await t.acquire(); await t.release(pressure=True, ok=False)
    assert t.allowed == 1

    await t.acquire()                       # ocupa a única vaga
    segunda = asyncio.create_task(t.acquire())
    await asyncio.sleep(0)
    assert not segunda.done(), "a segunda execução passou apesar da permissão ser 1"

    await t.release(ok=True)                # libera; agora a segunda entra
    await asyncio.wait_for(segunda, timeout=1)
    assert t.in_flight == 1


async def test_a_failure_does_not_leak_its_slot():
    """A vaga é devolvida no `finally`, não no caminho de sucesso.

    Sem isto, depois de `allowed` falhas o pool inteiro travaria — pior que o problema que
    a contrapressão existe para resolver.

    MUTAÇÃO: mover o `release` do `finally` para o `else` em `Runner._execute`.
    """
    from lithium.worker.runner import Runner

    import inspect
    fonte = inspect.getsource(Runner._execute)
    corpo = fonte[fonte.index("finally:"):] if "finally:" in fonte else ""
    assert "throttle.release" in corpo, (
        "o release não está no `finally`: uma falha vaza a vaga e o pool trava"
    )
