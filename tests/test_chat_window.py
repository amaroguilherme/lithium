"""O turno de chat cabe na janela — com TODOS os blocos nos seus tetos.

Isto é conserto de um defeito que já existia ANTES da Fase C, e é pré-requisito dela:
acrescentar dois blocos a um prompt que já estourava mata o chat sem diagnóstico.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence

import pytest

from lithium.chat import (
    CONSTRAINT_NOTES_CHAR_BUDGET,
    HISTORY_CHAR_BUDGET,
    MAX_REPLY_TOKENS,
    MEMORY_CHAR_BUDGET,
    RECON_NOTES_CHAR_BUDGET,
    ChatEngine,
)
from lithium.db import Store
from lithium.llm.prompts import (
    BUDGET_MARGIN,
    CHARS_PER_TOKEN,
    PromptTooLarge,
    estimate_tokens,
    render,
)

from conftest import prod_profile

PROFILE = prod_profile()
DIM = 8
N_CTX = 8192


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


class CapturingLLM:
    def __init__(self) -> None:
        self.messages: list[list[dict]] = []

    async def complete(self, messages, **kw):
        self.messages.append(messages)
        return "ok"

    async def structured(self, *a, **k):
        raise AssertionError("não deve detectar memória aqui")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "w.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _engine(store, llm=None, n_ctx=N_CTX) -> ChatEngine:
    return ChatEngine(store, llm or CapturingLLM(), FakeEmbedder(), profile=PROFILE,
                      n_ctx=n_ctx)


def _fill(store) -> None:
    """Enche TODOS os blocos até os tetos."""
    for i in range(40):
        store.conn.execute(
            "INSERT INTO memories(text, kind, source, confirmed, active, text_key) "
            "VALUES(?, 'constraint', 'chat', 1, 1, ?)",
            ("o usuário evita lítio " + "x" * 200 + str(i), f"k{i}"))
    for i in range(20):
        store.conn.execute(
            "INSERT INTO memories(text, kind, source, confirmed, active, focus_id, "
            "                     provenance, text_key) "
            "VALUES(?, 'fact', 'recon', 1, 1, 1, '{\"url\":\"https://x.invalid/p\"}', ?)",
            ("a página diz " + "y" * 200 + str(i), f"r{i}"))
    for i in range(40):
        store.conn.execute(
            "INSERT INTO messages(role, content) VALUES('user', ?)", ("z" * 900,))


def test_the_worst_case_turn_fits_the_window_with_every_block_at_its_cap():
    """MEDIDO renderizando o `chat.md` real com o perfil real:

        sistema vazio                       3.819 chars =   954 tok
        pior caso com os blocos da Fase C  14.919 chars = 3.729 tok
        + histórico CONSTANTE de 12.000 chars + 1.536 de saída = 8.265 tok
        janela utilizável (8192 − 256)                         = 7.936 tok  → ESTOURA

    Ou seja o chat já estourava ANTES desta fase, e o `except PromptTooLarge` que daria
    a mensagem certa era INALCANÇÁVEL porque `send()` nunca chamava `budget_guard`: o
    llama-server devolvia 400, subia como `LLMError`, e o usuário lia "falha na chamada".

    MUTAÇÃO: voltar `_history_budget` para a constante `HISTORY_CHAR_BUDGET`. A soma
    abaixo passa dos utilizáveis.
    """
    worst = render("chat", **PROFILE.prompt_blocks("chat"),
                   memories="m" * MEMORY_CHAR_BUDGET,
                   evidence="e" * 3_000,
                   constraint_notes="c" * CONSTRAINT_NOTES_CHAR_BUDGET,
                   recon_notes="r" * RECON_NOTES_CHAR_BUDGET,
                   safety="s" * 800)
    usable = N_CTX - BUDGET_MARGIN
    with_constant_history = (estimate_tokens(worst)
                             + HISTORY_CHAR_BUDGET // CHARS_PER_TOKEN
                             + MAX_REPLY_TOKENS)
    assert with_constant_history > usable, (
        "a premissa deste conserto evaporou; recalibre os tetos antes de simplificar"
    )

    derived = max(0, min(HISTORY_CHAR_BUDGET,
                         (usable - MAX_REPLY_TOKENS - estimate_tokens(worst))
                         * CHARS_PER_TOKEN))
    assert estimate_tokens(worst) + derived // CHARS_PER_TOKEN + MAX_REPLY_TOKENS \
        <= usable


async def test_a_full_turn_does_not_raise_and_the_history_shrinks_by_itself(store):
    """Ponta a ponta pelo `send()` real, com os blocos cheios."""
    _fill(store)
    llm = CapturingLLM()
    await _engine(store, llm).send("e agora?", detect_memory=False)

    messages = llm.messages[0]
    total = estimate_tokens("".join(m["content"] for m in messages))
    assert total + MAX_REPLY_TOKENS <= N_CTX - BUDGET_MARGIN, total
    assert len(messages) > 1, "o histórico foi zerado em vez de encolher"


async def test_the_guard_fires_before_the_post_instead_of_after_the_400(store):
    """`budget_guard` no `send()`. Sem ele, o llama-server devolve 400, o cliente
    corretamente não repete 4xx, e o `LLMError` sobe sem diagnóstico."""
    _fill(store)
    engine = _engine(store, n_ctx=2048)
    with pytest.raises(PromptTooLarge, match="chat"):
        await engine.send("e agora?", detect_memory=False)


async def test_the_message_distinguishes_clear_the_history_from_nothing_fits(store):
    """Duas causas, dois remédios. Com `n_ctx` minúsculo nem o sistema cabe, e mandar o
    usuário rodar `/novo` seria prescrever o que não resolve."""
    _fill(store)
    assert _engine(store, n_ctx=2048).system_alone_overflows() is True
    assert _engine(store, n_ctx=N_CTX).system_alone_overflows() is False


def test_the_history_budget_is_derived_not_constant(store):
    """MUTAÇÃO: `_history_budget` devolver `HISTORY_CHAR_BUDGET` sempre."""
    engine = _engine(store)
    assert engine._history_budget("x" * 100) == HISTORY_CHAR_BUDGET
    assert engine._history_budget("x" * 20_000) < HISTORY_CHAR_BUDGET
    assert engine._history_budget("x" * 40_000) == 0


async def test_a_dead_embedder_does_not_lose_a_confirmed_memory(store):
    """`ChatEngine.remember` fazia `embed()` ANTES de qualquer INSERT — literalmente o
    bug de `test_escalation_memory.py::test_a_dead_embedder_still_records_and_escalates`
    ("a pergunta sumia inteira"). O estado normal depois de `lithium mode off` é o
    llama-server fora do ar, e é justamente quando o usuário diz "sim".
    """
    from lithium.llm.schemas import MemoryProposal

    class Dead:
        async def embed(self, texts):
            raise RuntimeError("fora do ar")

    engine = ChatEngine(store, CapturingLLM(), Dead(), profile=PROFILE)
    memory_id = await engine.remember(
        MemoryProposal(worth_remembering=True, text="prefere manhã", kind="preference",
                       rationale="r"))
    row = store.conn.execute(
        "SELECT text, embedding, text_key FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    assert row["text"] == "prefere manhã"
    assert row["embedding"] is None
    assert row["text_key"], "sem chave a memória escapa da deduplicação"
