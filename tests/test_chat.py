"""Conversa e memória.

A propriedade que mais importa aqui é a que o usuário pediu explicitamente: o sistema
**propõe** memórias e nunca grava sozinho. Num domínio de saúde, acumular fatos sobre a
pessoa sem autorização é diferente de acumular papers.

A segunda é a separação entre memória e claim. Memória é sobre você e não tem citação;
claim é sobre o mundo e tem PMID. Misturá-las contaminaria o peso de evidência com
material não-publicado.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from collections.abc import Sequence

import pytest

from lithium.chat import ChatEngine
from lithium.db import Store
from lithium.llm import LLMError
from lithium.llm.schemas import MemoryProposal
from lithium.types import Directness, Grade

from conftest import onco_profile, prod_profile

PROFILE = onco_profile()


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


class ScriptedLLM:
    def __init__(self, replies=("resposta",), proposals=()) -> None:
        self._replies = list(replies)
        self._proposals = list(proposals)
        self.systems: list[str] = []
        self.histories: list[list] = []

    async def complete(self, messages, **kw):
        self.systems.append(messages[0]["content"])
        self.histories.append(messages[1:])
        return self._replies.pop(0) if self._replies else "resposta"

    async def structured(self, messages, schema, **kw):
        if schema is MemoryProposal:
            item = self._proposals.pop(0) if self._proposals else _no_memory()
            if isinstance(item, Exception):
                raise item
            return item
        raise AssertionError(f"schema inesperado: {schema}")


def _no_memory() -> MemoryProposal:
    return MemoryProposal(worth_remembering=False, text="", kind="", rationale="")


def _memory(text="O usuário evita fármacos com monitoramento sérico",
            kind="constraint") -> MemoryProposal:
    return MemoryProposal(worth_remembering=True, text=text, kind=kind,
                          rationale="restrição durável que muda o que investigar")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "c.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _engine(store, llm) -> ChatEngine:
    return ChatEngine(store, llm, FakeEmbedder(), profile=PROFILE)


def _seed_claim(store, statement: str, intervention: str = "quetiapine") -> None:
    sid = store.upsert_source(kind="pubmed", external_id="31415926", raw={},
                              title="Estudo", year=2021)
    cid = store.add_chunk(source_id=sid, ord=0,
                          text=f"{statement} Trecho longo o suficiente para indexar.")
    _c = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "grade, scale_id, confidence, verified) VALUES(?,?,?,?,'positive',?,1,1.0,1) "
        "RETURNING id",
        (sid, json.dumps([cid]), statement, intervention, Grade.RCT.value),
    )
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (int(_c.fetchone()["id"]), Directness.PARTIAL.value),
    )
    return cid


# ────────────────────────────────────────────────────────────── turno básico


async def test_send_persists_both_sides_of_the_exchange(store):
    engine = _engine(store, ScriptedLLM(["olá de volta"]))
    turn = await engine.send("olá")

    assert turn.reply == "olá de volta"
    rows = store.conn.execute("SELECT role, content FROM messages ORDER BY id").fetchall()
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", "olá"), ("assistant", "olá de volta")
    ]


async def test_history_is_replayed_in_order(store):
    llm = ScriptedLLM(["um", "dois"])
    engine = _engine(store, llm)
    await engine.send("primeira")
    await engine.send("segunda")

    segunda = llm.histories[1]
    assert [m["content"] for m in segunda] == ["primeira", "um", "segunda"]


async def test_history_is_windowed(store):
    llm = ScriptedLLM(["r"] * 10)
    engine = ChatEngine(store, llm, FakeEmbedder(), history_turns=2, profile=PROFILE)
    for i in range(5):
        await engine.send(f"mensagem {i}")
    assert len(llm.histories[-1]) == 4  # 2 turnos = 4 mensagens


async def test_clear_history_wipes_the_conversation(store):
    engine = _engine(store, ScriptedLLM())
    await engine.send("oi")
    assert engine.clear_history() == 2
    assert store.conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"] == 0


# ───────────────────────────────────────────────────── acesso ao corpus


async def test_relevant_claims_are_injected_with_pmids(store):
    _seed_claim(store, "Quetiapina reduziu escores de ansiedade frente a placebo.")
    llm = ScriptedLLM()
    turn = await _engine(store, llm).send("o que se sabe sobre quetiapina e ansiedade?")

    assert "PMID:31415926" in llm.systems[0]
    assert "rct" in llm.systems[0] and "partial" in llm.systems[0]
    assert turn.cited == ["31415926"]


async def test_empty_corpus_is_stated_explicitly(store):
    """Silêncio do corpus precisa chegar ao modelo como fato, não como ausência —
    senão ele preenche o vazio."""
    llm = ScriptedLLM()
    await _engine(store, llm).send("e sobre pregabalina?")
    assert "corpus is empty" in llm.systems[0]
    assert "offer to start the research" in llm.systems[0]


async def test_retrieval_failure_degrades_instead_of_crashing(store):
    """Sem servidor de embeddings a conversa continua, sem corpus."""

    class BrokenEmbedder:
        async def embed(self, texts):
            raise RuntimeError("servidor caiu")

    engine = ChatEngine(store, ScriptedLLM(["ainda respondo"]), BrokenEmbedder(), profile=PROFILE)
    turn = await engine.send("oi", detect_memory=False)
    assert turn.reply == "ainda respondo"
    assert turn.cited == []


def test_prompt_forbids_blurring_evidence_and_opinion(store):
    from lithium.llm.prompts import render

    prompt = render("chat", memories="", evidence="", constraint_notes="",
                    recon_notes="", safety="",
                    **prod_profile().prompt_blocks("chat"))
    assert "must never be blurred" in prompt
    # Conceito, não grafia: a invariante de que os prompts substantivos nomeiam o risco
    # de virada mora em test_prompt_contract.py; aqui só confirmamos que chat.md o traz.
    assert re.search(r"manic[- ]switch", prompt)


# ────────────────────────────────────────────── memória: propor, nunca gravar


async def test_memory_is_proposed_not_saved(store):
    """O requisito central: ele pergunta antes de guardar."""
    engine = _engine(store, ScriptedLLM(proposals=[_memory()]))
    turn = await engine.send("não quero nada que exija monitoramento sérico")

    assert turn.proposal is not None
    assert turn.proposal.kind == "constraint"
    assert store.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0


async def test_confirmed_memory_becomes_live(store):
    engine = _engine(store, ScriptedLLM(proposals=[_memory()]))
    turn = await engine.send("evito monitoramento sérico")
    memory_id = await engine.remember(turn.proposal)

    row = store.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    assert row["confirmed"] == 1 and row["active"] == 1
    assert row["confirmed_at"] is not None
    assert row["embedding"] is not None
    assert [m["id"] for m in engine.memories()] == [memory_id]


async def test_declined_memory_is_recorded_but_inactive(store):
    """Registrar a recusa evita repropor a mesma coisa toda conversa."""
    engine = _engine(store, ScriptedLLM(proposals=[_memory()]))
    turn = await engine.send("evito monitoramento sérico")
    engine.decline(turn.proposal)

    assert engine.memories() == []
    row = store.conn.execute("SELECT confirmed, active FROM memories").fetchone()
    assert row["confirmed"] == 0 and row["active"] == 0


async def test_no_proposal_for_ordinary_conversation(store):
    engine = _engine(store, ScriptedLLM(proposals=[_no_memory()]))
    assert (await engine.send("quetiapina funciona?")).proposal is None


async def test_empty_text_never_becomes_a_proposal(store):
    """`worth_remembering: true` com texto vazio é saída malformada, não memória."""
    bad = MemoryProposal(worth_remembering=True, text="  ", kind="fact", rationale="x")
    engine = _engine(store, ScriptedLLM(proposals=[bad]))
    assert (await engine.send("oi")).proposal is None


async def test_an_invalid_kind_is_rejected_not_coerced(store):
    """O contrato INVERTEU, e o motivo é a interação com a ordenação do bloco.

    Antes coagia para `'fact'`, o que parecia conservador — "perder a memória inteira
    por um rótulo errado seria desproporcional". Virou o contrário quando o bloco de
    memórias passou a ser ordenado por tipo: `'fact'` é o ÚLTIMO tier, então uma
    restrição que o modelo rotulou errado ia para o fim da fila **por construção**. A
    coerção e a ordenação se combinavam justamente contra a categoria mais crítica, sem
    deixar rastro.

    Perder uma proposta é recuperável e visível (fica no log, e dá para gravar com
    `lithium memories --add --kind`). Enterrar uma restrição é invisível.
    """
    odd = MemoryProposal(worth_remembering=True, text="não pode fazer coleta de sangue",
                         kind="inventado", rationale="x")
    engine = _engine(store, ScriptedLLM(proposals=[odd]))
    turn = await engine.send("oi")

    assert turn.proposal is None
    assert turn.reply, "a rejeição da proposta não pode derrubar o turno"


async def test_the_memory_block_puts_constraints_before_incidental_facts(store):
    """A ordenação que torna a coerção perigosa — e que por isso tem que existir."""
    for text, kind in [("mora sozinho", "fact"),
                       ("evita coleta de sangue frequente", "constraint"),
                       ("prefere manhã", "preference")]:
        await engine_remember(store, text, kind)

    block = _engine(store, ScriptedLLM())._memory_block()
    assert block.index("coleta de sangue") < block.index("prefere manhã")
    assert block.index("prefere manhã") < block.index("mora sozinho")


async def test_the_memory_block_declares_what_it_hides(store):
    """Teto em caracteres, não em contagem: o texto de uma memória não tem limite de
    tamanho, então contar linhas é contar a unidade errada — o mesmo erro que
    `HISTORY_CHAR_BUDGET` já corrigiu uma vez."""
    from lithium.chat import MEMORY_CHAR_BUDGET

    long_text = "x" * 500
    for i in range(20):
        await engine_remember(store, f"{long_text} {i}", "fact")

    block = _engine(store, ScriptedLLM())._memory_block()
    assert len(block) < MEMORY_CHAR_BUDGET + 1_000
    assert "não mostradas — este quadro é parcial" in block


async def engine_remember(store, text: str, kind: str) -> int:
    return await _engine(store, ScriptedLLM()).remember(
        MemoryProposal(worth_remembering=True, text=text, kind=kind, rationale="r"),
        source="chat",
    )


async def test_detection_failure_does_not_break_the_turn(store):
    engine = _engine(store, ScriptedLLM(["resposta"], [LLMError("caiu")]))
    turn = await engine.send("oi")
    assert turn.reply == "resposta" and turn.proposal is None


# ──────────────────────────────────────────────── memória no contexto


async def test_live_memories_are_injected_into_the_prompt(store):
    engine = _engine(store, ScriptedLLM(proposals=[_memory(), _no_memory()]))
    turn = await engine.send("evito monitoramento sérico")
    await engine.remember(turn.proposal)

    llm = ScriptedLLM()
    await _engine(store, llm).send("o que sugerir?")
    assert "monitoramento sérico" in llm.systems[0]
    assert "[constraint]" in llm.systems[0]


async def test_forgotten_memories_leave_the_context(store):
    engine = _engine(store, ScriptedLLM(proposals=[_memory()]))
    turn = await engine.send("evito monitoramento sérico")
    memory_id = await engine.remember(turn.proposal)

    assert engine.forget(memory_id) is True
    assert engine.forget(memory_id) is False, "esquecer duas vezes não deve reportar sucesso"

    llm = ScriptedLLM()
    await _engine(store, llm).send("e agora?")
    assert "monitoramento sérico" not in llm.systems[0]


async def test_existing_memories_are_shown_to_the_detector(store):
    """Sem isto ele repropõe a mesma memória toda conversa."""
    engine = _engine(store, ScriptedLLM(proposals=[_memory()]))
    await engine.remember(_memory())

    from lithium.llm.prompts import render

    existing = "\n".join(f"  - {m['text']}" for m in engine.memories())
    prompt = render("detect_memory", existing=existing, message="qualquer coisa")
    assert "monitoramento sérico" in prompt
    assert "do not propose again" in prompt.lower()


async def test_manual_memory_is_confirmed_immediately(store):
    engine = _engine(store, ScriptedLLM())
    memory_id = await engine.add_manual("Priorizar desfecho funcional", "preference")

    row = store.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    assert row["confirmed"] == 1 and row["source"] == "manual"
    assert row["kind"] == "preference"


# ───────────────────────────────────── memórias e claims ficam separadas


async def test_memories_never_enter_the_evidence_tables(store):
    """Memória é sobre você e não tem citação; claim é sobre o mundo e tem PMID.
    Misturá-las contaminaria o peso de evidência com material não-publicado."""
    engine = _engine(store, ScriptedLLM(proposals=[_memory()]))
    turn = await engine.send("evito monitoramento sérico")
    await engine.remember(turn.proposal)

    assert store.conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"] == 0
    assert store.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 1


async def test_nonempty_corpus_is_never_reported_as_empty(store):
    """Regressão de um caso real: com 1 claim indexada e nenhuma correspondência, o
    prompt dizia "nothing matches" e o modelo respondeu "o corpus está vazio" — uma
    afirmação diferente, e falsa. A contagem real torna a distinção impossível de
    perder."""
    _seed_claim(store, "Quetiapina reduziu HAM-A frente a placebo.")
    llm = ScriptedLLM()

    class NoMatchEmbedder(FakeEmbedder):
        pass

    engine = ChatEngine(store, llm, NoMatchEmbedder(), evidence_k=6, profile=PROFILE)
    # consulta ortogonal: o hash bag-of-words não casa com nada do chunk
    await engine.send("zzz qqq xyz", detect_memory=False)

    system = llm.systems[0]
    if "none matched" in system:
        assert "1 verified claim" in system
        assert "Do NOT say the corpus is empty" in system


async def test_truly_empty_corpus_says_so(store):
    llm = ScriptedLLM()
    await _engine(store, llm).send("qualquer coisa", detect_memory=False)
    assert "corpus is empty" in llm.systems[0]
