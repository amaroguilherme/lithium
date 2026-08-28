"""Perguntas que só você pode responder: memória anexada, escalada mesmo assim.

`chat.py` prometia isto na docstring e o código não fazia: *"quando o loop de pesquisa
trava numa pergunta CONTEXT, ele consulta as memórias antes de escalar"*. A promessa
estava certa e faltava a metade que importa — **ele não responde sozinho.**

A cadeia que proíbe auto-responder está verificada no código: o único escritor de
`answer_origin='human'` é o CLI, e `training_examples` dá peso 3.0 a material humano.
Uma síntese de LLM rotulada como humana ali é irreversível — uma memória pode receber
`--forget`, uma atualização de pesos não.

E um bug vivo que estava adjacente: `add()` chamava `embed()` **antes** de qualquer
INSERT, sem guarda. Embedder fora do ar significava pergunta nunca gravada, nunca
escalada — ausência, que é a falha que ninguém nota.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence

import pytest

from lithium.db import Store
from lithium.llm.schemas import GeneratedQuestion, MemoryProposal
from lithium.pipeline.question import MAX_RECALLED, QuestionEngine
from lithium.types import QuestionKind, QuestionStatus

from conftest import onco_profile

PROFILE = onco_profile()


DIM = 16


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
        raise RuntimeError("servidor de embeddings caiu")


class NullLLM:
    async def structured(self, *a, **k):
        raise AssertionError("não deve chamar o LLM")

    async def complete(self, *a, **k):
        raise AssertionError("não deve chamar o LLM")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "q.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _engine(store, embedder=None) -> QuestionEngine:
    return QuestionEngine(store, NullLLM(), embedder or FakeEmbedder(), profile=PROFILE)


def _question(kind: QuestionKind, text: str = "o que já foi tentado?") -> GeneratedQuestion:
    return GeneratedQuestion(
        text=text, kind=kind, rationale="preciso saber o histórico", targets="",
    )


async def _remember(store, text: str, kind: str = "context") -> int:
    from lithium.chat import ChatEngine

    return await ChatEngine(store, NullLLM(), FakeEmbedder(), profile=PROFILE).remember(
        MemoryProposal(worth_remembering=True, text=text, kind=kind, rationale="r"),
        source="chat",
    )


def _partial_work(store, question_id: int) -> str:
    return store.conn.execute(
        "SELECT partial_work FROM questions WHERE id = ?", (question_id,)
    ).fetchone()["partial_work"]


# ─────────────────────────────────────────────── a memória chega, e não responde


async def test_a_context_question_carries_what_you_already_told_the_system(store):
    await _remember(store, "já tentou lamotrigina e teve rash")
    record = await _engine(store).add(_question(QuestionKind.CONTEXT))

    work = _partial_work(store, record.id)
    assert "lamotrigina" in work
    assert "memória #" in work, "a origem precisa estar visível"


async def test_the_recalled_block_says_it_is_memory_not_evidence(store):
    """O rótulo é o ponto. Sem ele, um bloco de memória ao lado de `partial_work`
    parece achado de pesquisa — e memória não tem PMID."""
    await _remember(store, "prefere evitar sedação diurna", kind="preference")
    record = await _engine(store).add(_question(QuestionKind.PREFERENCE))

    work = _partial_work(store, record.id)
    assert "memória, não evidência verificada" in work


async def test_it_escalates_anyway(store):
    """A metade que a docstring não dizia. Recuperar memória não é responder."""
    await _remember(store, "já tentou lítio, abandonou por tremor")
    record = await _engine(store).add(_question(QuestionKind.CONTEXT))

    row = store.conn.execute(
        "SELECT status, answer, answer_origin FROM questions WHERE id = ?", (record.id,)
    ).fetchone()
    assert row["status"] == QuestionStatus.ESCALATED.value
    assert row["answer"] is None
    assert row["answer_origin"] is None


async def test_nothing_in_this_path_can_write_answer_origin_human(store):
    """A trava contra o vazamento irreversível para o LoRA.

    `training_examples` dá peso 3.0 a material humano. Uma síntese de LLM rotulada como
    humana ali não tem `--forget`.
    """
    await _remember(store, "já tentou de tudo")
    engine = _engine(store)
    for kind in (QuestionKind.CONTEXT, QuestionKind.PREFERENCE,
                 QuestionKind.METHODOLOGICAL):
        await engine.add(_question(kind, f"pergunta {kind.value}"))

    origins = [
        r["answer_origin"]
        for r in store.conn.execute("SELECT answer_origin FROM questions")
    ]
    assert origins and all(o is None for o in origins)


HUMAN_ORIGIN_WRITERS = {
    ("lithium/cli.py", "answer"),
    ("lithium/pipeline/question.py", "answer_from_human"),
}
"""Quem pode carimbar uma resposta como humana.

São dois, não um — descobri escrevendo este teste, depois de ter afirmado o contrário
num docstring. Os dois recebem texto que veio de uma pessoa por um caminho síncrono
(`lithium answer <id> "<texto>"`), que é a propriedade que importa. O registro é literal
para que um terceiro escritor precise de uma decisão explícita: `training_examples` dá
peso 3.0 a material humano, e uma síntese de LLM rotulada assim não tem `--forget`."""


def test_only_registered_functions_can_stamp_an_answer_as_human():
    """Por AST, não por substring.

    A primeira versão deste teste procurava a string no arquivo inteiro — e casou o
    **docstring** que explicava a regra. Um teste que se acusa ao ser documentado é um
    teste que vai ser desligado.
    """
    import ast
    from pathlib import Path

    pkg = Path(__file__).resolve().parent.parent / "lithium"
    found: set[tuple[str, str]] = set()

    for path in pkg.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(pkg.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Constant):
                    continue                      # docstring: menciona, não escreve
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if "answer_origin = 'human'" in sub.value:
                        found.add((rel, node.name))

    assert found == HUMAN_ORIGIN_WRITERS, (
        f"o conjunto de escritores de answer_origin='human' mudou.\n"
        f"  novos={sorted(found - HUMAN_ORIGIN_WRITERS)}\n"
        f"  sumiram={sorted(HUMAN_ORIGIN_WRITERS - found)}\n"
        "Esse rótulo entra em `training_examples` com peso 3.0 e não é reversível: "
        "um escritor novo precisa de decisão explícita, não de um merge."
    )


# ──────────────────────────────────────────────────────── truncamento declarado


async def test_the_recalled_block_declares_what_it_hides(store):
    for i in range(MAX_RECALLED + 4):
        await _remember(store, f"fato número {i} sobre o caso")
    record = await _engine(store).add(_question(QuestionKind.CONTEXT))

    work = _partial_work(store, record.id)
    assert work.count("memória #") == MAX_RECALLED
    assert "+4 outras" in work


async def test_constraints_come_before_incidental_facts(store):
    await _remember(store, "mora sozinho", kind="fact")
    await _remember(store, "não pode fazer coleta de sangue frequente", kind="constraint")
    record = await _engine(store).add(_question(QuestionKind.CONTEXT))

    work = _partial_work(store, record.id)
    assert work.index("coleta de sangue") < work.index("mora sozinho")


async def test_an_auto_answerable_question_gets_no_memory_block(store):
    """Pergunta que a pesquisa responde não é lugar de preferência do usuário — seria
    a memória entrando no caminho que decide o que buscar."""
    await _remember(store, "já tentou lamotrigina")
    record = await _engine(store).add(_question(QuestionKind.FACTUAL, "quetiapina em TAG?"))

    work = _partial_work(store, record.id)
    assert work is None or "memória #" not in work


async def test_no_memory_means_no_block(store):
    record = await _engine(store).add(_question(QuestionKind.CONTEXT))
    work = _partial_work(store, record.id)
    assert "memória" not in (work or "")


# ────────────────────────────────────────── o embedder fora do ar não custa a pergunta


async def test_a_dead_embedder_still_records_and_escalates(store):
    """O bug vivo. `embed()` levantava antes de qualquer INSERT: a pergunta sumia
    inteira, sem linha no banco e sem entrada na fila humana."""
    record = await _engine(store, DeadEmbedder()).add(_question(QuestionKind.CONTEXT))

    assert record is not None
    row = store.conn.execute(
        "SELECT status, embedding FROM questions WHERE id = ?", (record.id,)
    ).fetchone()
    assert row["status"] == QuestionStatus.ESCALATED.value
    assert row["embedding"] is None, "sem embedder, sem vetor — mas a pergunta existe"


async def test_a_dead_embedder_degrades_dedup_not_persistence(store):
    """Sem dedup, o pior caso é uma paráfrase repetida na fila: visível e reversível.
    Perder a pergunta é invisível e não é."""
    engine = _engine(store, DeadEmbedder())
    first = await engine.add(_question(QuestionKind.FACTUAL, "quetiapina funciona em TAG?"))
    second = await engine.add(_question(QuestionKind.FACTUAL, "quetiapina funciona em TAG?"))

    assert first is not None and second is not None
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM questions"
    ).fetchone()["n"] == 2


async def test_dedup_still_works_when_the_embedder_is_alive(store):
    """A contrapartida: a degradação não pode virar o comportamento normal."""
    engine = _engine(store)
    first = await engine.add(_question(QuestionKind.FACTUAL, "quetiapina funciona em TAG?"))
    second = await engine.add(_question(QuestionKind.FACTUAL, "quetiapina funciona em TAG?"))

    assert first is not None
    assert second is None
