"""O que a trilha de especulação recebe — e o que ela nunca pode receber.

`generate_speculation.md` instrui "não caia num modo de falha já nomeado aqui". Isso faz
de cada linha injetada uma **exclusão**: ela remove uma classe de mecanismo do espaço de
busca. E é o poder mais difícil de auditar de todo o sistema, porque nada reporta
"hipóteses que nunca foram geradas" — a falha não tem assinatura observável. Um teste
sobre a saída jamais a pegaria; só um teste sobre o *bloco injetado* pega.

Dois modos de falha simétricos, e os dois são silenciosos:

1. **Excesso de exclusão.** A prosa que o modelo escreveu num `dead_end` entra como
   instrução. "...logo TODOS os mecanismos glutamatérgicos são becos sem saída e nunca
   devem ser propostos" apaga uma classe inteira por um salto que ninguém revisou.
2. **Perda de capacidade.** O `pattern` — o único canal substantivo com portão de fonte
   — renderizado sob um cabeçalho que diz "não afirmam nada sobre biologia e não devem
   estreitar o espaço de mecanismos". O sistema continua gravando patterns e nada
   reporta que eles deixaram de ter efeito.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from lithium.db import Store
from lithium.pipeline.reflect import Lesson, Reflector, Ref
from lithium.types import Directness, Grade

from conftest import onco_profile

PROFILE = onco_profile()


DIM = 8

LEAP = (
    "A ansiólise do dextrometorfano nunca foi dissociada do antagonismo NMDA; "
    "portanto TODOS os mecanismos glutamatérgicos são becos sem saída e nunca "
    "devem ser propostos de novo."
)
FLAW = "A cadeia confunde efeito agudo com efeito sustentado no elo 3."


class FakeEmbedder:
    async def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(zlib.crc32(t.encode()) % (2**32))
            v = rng.normal(size=DIM).astype(np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return out


class NullLLM:
    async def structured(self, *a, **k):
        raise AssertionError("não deve chamar o LLM")


class ApprovingPatternGate:
    """LLM mínimo que aprova todo `pattern` e CONTA as chamadas.

    Necessário porque `remember()` de um `pattern` agora passa pelo portão de derivação.
    Contar é o ponto: um teste que só aprova não distingue "o portão rodou e aprovou" de
    "o portão não existe mais".
    """

    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, messages, schema, **kw):
        from lithium.llm.schemas import PatternVerdict

        assert schema is PatternVerdict, f"schema inesperado: {schema}"
        self.calls += 1
        return PatternVerdict(
            follows_from_cited_claims_alone=True,
            population_scope_exceeded=False,
            contradicts_a_cited_claim_direction=False,
            restates_a_premise=False,
            reason="segue das claims citadas",
        )

    async def complete(self, *a, **k):
        raise AssertionError("não deve chamar complete")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "c.db", embedding_dim=DIM)
    s.init_schema()
    s.conn.execute(
        "INSERT INTO sources(kind, external_id, title, raw_json) "
        "VALUES('pubmed', '1', 't', '{}')"
    )
    yield s
    s.close()


def _reflector(store) -> Reflector:
    return Reflector(store, NullLLM(), FakeEmbedder())


def _claim(store, claim_id: int, directness: Directness, grade=Grade.RCT) -> int:
    store.conn.execute(
        "INSERT INTO claims(id, source_id, chunk_ids, statement, direction, grade, "
        "  scale_id, confidence, verified) "
        "VALUES(?, 1, '[]', ?, 'positive', ?, 1, 0.9, 1)",
        (claim_id, f"claim {claim_id}", grade.value),
    )
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (claim_id, directness.value),
    )
    return claim_id


def _hypothesis(store, hid: int, *, refuted: bool, flaw: str = FLAW) -> int:
    store.conn.execute(
        "INSERT INTO hypotheses(id, focus_id, statement, tier, survives_critique, "
        "  critique_json) VALUES(?, 1, ?, 'speculative', ?, ?)",
        (hid, "dextrometorfano transdérmico em bipolar I com TAG",
         0 if refuted else 1, json.dumps({"fatal_flaw": flaw})),
    )
    return hid


def _lesson(kind: str, text: str, *, refs=None, claim_ids=None) -> Lesson:
    return Lesson(
        id=1, text=text, kind=kind,
        provenance=json.dumps({
            "note": "n",
            "claim_ids": claim_ids or [],
            "refs": {str(k): list(v) for k, v in (refs or {}).items()},
        }),
    )


# ─────────────────────────────────────── 1. a prosa do modelo não vira instrução


def test_the_model_prose_of_a_dead_end_never_reaches_the_prompt(store):
    """O ataque: um salto de classe com ordem imperativa dentro do texto da lição."""
    _hypothesis(store, 7, refuted=True)
    block = _reflector(store).lessons_for_speculation(
        [_lesson("dead_end", LEAP, refs={1: ["hypothesis", 7]})]
    )

    assert "TODOS os mecanismos glutamatérgicos" not in block
    assert "nunca devem ser propostos" not in block
    assert FLAW in block, "a exclusão real, a do banco, tem que aparecer"


def test_the_exclusion_comes_from_the_database_not_the_lesson(store):
    _hypothesis(store, 7, refuted=True)
    block = _reflector(store).lessons_for_speculation(
        [_lesson("dead_end", "qualquer prosa", refs={1: ["hypothesis", 7]})]
    )

    assert "dextrometorfano transdérmico em bipolar I" in block
    assert "qualquer prosa" not in block


def test_a_dead_end_whose_referent_stopped_being_refuted_disappears(store):
    """E não pode voltar como prosa do modelo — o fallback é justamente a porta que
    esta contenção fecha."""
    _hypothesis(store, 7, refuted=False)
    lesson = _lesson("dead_end", LEAP, refs={1: ["hypothesis", 7]})
    block = _reflector(store).lessons_for_speculation([lesson])

    assert FLAW not in block
    assert "TODOS os mecanismos glutamatérgicos" not in block, (
        "a lição saiu de cena, mas voltou pela porta do fallback"
    )
    assert block == "(none)"


def test_a_dead_end_without_a_hypothesis_reference_is_dropped(store):
    """Sem referente não há exclusão auditável — e a alternativa seria injetar a prosa."""
    block = _reflector(store).lessons_for_speculation([_lesson("dead_end", LEAP)])
    assert block == "(none)"


def test_the_exclusion_says_a_refutation_is_not_class_wide(store):
    """A instrução que impede o modelo de fazer sozinho o salto que a contenção
    impediu a lição de fazer."""
    _hypothesis(store, 7, refuted=True)
    block = _reflector(store).lessons_for_speculation(
        [_lesson("dead_end", "x", refs={1: ["hypothesis", 7]})]
    )
    assert "NOT a refutation of its whole class" in block


# ──────────────────────────────── 2. o canal bem fundamentado não é declawed


def test_a_pattern_is_not_rendered_under_the_process_heading(store):
    """A perda de capacidade invisível: o pattern continua sendo gravado, e o prompt
    passa a mandar o gerador ignorá-lo."""
    _claim(store, 1, Directness.DIRECT)
    block = _reflector(store).lessons_for_speculation(
        [_lesson("pattern", "tônus glutamatérgico agudo falhou em três hipóteses",
                 claim_ids=[1])]
    )

    pattern_at = block.index("tônus glutamatérgico agudo")
    process_heading = block.find("### Notes about how to search")
    assert process_heading == -1 or pattern_at < process_heading, (
        "o pattern caiu na seção que diz 'não afirmam nada sobre biologia'"
    )
    assert "Substantive patterns" in block


def test_a_pattern_and_a_process_note_land_in_different_sections(store):
    _claim(store, 1, Directness.DIRECT)
    block = _reflector(store).lessons_for_speculation([
        _lesson("pattern", "PADRAOMARK", claim_ids=[1]),
        _lesson("search_lesson", "NOTAMARK"),
    ])

    assert block.index("PADRAOMARK") < block.index("### Notes about how to search")
    assert block.index("### Notes about how to search") < block.index("NOTAMARK")


# ───────────────────────────── 3. o qualificador sobrevive à abstração


def test_the_pattern_carries_the_weakest_directness_not_the_best(store):
    """Três claims fracas não podem virar uma linha com a autoridade de uma forte."""
    _claim(store, 1, Directness.DIRECT)
    _claim(store, 2, Directness.EXTRAPOLATED)
    _claim(store, 3, Directness.PARTIAL)

    block = _reflector(store).lessons_for_speculation(
        [_lesson("pattern", "padrão", claim_ids=[1, 2, 3])]
    )
    assert "[pattern · extrapolated]" in block
    assert "direct]" not in block


def test_the_pattern_declares_how_many_claims_support_it(store):
    _claim(store, 1, Directness.DIRECT)
    _claim(store, 2, Directness.DIRECT)
    block = _reflector(store).lessons_for_speculation(
        [_lesson("pattern", "padrão", claim_ids=[1, 2])]
    )
    assert "from 2 verified claims" in block


def test_a_process_lesson_never_gets_a_strength_qualifier(store):
    """O selo é para o canal com portão. Um `search_lesson` estampado
    `· direct` seria autoridade emprestada — e `dead_end` é justamente a categoria
    capaz de apagar uma classe de mecanismo."""
    _claim(store, 1, Directness.DIRECT, Grade.META_ANALYSIS)
    _hypothesis(store, 7, refuted=True)

    block = _reflector(store).lessons_for_speculation([
        _lesson("search_lesson", "queries com 'novel' voltam vazias", claim_ids=[1]),
        _lesson("source_lesson", "fonte X não tem texto completo", claim_ids=[1]),
        _lesson("dead_end", "x", refs={1: ["hypothesis", 7]}, claim_ids=[1]),
    ])

    assert "· direct" not in block
    assert "search_lesson ·" not in block
    assert "dead_end ·" not in block


def test_weakest_directness_ignores_a_claim_id_that_does_not_exist(store):
    _claim(store, 1, Directness.PARTIAL)
    worst = _reflector(store).weakest_directness([1, 9999])
    assert worst is Directness.PARTIAL


def test_weakest_directness_is_none_without_claims(store):
    assert _reflector(store).weakest_directness([]) is None


# ──────────────────────────────────────────────────── invariante de renderização


def test_the_block_never_carries_a_real_hypothesis_id(store):
    """Ids reais não voltam para dentro de um prompt que usa índices locais — senão
    dois namespaces coexistem e uma conflação resolve para o objeto errado."""
    _hypothesis(store, 7, refuted=True)
    block = _reflector(store).lessons_for_speculation(
        [_lesson("dead_end", "x", refs={1: ["hypothesis", 7]})]
    )
    assert "hypothesis 7" not in block
    assert "#7" not in block


def test_an_empty_lesson_list_renders_the_sentinel(store):
    assert _reflector(store).lessons_for_speculation([]) == "(none)"


# ──────────────────────── 4. o WIRING, não só o renderizador

async def test_the_speculation_track_receives_the_contained_block(store):
    """A lacuna que a bateria de mutação achou: eu testava o renderizador e deixava a
    LIGAÇÃO livre.

    Reverter `explore._lessons_block()` para `"  [{kind}] {text}"` passava com os 13
    testes acima verdes — exatamente a mesma classe de furo do wiring memória→screen na
    Fase 1. Um renderizador correto que ninguém chama não contém nada.
    """
    from lithium.pipeline.explore import Explorer

    _hypothesis(store, 7, refuted=True)
    reflector = _reflector(store)
    await reflector.remember(
        LEAP, "dead_end", "de [1]", shown={1: __import__(
            "lithium.pipeline.reflect", fromlist=["Ref"]
        ).Ref(kind="hypothesis", id=7, label="dextrometorfano transdérmico")},
    )

    engine = Explorer(store, NullLLM(), reflector, profile=PROFILE)
    block = await engine._lessons_block()

    assert "TODOS os mecanismos glutamatérgicos" not in block, (
        "a prosa do modelo chegou à trilha de especulação"
    )
    assert FLAW in block


async def test_a_pattern_reaches_the_speculation_track_with_its_qualifier(store):
    """A metade recíproca: a ligação não pode ser "contém tudo", tem que passar o
    canal fundamentado adiante — com o selo."""
    from lithium.pipeline.explore import Explorer

    _claim(store, 1, Directness.EXTRAPOLATED)
    reflector = Reflector(store, ApprovingPatternGate(), FakeEmbedder())
    await reflector.remember(
        "tônus glutamatérgico agudo falhou em três hipóteses", "pattern", "de [1]",
        [1], shown={1: Ref(kind="claim", id=1, label="claim citada")},
    )

    block = await Explorer(store, NullLLM(), reflector, profile=PROFILE)._lessons_block()
    assert "tônus glutamatérgico agudo" in block
    assert "· extrapolated" in block
