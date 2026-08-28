"""Reflexão: o sistema aprendendo sobre o próprio trabalho, sem confirmação.

Por decisão do usuário, memória de pesquisa grava sozinha e memória de conversa exige
confirmação. Essa autonomia só é defensável por causa de dois limites estruturais, e
estes testes travam os dois:

* **portão de citação por categoria** — lição substantiva (`pattern`) exige `claim_ids`
  de claims verificadas; sem eles, uma asserção sem fonte entraria e seria injetada em
  todo prompt seguinte, o sistema ensinando as próprias suposições a si mesmo.
* **recuperação por relevância** — lições auto-gravadas compõem. Injetar todas faria
  cada decisão futura passar pelo filtro das crenças anteriores do sistema.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence

import pytest

from lithium.db import Store
from lithium.llm import LLMError
from lithium.llm.schemas import ResearchLesson, ResearchLessons
from lithium.pipeline.reflect import (
    MAX_LESSONS_IN_PROMPT,
    FabricatedReference,
    PROCESS_KINDS,
    SUBSTANTIVE_KINDS,
    Reflector,
)
from lithium.types import Directness, Grade

DIM = 16


def _bucket(word: str) -> int:
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
    def __init__(self, results=()) -> None:
        self._results = list(results)
        self.prompts: list[str] = []
        self.pattern_gate_calls = 0
        self.pattern_verdict = None

    async def structured(self, messages, schema, **kw):
        from lithium.llm.schemas import PatternVerdict

        # O pipeline faz DUAS chamadas ao gravar um `pattern`: a reflexão e o portão de
        # derivação. Um dublê que só modela a primeira faz todo teste de pattern
        # explodir em "schema inesperado" — e a tentação seria afrouxar o portão.
        if schema is PatternVerdict:
            self.pattern_gate_calls += 1
            return self.pattern_verdict or PatternVerdict(
                follows_from_cited_claims_alone=True,
                population_scope_exceeded=False,
                contradicts_a_cited_claim_direction=False,
                restates_a_premise=False,
                reason="segue das claims citadas",
            )
        self.prompts.append(messages[0]["content"])
        item = self._results.pop(0) if self._results else ResearchLessons(lessons=[])
        if isinstance(item, Exception):
            raise item
        return item


def _lessons(*items: dict) -> ResearchLessons:
    return ResearchLessons(lessons=[ResearchLesson.model_validate(i) for i in items])


def _process(text="Queries pareando sigma-1 com desfecho clínico voltam vazias",
             kind="search_lesson") -> dict:
    return {"text": text, "kind": kind, "provenance_note": "hipótese 1", "claim_ids": []}


def _pattern(text="Três hipóteses refutadas falharam por elevar tônus glutamatérgico",
             claim_ids=None) -> dict:
    return {"text": text, "kind": "pattern", "provenance_note": "hipóteses 1-3",
            "claim_ids": claim_ids if claim_ids is not None else []}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "r.db", embedding_dim=DIM)
    s.init_schema()
    # atividade mínima, senão `reflect` sai cedo por não ter o que generalizar
    s.conn.execute(
        "INSERT INTO hypotheses(focus_id, statement, tier, survives_critique, "
        "  critique_json) "
        "VALUES(1, 'hipótese ruim', 'speculative', 0, ?)",
        (json.dumps({"fatal_flaw": "circular", "weakest_link": "elo 2"}),),
    )
    yield s
    s.close()


def _reflector(store, llm) -> Reflector:
    return Reflector(store, llm, FakeEmbedder())


def _seed_claim(store, external_id="111") -> int:
    sid = store.upsert_source(kind="pubmed", external_id=external_id, raw={})
    cid = store.add_chunk(source_id=sid, ord=0,
                          text="Um trecho de apoio com tamanho suficiente para indexar.")
    cur = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "grade, scale_id, confidence, verified) "
        "VALUES(?,?,'achado','x','positive',?,1,1.0,1) RETURNING id",
        (sid, json.dumps([cid]), Grade.RCT.value),
    )
    claim_id = int(cur.fetchone()["id"])
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (claim_id, Directness.PARTIAL.value),
    )
    return claim_id


# ─────────────────────────────────── autonomia: grava sem pedir confirmação


async def test_process_lesson_is_saved_without_confirmation(store):
    """O regime de consentimento da pesquisa. Pedir permissão para "aprendi que MeSH
    funciona melhor" treinaria o usuário a clicar sem ler — e aí a confirmação que
    importa, sobre a própria pessoa, perderia valor."""
    llm = ScriptedLLM([_lessons(_process())])
    [lesson] = await _reflector(store, llm).reflect()

    row = store.conn.execute("SELECT * FROM memories WHERE id = ?", (lesson.id,)).fetchone()
    assert row["source"] == "research"
    assert row["confirmed"] == 1 and row["active"] == 1
    assert row["confirmed_at"] is not None


async def test_research_lessons_stay_out_of_user_memories(store):
    """Servem a prompts diferentes: o que ele sabe sobre VOCÊ vai ao chat; o que
    aprendeu sobre o próprio trabalho vai aos prompts de pesquisa."""
    llm = ScriptedLLM([_lessons(_process())])
    await _reflector(store, llm).reflect()

    assert store.conn.execute("SELECT COUNT(*) AS n FROM research_lessons").fetchone()["n"] == 1
    assert store.conn.execute("SELECT COUNT(*) AS n FROM user_memories").fetchone()["n"] == 0


# ───────────────────────────── o portão de citação da categoria substantiva


async def test_pattern_without_claim_ids_is_discarded(store):
    """Sem o portão, um padrão substantivo entraria sem fonte e seria injetado em
    todo prompt futuro — a falha que os dois portões da extração previnem, entrando
    pela porta de trás."""
    llm = ScriptedLLM([_lessons(_pattern(claim_ids=[]))])
    assert await _reflector(store, llm).reflect() == []
    assert store.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0


async def test_pattern_with_invented_claim_ids_is_discarded(store):
    llm = ScriptedLLM([_lessons(_pattern(claim_ids=[9999, 8888]))])
    assert await _reflector(store, llm).reflect() == []


async def test_pattern_with_verified_claims_is_saved_with_provenance(store):
    claim_id = _seed_claim(store)
    reflector = _reflector(store, ScriptedLLM())
    # O que o modelo cita é o índice local do bloco de claims, não o id de claim: ele
    # nunca vê id real. `verify_claim_ids` traduz, e a proveniência guarda o id.
    k = next(i for i, r in reflector.recent_activity().shown.items()
             if r.kind == "claim" and r.id == claim_id)
    reflector.llm = ScriptedLLM([_lessons(_pattern(claim_ids=[k]))])
    [lesson] = await reflector.reflect()

    row = store.conn.execute("SELECT provenance FROM memories WHERE id = ?",
                             (lesson.id,)).fetchone()
    assert json.loads(row["provenance"])["claim_ids"] == [claim_id]


async def test_unverified_claims_do_not_count_as_provenance(store):
    """Mesma disciplina de `claim_weight`: só material verificado tem peso."""
    sid = store.upsert_source(kind="pubmed", external_id="222", raw={})
    cur = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, grade, scale_id, verified) "
        "VALUES(?, '[]', 's', 'rct', 1, 0) RETURNING id", (sid,)
    )
    unverified = int(cur.fetchone()["id"])
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) "
        "VALUES(?, 1, 'direct')", (unverified,),
    )

    llm = ScriptedLLM([_lessons(_pattern(claim_ids=[unverified]))])
    assert await _reflector(store, llm).reflect() == []


def test_verify_claim_ids_closes_over_what_was_shown(store):
    """Substitui `..._filters_rather_than_trusting`: filtrar era o defeito.

    O contrato mudou de "descarta o id que não existe" para "reprova a lição inteira".
    Filtrar dá crédito parcial — a lição entra com o subconjunto que por acaso existia
    e a fabricação não deixa rastro. A bateria completa está em
    tests/test_reflect_shown_set.py; aqui fica só o contrato mínimo.
    """
    real = _seed_claim(store)
    reflector = _reflector(store, ScriptedLLM())
    shown = reflector.recent_activity().shown
    k = next(i for i, r in shown.items() if r.kind == "claim" and r.id == real)

    assert reflector.verify_claim_ids([k], shown=shown) == [real]
    assert reflector.verify_claim_ids([], shown=shown) == []
    with pytest.raises(FabricatedReference):
        reflector.verify_claim_ids([k, 9999], shown=shown)


def test_process_kinds_need_no_citation():
    assert SUBSTANTIVE_KINDS == {"pattern"}
    assert "search_lesson" in PROCESS_KINDS and "pattern" not in PROCESS_KINDS


async def test_invalid_kind_is_discarded_not_normalised(store):
    """Ao contrário da memória de conversa, aqui um `kind` fora do conjunto costuma
    significar que o modelo escorregou para conclusão sobre o mundo — normalizar
    deixaria a asserção entrar."""
    llm = ScriptedLLM([_lessons({"text": "sigma-1 não funciona em bipolar",
                                 "kind": "conclusion", "provenance_note": "x",
                                 "claim_ids": []})])
    assert await _reflector(store, llm).reflect() == []


# ────────────────────────────── recuperação por relevância, não injeção em massa


async def test_relevant_lessons_are_capped(store):
    """Lições auto-gravadas compõem. Injetar todas faria cada decisão futura passar
    pelo filtro das crenças anteriores do sistema — caminho silencioso para ele
    convergir nas próprias opiniões."""
    reflector = _reflector(store, ScriptedLLM())
    for i in range(MAX_LESSONS_IN_PROMPT + 6):
        await reflector.remember(f"lição número {i} sobre buscas", "search_lesson", "x")

    relevant = await reflector.relevant_lessons(["buscas"])
    assert len(relevant) == MAX_LESSONS_IN_PROMPT
    assert len(reflector.lessons()) == MAX_LESSONS_IN_PROMPT + 6


async def test_relevance_ranks_the_related_lesson_first(store):
    reflector = _reflector(store, ScriptedLLM())
    await reflector.remember("consultas com termos MeSH rendem mais precisão",
                             "search_lesson", "x")
    for i in range(MAX_LESSONS_IN_PROMPT + 3):
        await reflector.remember(f"assunto totalmente distinto numero {i}",
                                 "source_lesson", "x")

    top = await reflector.relevant_lessons(["consultas com termos MeSH"])
    assert "MeSH" in top[0].text


async def test_small_lesson_set_skips_the_embedding_call(store):
    """Com poucas lições não há o que ranquear; gastar uma chamada de embedding
    seria desperdício."""
    reflector = _reflector(store, ScriptedLLM())
    await reflector.remember("uma lição só", "search_lesson", "x")

    class ExplodingEmbedder:
        async def embed(self, texts):
            raise AssertionError("não deveria embeddar a consulta")

    reflector.embedder = ExplodingEmbedder()
    assert len(await reflector.relevant_lessons(["qualquer coisa"])) == 1


# ─────────────────────────────────────────────────────── comportamento geral


async def test_no_activity_means_no_reflection(store):
    """Generalizar sobre pesquisa que não aconteceu produz ruído, não lição."""
    store.conn.execute("DELETE FROM hypotheses")
    llm = ScriptedLLM([_lessons(_process())])
    assert await _reflector(store, llm).reflect() == []
    assert llm.prompts == [], "não deveria nem chamar o LLM"


async def test_empty_lesson_list_is_a_valid_outcome(store):
    llm = ScriptedLLM([_lessons()])
    assert await _reflector(store, llm).reflect() == []


async def test_llm_failure_is_not_fatal(store):
    llm = ScriptedLLM([LLMError("caiu")])
    assert await _reflector(store, llm).reflect() == []


async def test_prior_lessons_are_shown_so_they_are_not_repeated(store):
    llm = ScriptedLLM([_lessons(_process()), _lessons()])
    reflector = _reflector(store, llm)
    await reflector.reflect()
    await reflector.reflect()

    assert "voltam vazias" in llm.prompts[1]
    assert "do not repeat" in llm.prompts[1].lower()


async def test_activity_reports_refutation_reasons_not_just_the_fact(store):
    """"Refutada" sozinha não ensina nada; o motivo é o que impede o gerador de
    repetir a mesma jogada inferencial."""
    llm = ScriptedLLM([_lessons()])
    await _reflector(store, llm).reflect()

    assert "fatal flaw: circular" in llm.prompts[0]
    assert "weakest link: elo 2" in llm.prompts[0]


async def test_lessons_can_be_forgotten_like_any_memory(store):
    """Auto-gravada não significa inauditável."""
    llm = ScriptedLLM([_lessons(_process())])
    [lesson] = await _reflector(store, llm).reflect()

    store.conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (lesson.id,))
    assert store.conn.execute("SELECT COUNT(*) AS n FROM research_lessons").fetchone()["n"] == 0


async def test_relevant_lessons_refuses_a_bare_string(store):
    """O call site antigo passava uma frase. `Sequence[str]` aceita `str`, então
    `list("uma frase")` daria uma consulta por caractere e ranking puro ruído — sem
    levantar nada. Reverter o chamador tem que doer, não degradar em silêncio."""
    reflector = _reflector(store, ScriptedLLM())
    with pytest.raises(TypeError):
        await reflector.relevant_lessons("mechanistic hypotheses for bipolar I")
