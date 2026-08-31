"""O estado do corpus como fato, e o portão que exige contato com a literatura.

Esta é a Fase 4 depois da medição, e a medição mudou o que ela é. O plano especificava
uma coluna `claims.surprise` (escalar) mais um gatilho por soma de surpresa. Três
resultados mataram as duas coisas — estão em `_corpus_note` e no `PLAN.md`; o resumo:

* o escalar **satura antes de servir**: 100% das claims em 1.0 com corpus de 10, 99% com
  100, 83% com 400;
* **estreia e notícia colidem no máximo**: cinco grafias de quetiapina com o mesmo achado
  dão o mesmo valor que um salto `extrapolated → direct`;
* o gatilho é **inerte na vazão de projeto** (T=6 e T=24 dão a mesma cadência: quem manda
  é o piso de intervalo) e **auto-extintor num corpus maduro** (72 h derivam para 207 h).

O que sobrevive é o que sempre foi verificável: quantas claims existem para aquela
intervenção e se elas concordam — renderizado como fato, sem escala para o modelo
interpretar errado. Mais o portão que impede um `pattern` de nascer de uma janela cujo
conteúdo inteiro é opinião do próprio sistema.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from lithium.db import Store
from lithium.pipeline.reflect import Reflector
from lithium.types import Directness, Grade

DIM = 8


class FakeEmbedder:
    async def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(zlib.crc32(t.encode()) % (2**32))
            v = rng.normal(size=DIM).astype(np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return out


class ApprovingGate:
    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, messages, schema, **kw):
        from lithium.llm.schemas import PatternVerdict

        assert schema is PatternVerdict
        self.calls += 1
        return PatternVerdict(
            follows_from_cited_claims_alone=True,
            population_scope_exceeded=False,
            contradicts_a_cited_claim_direction=False,
            restates_a_premise=False,
            reason="r",
        )


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "n.db", embedding_dim=DIM)
    s.init_schema()
    s.conn.execute(
        "INSERT INTO sources(kind, external_id, title, raw_json) "
        "VALUES('pubmed', '1', 't', '{}')"
    )
    yield s
    s.close()


def _reflector(store, llm=None) -> Reflector:
    return Reflector(store, llm or ApprovingGate(), FakeEmbedder())


def _claim(store, statement, intervention, *, direction="positive",
           grade=Grade.RCT, directness=Directness.DIRECT) -> int:
    cur = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "  grade, scale_id, confidence, verified) "
        "VALUES(1, '[]', ?, ?, ?, ?, 1, 0.9, 1) RETURNING id",
        (statement, intervention, direction, grade.value),
    )
    claim_id = int(cur.fetchone()["id"])
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (claim_id, directness.value),
    )
    return claim_id


def _barren_source(store, pmid: str) -> None:
    """Fonte que rendeu ZERO claims verificadas — o processo escrevendo sobre si mesmo."""
    store.conn.execute(
        "INSERT INTO sources(kind, external_id, title, raw_json) "
        "VALUES('pubmed', ?, 't', '{}')", (pmid,),
    )


# ═══════════════════════════════ 1. o estado do corpus, renderizado como fato


def test_an_intervention_with_no_prior_says_so(store):
    _claim(store, "Cetamina reduziu ansiedade.", "ketamine")
    activity = _reflector(store).recent_activity()

    assert "no other verified claim for this intervention" in activity.text


def test_a_disagreement_is_named_with_the_counts(store):
    _claim(store, "Quetiapina reduziu ansiedade.", "quetiapine")
    _claim(store, "Quetiapina não separou de placebo.", "quetiapine",
           direction="negative")
    activity = _reflector(store).recent_activity()

    assert "the corpus disagrees on this intervention" in activity.text
    assert "1 positive, 1 negative or null" in activity.text


def test_an_intervention_with_agreeing_priors_gets_no_note(store):
    """Silêncio quando não há notícia. Uma nota em toda linha é ruído, e ruído no bloco
    que o portão de `pattern` usa como referente é pior que ruído em qualquer outro."""
    for i in range(3):
        _claim(store, f"Lítio reduziu recaída, estudo {i}.", "lithium")
    activity = _reflector(store).recent_activity()

    assert "no other verified claim" not in activity.text
    assert "corpus disagrees" not in activity.text


def test_no_number_ever_reaches_the_prompt(store):
    """O corolário que o plano registra e que a medição confirmou.

    Um 12B lendo `surprise: 1.0` conclui "achado marcante" quando o significado real é
    *nós nunca buscamos isto*. E como cinco grafias do mesmo fármaco produziam esse mesmo
    1.0, o número seria falso na maioria das vezes em que aparecesse.
    """
    import re

    _claim(store, "Cetamina reduziu ansiedade.", "ketamine")
    _claim(store, "Quetiapina reduziu ansiedade.", "quetiapine")
    _claim(store, "Quetiapina não separou.", "quetiapine", direction="negative")
    text = _reflector(store).recent_activity().text

    assert not re.search(r"surprise|frontier|magnitude|novelty\s*[:=]", text, re.I)
    assert not re.search(r"\b(?:0\.\d+|1\.0)\b", text), (
        "um score decimal chegou ao prompt de reflexão"
    )


def test_spelling_variants_are_not_presented_as_separate_frontiers(store):
    """A medição que matou o escalar, virada em teste.

    Cinco grafias do mesmo fármaco com o mesmo achado davam `frontier = 1.0` cada uma —
    o mesmo valor de um salto `extrapolated → direct`. Aqui elas continuam sendo grupos
    distintos (a cardinalidade de texto livre não converge, e o plano já registra isso em
    `MAX_COVERAGE_ROWS`), mas o que o modelo lê é "não há outra claim para esta
    intervenção" — uma afirmação sobre o CORPUS, que é verdadeira, e não uma nota de
    importância, que seria falsa.
    """
    for spelling in ("quetiapine", "quetiapina", "Quetiapine XR",
                     "adjunctive quetiapine", "Seroquel"):
        _claim(store, f"Estudo com {spelling} reduziu ansiedade.", spelling)

    text = _reflector(store).recent_activity().text
    assert text.count("no other verified claim for this intervention") >= 2
    assert "marcante" not in text and "notable" not in text.lower()


def test_a_claim_with_no_intervention_is_not_described_as_an_intervention(store):
    """A nota é FATO do corpus, e sobre claim sem intervenção o fato é outro.

    O `GROUP BY` cai para `id:<claim_id>` quando `intervention` está vazio, então
    `n_group` é 1 e a nota afirmava "não há outra claim verificada para ESTA
    INTERVENÇÃO" sobre uma claim em que o campo não nomeia intervenção nenhuma.

    MEDIDO no corpus real: 61 das 213 claims caem neste grupo, e 2 dos 12 slots da
    janela default eram ocupados por elas (`id:30` e `id:33`, ambas sobre prevalência de
    ansiedade comórbida — o assunto MAIS coberto do corpus). Importa porque a saída deste
    passo com `kind='pattern'` é gravada de forma durável em `research_lessons`, e este
    repo não reescreve julgamento passado.

    MUTAÇÃO QUE MATA: apagar o ramo `if str(row["group_key"]).startswith("id:")` de
    `_corpus_note` (a nota volta a cair no `n_group <= 1`).
    """
    _claim(store, "Uma coorte observou associação, sem braço de tratamento.", None)
    text = _reflector(store).recent_activity().text

    assert "this claim records no intervention" in text
    assert "no other verified claim for this intervention" not in text, (
        "o prompt afirmou ausência de outra claim PARA UMA INTERVENÇÃO que o campo não nomeia"
    )


# ═════════════ 2. o portão: um `pattern` exige contato novo com a literatura


async def test_a_pattern_needs_a_new_verified_claim_in_the_window(store):
    from lithium.pipeline.reflect import Ref

    cid = _claim(store, "Lítio reduziu recaída.", "lithium")
    reflector = _reflector(store)
    shown = {1: Ref("claim", cid, "a claim")}

    # janela aberta: a claim é nova
    first = await reflector.remember("um padrão", "pattern", "n", [1], shown=shown)
    assert first is not None

    # fecha a janela e tenta de novo, sem colheita nova
    reflector.mark_literature_seen()
    again = await reflector.remember("outro padrão", "pattern", "n", [1], shown=shown)
    assert again is None, (
        "um pattern nasceu de uma janela sem claim nova — seria o sistema "
        "generalizando sobre a própria opinião"
    )


async def test_a_barren_source_does_not_open_the_gate(store):
    """O furo que a auditoria mediu: contar fontes deixava o portão aberto exatamente na
    janela estéril que ele existe para fechar.

    Uma fonte árida rendeu zero claims verificadas — é o processo escrevendo sobre si
    mesmo, e são justamente as fontes que `recent_activity()` lista sob "yielded zero
    verified claims".
    """
    from lithium.pipeline.reflect import Ref

    cid = _claim(store, "Lítio reduziu recaída.", "lithium")
    reflector = _reflector(store)
    reflector.mark_literature_seen()

    for i in range(30):
        _barren_source(store, f"9000{i}")

    shown = {1: Ref("claim", cid, "a claim")}
    assert await reflector.remember("padrão", "pattern", "n", [1], shown=shown) is None


async def test_a_new_verified_claim_reopens_the_gate(store):
    from lithium.pipeline.reflect import Ref

    cid = _claim(store, "Lítio reduziu recaída.", "lithium")
    reflector = _reflector(store)
    reflector.mark_literature_seen()

    fresh = _claim(store, "Cetamina reduziu ansiedade.", "ketamine")
    shown = {1: Ref("claim", fresh, "a claim")}
    assert await reflector.remember("padrão", "pattern", "n", [1], shown=shown) is not None
    assert cid  # a antiga não é o que reabriu


async def test_a_process_lesson_never_needs_the_gate(store):
    """O portão é do canal substantivo. `search_lesson` fala do processo, e a evidência
    dele — query estéril, fonte árida — não é contato com a literatura por definição.
    Exigir isso silenciaria a categoria para sempre."""
    reflector = _reflector(store)
    reflector.mark_literature_seen()

    assert await reflector.remember("query volta vazia", "search_lesson", "n") is not None


async def test_the_gate_does_not_spend_an_llm_call_when_it_refuses(store):
    """Reprovar por janela é determinístico e tem que vir ANTES do portão de derivação —
    senão o sistema paga uma chamada para descobrir algo que o banco já sabia."""
    from lithium.pipeline.reflect import Ref

    cid = _claim(store, "Lítio reduziu recaída.", "lithium")
    gate = ApprovingGate()
    reflector = _reflector(store, gate)
    reflector.mark_literature_seen()

    await reflector.remember("padrão", "pattern", "n", [1],
                             shown={1: Ref("claim", cid, "a claim")})
    assert gate.calls == 0


# ══════════════════════════════════════ 3. A LIGAÇÃO: o handler fecha a janela


async def test_the_reflect_handler_closes_the_window(store, tmp_path):
    """Quarta ocorrência da classe: reverter `mark_literature_seen()` do handler deixava
    o portão PERMANENTEMENTE ABERTO e a suíte verde. Um portão que nunca fecha é
    indistinguível, no verde do teste, de um portão que não existe.
    """
    from lithium.config import Config
    from lithium.worker.handlers import reflect_tick

    _claim(store, "Lítio reduziu recaída.", "lithium")

    class NoLessons:
        async def structured(self, messages, schema, **kw):
            from lithium.llm.schemas import ResearchLessons

            return ResearchLessons(lessons=[])

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.llm = NoLessons()
            self.embedder = FakeEmbedder()
            self.config = Config(data_dir=tmp_path)

    reflector = _reflector(store)
    assert reflector._window_touched_literature() is True

    await reflect_tick({}, Ctx())

    assert _reflector(store)._window_touched_literature() is False, (
        "o handler não fechou a janela: o portão de pattern fica aberto para sempre"
    )


def test_the_mark_advances_only_to_verified_claims(store):
    """Claim não verificada não é contato com a literatura: ela não passou os portões."""
    store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, direction, grade, "
        "  scale_id, confidence, verified) "
        "VALUES(1, '[]', 'não verificada', 'positive', 'rct', 1, 0.5, 0)"
    )
    reflector = _reflector(store)
    reflector.mark_literature_seen()
    assert reflector._window_touched_literature() is False

    _claim(store, "verificada", "lithium")
    assert reflector._window_touched_literature() is True


# ═════════════════════════════ o portão da literatura mede o corpus DESTE foco


def test_the_pattern_gate_ignores_claims_that_carry_no_weight(store):
    """MUTAÇÃO: reverter o portão para `SELECT MAX(id) FROM claims WHERE verified = 1`.

    Claim sem julgamento é a versão nova das "30 fontes áridas" que o docstring de
    `_window_touched_literature` registra ter deixado um `pattern` passar: corpus que
    cresce sem que este foco tenha aprendido nada com ele.
    """
    reflector = _reflector(store)
    reflector.mark_literature_seen()
    assert reflector._window_touched_literature() is False

    cur = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "  grade, scale_id, confidence, verified) "
        "VALUES(1, '[]', 'sem julgamento', 'lítio', 'positive', 'rct', 1, 0.9, 1) "
        "RETURNING id")
    claim_id = int(cur.fetchone()["id"])
    assert reflector._window_touched_literature() is False, (
        "uma claim SEM PESO abriu o portão de `pattern`")

    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) "
        "VALUES(?, 1, 'direct')", (claim_id,))
    assert reflector._window_touched_literature() is True


def test_the_literature_mark_is_scoped_to_the_focus(store):
    """MUTAÇÃO: reverter `LITERATURE_MARK` para a chave literal global. Nenhum dos 8
    testes existentes deste módulo asserta essa string — eles usam só a API — então sem
    este teste a mudança é livre nos dois sentidos e a reversão não é detectada."""
    _claim(store, "verificada", "lítio")
    reflector = _reflector(store)
    reflector.mark_literature_seen()
    assert reflector._window_touched_literature() is False

    sid = int(store.conn.execute("SELECT id FROM evidence_scales").fetchone()["id"])
    store.conn.execute(
        "INSERT INTO focuses(id, slug, target, scale_id) VALUES(2,'b','outro alvo',?)",
        (sid,))
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) "
        "SELECT id, 2, 'direct' FROM claims WHERE verified = 1")
    store.conn.execute("UPDATE meta SET value = '2' WHERE key = 'active_focus'")

    assert reflector._window_touched_literature() is True, (
        "o foco #2 herdou a marca do #1 e nasceu achando que já viu esta literatura")
