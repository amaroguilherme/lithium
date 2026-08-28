"""A recuperação de lições deixa de ser constante — e o bloco chega ao prompt.

O bug: `relevant_lessons` tinha um único chamador, passando o literal fixo
`"mechanistic hypotheses for bipolar I with comorbid generalized anxiety"`. O ranking era
**idêntico em toda rodada pela vida do sistema**: as mesmas oito lições em todo prompt de
especulação, para sempre. Não era recuperação por relevância, era injeção de um conjunto
congelado — e cada rodada nova passava pelo filtro das conclusões das anteriores.

**A ligação é testada aqui, não só o renderizador.** Esse furo escapou três vezes neste
projeto: reverter o *call site* (`lessons="(none)"`, ou pegar só a primeira linha do
bloco) passava com a suíte inteira verde enquanto as consultas eram embedadas, a cota
aplicada e o renderizador rodava — com o resultado jogado no lixo. Por isso os testes
abaixo afirmam sobre `llm.prompts[0]`, o texto que de fato chegou ao modelo.

**Rodízio, não RRF.** Escolhi RRF primeiro. A auditoria mostrou que a garantia que eu
queria dele não existia: o teste que dizia "uma lacuna cuja melhor lição é globalmente
mediana ainda recebe seu slot" passava por causa da *cota por kind*. Bastava dar às
concorrentes o mesmo `kind` da lição solitária para a lacuna ficar sem resposta nenhuma.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence

import pytest

from lithium.db import Store
from lithium.pipeline.explore import (
    MAX_GAP_QUERIES,
    Explorer,
    _lessons_queries,
)
from lithium.pipeline.reflect import (
    MAX_LESSONS_IN_PROMPT,
    SUBSTANTIVE_SLOT_CAP,
    Reflector,
)
from lithium.pipeline.state import build_state
from lithium.types import Directness, Grade

from conftest import onco_profile

PROFILE = onco_profile()



DIM = 16


class FakeEmbedder:
    """Registra cada lote, para o teste poder afirmar sobre o que foi consultado."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class DictatedEmbedder:
    """Vetores ditados pelo teste, para separar posição de magnitude.

    Necessário porque as fixtures de cosseno "redondo" (0, 0.3, 0.8, 1) nunca dissociam
    posição de magnitude — e dissociar as duas é a razão de existir do rodízio.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    async def embed(self, texts):
        return [self.table[t] for t in texts]


class CapturingLLM:
    def __init__(self, batch=None) -> None:
        self.prompts: list[str] = []
        self._batch = batch

    async def structured(self, messages, schema, **kw):
        from lithium.llm.schemas import SpeculationBatch

        self.prompts.append(messages[0]["content"])
        return self._batch or SpeculationBatch(speculations=[])


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "g.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _reflector(store, embedder=None) -> Reflector:
    return Reflector(store, CapturingLLM(), embedder or FakeEmbedder())


async def _lesson(reflector, text: str, kind: str = "search_lesson") -> None:
    await reflector.remember(text, kind, "x")


def _claim(store, pmid: str, statement: str, intervention: str,
           *, direction: str = "positive") -> None:
    sid = store.upsert_source(kind="pubmed", external_id=pmid, raw={},
                              title="t", year=2020)
    cid = store.add_chunk(source_id=sid, ord=0,
                          text=f"{statement} texto longo o suficiente para indexar.")
    _cl = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "  grade, scale_id, confidence, verified) VALUES(?,?,?,?,?,?,1,0.9,1) "
        "RETURNING id",
        (sid, json.dumps([cid]), statement, intervention, direction, Grade.RCT.value),
    ).fetchone()["id"]
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (int(_cl), Directness.DIRECT.value),
    )


# ══════════════════════════════════════ 1. a consulta deixa de ser constante


def test_the_query_differs_between_two_knowledge_states(store):
    """O critério de aceite do plano. Antes era o mesmo literal byte a byte nos dois."""
    empty = _lessons_queries(build_state(store, PROFILE), PROFILE)

    _claim(store, "111", "Quetiapine reduced anxiety.", "quetiapine")
    _claim(store, "222", "Quetiapine increased anxiety.", "quetiapine",
           direction="negative")
    populated = _lessons_queries(build_state(store, PROFILE), PROFILE)

    assert empty != populated, (
        "a consulta de lições é a mesma em dois estados de conhecimento diferentes — "
        "é o bug: o ranking fica congelado pela vida do sistema"
    )


def test_a_conflict_becomes_its_own_query(store):
    _claim(store, "111", "Quetiapine reduced anxiety.", "quetiapine")
    _claim(store, "222", "Quetiapine increased anxiety.", "quetiapine",
           direction="negative")

    queries = _lessons_queries(build_state(store, PROFILE), PROFILE)
    assert any("disagree on the direction of quetiapine" in q for q in queries)


def test_untouched_gaps_are_sorted_before_slicing(store):
    """A fatia tem que ser estável. `state.untouched` chega na ordem do dict literal que
    o construiu: fatiar sem ordenar faz a mesma lacuna entrar ou sair conforme uma chave
    nova apareça no meio do dict."""
    queries = _lessons_queries(build_state(store, PROFILE), PROFILE)
    gaps = [q for q in queries if "no evidence at all on" in q]
    assert gaps == sorted(gaps)


def test_the_number_of_queries_is_capped(store):
    queries = _lessons_queries(build_state(store, PROFILE), PROFILE)
    assert len(queries) <= MAX_GAP_QUERIES


def test_there_is_always_at_least_one_query(store):
    """Corpus vazio não pode virar zero consultas — cairia no braço fail-open sem que
    ninguém tivesse pedido."""
    assert _lessons_queries(build_state(store, PROFILE), PROFILE)


# ══════════════════════════════════════════ 2. o rodízio, com magnitude dissociada


async def test_a_gap_with_only_a_weak_answer_still_gets_a_slot(store):
    """O caso que separa rodízio de qualquer fusão por score.

    Doze lições fortes na lacuna A (cosseno alto) e **uma** resposta fraca à lacuna B
    (cosseno baixo) — todas do MESMO kind, para a cota não poder carregar a asserção.
    Sob soma de cossenos, ou sob RRF sobre rankings completos, a resposta da lacuna B não
    entra e a lacuna fica sem cobertura. No rodízio ela entra na primeira volta.
    """
    gap_a = [1.0] + [0.0] * (DIM - 1)
    gap_b = [0.0, 1.0] + [0.0] * (DIM - 2)
    table = {"consulta A": gap_a, "consulta B": gap_b}

    reflector = _reflector(store, DictatedEmbedder(table))
    for i in range(12):
        # colineares com a lacuna A: cosseno ~0,9
        table[f"forte {i}"] = [0.9, 0.05] + [0.01] * (DIM - 2)
        await _lesson(reflector, f"forte {i}", "search_lesson")
    table["fraca só da B"] = [0.05, 0.15] + [0.0] * (DIM - 2)
    await _lesson(reflector, "fraca só da B", "search_lesson")

    # o embedder de gravação é outro; recarrega os vetores ditados
    reflector.embedder = DictatedEmbedder(table)
    got = await reflector.relevant_lessons(["consulta A", "consulta B"])

    assert "fraca só da B" in [lesson.text for lesson in got], (
        "a lacuna B ficou sem resposta: o rodízio não garantiu o slot da primeira volta"
    )


async def test_every_gap_is_served_before_any_gap_gets_seconds(store):
    """A propriedade estrutural do rodízio, afirmada diretamente."""
    vectors = {}
    reflector = _reflector(store, DictatedEmbedder(vectors))
    for i in range(6):
        vectors[f"lição {i}"] = [1.0 if i % 2 == 0 else 0.0,
                                 1.0 if i % 2 else 0.0] + [0.0] * (DIM - 2)
        await _lesson(reflector, f"lição {i}", "search_lesson")
    vectors["q0"] = [1.0] + [0.0] * (DIM - 1)
    vectors["q1"] = [0.0, 1.0] + [0.0] * (DIM - 2)

    reflector.embedder = DictatedEmbedder(vectors)
    got = await reflector.relevant_lessons(["q0", "q1"], k=2)
    ids = [int(lesson.text.split()[-1]) for lesson in got]

    assert len(got) == 2
    assert (ids[0] % 2) != (ids[1] % 2), (
        "os dois slots foram para a mesma lacuna — o rodízio não distribuiu"
    )


# ══════════════════════════════════════════════════ 3. a cota, nos dois braços


async def test_the_quota_limits_self_generated_content(store):
    reflector = _reflector(store)
    for i in range(6):
        await _lesson(reflector, f"padrão de conteúdo {i}", "dead_end")
    for i in range(6):
        await _lesson(reflector, f"nota de processo {i}", "search_lesson")

    got = await reflector.relevant_lessons(["qualquer lacuna"])
    content = sum(1 for lesson in got if lesson.kind in {"pattern", "dead_end"})
    assert content <= SUBSTANTIVE_SLOT_CAP


async def test_the_quota_also_applies_on_the_fail_open_path(store):
    """O caminho que existe para sobreviver à queda do embedder não pode ser justamente
    o que enche o prompt de conteúdo auto-gerado."""
    reflector = _reflector(store)
    for i in range(5):
        await _lesson(reflector, f"conteúdo {i}", "dead_end")

    class DeadEmbedder:
        async def embed(self, texts):
            raise AssertionError("o braço fail-open não deve embeddar")

    reflector.embedder = DeadEmbedder()
    got = await reflector.relevant_lessons(["lacuna"])   # 5 <= k=8, curto-circuito
    content = sum(1 for lesson in got if lesson.kind in {"pattern", "dead_end"})
    assert content <= SUBSTANTIVE_SLOT_CAP


async def test_the_block_is_still_capped(store):
    reflector = _reflector(store)
    for i in range(MAX_LESSONS_IN_PROMPT + 6):
        await _lesson(reflector, f"nota {i}", "search_lesson")

    got = await reflector.relevant_lessons(["uma lacuna", "outra lacuna"])
    assert len(got) == MAX_LESSONS_IN_PROMPT


# ═══════════════════════════ 4. A LIGAÇÃO — o bloco chega ao prompt de verdade


async def test_the_rendered_lessons_reach_the_speculation_prompt(store):
    """O furo que escapou três vezes: renderizador testado, ligação livre.

    Reverter o call site para `lessons="(none)"` — ou para a primeira linha do bloco —
    passava com a suíte inteira verde. As consultas eram embedadas, a cota aplicada, o
    renderizador rodava, e o resultado ia para o lixo.
    """
    reflector = _reflector(store)
    await _lesson(reflector, "SENTINELA DA LICAO", "search_lesson")

    llm = CapturingLLM()
    await Explorer(store, llm, reflector, profile=PROFILE).generate(max_items=1)

    assert llm.prompts, "generate() não chegou a montar prompt nenhum"
    prompt = llm.prompts[0]
    assert "SENTINELA DA LICAO" in prompt, "o texto da lição não chegou ao prompt"
    assert "### Notes about how to search" in prompt, (
        "o cabeçalho de seção do renderizador não chegou — o bloco foi substituído"
    )


async def test_the_gap_queries_reach_the_embedder_one_per_gap(store):
    """E não um saco só: um vetor médio de dez rótulos não representa nenhum deles."""
    _claim(store, "111", "Quetiapine reduced anxiety.", "quetiapine")
    embedder = FakeEmbedder()
    reflector = _reflector(store, embedder)
    for i in range(MAX_LESSONS_IN_PROMPT + 2):
        await _lesson(reflector, f"nota {i}", "search_lesson")

    embedder.batches.clear()
    await Explorer(store, CapturingLLM(), reflector, profile=PROFILE).generate(max_items=1)

    assert embedder.batches, "nenhuma consulta foi embedada"
    queries = embedder.batches[0]
    assert len(queries) > 1, f"veio um saco só: {queries}"


async def test_an_explorer_without_a_reflector_still_renders(store):
    llm = CapturingLLM()
    await Explorer(store, llm, None, profile=PROFILE).generate(max_items=1)
    assert "(none)" in llm.prompts[0]


def test_a_conflict_is_never_crowded_out_by_untouched_labels(store):
    """O bug que este teste pegou no meu primeiro desenho.

    Escrevi `untouched` antes de `conflicts` e cortei em 8. Num corpus jovem há muitos
    rótulos sem nada, então eles enchiam os oito slots e **nenhum conflito entrava
    nunca** — e conflito é o sinal mais informativo dos dois: significa que há dado e ele
    discorda. Perda silenciosa por ordem de lista.
    """
    _claim(store, "111", "Quetiapine reduced anxiety.", "quetiapine")
    _claim(store, "222", "Quetiapine increased anxiety.", "quetiapine",
           direction="negative")

    queries = _lessons_queries(build_state(store, PROFILE), PROFILE)
    untouched_count = sum(1 for q in queries if "no evidence at all" in q)

    assert untouched_count >= MAX_GAP_QUERIES - 2, (
        "o cenário precisa saturar os slots, senão o teste não exercita a disputa"
    )
    assert any("disagree on the direction of quetiapine" in q for q in queries)


# ══════════════════════════════════════════════ 3. o prefixo vem do PERFIL ATIVO


def test_the_lesson_query_carries_the_prefix_of_the_active_profile(store):
    """O prefixo de recuperação vem do PERFIL, e este teste não pode ler a mesma fonte.

    As duas asserções que existiam antes (`all(DOMAIN_PREFIX in q for q in queries)`)
    eram TAUTOLÓGICAS por construção — comparavam a saída contra a constante que a
    produziu — e `'' in q` é sempre verdadeiro. MUTAÇÃO EXECUTADA na Fase A: com
    `DOMAIN_PREFIX = ''` E com `DOMAIN_PREFIX = 'cancer immunotherapy in dogs'`, os 14
    testes deste arquivo passavam nos DOIS casos.

    Aqui a expectativa é um literal de oncologia veterinária, que só pode ter vindo do
    perfil recebido: não existe essa string em `lithium/`.

    MUTAÇÃO que isto mata: remover o prefixo de `_lessons_queries`, ou voltá-lo a uma
    constante de módulo.
    """
    queries = _lessons_queries(build_state(store, PROFILE), PROFILE)
    assert queries
    assert all(q.startswith("canine multicentric lymphoma with renal impairment")
               for q in queries), queries
    assert not any("bipolar" in q for q in queries), (
        "vazou o prefixo do perfil de produção"
    )


def test_two_profiles_produce_two_different_lesson_queries(store):
    """A mesma lacuna, dois perfis, dois textos. Fecha o caso do prefixo VAZIO, que a
    asserção antiga deixava passar: com `prefix = ''` os dois renders são idênticos."""
    from conftest import prod_profile

    a = _lessons_queries(build_state(store, PROFILE), PROFILE)
    b = _lessons_queries(build_state(store, prod_profile()), prod_profile())
    assert a != b
