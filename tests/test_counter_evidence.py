"""O assento reservado de contra-evidência.

A ordenação por `relevance × (1 + weight)` empurra o achado nulo para baixo de forma
sistemática, e não por acidente: resultado negativo raramente vem de meta-análise grande,
e a literatura publica menos deles. Sem assento reservado, o bloco de evidência mostra ao
revisor uma versão do corpus **onde tudo funciona** — que é o pior viés possível numa
ferramenta cujo produto é decidir o que ainda vale investigar.

Duas armadilhas, as duas medidas numa versão anterior deste código, e as duas cobertas
aqui porque um canal de contra-evidência que erra em qualquer das duas é pior que nenhum:

1. **Sem piso, o assento vira porta dos fundos.** O horizonte é alimentado por um BM25
   sem piso, então qualquer negativa que compartilhe um token entra. Medido: numa busca
   sobre quetiapina, o assento foi preenchido por "anticoagulação oral não reduz
   mortalidade em fibrilação atrial" (relevância 0,05), **deslocando um RCT no assunto** e
   apresentado sob o cabeçalho de contra-evidência.
2. **O canal expulsava contra-evidência.** Truncando pela cauda, a vítima era exatamente
   a negativa de grade baixa que já havia entrado organicamente — 8% das buscas trocavam
   uma contra-claim por outra, com ganho líquido zero.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence

import pytest

from lithium.db import Store
from lithium.pipeline.retrieval import CONTRA_SEATS, Retriever
from lithium.types import Directness, Grade

from conftest import prod_profile

# Perfil de PRODUÇÃO: o assunto deste arquivo é o vocabulário de segurança real
# (lítio, valproato, benzodiazepínico). Rodá-lo contra o perfil de teste — que declara
# `safety = false` — o deixaria verde afirmando sobre um ruleset que não existe.
PROFILE = prod_profile()


DIM = 128
"""Grande o suficiente para colisão de hash não fabricar cosseno. Com 16 dimensões o
bag-of-words colide e um chunk fora do assunto ganha similaridade que não tem."""
QUERY = "quetiapina reduz ansiedade no bipolar I com TAG"


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[zlib.crc32(word.strip(".,;:").encode()) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "ce.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


async def _claim(store, pmid, statement, *, direction="positive",
                 grade=Grade.RCT, directness=Directness.DIRECT, on_topic=True):
    """Semeia claim COM embedding do chunk.

    Sem `set_chunk_embedding` o ranking vetorial fica vazio e a recuperação vira BM25
    puro — que é como esta fixture nasceu, e por isso ela não conseguia distinguir
    "no assunto" de "compartilha um token": o único sinal semântico do sistema nunca
    era exercitado.
    """
    sid = store.upsert_source(kind="pubmed", external_id=pmid, raw={},
                              title="t", year=2020)
    filler = ("quetiapina ansiedade bipolar TAG reduz" if on_topic
              else "anticoagulação fibrilação atrial mortalidade")
    cid = store.add_chunk(source_id=sid, ord=0, text=f"{statement} {filler} contexto.")
    [vector] = await FakeEmbedder().embed([f"{statement} {filler}"])
    store.set_chunk_embedding(cid, vector)
    _cl = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "  grade, scale_id, confidence, verified) VALUES(?,?,?,?,?,?,1,0.9,1) "
        "RETURNING id",
        (sid, json.dumps([cid]), statement, "quetiapine", direction, grade.value),
    ).fetchone()["id"]
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (int(_cl), directness.value),
    )


async def _search(store, k=5):
    return await Retriever(store, FakeEmbedder()).search_claims(QUERY, k=k)


# ────────────────────────────────── o critério de aceite: a negativa sobe


async def test_a_negative_claim_just_below_the_cut_is_seated(store):
    """Uma negativa logo abaixo do corte sobe. É o que o assento entrega.

    Note o "logo abaixo": o alcance é curto de propósito, e o teste seguinte documenta o
    recall que isso custa — com a medição que explica por quê.
    """
    for i in range(6):
        await _claim(store, f"1000000{i}", f"Quetiapina reduziu ansiedade, estudo {i}.",
               grade=Grade.META_ANALYSIS)
    await _claim(store, "29999999", "Quetiapina não separou de placebo em ansiedade.",
           direction="negative", grade=Grade.CASE_SERIES,
           directness=Directness.PARTIAL)

    hits = await _search(store, k=5)
    pmids = [h.external_id for h in hits]

    assert "29999999" in pmids, f"a contra-evidência ficou fora do bloco: {pmids}"
    assert len(hits) == 5, "o assento reserva, não amplia o bloco"


async def test_a_deeply_buried_negative_reaches_the_block(store):
    """O critério de aceite da Fase 3, agora atendido — e o alcance não é posicional.

    A negativa fica em 14ª numa lista de 14, ou seja no PIOR lugar possível, e entra no
    top-5. A Fase 3 não conseguia isto: lá o assento repescava por janela de posição, e
    qualquer janela larga o bastante para alcançar a 14ª alcançava também a negativa de
    cardiologia — porque `_minmax` faz de `relevance` uma quantidade derivada de posição,
    e as duas chegam no mesmo `MINMAX_FLOOR`.

    O que mudou é o portão: o aval do ranker SEMÂNTICO, que já existia e era descartado na
    fusão RRF. Medido nesta fixture: a negativa no assunto tem cosseno 0,600 e passa
    `MIN_COSINE`; a de cardiologia tem 0,000 e nunca entra no ranking vetorial — ela só
    chegava ao páreo por sobreposição de token no BM25.
    """
    for i in range(13):
        await _claim(store, f"1000000{i}", f"Quetiapina reduziu ansiedade, estudo {i}.",
                     grade=Grade.META_ANALYSIS)
    await _claim(store, "29999999", "Quetiapina não separou de placebo em ansiedade.",
                 direction="negative", grade=Grade.CASE_SERIES)

    hits = await _search(store, k=5)
    assert "29999999" in [h.external_id for h in hits], (
        "a negativa no assunto continua inalcançável: o portão semântico não funcionou"
    )
    assert len(hits) == 5, "o assento reserva, não amplia"


async def test_counter_evidence_that_is_only_off_topic_is_still_reported_as_absent(store):
    """O caso em que a perda é real, e continua declarada.

    O corpus TEM negativa, mas nenhuma sobre o assunto perguntado. O bloco não pode
    inventar contra-evidência nem calar: ele diz que existem e que nenhuma foi alcançada.
    """
    from lithium.chat import ChatEngine

    for i in range(4):
        await _claim(store, f"1000000{i}", f"Quetiapina reduziu ansiedade {i}.")
    await _claim(store, "39999999", "Anticoagulação não reduziu mortalidade.",
                 direction="negative", on_topic=False)

    hits = await _search(store, k=5)
    assert not [h for h in hits if h.direction in {"negative", "null"}]

    class NullLLM:
        async def structured(self, *a, **k):
            raise AssertionError("sem LLM")

    evidence, _ = await ChatEngine(store, NullLLM(), FakeEmbedder(), profile=PROFILE)._evidence_for(QUERY)
    assert "no counter-evidence in this retrieval" in evidence
    assert "1 negative or null claim(s) overall" in evidence


async def test_the_semantic_gate_is_what_excludes_the_off_topic_negative(store):
    """Prova direta do discriminador, para a razão não ficar só na prosa."""
    await _claim(store, "10000000", "Quetiapina reduziu ansiedade.")
    await _claim(store, "29999999", "Quetiapina não separou de placebo.",
                 direction="negative")
    await _claim(store, "39999999", "Anticoagulação não reduziu mortalidade.",
                 direction="negative", on_topic=False)

    hits = await Retriever(store, FakeEmbedder()).search_claims(QUERY, k=50)
    by_pmid = {h.external_id: h for h in hits}

    assert by_pmid["29999999"].semantic is True, (
        "a negativa NO ASSUNTO perdeu o aval semântico — o portão a excluiria"
    )
    # A de cardiologia ou não é recuperada, ou é recuperada SEM aval. As duas contam
    # como excluída do assento; o que não pode acontecer é ela ter aval.
    off_topic = by_pmid.get("39999999")
    assert off_topic is None or off_topic.semantic is False, (
        "a claim de cardiologia recebeu aval semântico — o portão não discrimina"
    )


async def test_the_seat_does_not_import_an_off_topic_negative(store):
    """A armadilha 1: assento sem piso de relevância vira porta dos fundos.

    A negativa de cardiologia não pode deslocar um RCT no assunto — e muito menos ser
    apresentada como contra-evidência da pergunta.
    """
    for i in range(8):
        await _claim(store, f"1000000{i}", f"Quetiapina reduziu ansiedade, estudo {i}.")
    await _claim(store, "39999999",
           "Anticoagulação oral não reduziu mortalidade em fibrilação atrial.",
           direction="negative", grade=Grade.META_ANALYSIS, on_topic=False)

    hits = await _search(store, k=5)
    seated = [h.external_id for h in hits if h.direction == "negative"]

    assert "39999999" not in seated, (
        "uma negativa fora do assunto ocupou o assento de contra-evidência"
    )


async def test_the_channel_does_not_evict_counter_evidence(store):
    """A armadilha 2, e é a mais perversa: o canal expulsando o que ele existe para
    proteger. Duas negativas no bloco não podem virar uma."""
    for i in range(6):
        await _claim(store, f"1000000{i}", f"Quetiapina reduziu ansiedade, estudo {i}.",
               grade=Grade.META_ANALYSIS)
    await _claim(store, "29999901", "Quetiapina não separou de placebo, série A.",
           direction="negative", grade=Grade.CASE_SERIES)
    await _claim(store, "29999902", "Quetiapina sem efeito em ansiedade, série B.",
           direction="null", grade=Grade.CASE_SERIES)

    hits = await _search(store, k=6)
    contra = [h.external_id for h in hits if h.direction in {"negative", "null"}]

    assert len(contra) >= 2, f"o canal perdeu contra-evidência: {contra}"


async def test_the_seats_are_capped(store):
    """O assento reserva, não domina: sempre sobra espaço orgânico."""
    for i in range(4):
        await _claim(store, f"1000000{i}", f"Quetiapina reduziu ansiedade {i}.",
               grade=Grade.META_ANALYSIS)
    for i in range(8):
        await _claim(store, f"2000000{i}", f"Quetiapina não separou de placebo {i}.",
               direction="negative", grade=Grade.META_ANALYSIS)

    hits = await _search(store, k=5)
    positives = [h for h in hits if h.direction == "positive"]
    assert positives, "o bloco virou só contra-evidência"


async def test_a_corpus_without_counter_evidence_is_unchanged(store):
    """Sem negativa nenhuma, o corte é o de sempre — o assento não pode inventar."""
    for i in range(8):
        await _claim(store, f"1000000{i}", f"Quetiapina reduziu ansiedade {i}.")

    hits = await _search(store, k=5)
    assert len(hits) == 5
    assert all(h.direction == "positive" for h in hits)


# ──────────────────────────────────── a linha de ausência, e a marcação


async def test_the_block_marks_which_lines_are_counter_evidence(store):
    from lithium.chat import ChatEngine

    await _claim(store, "29999999", "Quetiapina não separou de placebo.",
           direction="negative")

    class NullLLM:
        async def structured(self, *a, **k):
            raise AssertionError("sem LLM")

    evidence, _ = await ChatEngine(store, NullLLM(), FakeEmbedder(), profile=PROFILE)._evidence_for(QUERY)
    assert "[counter-evidence]" in evidence


async def test_absence_of_counter_evidence_is_stated_not_implied(store):
    """"Nada contra" e "não procuramos" são afirmações diferentes, e o modelo preenche o
    silêncio com a primeira."""
    from lithium.chat import ChatEngine

    for i in range(3):
        await _claim(store, f"1000000{i}", f"Quetiapina reduziu ansiedade {i}.")
    # negativa existente, mas fora do assunto: o corpus TEM contra-evidência
    await _claim(store, "39999999", "Anticoagulação não reduziu mortalidade.",
           direction="negative", on_topic=False)

    class NullLLM:
        async def structured(self, *a, **k):
            raise AssertionError("sem LLM")

    evidence, _ = await ChatEngine(store, NullLLM(), FakeEmbedder(), profile=PROFILE)._evidence_for(QUERY)

    assert "no counter-evidence in this retrieval" in evidence
    assert "Do NOT read this as 'nothing contradicts it'" in evidence
    assert "negative or null claim(s) overall" in evidence, (
        "a contagem do corpus é o que torna a linha verificável em vez de retórica"
    )


def test_the_seat_count_is_declared(store):
    assert CONTRA_SEATS >= 1


class DictatedEmbedder:
    """Vetores ditados, para separar LÉXICO de SEMÂNTICO.

    Com bag-of-words os dois sinais são quase colineares: um texto que compartilha token
    com a consulta também tem cosseno alto. Isso torna impossível construir, com
    `FakeEmbedder`, o caso que o portão existe para pegar — uma claim que o BM25 encontra
    e o ranker semântico rejeita. Ditar os vetores separa as duas coisas, que é
    exatamente a propriedade sob teste.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    async def embed(self, texts):
        return [self.table[t] for t in texts]


async def test_a_negative_in_the_pool_without_semantic_endorsement_is_not_seated(store):
    """O teste que faltava, e a bateria de mutação apontou a falta.

    Desligar o portão (`and c.semantic` removido) ou fazê-lo aprovar tudo passava com a
    suíte verde, porque na outra fixture a claim fora do assunto nem chegava ao páreo.
    Aqui ela chega — o BM25 a encontra pelos tokens compartilhados — e o cosseno ditado a
    reprova. É a única forma do defeito que importa em produção, onde o BM25 traz
    material que o embedding não endossaria.
    """
    on_axis = [1.0] + [0.0] * (DIM - 1)
    off_axis = [0.0, 1.0] + [0.0] * (DIM - 2)

    table = {QUERY: on_axis}
    ids = {}
    for pmid, statement, direction, vector in [
        ("10000001", "Quetiapina reduziu ansiedade no bipolar I com TAG.", "positive", on_axis),
        ("10000002", "Quetiapina reduziu ansiedade no bipolar I, coorte.", "positive", on_axis),
        ("29999901", "Quetiapina não separou de placebo em ansiedade no bipolar I.",
         "negative", on_axis),
        # compartilha token com a consulta (o BM25 acha), mas o embedding é ortogonal
        ("39999901", "Quetiapina reduz ansiedade? Nota editorial sem dado no bipolar I.",
         "negative", off_axis),
    ]:
        sid = store.upsert_source(kind="pubmed", external_id=pmid, raw={},
                                  title="t", year=2020)
        cid = store.add_chunk(source_id=sid, ord=0, text=f"{statement} contexto.")
        store.set_chunk_embedding(cid, vector)
        table[f"{statement} contexto."] = vector
        _cl = store.conn.execute(
            "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
            "  grade, scale_id, confidence, verified) VALUES(?,?,?,?,?,?,1,0.9,1) "
            "RETURNING id",
            (sid, json.dumps([cid]), statement, "quetiapine", direction,
             Grade.META_ANALYSIS.value),
        ).fetchone()["id"]
        store.conn.execute(
            "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
            (int(_cl), Directness.DIRECT.value),
        )
        ids[pmid] = cid

    hits = await Retriever(store, DictatedEmbedder(table)).search_claims(QUERY, k=50)
    by_pmid = {h.external_id: h for h in hits}

    assert "39999901" in by_pmid, (
        "o cenário exige que a claim sem aval esteja NO PÁREO, senão o portão não é "
        "exercitado — foi assim que a mutação passou verde"
    )
    assert by_pmid["39999901"].semantic is False
    assert by_pmid["29999901"].semantic is True

    seated = await Retriever(store, DictatedEmbedder(table)).search_claims(QUERY, k=3)
    pmids = [h.external_id for h in seated]
    assert "39999901" not in pmids, (
        "a negativa sem aval semântico ocupou o assento reservado"
    )
