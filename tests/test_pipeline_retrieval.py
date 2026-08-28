"""Ingestão, chunking e retrieval híbrido.

O embedder é falso e determinístico (bag-of-words hasheada), o que dá similaridade
semântica plausível sem GPU e sem rede — suficiente para verificar a fusão de
rankings e a reordenação por evidência, que é o que estes testes existem para travar.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Sequence

import pytest

from lithium.db import Store
from lithium.pipeline.ingest import MAX_CHUNK_CHARS, Ingestor, split_passage
from lithium.pipeline.retrieval import Retriever, fts_query, reciprocal_rank_fusion
from lithium.sources.base import Passage, SourceRecord
from lithium.types import Directness, Grade

DIM = 32


def _bucket(word: str) -> int:
    """`hash()` de str é salinizado por processo: os vetores mudariam a cada
    execução e os testes ficariam flaky. crc32 é estável entre processos."""
    return zlib.crc32(word.encode()) % DIM



class FakeEmbedder:
    """Bag-of-words hasheada e normalizada: textos com palavras em comum ficam
    próximos, o que basta para exercitar o ranking."""

    dim = DIM

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[_bucket(word)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


@pytest.fixture
def embedder():
    return FakeEmbedder()


def _record(external_id: str, passages: list[Passage], **kw) -> SourceRecord:
    return SourceRecord(
        kind="pubmed",
        external_id=external_id,
        title=kw.pop("title", f"Estudo {external_id}"),
        passages=passages,
        raw={},
        **kw,
    )


# ─────────────────────────────────────────────────────────────────────── chunking


def test_short_passage_is_not_split():
    assert split_passage("Uma frase curta.") == ["Uma frase curta."]


def test_empty_passage_yields_nothing():
    assert split_passage("   ") == []


def test_long_passage_splits_on_sentence_boundaries():
    text = " ".join(f"Esta é a sentença número {i} do texto." for i in range(200))
    chunks = split_passage(text)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_chunks_overlap_so_quotes_survive_the_boundary():
    """A citação verbatim precisa caber inteira num chunk. Sem sobreposição, uma
    frase partida no limite vira claim que o verificador de ancoragem descarta —
    evidência real perdida por acidente de tokenização."""
    text = " ".join(f"Sentença {i} com conteúdo suficiente para ocupar espaço." for i in range(80))
    chunks = split_passage(text)
    tails = [c[-60:] for c in chunks[:-1]]
    assert any(tail.split()[0] in nxt for tail, nxt in zip(tails, chunks[1:], strict=True))


def test_single_sentence_longer_than_limit_is_force_split():
    """Tabelas e listas coladas viram uma 'sentença' gigante sem pontuação."""
    chunks = split_passage("palavra " * 2000)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


# ────────────────────────────────────────────────────────────────────── ingestão


async def test_ingest_persists_source_chunks_and_embeddings(store, embedder):
    result = await Ingestor(store, embedder).ingest(
        _record("1", [Passage("Quetiapina XR reduziu escores de ansiedade de forma significativa "
                        "frente a placebo em oito semanas.", "results")],
                year=2019, journal="Lancet", design=Grade.RCT)
    )
    assert len(result.chunk_ids) == 1
    assert result.embedded == 1

    row = store.conn.execute("SELECT * FROM sources").fetchone()
    assert row["design"] == "rct" and row["year"] == 2019
    assert store.conn.execute("SELECT section FROM chunks").fetchone()["section"] == "results"
    assert store.conn.execute("SELECT COUNT(*) AS n FROM chunk_vec").fetchone()["n"] == 1


async def test_ingest_is_idempotent_and_skips_reembedding(store, embedder):
    """O daemon reprocessa muito; re-embeddar tudo a cada passada custaria horas."""
    record = _record("1", [Passage("Um trecho de texto estável e longo o bastante para virar chunk.")])
    ingestor = Ingestor(store, embedder)

    first = await ingestor.ingest(record)
    calls_after_first = embedder.calls
    second = await ingestor.ingest(record)

    assert first.chunk_ids == second.chunk_ids
    assert second.embedded == 0 and second.skipped_existing
    assert embedder.calls == calls_after_first, "não pode re-embeddar chunk já indexado"
    assert store.conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"] == 1


async def test_reembed_flag_forces_recompute(store, embedder):
    record = _record("1", [Passage("Um trecho de texto com tamanho suficiente para indexar.")])
    ingestor = Ingestor(store, embedder)
    await ingestor.ingest(record)
    assert (await ingestor.ingest(record, reembed=True)).embedded == 1


async def test_multiple_passages_get_sequential_ordinals(store, embedder):
    await Ingestor(store, embedder).ingest(
        _record("1", [Passage("Primeira passagem, com tamanho suficiente para indexar.", "background"),
         Passage("Segunda passagem, também longa o bastante para virar chunk.", "results")])
    )
    rows = store.conn.execute("SELECT ord, section FROM chunks ORDER BY ord").fetchall()
    assert [(r["ord"], r["section"]) for r in rows] == [(0, "background"), (1, "results")]


async def test_boilerplate_sections_are_not_indexed(store, embedder):
    """Regressão de um piloto real: um chunk cujo texto era literalmente "None."
    (seção `funding`) apareceu em 2º lugar numa busca. Boilerplate curto tem
    embedding próximo de qualquer coisa, então ele emerge justamente quando não há
    resposta boa — o pior momento possível."""
    await Ingestor(store, embedder).ingest(
        _record("1", [
            Passage("Quetiapina reduziu escores de ansiedade frente a placebo.", "results"),
            Passage("None.", "funding"),
            Passage("The authors declare no competing interests.", "declaration of interest"),
            Passage("Os autores agradecem aos coordenadores do estudo.", "acknowledgements"),
        ])
    )
    sections = {r["section"] for r in store.conn.execute("SELECT section FROM chunks")}
    assert sections == {"results"}


async def test_limitations_section_is_kept(store, embedder):
    """`limitations` é conteúdo, e conteúdo que importa para graduar a evidência."""
    await Ingestor(store, embedder).ingest(
        _record("1", [
            Passage("Os dados eram transversais e nem todos os fármacos entraram.", "limitations")
        ])
    )
    assert store.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 1


async def test_chunks_below_minimum_length_are_dropped(store, embedder):
    """Pega o boilerplate que escapa da lista de seções: "Not applicable.", "None."."""
    await Ingestor(store, embedder).ingest(
        _record("1", [
            Passage("Not applicable.", "results"),
            Passage("Uma passagem de conteúdo com tamanho suficiente para ser indexada.", "results"),
        ])
    )
    assert store.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 1


# ──────────────────────────────────────────────────────────────────────── RRF


def test_rrf_rewards_agreement_between_rankers():
    """Item consistentemente bem colocado vence item que só um ranker adora.

    A margem é estreita de propósito no RRF: ser #1 num ranking e #3 no outro
    *supera* ser #2 nos dois (1/61 + 1/63 > 2/62). Por isso o caso de teste usa
    posições bem separadas — é onde a propriedade realmente aparece.
    """
    fused = reciprocal_rank_fusion([[1, 2, 3, 4, 5], [5, 2, 4, 3, 1]])
    assert fused[2] == max(fused.values())
    assert fused[2] > fused[1] and fused[2] > fused[5]


def test_minmax_never_zeroes_the_last_candidate():
    """Regressão: normalizar para [0, 1] cru dava exatamente 0.0 ao pior candidato,
    e o guarda `relevance <= 0.0` em search_claims o descartava. O último colocado
    sumia silenciosamente — 1 de 60 no caso normal, metade dos resultados quando só
    havia dois candidatos."""
    from lithium.pipeline.retrieval import MINMAX_FLOOR, _minmax

    out = _minmax({1: 0.03, 2: 0.02, 3: 0.01})
    assert min(out.values()) == pytest.approx(MINMAX_FLOOR)
    assert max(out.values()) == pytest.approx(1.0)
    assert all(v > 0 for v in out.values())


def test_minmax_ties_all_map_to_one():
    """Textos idênticos empatam; aí quem desempata é só o peso de evidência."""
    from lithium.pipeline.retrieval import _minmax

    assert set(_minmax({1: 0.5, 2: 0.5}).values()) == {1.0}


def test_rrf_handles_disjoint_rankings():
    fused = reciprocal_rank_fusion([[1], [2]])
    assert set(fused) == {1, 2}


def test_rrf_weights_shift_the_balance():
    only_text = reciprocal_rank_fusion([[9], [1, 2]], weights=[0.0, 1.0])
    assert only_text.get(9, 0.0) == 0.0


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([[], []]) == {}


# ──────────────────────────────────────────────────────────── sanitização do FTS


@pytest.mark.parametrize(
    "raw",
    [
        'quetiapina "monoterapia" (TAG)',
        "bipolar AND NOT unipolar",
        "efeito*",
        "NEAR/3 lítio",
        "e o que dizer de: 50 mg/dia?",
    ],
)
def test_fts_query_survives_punctuation_and_operators(store, raw):
    """Pergunta em linguagem natural derrubaria o MATCH com 'fts5: syntax error'."""
    store.conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?", (fts_query(raw),))


def test_fts_query_on_empty_text_is_still_valid(store):
    store.conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?", (fts_query("!!!"),))


# ─────────────────────────────────────────────────────────────── busca de chunks


async def test_search_chunks_finds_by_exact_term(store, embedder):
    ingestor = Ingestor(store, embedder)
    await ingestor.ingest(_record("1", [Passage("Pregabalina reduziu a pontuação HAM-A de forma significativa "
                              "em adultos com ansiedade generalizada.")]))
    await ingestor.ingest(_record("2", [Passage("Lítio exige monitoramento sérico regular por conta da "
                              "janela terapêutica estreita.")]))

    hits = await Retriever(store, embedder).search_chunks("pregabalina HAM-A", k=1)
    assert len(hits) == 1
    assert hits[0].external_id == "1"
    assert hits[0].title == "Estudo 1"


async def test_search_chunks_on_empty_index(store, embedder):
    assert await Retriever(store, embedder).search_chunks("qualquer coisa") == []


async def test_similarity_floor_returns_nothing_rather_than_noise(store, embedder):
    """Regressão de um piloto real: `search_vector` sempre devolve os k mais
    próximos, por pior que sejam. Sem piso, uma consulta sem resposta no corpus
    retornava com confiança o chunk menos irrelevante — e esse texto seguiria
    adiante como "evidência" para o sintetizador.

    O piso NÃO é filtro de relevância — medições com bge-m3 mostraram que consulta
    conversacional sobre o tema certo (0.705) pontua abaixo de pergunta fora do
    domínio (0.710), então nenhum limiar separa. Ele existe só como guarda contra
    caso degenerado. Ver a docstring de MIN_COSINE.
    """
    await Ingestor(store, embedder).ingest(
        _record("1", [Passage("Revascularização miocárdica em doença arterial coronariana.")])
    )
    # Piso alto o bastante para reprovar qualquer coisa: nada deve voltar.
    assert await Retriever(store, embedder).search_chunks(
        "quetiapina", min_cosine=0.99, text_weight=0.0
    ) == []
    # Sem piso, o mesmo chunk irrelevante volta — é o comportamento que o piso corta.
    assert await Retriever(store, embedder).search_chunks(
        "quetiapina", min_cosine=0.0, text_weight=0.0
    ) != []


async def test_bm25_catches_rare_token_embedding_would_dilute(store, embedder):
    """Identificadores como NCT01236411 são o caso onde só o BM25 salva."""
    ingestor = Ingestor(store, embedder)
    await ingestor.ingest(_record("1", [Passage("O ensaio NCT01236411 avaliou desfechos de ansiedade ao longo "
                              "de doze semanas de acompanhamento.")]))
    for i in range(2, 8):
        await ingestor.ingest(_record(str(i), [Passage(f"Texto genérico de preenchimento número {i}, sem relação com o "
                                  f"assunto pesquisado neste teste.")]))

    hits = await Retriever(store, embedder).search_chunks("NCT01236411", k=3)
    assert hits[0].external_id == "1"


# ──────────────────────────────────── busca de claims e reordenação por evidência


async def _add_claim(store, source_id: int, chunk_id: int, *, grade: Grade,
                     directness: Directness, statement: str) -> int:
    cur = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "grade, scale_id, confidence, verified) VALUES(?,?,?,?,?,?,1,1.0,1) RETURNING id",
        (source_id, json.dumps([chunk_id]), statement, "x", "positive", grade.value),
    )
    claim_id = int(cur.fetchone()["id"])
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (claim_id, directness.value),
    )
    return claim_id


async def test_claim_ranking_prefers_right_population_over_stronger_design(store, embedder):
    """A invariante do projeto, agora ponta a ponta no retrieval.

    Os dois chunks têm relevância textual quase idêntica. O desempate tem que vir do
    peso de evidência — senão o sistema recomenda com base em RCT de unipolar, que é
    o erro que ele existe para evitar.
    """
    ingestor = Ingestor(store, embedder)
    weak_right = await ingestor.ingest(
        _record("1", [Passage("Quetiapina reduziu ansiedade em pacientes estudados.")])
    )
    strong_wrong = await ingestor.ingest(
        _record("2", [Passage("Quetiapina reduziu ansiedade em pacientes estudados.")])
    )

    right_pop = await _add_claim(
        store, weak_right.source_id, weak_right.chunk_ids[0],
        grade=Grade.COHORT, directness=Directness.DIRECT,
        statement="coorte em bipolar I com TAG",
    )
    await _add_claim(
        store, strong_wrong.source_id, strong_wrong.chunk_ids[0],
        grade=Grade.META_ANALYSIS, directness=Directness.INDIRECT,
        statement="meta-análise em unipolar",
    )

    hits = await Retriever(store, embedder).search_claims("quetiapina ansiedade", k=5)
    assert hits[0].claim_id == right_pop
    assert hits[0].weight > hits[1].weight


async def test_unverified_claims_never_surface(store, embedder):
    result = await Ingestor(store, embedder).ingest(_record("1", [Passage("Um trecho de texto relevante e com tamanho suficiente para indexar.")]))
    store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, grade, scale_id, verified) "
        "VALUES(?,?,?,?,1,0)",
        (result.source_id, json.dumps(result.chunk_ids), "não verificada", "rct"),
    )
    assert await Retriever(store, embedder).search_claims("texto relevante") == []


async def test_offtopic_claim_ranks_below_ontopic_despite_stronger_evidence(store, embedder):
    """Uma claim herda a relevância do melhor chunk que a sustenta.

    Não há piso absoluto de similaridade — um limiar de cosseno seria dependente do
    modelo de embedding e envelheceria mal. A defesa contra evidência irrelevante é
    o corte por `k` mais o juiz de suficiência, que vê a evidência e pode declarar
    que ela não responde a pergunta. O que o retrieval garante é a *ordem*: fora do
    assunto fica atrás, mesmo com desenho de estudo mais forte.
    """
    ingestor = Ingestor(store, embedder)
    on_topic = await ingestor.ingest(
        _record("1", [Passage("Quetiapina em transtorno bipolar com ansiedade generalizada.")])
    )
    off_topic = await ingestor.ingest(
        _record("2", [Passage("Stent farmacológico em doença arterial coronariana.")])
    )

    relevant = await _add_claim(
        store, on_topic.source_id, on_topic.chunk_ids[0],
        grade=Grade.CASE_SERIES, directness=Directness.DIRECT, statement="no assunto",
    )
    await _add_claim(
        store, off_topic.source_id, off_topic.chunk_ids[0],
        grade=Grade.META_ANALYSIS, directness=Directness.DIRECT, statement="fora do assunto",
    )

    hits = await Retriever(store, embedder).search_claims("quetiapina bipolar ansiedade")
    assert hits[0].claim_id == relevant


async def test_evidence_weight_does_not_swamp_relevance(store, embedder):
    """Regressão: o peso de evidência engolia a relevância e a busca deixava de buscar.

    Scores RRF crus variam ~2x entre o 1º e o último candidato; o peso de evidência
    varia ~160x (`opinion × extrapolated` = 0.006 até `meta_analysis × direct` = 1.0).
    Multiplicando os dois sem normalizar, o peso decide sozinho — perguntar sobre TCC
    devolvia a claim de maior grade do corpus, sobre o assunto que fosse.

    Aqui a claim fraca está no assunto e a forte está em outro campo inteiro. A
    relevância precisa mandar.
    """
    ingestor = Ingestor(store, embedder)
    on_topic = await ingestor.ingest(
        _record("1", [Passage("Terapia cognitivo comportamental para ansiedade no bipolar.")])
    )
    off_topic = await ingestor.ingest(
        _record("2", [Passage("Anticoagulação oral em fibrilação atrial não valvar.")])
    )

    weakest = await _add_claim(
        store, on_topic.source_id, on_topic.chunk_ids[0],
        grade=Grade.OPINION, directness=Directness.EXTRAPOLATED, statement="TCC, evidência fraca",
    )
    await _add_claim(
        store, off_topic.source_id, off_topic.chunk_ids[0],
        grade=Grade.META_ANALYSIS, directness=Directness.DIRECT, statement="cardiologia, forte",
    )

    hits = await Retriever(store, embedder).search_claims("terapia cognitivo comportamental")
    assert hits[0].claim_id == weakest, "peso de evidência voltou a dominar a relevância"


async def test_k_limits_the_evidence_handed_to_the_llm(store, embedder):
    """Janela de contexto é finita; `k` é o que impede o corpus inteiro de entrar."""
    ingestor = Ingestor(store, embedder)
    for i in range(12):
        result = await ingestor.ingest(_record(str(i), [Passage(f"Quetiapina foi avaliada no estudo número {i} com desfecho de ansiedade.")]))
        await _add_claim(
            store, result.source_id, result.chunk_ids[0],
            grade=Grade.RCT, directness=Directness.DIRECT, statement=f"claim {i}",
        )
    assert len(await Retriever(store, embedder).search_claims("quetiapina", k=4)) == 4
