"""Testes do store. Rodam em macOS e Windows — ver .github/workflows/lithium-ci.yml.

O teste que mais importa aqui é `test_directness_outranks_design`: ele trava a
propriedade de domínio que justifica o modelo de dados inteiro.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lithium.db.store import Store

DIM = 8


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _source(store: Store, external_id: str = "1", **kw) -> int:
    return store.upsert_source(
        kind="pubmed", external_id=external_id, raw={"pmid": external_id},
        title=f"Estudo {external_id}", year=2020, **kw,
    )


def _claim(store: Store, source_id: int, grade: str, directness: str,
           *, verified: int = 1, confidence: float = 1.0,
           intervention: str = "quetiapina", direction: str = "positive") -> int:
    chunk_id = store.add_chunk(source_id=source_id, ord=0, text="texto de apoio")
    cur = store.conn.execute(
        "INSERT INTO claims(source_id, chunk_ids, statement, intervention, direction, "
        "                   grade, scale_id, confidence, verified) "
        "VALUES(?, ?, ?, ?, ?, ?, 1, ?, ?) RETURNING id",
        (source_id, json.dumps([chunk_id]), "reduz sintomas", intervention, direction,
         grade, confidence, verified),
    )
    claim_id = int(cur.fetchone()["id"])
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?, 1, ?)",
        (claim_id, directness),
    )
    return claim_id


# ─────────────────────────────────────────────────────────────────────── schema


def test_init_schema_is_idempotent(tmp_path):
    s = Store(tmp_path / "x.db", embedding_dim=DIM)
    s.init_schema()
    s.init_schema()  # não pode explodir nem duplicar seeds
    n = s.conn.execute("SELECT COUNT(*) AS n FROM grade_weight").fetchone()["n"]
    assert n == 9
    s.close()


def test_seeds_present(store):
    grades = {r["grade"] for r in store.conn.execute("SELECT grade FROM grade_weight")}
    assert "meta_analysis" in grades and "opinion" in grades
    dirs = {r["directness"] for r in store.conn.execute("SELECT directness FROM directness_weight")}
    assert dirs == {"direct", "partial", "indirect", "extrapolated"}


def test_invalid_enum_is_rejected(store):
    src = _source(store)
    with pytest.raises(Exception):
        store.conn.execute(
            "INSERT INTO claims(source_id, chunk_ids, statement, grade) "
            "VALUES(?, '[]', 'x', 'not_a_grade')", (src,),
        )


# ────────────────────────────────────────────────────────── a invariante de domínio


def test_directness_outranks_design(store):
    """Coorte na população certa deve pesar mais que meta-análise na população errada.

    É a razão de `directness` ser multiplicativo. Se este teste cair, o sistema
    passa a recomendar com base em RCTs de unipolar — exatamente o erro que o
    projeto existe para evitar.
    """
    weak_design_right_pop = _claim(store, _source(store, "a"), "cohort", "direct")
    strong_design_wrong_pop = _claim(store, _source(store, "b"), "meta_analysis", "indirect")

    weights = {
        r["claim_id"]: r["weight"]
        for r in store.conn.execute("SELECT claim_id, weight FROM claim_weight")
    }
    assert weights[weak_design_right_pop] > weights[strong_design_wrong_pop]


def test_unverified_claims_are_invisible_to_scoring(store):
    """Claim não verificada não pode influenciar placar — nem relatório, nem treino."""
    _claim(store, _source(store, "a"), "rct", "direct", verified=0)
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"] == 0


def test_hypothesis_scoreboard_sums_by_polarity(store):
    hyp = store.conn.execute(
        "INSERT INTO hypotheses(focus_id, statement) "
        "VALUES(1, 'quetiapina ajuda') RETURNING id"
    ).fetchone()["id"]
    pro = _claim(store, _source(store, "a"), "rct", "direct")          # 0.85
    con = _claim(store, _source(store, "b"), "case_report", "direct")  # 0.15
    store.conn.executemany(
        "INSERT INTO evidence_links(claim_id, hypothesis_id, polarity) VALUES(?, ?, ?)",
        [(pro, hyp, "support"), (con, hyp, "contra")],
    )
    row = store.conn.execute("SELECT * FROM hypothesis_scoreboard WHERE id = ?", (hyp,)).fetchone()
    assert row["support"] == pytest.approx(0.85)
    assert row["contra"] == pytest.approx(0.15)
    assert row["n_claims"] == 2


def test_scoreboard_includes_hypotheses_without_evidence(store):
    """LEFT JOIN: hipótese recém-criada precisa aparecer com placar zerado."""
    store.conn.execute("INSERT INTO hypotheses(focus_id, statement) "
                       "VALUES(1, 'sem evidência ainda')")
    row = store.conn.execute("SELECT * FROM hypothesis_scoreboard").fetchone()
    assert row["support"] == 0 and row["contra"] == 0 and row["n_claims"] == 0


# ───────────────────────────────────────────────────────────────────── upsert/CRUD


def test_upsert_source_is_idempotent(store):
    a = store.upsert_source(kind="pubmed", external_id="123", raw={}, title="v1")
    b = store.upsert_source(kind="pubmed", external_id="123", raw={}, title="v2")
    assert a == b
    assert store.conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"] == 1
    assert store.conn.execute("SELECT title FROM sources").fetchone()["title"] == "v2"


def test_add_chunk_upserts_on_reprocess(store):
    src = _source(store)
    a = store.add_chunk(source_id=src, ord=0, text="original")
    b = store.add_chunk(source_id=src, ord=0, text="reprocessado")
    assert a == b
    assert store.conn.execute("SELECT text FROM chunks").fetchone()["text"] == "reprocessado"


def test_cascade_delete_removes_chunks(store):
    src = _source(store)
    store.add_chunk(source_id=src, ord=0, text="alvo")
    store.conn.execute("DELETE FROM sources WHERE id = ?", (src,))
    assert store.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 0


# ──────────────────────────────────────────────────────────────────────── buscas


def test_fts_index_tracks_chunk_writes(store):
    src = _source(store)
    cid = store.add_chunk(source_id=src, ord=0, text="quetiapina reduziu sintomas de ansiedade")
    store.add_chunk(source_id=src, ord=1, text="lítio exige monitoramento sérico")

    assert [c for c, _ in store.search_text("quetiapina")] == [cid]

    # o trigger de UPDATE precisa reindexar, senão a busca fica mentindo
    store.add_chunk(source_id=src, ord=0, text="pregabalina em transtorno de ansiedade")
    assert store.search_text("quetiapina") == []
    assert [c for c, _ in store.search_text("pregabalina")] == [cid]


def test_vector_search_ranks_by_similarity(store):
    src = _source(store)
    ids = [store.add_chunk(source_id=src, ord=i, text=f"c{i}") for i in range(3)]
    vectors = [
        [1.0] + [0.0] * (DIM - 1),
        [0.0, 1.0] + [0.0] * (DIM - 2),
        [0.9, 0.1] + [0.0] * (DIM - 2),
    ]
    for cid, vec in zip(ids, vectors, strict=True):
        store.set_chunk_embedding(cid, vec)

    hits = store.search_vector([1.0] + [0.0] * (DIM - 1), k=3)
    assert [cid for cid, _ in hits][:2] == [ids[0], ids[2]]


def test_set_chunk_embedding_overwrites(store):
    src = _source(store)
    cid = store.add_chunk(source_id=src, ord=0, text="c")
    store.set_chunk_embedding(cid, [1.0] + [0.0] * (DIM - 1))
    store.set_chunk_embedding(cid, [0.0, 1.0] + [0.0] * (DIM - 2))
    assert len(store.search_vector([1.0] + [0.0] * (DIM - 1), k=10)) == 1


def test_embedding_dim_mismatch_fails_loudly(store):
    src = _source(store)
    cid = store.add_chunk(source_id=src, ord=0, text="c")
    with pytest.raises(ValueError, match="esperado"):
        store.set_chunk_embedding(cid, [1.0, 2.0])


def test_embedding_roundtrip_preserves_values(store):
    vec = np.random.default_rng(0).random(DIM).astype(np.float32)
    assert np.allclose(Store.unpack_embedding(Store.pack_embedding(vec)), vec)


# ──────────────────────────────────────────────────────────────────── concorrência


def test_thread_local_connections_see_committed_writes(store):
    """O worker pool chama o store de várias threads via asyncio.to_thread."""
    import threading

    src = _source(store)
    seen: list[int] = []

    def reader() -> None:
        seen.append(store.conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"])

    t = threading.Thread(target=reader)
    t.start()
    t.join()
    assert seen == [1] and src


def test_counts_reports_pipeline_state(store):
    _claim(store, _source(store, "a"), "rct", "direct", verified=1)
    _claim(store, _source(store, "b"), "rct", "direct", verified=0)
    counts = store.counts()
    assert counts["claims"] == 2
    assert counts["claims_verified"] == 1
    assert counts["questions_escalated"] == 0
