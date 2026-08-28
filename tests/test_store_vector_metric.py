"""A métrica de distância vetorial precisa ser cosseno nos dois caminhos.

`sqlite-vec` usa **L2 por padrão**; o fallback em numpy calcula cosseno. Sem declarar
`distance_metric=cosine` na tabela vec0, `search_vector` devolve números com
significados diferentes conforme a extensão tenha carregado ou não — e qualquer
limiar calibrado em cima disso fica errado em um dos dois ambientes.

Passou despercebido uma vez: medi "cosseno" num corpus real usando `1 - distância` com
a tabela em L2, e calibrei um piso com números que não eram cosseno.
"""

from __future__ import annotations

import math

import pytest

from lithium.db.store import Store

DIM = 4


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _seed(store: Store, vectors: list[list[float]]) -> list[int]:
    source_id = store.upsert_source(kind="pubmed", external_id="1", raw={})
    ids = []
    for i, vec in enumerate(vectors):
        cid = store.add_chunk(source_id=source_id, ord=i, text=f"chunk {i}")
        store.set_chunk_embedding(cid, vec)
        ids.append(cid)
    return ids


def test_orthogonal_vectors_have_cosine_distance_one(store):
    """Com L2 a resposta seria √2 ≈ 1.414. Com cosseno é exatamente 1.0."""
    [a, _b] = _seed(store, [[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    hits = dict(store.search_vector([1.0, 0, 0, 0], k=2))
    assert hits[a] == pytest.approx(0.0, abs=1e-5)
    assert min(v for k, v in hits.items() if k != a) == pytest.approx(1.0, abs=1e-5)


def test_opposite_vectors_have_cosine_distance_two(store):
    """Faixa do cosseno é [0, 2]. Com L2 daria 2.0 também aqui — este teste sozinho
    não distingue; é o de ortogonais que separa as métricas."""
    _seed(store, [[1.0, 0, 0, 0], [-1.0, 0, 0, 0]])
    distances = sorted(d for _, d in store.search_vector([1.0, 0, 0, 0], k=2))
    assert distances[-1] == pytest.approx(2.0, abs=1e-5)


def test_distance_is_scale_invariant(store):
    """Cosseno ignora magnitude; L2 não. Embeddings do llama.cpp não vêm
    necessariamente normalizados."""
    [a, b] = _seed(store, [[3.0, 4.0, 0, 0], [0.3, 0.4, 0, 0]])
    hits = dict(store.search_vector([30.0, 40.0, 0, 0], k=2))
    assert hits[a] == pytest.approx(0.0, abs=1e-5)
    assert hits[b] == pytest.approx(0.0, abs=1e-5)


def test_known_angle_matches_cosine_formula(store):
    """45° → cos = √2/2 ≈ 0.7071 → distância ≈ 0.2929. Com L2 seria ≈ 0.7654."""
    _seed(store, [[1.0, 1.0, 0, 0]])
    [(_, distance)] = store.search_vector([1.0, 0, 0, 0], k=1)
    assert distance == pytest.approx(1.0 - math.sqrt(2) / 2, abs=1e-4)


def test_both_backends_agree_on_the_metric(store, monkeypatch, tmp_path):
    """O mesmo par de vetores precisa dar a mesma distância com e sem sqlite-vec."""
    vectors = [[1.0, 0.5, 0.25, 0.125], [0.2, 0.9, 0.1, 0.4]]
    query = [0.7, 0.7, 0.0, 0.1]

    native = dict(_distances(store, vectors, query))

    fallback_store = Store(tmp_path / "fb.db", embedding_dim=DIM)
    fallback_store._vec_available = False
    monkeypatch.setattr(fallback_store, "_load_vec_extension", lambda conn: False)
    fallback_store.init_schema()
    assert not fallback_store.vec_available
    fb = dict(_distances(fallback_store, vectors, query))
    fallback_store.close()

    assert sorted(native.values()) == pytest.approx(sorted(fb.values()), abs=1e-5)


def _distances(store: Store, vectors, query):
    _seed(store, vectors)
    return store.search_vector(query, k=len(vectors))


def test_l2_table_is_migrated_to_cosine(tmp_path, caplog):
    """Bancos criados antes do fix ficariam em L2 para sempre — `IF NOT EXISTS`
    não conserta tabela já criada."""
    path = tmp_path / "legacy.db"
    legacy = Store(path, embedding_dim=DIM)
    legacy.init_schema()
    # simula o schema antigo, sem distance_metric
    legacy.conn.execute("DROP TABLE chunk_vec")
    legacy.conn.execute(
        f"CREATE VIRTUAL TABLE chunk_vec USING vec0("
        f"  chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{DIM}])"
    )
    _seed(legacy, [[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    assert dict(legacy.search_vector([1.0, 0, 0, 0], k=2))  # L2: ortogonais dão √2
    legacy.close()

    reopened = Store(path, embedding_dim=DIM)
    reopened.init_schema()
    sql = reopened.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunk_vec'"
    ).fetchone()["sql"]
    assert "distance_metric=cosine" in sql.lower()
    assert reopened.conn.execute("SELECT COUNT(*) AS n FROM chunk_vec").fetchone()["n"] == 0
    reopened.close()


def test_dimension_change_also_triggers_rebuild(tmp_path):
    """Trocar de modelo de embedding muda a dimensão; o índice antigo é lixo."""
    path = tmp_path / "dim.db"
    first = Store(path, embedding_dim=DIM)
    first.init_schema()
    first.close()

    second = Store(path, embedding_dim=DIM * 2)
    second.init_schema()
    sql = second.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunk_vec'"
    ).fetchone()["sql"]
    assert f"float[{DIM * 2}]" in sql.lower()
    second.close()
