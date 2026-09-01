"""O relatório periódico: delta, e o que não dá para saber.

Item 10. O motor é agnóstico — não sabe o que é fármaco nem ensaio, só lê contadores e as
corridas de extração; o alvo entra como texto vindo do banco.
"""

from __future__ import annotations

import pytest

from lithium.db import Store
from lithium.pipeline.extract import ExtractionResult, Rejection
from lithium.report import build, save

DIM = 8


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "r.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _run(store, sid=1, **kw):
    base = dict(source_id=sid, proposed=4, anchored=3, verified=2, chunks_seen=5)
    r = ExtractionResult(**{**base, **kw})
    return r


def test_the_window_is_the_delta_since_the_last_saved_report(store):
    """"213 claims" é teatro de crescimento: sobe com qualquer afrouxamento e sobe sozinho
    com o tempo. O que responde "o que aconteceu" é a diferença.

    MUTAÇÃO: em `build`, ignorar `_last_report_at` e sempre relatar o total.
    """
    store.upsert_source(kind="pubmed", external_id="1", raw={}, title="t")
    store.record_extraction(_run(store), focus_id=1)
    primeiro = build(store)
    assert "4" in primeiro.render(), "a primeira janela cobre tudo"

    save(store, primeiro)
    depois = build(store)

    assert depois.since is not None, "a janela não avançou com o relatório salvo"
    assert "Nenhuma extração nesta janela" in depois.render(), (
        "o segundo relatório recontou o que o primeiro já havia contado"
    )


def test_reading_a_report_does_not_move_the_window(store):
    """Ler por curiosidade não pode fazer o próximo relatório esconder o período.

    Por isso `build` não salva e `--save` é opt-in; quem salva é o job semanal.

    MUTAÇÃO: fazer `build` chamar `save` no fim.
    """
    store.upsert_source(kind="pubmed", external_id="1", raw={}, title="t")
    store.record_extraction(_run(store), focus_id=1)
    build(store)
    build(store)
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM reports").fetchone()["n"] == 0, (
        "montar o relatório persistiu-o: a janela andou sem ninguém pedir"
    )


def test_counter_evidence_gets_its_own_line(store):
    """Um relatório que só conta o que confirma descreve um sistema que parou de procurar
    o que contradiz.

    MUTAÇÃO: remover a linha de direção da seção "O que entrou".
    """
    import json

    sid = store.upsert_source(kind="pubmed", external_id="1", raw={}, title="t")
    cid = store.add_chunk(source_id=sid, ord=0, text="texto")
    for direcao in ("positive", "negative", "null"):
        store.conn.execute(
            "INSERT INTO claims(source_id, chunk_ids, statement, direction, grade, "
            "  scale_id, confidence, verified) VALUES(?,?,?,?,'cohort',1,0.9,1)",
            (sid, json.dumps([cid]), f"c-{direcao}", direcao))

    texto = build(store).render()
    assert "2 contra, nula ou mista" in texto, (
        "a contra-evidência não apareceu com espaço próprio"
    )


def test_the_report_names_no_domain_of_its_own(store):
    """O motor é agnóstico: o alvo entra como texto do banco, nada mais.

    MUTAÇÃO: escrever qualquer termo de domínio nas seções de `lithium/report.py`.
    """
    alvo = "liga Ti-6Al-4V sob fadiga criogênica"
    store.conn.execute("UPDATE focuses SET target = ? WHERE id = ?",
                       (alvo, store.active_focus()["id"]))
    texto = build(store).render()
    assert alvo in texto
    import re
    assert not re.search(r"bipolar|lítio|quetiapin|anxiety", texto, re.I), (
        "o relatório trouxe vocabulário de um foco que não é o ativo"
    )
