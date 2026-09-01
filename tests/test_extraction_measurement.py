"""O NÃO de cada portão passa a existir no banco.

Ver METRICS.md, MF1. Antes desta instrumentação, `anchored/proposed` só era mensurável
por grep num arquivo de log — e o único handler de log deste projeto é
`RichHandler(console)`, sem arquivo. A taxa das cinco primeiras fontes do corpus real
(27 claims, 12,7%) foi perdida exatamente assim: o log estava em `/tmp` e o macOS o
apagou no boot.
"""

from __future__ import annotations

import pytest

from lithium.db import Store
from lithium.pipeline.extract import ExtractionResult, Rejection

DIM = 8


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", embedding_dim=DIM)
    s.init_schema()
    s.upsert_source(kind="pubmed", external_id="1", raw={}, title="t")
    yield s
    s.close()


def _result(**kw) -> ExtractionResult:
    base = dict(source_id=1, proposed=3, anchored=2, verified=1, chunks_seen=4)
    r = ExtractionResult(**{**base, **kw})
    return r


def test_every_rejection_is_persisted_not_just_the_first_five(store):
    """O `log.debug(result.rejections[:5])` descartava a sexta em diante, em silêncio.

    Num chunk denso a extração propõe até 8 claims (`ClaimExtraction` tem
    `max_length=8`), então o truncamento não era hipotético.

    MUTAÇÃO: em `Store.record_extraction`, voltar a iterar `result.rejections[:5]`.
    """
    r = _result(proposed=8, anchored=0, verified=0)
    for i in range(8):
        r.rejections.append(Rejection(chunk_id=1, gate="anchor",
                                      reason=f"motivo {i}", quote=f"q{i}"))
    store.record_extraction(r, focus_id=1)

    n = store.conn.execute(
        "SELECT COUNT(*) AS n FROM extraction_rejections").fetchone()["n"]
    assert n == 8, f"{n} de 8 rejeições persistidas — o truncamento voltou"


def test_the_gate_that_rejected_is_recorded_not_flattened_into_prose(store):
    """`gate` separa "o gerador fabricou citação" de "a citação não sustentava".

    São dois defeitos com consertos diferentes, e a string formatada antiga os misturava
    numa frase. MF1 lê `gate`; sem ele a métrica não existe.

    MUTAÇÃO: gravar `str(r)` numa coluna só, em vez de `gate` separado.
    """
    r = _result()
    r.rejections.append(Rejection(chunk_id=7, gate="anchor", reason="não literal",
                                  statement="s", quote="citação inventada"))
    r.rejections.append(Rejection(chunk_id=7, gate="entailment", reason="não sustenta",
                                  statement="s2", quote="citação real"))
    store.record_extraction(r, focus_id=1)

    linhas = {x["gate"]: x for x in store.conn.execute(
        "SELECT gate, chunk_id, quote FROM extraction_rejections")}
    assert set(linhas) == {"anchor", "entailment"}
    assert linhas["anchor"]["chunk_id"] == 7
    assert linhas["anchor"]["quote"] == "citação inventada", (
        "a citação rejeitada não foi guardada — sem ela não dá para distinguir "
        "paráfrase de colagem de três palavras seguras"
    )


def test_a_sterile_chunk_is_told_apart_from_an_annihilated_one(store):
    """Duas ausências que hoje são a mesma linha em falta, e pedem ações OPOSTAS.

    Aniquilado = propôs e perdeu tudo: o paper certo, a citação ruim -> apertar o
    extrator. Estéril = não propôs nada: o paper errado -> trocar a frente de busca.
    No corpus real são 76 de 160 chunks indistinguíveis.

    MUTAÇÃO: fazer `chunks_sterile` devolver `chunks_seen - len(chunks_yielding)`, que
    conta os aniquilados como estéreis.
    """
    r = _result(proposed=2, anchored=0, verified=1, chunks_seen=4)
    r.chunks_yielding.add(1)                        # chunk 1 rendeu
    r.rejections.append(Rejection(chunk_id=2, gate="anchor", reason="x"))  # aniquilado
    # chunks 3 e 4 não aparecem em lugar nenhum: estéreis
    store.record_extraction(r, focus_id=1)

    row = store.conn.execute(
        "SELECT chunks_annihilated, chunks_sterile FROM extraction_runs").fetchone()
    assert row["chunks_annihilated"] == 1, "o chunk que propôs e perdeu não foi contado"
    assert row["chunks_sterile"] == 2, "os chunks que nada propuseram não foram contados"


def test_a_backfilled_run_is_marked_as_second_hand(store):
    """Retro-encaixe de log é medição de SEGUNDA MÃO e não pode entrar numa média junto
    com a de primeira sem dizer.

    MUTAÇÃO: remover o CHECK de `origin`, ou fazer o default ser 'log'.
    """
    store.record_extraction(_result(), focus_id=1, origin="log")
    assert store.conn.execute(
        "SELECT origin FROM extraction_runs").fetchone()["origin"] == "log"

    store.record_extraction(_result(), focus_id=1)
    origens = [x["origin"] for x in store.conn.execute(
        "SELECT origin FROM extraction_runs ORDER BY id")]
    assert origens == ["log", "run"], "a origem deixou de distinguir as duas medições"


async def test_the_handler_persists_the_run_not_only_a_log_line(store, tmp_path):
    """A FIAÇÃO. Reverter `record_extraction` para `log.info` deixa todos os testes acima
    verdes — eles exercitam o Store isolado. Esta é a classe de defeito que este repo
    cometeu seis vezes.

    MUTAÇÃO: apagar a chamada `ctx.store.record_extraction(...)` de `extract_source`.
    """
    from lithium.config import Config
    from lithium.pipeline.extract import ExtractionResult as ER
    from lithium.worker import handlers as H

    class FakeExtractor:
        def __init__(self, *a, **k):
            pass

        async def extract_source(self, source_id):
            r = ER(source_id=source_id, proposed=2, anchored=1, verified=1, chunks_seen=1)
            r.chunks_yielding.add(1)
            r.rejections.append(Rejection(chunk_id=1, gate="anchor", reason="não literal"))
            return r

    class Ctx:
        def __init__(self):
            self.store = store
            self.llm = None
            self.config = Config(data_dir=tmp_path)

    original = H.Extractor
    H.Extractor = FakeExtractor
    try:
        await H.HANDLERS["extract_source"]({"source_id": 1, "focus_id": 1}, Ctx())
    finally:
        H.Extractor = original

    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM extraction_runs").fetchone()["n"] == 1, (
        "o handler não persistiu a corrida: a taxa do portão volta a existir só no log"
    )
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM extraction_rejections").fetchone()["n"] == 1
