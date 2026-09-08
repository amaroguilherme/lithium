"""O que só existe enquanto o sistema OPERA, e portanto não reconstrói.

METRICS.md diz que instrumentar é só para o que não se reconstrói dos timestamps que já
existem. Estas duas são exatamente isso: o veredito de uma rodada e a retirada de uma
lição são EVENTOS. Ligar o daemon sem elas gasta a primeira semana de operação — a única
que ainda não aconteceu — produzindo dado que não dá para analisar depois.
"""

from __future__ import annotations

import json

import pytest

from lithium.db import Store

DIM = 8


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "o.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


# ═══════════════════════ MF5: o juiz contra o piso determinístico


async def test_the_answer_loop_persists_every_round_it_runs(store, profile, monkeypatch):
    """A FIAÇÃO. O teste acima exercita a tabela; este exercita quem escreve nela.

    Reverter `_record_round` deixa o anterior verde — é a classe "fiação-não-testada",
    a sexta ocorrência neste repo.

    MUTAÇÃO: apagar `self._record_round(...)` de `_round`.
    """
    from lithium.pipeline.answer import Answerer

    qid = int(store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status, origin, priority) "
        "VALUES(1,'q','FACTUAL','OPEN','auto',0.5) RETURNING id").fetchone()["id"])

    class FakeRetriever:
        def __init__(self, *a, **k):
            pass

        async def search_claims(self, *a, **k):
            return []

    class FakeLLM:
        async def structured(self, *a, **k):
            raise RuntimeError("o juiz não deve ser chamado sem evidência")

    import lithium.pipeline.answer as A
    monkeypatch.setattr(A, "Retriever", FakeRetriever)

    a = Answerer(store, FakeLLM(), embedder=None, profile=profile)
    monkeypatch.setattr(a, "_judge", lambda *args, **kw: _verdict_none())
    await a.round(qid)

    n = store.conn.execute(
        "SELECT COUNT(*) AS n FROM answer_rounds WHERE question_id = ?", (qid,)
    ).fetchone()["n"]
    assert n == 1, "a rodada rodou e não deixou linha: o veredito voltou a sumir"


async def _verdict_none():
    return None


# ═══════════════════════ MF6: quem retirou a lição


@pytest.mark.parametrize("caminho,esperado", [("cli", "human_cli"), ("chat", "human_chat")])
def test_retiring_a_memory_records_who_did_it(store, caminho, esperado, tmp_path):
    """A ÂNCORA de MF6: quem retira é a PESSOA, e nenhum código do sistema escreve aqui.

    Sem `retired_by`, `active = 0` diz que a lição saiu e não diz por quem — então
    "o modelo reinseriu o que a pessoa negou", a assinatura mais direta de delírio que o
    plano captura, não tem como ser distinguida de uma limpeza automática.

    MUTAÇÃO: voltar `UPDATE memories SET active = 0` sem as duas colunas, em qualquer um
    dos dois escritores.
    """
    mid = int(store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active, focus_id) "
        "VALUES('lição','fact','recon',1,1,1) RETURNING id").fetchone()["id"])

    if caminho == "chat":
        from lithium.chat import ChatEngine

        engine = ChatEngine.__new__(ChatEngine)   # sem LLM nem embedder: `forget` é SQL
        engine.store = store
        assert engine.forget(mid)
    else:
        from typer.testing import CliRunner

        from lithium import cli
        cfg = tmp_path / "c.toml"
        cfg.write_text(f'data_dir = "{tmp_path / "d"}"\n', encoding="utf-8")
        CliRunner().invoke(cli.app, ["init", "-c", str(cfg)])
        # o CLI escreve no banco dele; refaz a lição lá
        from lithium.config import load_config
        s2 = Store(load_config(cfg).db_path, embedding_dim=1024)
        s2.init_schema()
        mid = int(s2.conn.execute(
            "INSERT INTO memories(text, kind, source, confirmed, active, focus_id) "
            "VALUES('lição','fact','recon',1,1,1) RETURNING id").fetchone()["id"])
        s2.close()
        CliRunner().invoke(cli.app, ["memories", "-c", str(cfg), "--forget", str(mid)])
        store = Store(load_config(cfg).db_path, embedding_dim=1024)

    row = store.conn.execute(
        "SELECT active, retired_by, retired_at FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["active"] == 0
    assert row["retired_by"] == esperado, (
        f"a retirada não registrou o caminho humano ({esperado})"
    )
    assert row["retired_at"], "a retirada não tem data: a série de MF6 não se forma"


# ═══════════════════════ MF6: o modelo reescrevendo o que a pessoa negou


def test_a_lesson_that_repeats_a_retired_one_is_counted_as_reinsertion(store):
    """A assinatura mais direta de delírio que este plano captura.

    `retired_by` só é escrito pelos dois caminhos HUMANOS. Uma lição nova cujo `text_key`
    casa uma retirada é o modelo reescrevendo o que a pessoa negou — e sem `retired_at`
    isso seria indistinguível de uma limpeza automática.

    Casa por `text_key`, que é o MESMO normalizador da deduplicação de memória — e ele
    só faz `lower()`. LIMITE DECLARADO, verificado: reinserir com outra pontuação
    ("...livre!" contra "...livre.") produz chave diferente e NÃO é detectado. Reusar o
    normalizador existente é deliberado: uma segunda noção de igualdade de texto no mesmo
    banco divergiria da primeira na primeira edição.

    MUTAÇÃO: em `record_reflect_tick`, tirar `AND retired_at IS NOT NULL` — toda lição
    repetida passa a contar como reinserção, inclusive as que ninguém negou.
    """
    store.conn.execute(
        "INSERT INTO memories(text, text_key, kind, source, confirmed, active, "
        "  focus_id, retired_at, retired_by) "
        "VALUES('MeSH funciona melhor que texto livre.', "
        "       'mesh funciona melhor que texto livre.', 'fact','recon',1,0,1,"
        "       '2026-08-01T00:00:00Z','human_cli')")
    # uma lição ATIVA, nunca retirada: repeti-la não é reinserção
    store.conn.execute(
        "INSERT INTO memories(text, text_key, kind, source, confirmed, active, focus_id) "
        "VALUES('Buscas por via tópica não retornam nada.', "
        "       'buscas por via tópica não retornam nada.','fact','recon',1,1,1)")

    tid = store.record_reflect_tick(1, [
        "MeSH funciona melhor que texto livre.",          # reinsere a retirada
        # Repete uma lição ATIVA, nunca retirada. É o CONTROLE: sem ele, remover o
        # filtro `retired_at IS NOT NULL` não muda o resultado e a mutação sobrevive —
        # foi o que aconteceu na primeira versão, porque a chave semeada aqui não casava
        # a normalização do texto proposto.
        "Buscas por via tópica não retornam nada.",
        "Uma lição inédita.",
    ])
    row = store.conn.execute(
        "SELECT proposed, reinserted FROM reflect_ticks WHERE id = ?", (tid,)).fetchone()
    assert row["proposed"] == 3
    assert row["reinserted"] == 1, (
        "a contagem de reinserção pegou lição que ninguém retirou, ou perdeu a que foi"
    )


async def test_the_reflect_handler_persists_the_tick(store, monkeypatch, tmp_path):
    """A FIAÇÃO do tique. Reverter a chamada no handler deixa o teste acima verde — ele
    exercita o Store isolado.

    MUTAÇÃO: apagar `ctx.store.record_reflect_tick(...)` de `reflect_tick`.
    """
    from lithium.config import Config
    from lithium.worker import handlers as H

    class FakeLesson:
        kind, text = "search_lesson", "uma lição qualquer"

    class FakeReflector:
        def __init__(self, *a, **k):
            pass

        async def reflect(self, max_lessons=3):
            return [FakeLesson()]

        def mark_literature_seen(self):
            pass

    class Ctx:
        def __init__(self):
            self.store = store
            self.llm = None
            self.embedder = None
            self.config = Config(data_dir=tmp_path)

    monkeypatch.setattr(H, "Reflector", FakeReflector)
    await H.HANDLERS["reflect_tick"]({"focus_id": 1}, Ctx())

    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM reflect_ticks").fetchone()["n"] == 1, (
        "o tique de reflexão rodou e não deixou linha: MF6 volta a ser impossível"
    )


# ═══════════════════════ MF5: os CONJUNTOS, não só a conjunção


async def test_a_refusal_records_which_condition_failed(store, profile, monkeypatch):
    """`_is_sufficient` exige QUATRO condições; gravar só o resultado não diz qual reprovou.

    MEDIDO na primeira operação real: 15 rodadas recusadas, `blocked_reason` NULL em todas,
    e a resposta só saiu inspecionando as 10 claims à mão — elas tinham peso até 0,85 e
    NENHUMA mencionava o tópico perguntado, o que aponta `addresses_question_directly`.
    Sem os conjuntos, essa inferência não é reproduzível a partir do banco.

    MUTAÇÃO: voltar a gravar só `judge_sufficient`, com as cinco colunas em NULL.
    """
    from lithium.llm.schemas import SufficiencyVerdict
    from lithium.pipeline.answer import Answerer

    qid = int(store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status, origin, priority) "
        "VALUES(1,'q','FACTUAL','OPEN','auto',0.5) RETURNING id").fetchone()["id"])

    # o caso da operação: evidência FORTE que não responde a pergunta
    veredito = SufficiencyVerdict(
        sufficient=True, n_independent_sources=4,
        addresses_question_directly=False, sources_agree=True,
        missing="nada sobre o tópico perguntado", blocked_reason=None)

    class FakeRetriever:
        def __init__(self, *a, **k):
            pass

        async def search_claims(self, *a, **k):
            return []

    import lithium.pipeline.answer as A
    monkeypatch.setattr(A, "Retriever", FakeRetriever)
    a = Answerer(store, object(), embedder=None, profile=profile)

    async def _judge(*args, **kw):
        return veredito

    monkeypatch.setattr(a, "_judge", _judge)
    await a.round(qid)

    r = store.conn.execute(
        "SELECT judge_sufficient, v_sufficient, v_addresses, v_n_sources, "
        "       v_sources_agree, v_missing FROM answer_rounds").fetchone()
    assert r["judge_sufficient"] == 0, "a conjunção deveria reprovar"
    assert r["v_sufficient"] == 1, (
        "o conjunto `sufficient` não foi gravado — a recusa fica sem causa identificável"
    )
    assert r["v_addresses"] == 0, "o campo que de fato reprovou não foi distinguido"
    assert r["v_n_sources"] == 4, "a contagem DO JUIZ não foi gravada; MF5 precisa dela"
    assert r["v_missing"] == "nada sobre o tópico perguntado"


async def test_an_unavailable_judge_leaves_the_conjuncts_null(store, profile, monkeypatch):
    """Juiz indisponível não é juiz que reprovou. Gravar 0 nos conjuntos confundiria
    "ele disse não" com "ele não disse nada", e a taxa de recusa passaria a incluir falha
    de infraestrutura.

    MUTAÇÃO: gravar `int(False)` em vez de `None` quando `verdict is None`.
    """
    from lithium.pipeline.answer import Answerer

    qid = int(store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status, origin, priority) "
        "VALUES(1,'q2','FACTUAL','OPEN','auto',0.5) RETURNING id").fetchone()["id"])

    class FakeRetriever:
        def __init__(self, *a, **k):
            pass

        async def search_claims(self, *a, **k):
            return []

    import lithium.pipeline.answer as A
    monkeypatch.setattr(A, "Retriever", FakeRetriever)
    a = Answerer(store, object(), embedder=None, profile=profile)

    async def _sem_juiz(*args, **kw):
        return None

    monkeypatch.setattr(a, "_judge", _sem_juiz)
    await a.round(qid)

    r = store.conn.execute(
        "SELECT judge_sufficient, v_sufficient, v_addresses FROM answer_rounds "
        " WHERE question_id = ?", (qid,)).fetchone()
    assert r["judge_sufficient"] == 0
    assert r["v_sufficient"] is None and r["v_addresses"] is None, (
        "juiz ausente foi gravado como juiz que reprovou"
    )
