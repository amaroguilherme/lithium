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
