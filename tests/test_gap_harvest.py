"""O laço que não fechava: `SEARCH_AGAIN` passa a enfileirar busca.

MEDIDO na primeira operação real (1-8/9): 15 rodadas de julgamento, zero findings, 11
perguntas em impasse PERMANENTE. Os três `return RoundResult(Action.SEARCH_AGAIN, ...)` só
retornavam, e existiam apenas dois produtores de `harvest_query` — as queries fixas do
perfil e as hipóteses do `pursue`. Nenhum vinha de pergunta.

A pergunta #1 pedia dieta cetogênica; o corpus tinha ZERO ocorrências de `ketogenic` em
1042 claims. Não porque a pergunta fosse má — nenhuma das 19 queries fixas menciona dieta,
e ninguém nunca perguntou ao PubMed.
"""

from __future__ import annotations

import pytest

from lithium.config import Config
from lithium.db import Store
from lithium.pipeline.answer import Action, RoundResult
from lithium.worker import handlers as H

DIM = 8


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "g.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


class Fila:
    def __init__(self):
        self.enfileiradas: list[tuple[str, dict, str | None]] = []
        self._chaves: set[str] = set()

    def enqueue(self, kind, payload, *, priority=0.5, dedup_key=None, origin=None):
        if dedup_key and dedup_key in self._chaves:
            return None                      # o dedup real devolve None
        if dedup_key:
            self._chaves.add(dedup_key)
        self.enfileiradas.append((kind, payload, dedup_key))
        return len(self.enfileiradas)


def _ctx(store, fila, tmp_path):
    class Ctx:
        def __init__(self):
            self.store = store
            self.queue = fila
            self.llm = None
            self.embedder = None
            self.config = Config(data_dir=tmp_path)
    return Ctx()


def _pergunta(store, origin="auto", texto="q") -> int:
    return int(store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status, origin, priority) "
        "VALUES(1,?,'FACTUAL','OPEN',?,0.5) RETURNING id", (texto, origin)
    ).fetchone()["id"])


async def _roda(store, fila, tmp_path, qid, resultado, monkeypatch):
    """Executa `answer_question` com um RoundResult fixo."""
    class FakeAnswerer:
        def __init__(self, *a, **k):
            pass

        async def round(self, question_id):
            return resultado

    monkeypatch.setattr(H, "Answerer", FakeAnswerer)
    await H.HANDLERS["answer_question"]({"question_id": qid}, _ctx(store, fila, tmp_path))


async def test_a_named_gap_becomes_a_directed_search(store, tmp_path, monkeypatch):
    """O CONSERTO. Sem isto, a pergunta pede material e ninguém vai buscar.

    MUTAÇÃO: apagar a chamada `_harvest_the_gap(...)` de `answer_question`.
    """
    qid = _pergunta(store)
    fila = Fila()
    await _roda(store, fila, tmp_path, qid,
                RoundResult(Action.SEARCH_AGAIN, reason="x",
                            missing="Trials of the ketogenic diet in bipolar I disorder"),
                monkeypatch)

    assert len(fila.enfileiradas) == 1, "a lacuna não virou busca: o laço continua aberto"
    kind, payload, chave = fila.enfileiradas[0]
    assert kind == "harvest_query"
    assert "ketogenic" in payload["query"]
    assert payload["label"] == f"gap:{qid}"
    assert chave.startswith("gap:")


@pytest.mark.parametrize("razao", ["corpus inalterado", "juiz indisponível"])
async def test_a_search_again_without_a_gap_searches_nothing(store, tmp_path, monkeypatch,
                                                             razao):
    """Dois dos três `SEARCH_AGAIN` NÃO são lacuna de corpus.

    "corpus inalterado" significa que a evidência é a mesma — buscar por isso é ruído.
    "juiz indisponível" é falha de infraestrutura, não lacuna. Por isso `missing` é campo
    próprio em vez de casar `reason` por string.

    MUTAÇÃO: trocar a condição por `result.action is Action.SEARCH_AGAIN` sozinha, ou
    passar `reason` no lugar de `missing`.
    """
    qid = _pergunta(store)
    fila = Fila()
    await _roda(store, fila, tmp_path, qid,
                RoundResult(Action.SEARCH_AGAIN, reason=razao), monkeypatch)
    assert not fila.enfileiradas, f"{razao!r} virou busca — é ruído, não lacuna"


async def test_the_same_gap_does_not_queue_twice(store, tmp_path, monkeypatch):
    """Dedup pela LACUNA, não pela pergunta: duas perguntas com a mesma lacuna
    compartilham uma busca, e a mesma lacuna não vira query nova a cada rodada.

    Sem isto, 11 perguntas x 3 rodadas = até 33 buscas para poucas lacunas distintas.

    MUTAÇÃO: usar `f"gap:{question_id}:..."` na `dedup_key`, ou omiti-la.
    """
    fila = Fila()
    lacuna = "Head-to-head trials of pulsed versus continuous dosing"
    for _ in range(3):
        qid = _pergunta(store)
        await _roda(store, fila, tmp_path, qid,
                    RoundResult(Action.SEARCH_AGAIN, reason="x", missing=lacuna),
                    monkeypatch)
    assert len(fila.enfileiradas) == 1, (
        f"a mesma lacuna virou {len(fila.enfileiradas)} buscas"
    )


async def test_a_human_question_does_not_leak_to_the_search_api(store, tmp_path,
                                                                monkeypatch):
    """A query vai para um TERCEIRO (a API do NCBI), e o `missing` é derivado da pergunta —
    que, se digitada em `lithium ask`, pode carregar contexto clínico do caso.

    É a mesma decisão que `recon.queries_for` tomou, pela mesma razão. Ser inconsistente
    aqui seria pior que ser conservador.

    MUTAÇÃO: remover a checagem de `origin` em `_harvest_the_gap`.
    """
    qid = _pergunta(store, origin="human", texto="meu paciente tomou X e teve Y")
    fila = Fila()
    await _roda(store, fila, tmp_path, qid,
                RoundResult(Action.SEARCH_AGAIN, reason="x",
                            missing="Studies on X in patients who had Y"), monkeypatch)
    assert not fila.enfileiradas, (
        "lacuna de pergunta HUMANA virou query para terceiro"
    )
