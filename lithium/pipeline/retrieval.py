"""Retrieval híbrido: vetorial + BM25, fundidos por RRF, reordenados por evidência.

Por que híbrido e não só vetorial: nomes de fármaco e escalas ("HAM-A", "pregabalina",
"NCT01236411") são exatamente o tipo de token que embedding dilui e BM25 acerta. E
por que não só BM25: "alternativa a antidepressivo" precisa casar com "mood
stabilizer monotherapy", que não compartilha um único termo.

**RRF em vez de soma ponderada de scores.** Distância de cosseno e score BM25 vivem
em escalas incomparáveis e instáveis entre consultas; normalizar exigiria calibração
que envelheceria. RRF usa só a *posição* no ranking, então não há escala para
calibrar.

**O reordenamento é o passo que carrega o domínio.** Relevância textual sozinha
colocaria em primeiro um RCT grande em depressão unipolar. Ponderar por
grade × directness empurra para cima o coorte pequeno em bipolar I com TAG — que é a
evidência que interessa. Ver `types.evidence_weight`.

**Relevância é normalizada min-max antes de encontrar o peso, e o peso entra como
`(1 + w)`, não como fator direto.** Isso corrige um erro que passou despercebido na
primeira versão: scores RRF crus vivem numa faixa de ~2x, enquanto o peso de
evidência varia ~160x (de `opinion × extrapolated` a `meta_analysis × direct`).
Multiplicando os dois crus, o peso domina e a busca deixa de ser busca — pergunte
sobre TCC e volta a claim de maior grade do corpus, sobre o que for. Normalizar
iguala o alcance das duas dimensões: **relevância decide quem está no páreo, peso de
evidência decide a ordem dentro dele.**
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from lithium.db import Store
from lithium.llm import EmbeddingClient
from lithium.types import Directness, Grade

RRF_K = 60
"""Constante padrão do RRF. Amortece a diferença entre a 1ª e a 2ª posição, o que
importa quando os dois rankers discordam bastante — que é o caso normal aqui."""

MIN_COSINE = 0.45
"""Guarda contra lixo degenerado. **Não é filtro de relevância** — não pode ser.

Tentei calibrá-lo como filtro e estava errado. Duas medições, com bge-m3:

Primeira rodada, só com consultas em estilo de pesquisa, sugeriu 0.70 como piso:

    "HAM-A" (nenhum chunk contém)                          → 0.631
    "anticoagulação em fibrilação atrial" (fora do domínio) → 0.704
    "antidepressivo causa virada maníaca em bipolar I"      → 0.845

Segunda rodada, incluindo consultas conversacionais do chat, destruiu essa conclusão:

    "O que o corpus já tem sobre quetiapina?"  (assunto CERTO) → 0.705
    "me conta o que você sabe de quetiapina"   (assunto CERTO) → 0.706
    "anticoagulação em fibrilação atrial"      (assunto ERRADO) → 0.710

Pergunta conversacional sobre o tema correto pontua **abaixo** de uma pergunta de
cardiologia. O enquadramento ("o que você sabe sobre…") domina o embedding e afoga o
conteúdo médico. Não existe limiar que separe os dois casos — as faixas se sobrepõem
por completo, e um piso em 0.70 apagava o corpus inteiro numa conversa normal, fazendo
o assistente afirmar que não tinha dados que tinha.

Então o piso desce para 0.45, onde nunca dispara com bge-m3 (cuja faixa começa em
~0.6) exceto em casos degenerados. A discriminação de relevância fica onde tem
informação para fazê-la: o corte por `k`, o **juiz de suficiência** (que vê a
evidência inteira e pode declarar que ela não responde a pergunta), e o prompt, que
obriga a admitir quando a evidência recuperada não serve.

O problema original — boilerplate curto emergindo quando não há resposta boa — foi
resolvido na origem por `SKIP_SECTIONS` e `MIN_CHUNK_CHARS` no ingest, que é onde
deveria ter sido resolvido desde o começo.
"""


@dataclass(slots=True)
class Hit:
    chunk_id: int
    source_id: int
    text: str
    score: float
    title: str | None = None
    section: str | None = None
    external_id: str | None = None
    semantic: bool = False
    """O ranker VETORIAL endossou este chunk (cosseno acima de `MIN_COSINE`)?

    A fusão RRF descarta de qual ranking cada candidato veio, e essa informação é
    justamente o que separa "no assunto" de "compartilha um token". Guardá-la custa um
    booleano e é o único sinal do sistema que não é derivado de posição na lista final."""


@dataclass(slots=True)
class ClaimHit:
    claim_id: int
    statement: str
    grade: Grade
    directness: Directness
    confidence: float
    source_id: int
    external_id: str
    title: str | None
    year: int | None
    relevance: float
    weight: float
    direction: str = "positive"
    population: str | None = None
    semantic: bool = False
    """Algum chunk que sustenta esta claim passou o portão semântico."""
    """`directness` é o julgamento contra o alvo do FOCO ATIVO, não uma propriedade
    absoluta da claim: a mesma claim vale `direct` num foco e `extrapolated` noutro. O
    INNER JOIN em `claim_directness` é consistente com o de `claim_weight` (que já exclui
    claim sem julgamento), então o campo nunca fica None — o que preserva
    `h.directness.value` em chat.py, answer.py e explore.py sem tocar nos três.

    `direction` é o que torna o assento de contra-evidência possível, e `population` é
    o que o portão de derivação da Fase 2 precisa policiar. Os dois são **transporte**,
    não consumo: a invariante reserva à view `claim_weight` o cálculo do peso, e nenhum
    dos dois entra em expressão de ranking."""

    @property
    def score(self) -> float:
        return self.relevance * (1.0 + self.weight)


_FTS_TOKEN = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\-]*")


def fts_query(text: str) -> str:
    """Converte texto livre em consulta FTS5 segura.

    Aspas, parênteses, `*` e `NEAR` são sintaxe do FTS5 — uma pergunta em linguagem
    natural com pontuação derruba o MATCH com "fts5: syntax error". Tokenizamos e
    citamos cada termo, o que também neutraliza os operadores.
    """
    tokens = _FTS_TOKEN.findall(text)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else '""'


MINMAX_FLOOR = 0.05
"""Piso da normalização. Mapear para [0, 1] cru daria exatamente 0.0 ao pior
candidato, zerando seu score final e o colocando atrás de qualquer coisa
independentemente do peso de evidência — inclusive atrás de nada. Com [0.05, 1] o
último colocado continua sendo o último, mas continua existindo."""


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Espalha os scores em [MINMAX_FLOOR, 1] dentro do conjunto de candidatos.

    Quando todos empatam (textos idênticos, ou candidato único), devolve 1.0 para
    todos — aí quem desempata é inteiramente o peso de evidência, que é o
    comportamento desejado.
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return dict.fromkeys(scores, 1.0)
    span = 1.0 - MINMAX_FLOOR
    return {k: MINMAX_FLOOR + span * (v - lo) / (hi - lo) for k, v in scores.items()}


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, k: int = RRF_K, weights: list[float] | None = None
) -> dict[int, float]:
    """Funde múltiplos rankings de ids. Score maior = melhor."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k + rank + 1)
    return fused


CONTRA_DIRECTIONS = frozenset({"negative", "null", "no_effect"})
"""Direções que constituem contra-evidência: o estudo olhou e não achou, ou achou o
contrário."""

CONTRA_SEATS = 2
"""Quantos assentos o corte final reserva para contra-evidência.

Existe porque a ordenação por `relevance × (1 + weight)` sistematicamente empurra o
achado nulo para baixo: um resultado negativo raramente vem de meta-análise grande, e a
literatura publica menos deles. Sem assento reservado, o bloco de evidência mostra ao
revisor uma versão do corpus onde tudo funciona."""


CONTRA_SEAT_GATE = "semantic"
"""O que qualifica uma claim para o assento reservado: o aval do ranker **semântico**.

Isto substitui uma janela de posição, e a troca resolve um limite que a versão anterior
declarava não conseguir vencer. Vale registrar a sequência inteira porque ela é sobre onde
o sinal estava, não sobre esforço:

**Primeira tentativa: piso de relevância.** Não funciona. `_minmax` espalha a relevância
linearmente pelo conjunto de candidatos, então o último recebe sempre exatamente
`MINMAX_FLOOR`. Medido: a negativa NO ASSUNTO em 14ª e uma negativa de cardiologia dão as
duas `relevance = 0.0500`. **`relevance` é derivada de posição, não de pertinência.**

**Segunda: janela de posição curta.** Segura, e custava recall — a negativa em 14ª ficava
inalcançável, e o critério de aceite do plano ficou sem ser atendido. Qualquer janela larga
o bastante para alcançá-la alcançava também a de cardiologia.

**O que estava disponível todo esse tempo:** `search_chunks` calcula um `vector_ranking`
já filtrado por `MIN_COSINE` e **descarta de qual ranking cada candidato veio** na fusão
RRF. Esse booleano é o único sinal do sistema que não é derivado de posição na lista final.
Medido nesta base:

    negativa no assunto (14ª de 14)      cosseno 0,600   passa MIN_COSINE
    negativa de cardiologia              cosseno 0,000   nunca entra no ranking vetorial

A de cardiologia só chegava ao páreo por sobreposição de token no BM25 — que é exatamente
o que a doutrina de `MIN_COSINE` diz que o BM25 faz. Com o portão semântico o alcance pode
ser a cauda inteira: profundidade deixa de importar, porque o que qualifica é o julgamento
do ranker semântico, não a posição numa lista ordenada por peso de evidência.

**Limite honesto que permanece:** com bge-m3 os cossenos começam em ~0,6 e `MIN_COSINE` é
0,45, então em produção o portão admite mais do que nesta medição com bag-of-words. Ele é
estritamente melhor que a janela de posição — é um julgamento semântico em vez de uma
posição — mas não é um filtro de tópico calibrado, e `MIN_COSINE` foi calibrado errado duas
vezes antes justamente por tentar ser um.
"""


def _seat_counter_evidence(results: list[ClaimHit], k: int) -> list[ClaimHit]:
    """Corta em `k` reservando assentos para contra-evidência — sem promover lixo.

    Duas armadilhas, as duas medidas numa versão anterior deste código:

    **1. Sem piso de relevância, o assento vira porta dos fundos.** O horizonte de
    candidatos é alimentado por um BM25 sem piso, então qualquer negativa que compartilhe
    um token entra. Medido: numa busca sobre quetiapina em bipolar I, o assento foi
    preenchido por *"anticoagulação oral não reduz mortalidade em fibrilação atrial"* —
    relevância 0,05 — **deslocando um RCT no assunto**, e apresentado sob o cabeçalho de
    contra-evidência. Por isso o assento só aceita claim que já estava no corte orgânico
    de `k` posições: ele **reordena**, nunca importa de fora.

    **2. O canal expulsava contra-evidência.** Contando quantas já estavam em
    `results[:k]` e truncando `k - assentos` pela cauda, a vítima do truncamento era
    exatamente a negativa de grade baixa que já havia entrado organicamente. Medido em
    120 corpora aleatórios: **8% das buscas trocavam uma contra-claim por outra, com
    ganho líquido zero.** Por isso as vítimas do deslocamento são escolhidas **só entre
    claims não-contra-direcionais**, de baixo para cima.
    """
    head = results[:k]
    if len(results) <= k:
        return head

    seated = [c for c in head if c.direction in CONTRA_DIRECTIONS]
    if len(seated) >= CONTRA_SEATS:
        return head

    # O ALCANCE É A CAUDA INTEIRA, e o portão é o aval do ranker SEMÂNTICO — não a
    # posição. Ver `CONTRA_SEAT_GATE` para o que isto substitui e por quê.
    wanted = [
        c for c in results[k:]
        if c.direction in CONTRA_DIRECTIONS and c.semantic
    ][: CONTRA_SEATS - len(seated)]
    if not wanted:
        return head

    # Vítimas: as mais fracas que NÃO são contra-evidência. Nunca trocar contra por
    # contra — foi o modo de falha da versão anterior.
    keep = list(head)
    for claim in wanted:
        victims = [c for c in keep if c.direction not in CONTRA_DIRECTIONS]
        if not victims:
            break                      # bloco todo contra-direcional: nada a deslocar
        keep.remove(victims[-1])
        keep.append(claim)
    keep.sort(key=lambda c: c.score, reverse=True)
    return keep


class Retriever:
    def __init__(self, store: Store, embedder: EmbeddingClient) -> None:
        self.store = store
        self.embedder = embedder

    # ─────────────────────────────────────────────────────────────────── chunks

    async def search_chunks(
        self, query: str, *, k: int = 12, candidates: int = 40,
        vector_weight: float = 1.0, text_weight: float = 1.0,
        min_cosine: float = MIN_COSINE,
    ) -> list[Hit]:
        vector_ranking: list[int] = []
        if query.strip():
            [embedding] = await self.embedder.embed([query])
            # O piso corta aqui, antes da fusão: um candidato irrelevante que entre
            # no RRF ainda ganha score e pode subir se o BM25 não trouxer nada.
            vector_ranking = [
                cid
                for cid, distance in self.store.search_vector(embedding, k=candidates)
                if (1.0 - distance) >= min_cosine
            ]

        text_ranking = [cid for cid, _ in self.store.search_text(fts_query(query), k=candidates)]

        fused = reciprocal_rank_fusion(
            [vector_ranking, text_ranking], weights=[vector_weight, text_weight]
        )
        if not fused:
            return []

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return self._hydrate_chunks(top, semantic=set(vector_ranking))

    def _hydrate_chunks(
        self, scored: list[tuple[int, float]], *, semantic: set[int] | None = None
    ) -> list[Hit]:
        ids = [cid for cid, _ in scored]
        holes = ",".join("?" for _ in ids)
        rows = {
            r["id"]: r
            for r in self.store.conn.execute(
                f"SELECT c.id, c.source_id, c.text, c.section, s.title, s.external_id "
                f"FROM chunks c JOIN sources s ON s.id = c.source_id "
                f"WHERE c.id IN ({holes})",
                ids,
            )
        }
        return [
            Hit(
                chunk_id=cid,
                source_id=rows[cid]["source_id"],
                text=rows[cid]["text"],
                score=score,
                title=rows[cid]["title"],
                section=rows[cid]["section"],
                external_id=rows[cid]["external_id"],
                semantic=cid in (semantic or set()),
            )
            for cid, score in scored
            if cid in rows
        ]

    # ──────────────────────────────────────────────────────────────────── claims

    async def search_claims(self, query: str, *, k: int = 15, candidates: int = 60) -> list[ClaimHit]:
        """Busca evidência para responder uma pergunta.

        Recupera chunks e sobe para as claims verificadas ancoradas neles, em vez de
        manter um segundo índice vetorial sobre claims. Uma claim herda a relevância
        do melhor chunk que a sustenta, e o ranking final multiplica isso pelo peso
        de evidência.
        """
        hits = await self.search_chunks(query, k=candidates, candidates=candidates)
        if not hits:
            return []

        relevance_by_chunk = _minmax({h.chunk_id: h.score for h in hits})
        semantic_chunks = {h.chunk_id for h in hits if h.semantic}
        # O peso vem da view `claim_weight`, não de um cálculo local.
        #
        # `evidence_weight()` produziria o mesmo número hoje — verificado nas 324
        # combinações. O motivo de não usá-lo é que duas implementações vivas podem
        # divergir: `_seed_weights` escreve as tabelas de peso a partir dos dicts de
        # `types.py`, e no dia em que uma delas mudar sem a outra, o ranking do chat
        # discorda do placar de hipóteses sem nada reclamar. A invariante não é
        # "chame a função certa" — é **o peso de claim tem uma origem só**.
        rows = self.store.conn.execute(
            "SELECT cl.id, cl.statement, cl.grade, cd.directness, cl.confidence, "
            "       cl.direction, cl.population, "
            "       cl.chunk_ids, cl.source_id, cw.weight, "
            "       s.external_id, s.title, s.year "
            "  FROM claims cl "
            "  JOIN claim_weight cw ON cw.claim_id = cl.id "
            "  JOIN claim_directness cd ON cd.claim_id = cl.id "
            "                          AND cd.focus_id = (SELECT id FROM active_focus) "
            "  JOIN sources s       ON s.id        = cl.source_id "
            " WHERE cl.verified = 1 AND cl.source_id IN "
            f"      ({','.join('?' for _ in {h.source_id for h in hits})})",
            sorted({h.source_id for h in hits}),
        ).fetchall()

        results: list[ClaimHit] = []
        for row in rows:
            chunk_ids = json.loads(row["chunk_ids"])
            # Teste de PERTINÊNCIA, não de valor: "não foi recuperado" e "foi
            # recuperado em último lugar" são coisas diferentes, e comparar o score
            # contra zero confundia as duas.
            relevances = [relevance_by_chunk[c] for c in chunk_ids if c in relevance_by_chunk]
            if not relevances:
                continue
            results.append(
                ClaimHit(
                    claim_id=row["id"],
                    statement=row["statement"],
                    grade=Grade(row["grade"]),
                    directness=Directness(row["directness"]),
                    confidence=row["confidence"],
                    source_id=row["source_id"],
                    external_id=row["external_id"],
                    title=row["title"],
                    year=row["year"],
                    relevance=max(relevances),
                    weight=row["weight"],
                    semantic=any(c in semantic_chunks for c in chunk_ids),
                    direction=row["direction"],
                    population=row["population"],
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return _seat_counter_evidence(results, k)
