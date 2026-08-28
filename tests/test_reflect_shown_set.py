"""Fechamento por conjunto-mostrado: a lição só pode citar o que foi renderizado.

Antes desta frente, `verify_claim_ids` segurava por **inanição de informação**, não por
projeto: nenhum bloco do prompt de reflexão imprimia id de claim, então o modelo tinha
de adivinhar inteiros — e o inteiro mais disponível era um id de HIPÓTESE, que vive no
mesmo espaço numérico (`INTEGER PRIMARY KEY` em duas tabelas). Reproduzido: com claims
1,2,3 verificadas, `[3, 9999]` gravava com `[3]`. Crédito parcial, nenhum log.

O fechamento tem três peças, e cada teste aqui foi visto FALHAR contra o defeito que ele
existe para pegar (o comando e a saída vermelha estão na especificação):

1. `recent_activity()` devolve o texto **e** o conjunto exato do que imprimiu;
2. um único espaço de índices locais `[k]`, com nenhum id real ao lado de um `[k]`;
3. `verify_claim_ids(ids, *, shown)` — `shown` obrigatório, sem default — tudo ou nada.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import zlib
from collections.abc import Sequence

import pytest

from lithium.db import Store
from lithium.llm.prompts import PromptTooLarge, budget_guard, estimate_tokens, render
from lithium.llm.schemas import ResearchLesson, ResearchLessons
from lithium.pipeline.reflect import (
    CLAIMS_IN_ACTIVITY,
    FabricatedReference,
    Ref,
    Reflector,
)
from lithium.types import Directness, Grade

DIM = 16

BRACKET = re.compile(r"\[([^\]]*)\]")
"""Tudo que aparece entre colchetes no texto de atividade. O teste de namespace usa o
conteúdo cru: `[spec:3]` e `[hypothesis 41]` só são visíveis assim."""


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class ScriptedLLM:
    def __init__(self, results=()) -> None:
        self._results = list(results)
        self.prompts: list[str] = []
        self.pattern_gate_calls = 0
        self.pattern_verdict = None

    async def structured(self, messages, schema, **kw):
        from lithium.llm.schemas import PatternVerdict

        # O pipeline faz DUAS chamadas ao gravar um `pattern`: a reflexão e o portão de
        # derivação. Um dublê que só modela a primeira faz todo teste de pattern
        # explodir em "schema inesperado" — e a tentação seria afrouxar o portão.
        if schema is PatternVerdict:
            self.pattern_gate_calls += 1
            return self.pattern_verdict or PatternVerdict(
                follows_from_cited_claims_alone=True,
                population_scope_exceeded=False,
                contradicts_a_cited_claim_direction=False,
                restates_a_premise=False,
                reason="segue das claims citadas",
            )
        self.prompts.append(messages[0]["content"])
        item = self._results.pop(0) if self._results else ResearchLessons(lessons=[])
        if isinstance(item, Exception):
            raise item
        return item


def _lessons(*items: dict) -> ResearchLessons:
    return ResearchLessons(lessons=[ResearchLesson.model_validate(i) for i in items])


def _pattern(refs, text="Três hipóteses refutadas falharam por elevar tônus glutamatérgico",
             note="generalizando de um bloco") -> dict:
    return {"text": text, "kind": "pattern", "provenance_note": note, "claim_ids": refs}


# ──────────────────────────────────────────────────────────────────── fixtures


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "shown.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _hypothesis(store, statement: str, *, hid: int | None = None) -> int:
    cur = store.conn.execute(
        "INSERT INTO hypotheses(id, focus_id, statement, tier, survives_critique, "
        "  critique_json) "
        "VALUES(?, 1, ?, 'speculative', 0, ?) RETURNING id",
        (hid, statement, json.dumps({"fatal_flaw": "inferiu agudo de crônico",
                                     "weakest_link": "elo 2 sem evidência humana"})),
    )
    return int(cur.fetchone()["id"])


def _claim(store, statement: str, *, cid: int | None = None, pmid: str | None = None,
           grade=Grade.RCT, directness=Directness.DIRECT, confidence=1.0,
           intervention: str | None = None, verified: int = 1) -> int:
    pmid = pmid or f"9{(cid or 0) + 100000:07d}"
    sid = store.upsert_source(kind="pubmed", external_id=pmid, raw={},
                              journal="J Clin Psychiatry")
    cur = store.conn.execute(
        "INSERT INTO claims(id, source_id, chunk_ids, statement, intervention, "
        "  direction, grade, scale_id, confidence, verified) "
        "VALUES(?, ?, '[]', ?, ?, 'positive', ?, 1, ?, ?) RETURNING id",
        (cid, sid, statement, intervention or statement.split()[0],
         grade.value, confidence, verified),
    )
    claim_id = int(cur.fetchone()["id"])
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
        (claim_id, directness.value),
    )
    return claim_id


def _search(store, query: str, *, label: str = "intersecao_direta") -> None:
    store.conn.execute(
        "INSERT INTO tasks(kind, payload_json, status, dedup_key) "
        "VALUES('harvest_query', ?, 'done', ?)",
        (json.dumps({"label": label, "query": query}), f"dk-{query[:20]}"),
    )


def _reflector(store, llm=None, **kw) -> Reflector:
    return Reflector(store, llm or ScriptedLLM(), FakeEmbedder(), **kw)


# ═══════════════════════════════════════════ 1. mostra claims e devolve o conjunto


def test_activity_prints_claims_the_model_can_cite(store):
    """Sem este bloco `pattern` é impossível de produzir honestamente: os três blocos
    antigos não mencionam claim nenhuma, então citar exige adivinhar inteiro."""
    _claim(store, "Quetiapina reduziu HAM-A versus placebo em bipolar I com TAG")
    activity = _reflector(store).recent_activity()

    claims = {k: r for k, r in activity.shown.items() if r.kind == "claim"}
    assert claims, "nenhuma claim citável foi mostrada"
    assert "Quetiapina reduziu HAM-A" in activity.text
    for k in claims:
        assert f"[{k}]" in activity.text


def test_shown_is_exactly_what_was_printed(store):
    """A igualdade de conjuntos é o contrato. Um `shown` montado por uma segunda query
    (ou herdado de uma renderização anterior) autoriza id que o modelo nunca viu — e é
    isso que uma regra tudo-ou-nada passa a tratar como legítimo."""
    for i in range(4):
        _hypothesis(store, f"Hipótese refutada número {i} sobre sigma-1")
        _claim(store, f"Achado verificado número {i} sobre pregabalina",
               intervention=f"droga-{i}")
    _search(store, "pregabalin AND generalized anxiety")

    activity = _reflector(store).recent_activity()

    printed = {int(b) for b in BRACKET.findall(activity.text) if b.isdigit()}
    assert printed == set(activity.shown), (
        f"impresso {sorted(printed)} != devolvido {sorted(activity.shown)}"
    )
    assert printed, "nada foi impresso com índice local"


def test_shown_claims_are_real_verified_claims(store):
    """Todo `[k]` de tipo claim resolve para uma linha verificada de `claims`."""
    good = _claim(store, "Lamotrigina reduziu ansiedade em bipolar I")
    _claim(store, "Achado não verificado sobre riluzol", verified=0,
           intervention="riluzol")
    activity = _reflector(store).recent_activity()

    resolved = [r.id for r in activity.shown.values() if r.kind == "claim"]
    assert resolved == [good]
    for cid in resolved:
        row = store.conn.execute(
            "SELECT verified FROM claims WHERE id = ?", (cid,)
        ).fetchone()
        assert row is not None and row["verified"] == 1


def test_local_indices_are_one_monotonic_space_across_blocks(store):
    """Um espaço só, 1..N sem buraco: dois contadores por bloco fariam `[3]` designar
    duas coisas ao mesmo tempo."""
    for i in range(3):
        _hypothesis(store, f"Hipótese {i} sobre alfa-2-delta")
        _claim(store, f"Claim {i} sobre gabapentina", intervention=f"gaba-{i}")
        _search(store, f"gabapentin query {i}")

    activity = _reflector(store).recent_activity()
    assert sorted(activity.shown) == list(range(1, len(activity.shown) + 1))
    kinds = [activity.shown[k].kind for k in sorted(activity.shown)]
    assert set(kinds) == {"hypothesis", "search", "claim"}


# ═══════════════════════════════════════════════ 2. um namespace só no prompt


def test_brackets_hold_only_local_indices(store):
    """`[k]` é reservado. `[spec:3]` (rótulo de busca dirigida) e `[hypothesis 41]`
    ocupam a MESMA posição visual e carregam um id real de hipótese na faixa dos
    índices locais — a conflação que resolve com sucesso para o objeto errado."""
    _hypothesis(store, "Hipótese refutada sobre NMDA", hid=3)
    _claim(store, "Claim sobre cetamina")
    _search(store, "ketamine AND bipolar", label="spec:3")

    activity = _reflector(store).recent_activity()

    for content in BRACKET.findall(activity.text):
        assert content.isdigit(), (
            f"conteúdo não numérico entre colchetes: [{content}] — colchete é o "
            f"namespace do índice local e de mais nada"
        )
        assert int(content) in activity.shown, f"[{content}] não está em shown"


def test_no_real_id_appears_on_a_line_that_carries_a_local_index(store):
    """Se o id real e o `[k]` aparecem juntos, o modelo tem dois números plausíveis
    para copiar no mesmo lugar — e o errado resolve."""
    hid = _hypothesis(store, "Hipótese refutada sobre mGluR5", hid=417)
    cid = _claim(store, "Riluzol reduziu ansiedade em modelo animal", cid=938)
    activity = _reflector(store).recent_activity()

    reais = {str(hid), str(cid)}
    for line in activity.text.splitlines():
        if not BRACKET.search(line):
            continue
        numeros = set(re.findall(r"\d+", line))
        leaked = numeros & reais
        assert not leaked, f"id real {leaked} na mesma linha de um [k]: {line!r}"


def test_pmids_live_only_on_lines_without_a_local_index(store):
    """A exceção argumentada. PMID é o único identificador que mantém um
    `source_lesson` auditável um mês depois — e a fonte estéril, por definição sem
    claim verificada, nunca é alvo legítimo de `claim_ids`. Então ela não recebe `[k]`,
    e a regra "id real nunca ao lado de índice local" continua literalmente verdadeira.
    """
    store.upsert_source(kind="pubmed", external_id="31234567", raw={},
                        journal="Bipolar Disord")
    _claim(store, "Cariprazina reduziu HAM-A em bipolar I")
    activity = _reflector(store).recent_activity()

    pmid_lines = [ln for ln in activity.text.splitlines() if "PMID:" in ln]
    assert pmid_lines, "o bloco de fontes estéreis desapareceu"
    for line in pmid_lines:
        assert not BRACKET.search(line), f"PMID ao lado de índice local: {line!r}"


async def test_a_saved_lesson_carries_no_real_id_back_into_the_prompt(store):
    """`[k]` gravado cru viraria ruído (designa outra coisa na rodada seguinte), e
    resolvido para `hypothesis #417` voltaria pelo `lessons_block` para dentro do mesmo
    prompt que tem índices locais — recriando os dois namespaces. O rótulo durável é um
    trecho CITADO, não um id."""
    hid = _hypothesis(store, "Elevar tônus glutamatérgico agudamente reduz ansiedade",
                      hid=417)
    reflector = _reflector(store)
    activity = reflector.recent_activity()
    k = next(i for i, r in activity.shown.items() if r.kind == "hypothesis")

    await reflector.remember(
        f"A classe glutamatérgica falhou em [{k}]", "dead_end",
        f"generalizando de [{k}]", shown=activity.shown,
    )
    row = store.conn.execute("SELECT text, provenance FROM research_lessons").fetchone()
    prov = json.loads(row["provenance"])

    assert f"[{k}]" not in row["text"] and f"[{k}]" not in prov["note"], (
        "índice local sobreviveu na gravação: ele designa outra coisa na próxima "
        "renderização"
    )
    assert str(hid) not in row["text"] and str(hid) not in prov["note"], (
        f"id real {hid} entrou no texto da lição — e `lessons_block` reinjeta isso no "
        f"mesmo prompt que tem índices locais"
    )
    assert prov["refs"] == {str(k): ["hypothesis", hid]}, (
        "a proveniência machine-readable é o que substitui o id no texto; sem ela a "
        "lição fica inauditável"
    )


# ══════════════════════════════════════════════ 3. tudo-ou-nada, `shown` obrigatório


def test_shown_is_keyword_only_and_has_no_default(store):
    """Sem default de propósito: um default tornaria silenciosamente permissivo todo
    call site que não foi atualizado — exatamente o estado anterior."""
    reflector = _reflector(store)
    _claim(store, "Alguma claim verificada")

    with pytest.raises(TypeError):
        reflector.verify_claim_ids([1])                     # shown ausente
    with pytest.raises(TypeError):
        reflector.verify_claim_ids([1], {1: Ref("claim", 1)})  # posicional

    param = inspect.signature(reflector.verify_claim_ids).parameters["shown"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_one_index_outside_the_shown_set_fails_the_whole_lesson(store):
    """Tudo ou nada. Filtrar dá crédito parcial: a lição entra com o subconjunto que
    por acaso existia, e a fabricação não deixa rastro."""
    cid = _claim(store, "Quetiapina reduziu HAM-A em bipolar I com TAG")
    reflector = _reflector(store)
    activity = reflector.recent_activity()
    k = next(i for i, r in activity.shown.items() if r.kind == "claim" and r.id == cid)

    assert reflector.verify_claim_ids([k], shown=activity.shown) == [cid]
    with pytest.raises(FabricatedReference):
        reflector.verify_claim_ids([k, 9999], shown=activity.shown)


def test_a_shown_index_of_the_wrong_type_also_fails(store):
    """O caso pior: o índice EXISTE no conjunto mostrado e resolve — para uma hipótese.
    Sem o tipo no mapa, `shown` autorizaria uma claim que não é claim."""
    _hypothesis(store, "Hipótese refutada sobre sigma-1")
    _claim(store, "Claim sobre buspirona")
    reflector = _reflector(store)
    activity = reflector.recent_activity()

    for kind in ("hypothesis", "search"):
        bad = [i for i, r in activity.shown.items() if r.kind == kind]
        if not bad:
            continue
        with pytest.raises(FabricatedReference, match=kind):
            reflector.verify_claim_ids(bad[:1], shown=activity.shown)


async def test_the_collision_that_used_to_be_written_is_now_rejected(store):
    """A reprodução exata do defeito medido, no novo sistema de coordenadas.

    Antes: claims 1,2,3 verificadas, hipótese id=3, `claim_ids=[3, 9999]` gravava com
    `[3]`. O id que o modelo tinha à mão no prompt era de HIPÓTESE. Agora o inteiro 3
    é um índice local que designa uma hipótese, e a claim de id real 3 continua
    verificada no banco — o banco deixou de ser a autoridade sobre o que é citável.
    """
    _hypothesis(store, "Hipótese refutada cujo id colide com o de uma claim", hid=3)
    for i in (1, 2, 3):
        _claim(store, f"Claim verificada de id real {i} sobre lítio", cid=i,
               intervention=f"agente-{i}")

    llm = ScriptedLLM([_lessons(_pattern([3, 9999]))])
    reflector = _reflector(store, llm)

    assert await reflector.reflect() == []
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM memories"
    ).fetchone()["n"] == 0, "gravou crédito parcial de uma citação fabricada"


async def test_a_plausible_small_integer_is_rejected_too(store):
    """O teste antigo usava 9999/8888 — verificava só onde adivinhar FALHA. Um id
    pequeno e plausível, que existe e está verificado, é o caso que passava."""
    _claim(store, "Claim de id real 1 sobre valproato", cid=1, intervention="valproato")
    _claim(store, "Claim de id real 2 sobre lurasidona", cid=2, intervention="lurasidona")
    for i in range(6):
        _hypothesis(store, f"Hipótese refutada de enchimento {i}")

    reflector = _reflector(store)
    activity = reflector.recent_activity()
    claim_indices = {i for i, r in activity.shown.items() if r.kind == "claim"}
    assert 1 not in claim_indices, "o cenário precisa que [1] NÃO seja claim"

    with pytest.raises(FabricatedReference):
        reflector.verify_claim_ids([1], shown=activity.shown)


async def test_a_process_lesson_with_a_fabricated_index_is_rejected_too(store):
    """A regra é sobre a lição, não sobre a categoria. Um `dead_end` que cita índice
    inventado é sinal de confabulação, e `claim_ids` vazio é o correto ali."""
    _claim(store, "Claim qualquer sobre topiramato")
    llm = ScriptedLLM([_lessons({"text": "queries com 'novel' voltam vazias",
                                 "kind": "search_lesson",
                                 "provenance_note": "x", "claim_ids": [4242]})])
    assert await _reflector(store, llm).reflect() == []
    assert store.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0


async def test_the_fabrication_is_logged(store, caplog):
    """"Nenhum log da fabricação" era metade do defeito: sem registro, a única
    evidência de que o modelo inventou id é a ausência de uma lição que ninguém
    esperava."""
    _claim(store, "Claim sobre carbamazepina")
    llm = ScriptedLLM([_lessons(_pattern([7777]))])
    with caplog.at_level(logging.WARNING, logger="lithium.pipeline.reflect"):
        await _reflector(store, llm).reflect()

    assert any("7777" in r.getMessage() for r in caplog.records), (
        f"a fabricação não foi registrada: {[r.getMessage() for r in caplog.records]}"
    )


async def test_remember_refuses_to_verify_without_the_shown_set(store):
    """Citar sem passar `shown` é erro de programação, não do modelo: não pode virar
    "grava sem verificar" nem "descarta em silêncio"."""
    cid = _claim(store, "Claim sobre oxcarbazepina")
    reflector = _reflector(store)
    with pytest.raises(ValueError):
        await reflector.remember("padrão", "pattern", "nota", [cid])
    # sem citação nenhuma, `shown` é irrelevante e o call site antigo continua válido
    assert await reflector.remember("lição de processo", "search_lesson", "n") is not None


# ═════════════════════════════════════════════════ 4. o teto e a aritmética de token


def _big_corpus(store, *, n_claims=400, n_lessons=0, lesson_chars=200):
    for i in range(40):
        _hypothesis(store, f"Hipótese refutada {i}: " + "adjuvante sobre alvo X " * 6)
    for i in range(60):
        _search(store, f"query {i} " + "AND termo[tiab] " * 8)
    for i in range(n_claims):
        _claim(store, f"Claim {i}: " + "desfecho reduzido versus placebo em populacao " * 4,
               intervention=f"agente-{i % 60}")
    for i in range(30):
        store.upsert_source(kind="pubmed", external_id=f"4000{i:04d}", raw={},
                            journal="Prog Neuropsychopharmacol Biol Psychiatry")


async def _seed_lessons(reflector, n, chars):
    for i in range(n):
        text = (f"Lição {i}: " + "vocabulario preclinico e clinico divergem. " * 20)[:chars]
        await reflector.remember(text, "search_lesson", "t")


async def test_the_claims_block_is_capped_and_the_prompt_still_fits(store):
    """Sem teto, o arranque produz centenas de linhas e `budget_guard` levanta
    `PromptTooLarge` — que NÃO é `LLMError`, portanto não é capturado por `reflect()`
    e a tarefa vai para o dead-letter a cada 24 h."""
    _big_corpus(store)
    reflector = _reflector(store)
    await _seed_lessons(reflector, 40, 200)

    activity = reflector.recent_activity()
    n_claims = sum(1 for r in activity.shown.values() if r.kind == "claim")
    assert n_claims == CLAIMS_IN_ACTIVITY, f"{n_claims} claims num corpus de 400"

    prompt = render("reflect", lessons=reflector.lessons_block(), activity=activity.text)
    budget_guard(prompt, label="reflect", max_tokens=1024, n_ctx=8192)


async def test_reflect_drops_the_claims_block_instead_of_dying(store):
    """O bloco de claims é o amortecedor: é o mais novo, o único cujo tamanho
    controlamos linha a linha, e o único que não é a razão de a reflexão existir.
    Sem o degrau, `PromptTooLarge` sobe de `reflect()` sem handler."""
    _big_corpus(store, n_claims=60)
    llm = ScriptedLLM([_lessons()])
    reflector = _reflector(store, llm)
    await _seed_lessons(reflector, 40, 200)

    # n_ctx apertado o suficiente para o bloco de claims não caber, e largo o
    # suficiente para o resto caber. As duas asserções abaixo tornam o teste
    # auto-verificável: se o prompt encolher, a primeira falha em vez de o teste
    # passar por vacuidade.
    com = render("reflect", lessons=reflector.lessons_block(),
                 activity=reflector.recent_activity().text)
    sem = render("reflect", lessons=reflector.lessons_block(),
                 activity=reflector.recent_activity(claim_limit=0).text)
    n_ctx = estimate_tokens(sem) + 1024 + 256 + 40
    assert estimate_tokens(com) + 1024 > n_ctx - 256, "o cenário não aperta nada"

    reflector.n_ctx = n_ctx
    await reflector.reflect()

    assert llm.prompts, "a reflexão morreu em vez de encolher"
    assert "Verified claims" not in llm.prompts[0]
    assert "Hypotheses refuted by the critique pass" in llm.prompts[0]


async def test_the_shown_set_shrinks_with_the_block_it_describes(store):
    """O degrau reconstrói a atividade, então `shown` reflete o que o prompt encolhido
    realmente mostrou. Um `shown` congelado autorizaria claim que saiu do prompt."""
    _big_corpus(store, n_claims=60)
    reflector = _reflector(store)
    cheia = reflector.recent_activity()
    vazia = reflector.recent_activity(claim_limit=0)

    assert any(r.kind == "claim" for r in cheia.shown.values())
    assert not any(r.kind == "claim" for r in vazia.shown.values())
    printed = {int(b) for b in BRACKET.findall(vazia.text) if b.isdigit()}
    assert printed == set(vazia.shown)


def test_the_claims_block_is_ordered_by_weight_not_by_harvest_order(store):
    """`ORDER BY c.id DESC` seria a escolha natural e está errada: `claims.id` é
    monotônico na inserção, logo é proxy exato de `extracted_at` — que a TRAVA 3
    proíbe por ser anti-correlacionado com directness. A trava varre tokens de tempo e
    não pega `id`."""
    forte = _claim(store, "AAA claim forte inserida primeiro", grade=Grade.META_ANALYSIS,
                   directness=Directness.DIRECT, confidence=1.0, intervention="forte")
    for i in range(CLAIMS_IN_ACTIVITY):
        _claim(store, f"MMM claim média {i}", grade=Grade.COHORT,
               directness=Directness.PARTIAL, confidence=0.5, intervention=f"med-{i}")
    fraca = _claim(store, "ZZZ claim fraca inserida por último", grade=Grade.OPINION,
                   directness=Directness.EXTRAPOLATED, confidence=0.2,
                   intervention="fraca")

    activity = _reflector(store).recent_activity()
    ordem = [r.id for _, r in sorted(activity.shown.items()) if r.kind == "claim"]

    assert ordem[0] == forte, "a claim mais forte não veio primeiro"
    assert fraca not in ordem, "a última colhida entrou apesar de ser a mais fraca"


def test_the_claims_block_does_not_repeat_one_intervention(store):
    """Doze restatements do mesmo achado tornam qualquer padrão derivado deles
    circular — e é o que sai de um corpus real, onde o topo por peso é todo
    `meta_analysis/direct` e o desempate alfabético decide."""
    for i in range(30):
        _claim(store, f"Quetiapina reduziu HAM-A, replicação {i}", grade=Grade.META_ANALYSIS,
               directness=Directness.DIRECT, intervention="quetiapina")
    for i in range(5):
        _claim(store, f"Outro agente {i} reduziu HAM-A", grade=Grade.RCT,
               directness=Directness.DIRECT, intervention=f"outro-{i}")

    activity = _reflector(store).recent_activity()
    cids = [r.id for r in activity.shown.values() if r.kind == "claim"]
    intervencoes = [
        store.conn.execute("SELECT intervention FROM claims WHERE id = ?",
                           (cid,)).fetchone()["intervention"]
        for cid in cids
    ]
    assert len(set(intervencoes)) == len(intervencoes), (
        f"intervenção repetida no bloco: {intervencoes}"
    )


async def test_a_pattern_can_now_be_recorded_honestly(store):
    """O ponto positivo da frente: com claims no prompt, `pattern` deixa de exigir
    adivinhação. Se este teste não existir, um portão que reprova tudo passa como
    sucesso."""
    _claim(store, "Aumentar tônus glutamatérgico agudamente piorou ansiedade",
           intervention="riluzol")
    _claim(store, "Cetamina em bolus elevou escores de ansiedade em bipolar I",
           intervention="cetamina")
    reflector = _reflector(store)
    activity = reflector.recent_activity()
    refs = [i for i, r in activity.shown.items() if r.kind == "claim"]

    llm = ScriptedLLM([_lessons(_pattern(refs, note=f"de [{refs[0]}] e [{refs[1]}]"))])
    reflector.llm = llm
    [lesson] = await reflector.reflect()

    prov = json.loads(store.conn.execute(
        "SELECT provenance FROM memories WHERE id = ?", (lesson.id,)
    ).fetchone()["provenance"])
    assert prov["claim_ids"] == sorted(activity.shown[i].id for i in refs)
    assert "[" not in prov["note"], f"índice local cru na proveniência: {prov['note']}"


# ═══════════════════════════════════ 5. o portão de derivação, por comportamento

_APPROVES = dict(
    follows_from_cited_claims_alone=True,
    population_scope_exceeded=False,
    contradicts_a_cited_claim_direction=False,
    restates_a_premise=False,
    reason="r",
)


def _verdict(**overrides):
    from lithium.llm.schemas import PatternVerdict

    return PatternVerdict(**{**_APPROVES, **overrides})


@pytest.mark.parametrize("flipped", [
    {"follows_from_cited_claims_alone": False},
    {"population_scope_exceeded": True},
    {"contradicts_a_cited_claim_direction": True},
    {"restates_a_premise": True},
])
async def test_each_answer_of_the_gate_can_reject_a_pattern(store, flipped):
    """Comportamental, não sobre a função pura.

    A auditoria construiu o defeito exato: deixar `pattern_is_entailed` perfeita e
    checar **uma** das quatro no call site (`if verdict is None or not
    verdict.follows_from_cited_claims_alone`). A suíte inteira ficou verde, inclusive as
    quatro parametrizações da versão anterior deste teste — três das quatro perguntas
    eram decorativas em produção, e a proveniência gravada contradizia a decisão que
    dizia justificar.
    """
    cid = _claim(store, "Claim sobre lítio em roedores")
    llm = ScriptedLLM()
    llm.pattern_verdict = _verdict(**flipped)
    reflector = _reflector(store, llm)
    shown = {1: Ref("claim", cid, 'the claim "lítio em roedores"')}

    result = await reflector.remember("padrão amplo", "pattern", "n", [1], shown=shown)

    assert result is None, f"a resposta {list(flipped)[0]} não reprovou nada"
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM research_lessons"
    ).fetchone()["n"] == 0
    assert llm.pattern_gate_calls == 1, "o portão nem foi chamado"


async def test_a_pattern_that_passes_the_gate_records_the_verdict(store):
    """Aprovar tem que deixar rastro: sem o veredito na proveniência, "passou pelo
    portão" e "não precisou de portão" ficam indistinguíveis na auditoria."""
    import json

    cid = _claim(store, "Claim sobre lítio")
    reflector = _reflector(store, ScriptedLLM())
    shown = {1: Ref("claim", cid, 'the claim "lítio"')}

    lesson = await reflector.remember("padrão", "pattern", "n", [1], shown=shown)
    assert lesson is not None

    prov = json.loads(store.conn.execute(
        "SELECT provenance FROM memories WHERE id = ?", (lesson.id,)
    ).fetchone()["provenance"])
    assert prov["entailment"]["follows_from_cited_claims_alone"] is True


async def test_the_gate_fails_closed_when_the_verifier_is_unavailable(store):
    """Mesma disciplina do portão 2 da extração: na dúvida, não entra."""
    from lithium.llm import LLMError

    class DeadVerifier(ScriptedLLM):
        async def structured(self, messages, schema, **kw):
            from lithium.llm.schemas import PatternVerdict
            if schema is PatternVerdict:
                raise LLMError("verificador fora do ar")
            return await super().structured(messages, schema, **kw)

    cid = _claim(store, "Claim")
    reflector = _reflector(store, DeadVerifier())
    shown = {1: Ref("claim", cid, "the claim")}

    assert await reflector.remember("padrão", "pattern", "n", [1], shown=shown) is None


async def test_a_process_lesson_does_not_pay_for_the_gate(store):
    """O custo é do canal substantivo. `search_lesson` não afirma nada sobre biologia,
    então gastar uma chamada nele seria pagar +1 por lição sem ganho nenhum."""
    llm = ScriptedLLM()
    reflector = _reflector(store, llm)

    assert await reflector.remember("query volta vazia", "search_lesson", "n") is not None
    assert llm.pattern_gate_calls == 0


async def test_a_rejected_rederivation_does_not_inflate_existing_support(store):
    """Por que o portão vem ANTES do dedup.

    O caminho de colisão chama `_merge_provenance`, que amplia `claim_ids` da lição que
    já existe. Com o portão depois, uma rederivação reprovada elevaria um `pattern`
    gravado de "apoiado em 1 claim" para "apoiado em 3" — sem que nenhuma das duas novas
    tivesse passado por verificação de derivação. É inflação de suporte, e ela sai
    estampada no selo que a trilha de especulação lê.
    """
    import json

    first_id = _claim(store, "Claim A sobre lítio")
    second_id = _claim(store, "Claim B sobre valproato")
    shown = {1: Ref("claim", first_id, "claim A"), 2: Ref("claim", second_id, "claim B")}

    approving = _reflector(store, ScriptedLLM())
    lesson = await approving.remember("o padrão", "pattern", "p1", [1], shown=shown)
    assert lesson is not None

    rejecting_llm = ScriptedLLM()
    rejecting_llm.pattern_verdict = _verdict(population_scope_exceeded=True)
    rejected = await _reflector(store, rejecting_llm).remember(
        "o padrão", "pattern", "p2", [1, 2], shown=shown
    )

    assert rejected is None
    prov = json.loads(store.conn.execute(
        "SELECT provenance FROM memories WHERE id = ?", (lesson.id,)
    ).fetchone()["provenance"])
    assert prov["claim_ids"] == [first_id], (
        "uma rederivação reprovada ampliou o suporte declarado do pattern existente"
    )
