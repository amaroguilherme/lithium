"""Extração e o portão duplo de verificação.

Os testes aqui protegem a fronteira mais importante do sistema: o que entra na
tabela `claims`. Material não verificado que passe daqui contamina o placar de
hipóteses, os relatórios e — pior — o dataset de treino da LoRA, fazendo o modelo
aprender com a própria alucinação.
"""

from __future__ import annotations

import json

import pytest

from lithium.db import Store
from lithium.llm import LLMError
from lithium.llm.schemas import CitationVerdict, ClaimExtraction
from lithium.pipeline.extract import Extractor, quote_is_anchored
from lithium.types import Direction, Directness, Grade

from conftest import onco_profile

PROFILE = onco_profile()


DIM = 8

CHUNK_TEXT = (
    "In this 8-week trial, 951 outpatients with generalized anxiety disorder were "
    "randomized to quetiapine XR or placebo. Quetiapine produced significantly "
    "greater reduction in HAM-A total score at week 8 than placebo (p<0.001)."
)


class ScriptedLLM:
    """Devolve respostas roteirizadas por tipo de schema pedido."""

    def __init__(self, extractions: list, verdicts: list | None = None) -> None:
        self._extractions = list(extractions)
        self._verdicts = list(verdicts or [])
        self.extraction_calls = 0
        self.verdict_calls = 0

    async def structured(self, messages, schema, **kw):
        if schema is ClaimExtraction:
            self.extraction_calls += 1
            item = self._extractions.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if schema is CitationVerdict:
            self.verdict_calls += 1
            item = self._verdicts.pop(0) if self._verdicts else CitationVerdict(
                supported=True, reason="ok"
            )
            if isinstance(item, Exception):
                raise item
            return item
        raise AssertionError(f"schema inesperado: {schema}")


def _claim(**kw) -> dict:
    return {
        "statement": "Quetiapina reduziu HAM-A mais que placebo.",
        "supporting_quote": "Quetiapine produced significantly greater reduction in HAM-A",
        "population": "adultos com TAG",
        "intervention": "quetiapina XR",
        "comparator": "placebo",
        "outcome": "HAM-A",
        "direction": Direction.POSITIVE,
        "effect": "p<0.001",
        "grade": Grade.RCT,
        # default JULGÁVEL: o caso normal. Os testes da escapatória passam False
        # explicitamente, para que a ausência de aresta seja sempre uma escolha visível.
        "directness_judgeable": True,
        "directness": Directness.PARTIAL,
        "confidence": 0.9,
        **kw,
    }


def _extraction(*claims: dict) -> ClaimExtraction:
    return ClaimExtraction.model_validate({"claims": list(claims)})


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", embedding_dim=DIM)
    s.init_schema()
    source_id = s.upsert_source(
        kind="pubmed", external_id="1", raw={}, title="Estudo",
        journal="J", year=2011, design="rct",
    )
    s.add_chunk(source_id=source_id, ord=0, text=CHUNK_TEXT, section="results")
    yield s
    s.close()


# ────────────────────────────────────────────────── portão 1: ancoragem literal


def test_exact_substring_is_anchored():
    assert quote_is_anchored("randomized to quetiapine XR", CHUNK_TEXT)


def test_whitespace_and_case_differences_are_tolerated():
    """O modelo normaliza quebra de linha ao copiar; punir isso descartaria
    citação legítima. Tolerância termina aí."""
    assert quote_is_anchored("randomized   to\n  QUETIAPINE xr", CHUNK_TEXT)


@pytest.mark.parametrize(
    "quote",
    [
        "randomized to olanzapine XR",                    # trocou o fármaco
        "Quetiapine reduced HAM-A scores",                # parafraseou
        "quetiapine cured generalized anxiety disorder",  # inventou
        "",                                               # vazio
        "   ",
    ],
)
def test_non_literal_quotes_are_rejected(quote):
    assert not quote_is_anchored(quote, CHUNK_TEXT)


def test_anchoring_does_not_accept_reordered_words():
    """Reordenar é parafrasear. Se passasse, o portão 1 não garantiria nada."""
    assert not quote_is_anchored("HAM-A in reduction greater significantly", CHUNK_TEXT)


# ──────────────────────────────────────────────────────── fluxo dos dois portões


async def test_anchored_and_entailed_claim_is_persisted(store):
    llm = ScriptedLLM([_extraction(_claim())])
    result = await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert (result.proposed, result.anchored, result.verified) == (1, 1, 1)
    row = store.conn.execute("SELECT * FROM claims").fetchone()
    assert row["verified"] == 1
    assert row["grade"] == "rct"
    # O directness deixou de ser coluna da claim: é a ARESTA, e ela nomeia o foco. Ler
    # `claims.directness` aqui não seria só desatualizado — seria deixar de exercitar a
    # única coisa que esta fase acrescenta.
    judged = store.conn.execute(
        "SELECT cd.directness, f.slug FROM claim_directness cd "
        "  JOIN focuses f ON f.id = cd.focus_id WHERE cd.claim_id = ?",
        (row["id"],),
    ).fetchone()
    assert judged["directness"] == "partial"
    assert judged["slug"] == "bipolar-tag"
    assert json.loads(row["chunk_ids"]) == [1]


async def test_unanchored_claim_never_reaches_the_database(store):
    llm = ScriptedLLM([_extraction(_claim(supporting_quote="quetiapina cura tudo"))])
    result = await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert result.proposed == 1 and result.anchored == 0 and result.verified == 0
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0
    assert "não literal" in result.rejections[0]


async def test_entailment_gate_runs_only_on_anchored_claims(store):
    """Portão 2 custa uma chamada de LLM; rodá-lo no que já falhou é desperdício."""
    llm = ScriptedLLM([_extraction(_claim(supporting_quote="inventado"), _claim())])
    await Extractor(store, llm, profile=PROFILE).extract_source(1)
    assert llm.verdict_calls == 1, "só a claim ancorada deveria chegar ao portão 2"


async def test_anchored_but_unentailed_claim_is_rejected(store):
    """O erro mais perigoso de um 12B: citar corretamente uma frase sobre TAG puro
    e escrever uma alegação sobre bipolar I."""
    llm = ScriptedLLM(
        [_extraction(_claim(statement="Quetiapina previne mania em bipolar I."))],
        [CitationVerdict(supported=False, reason="a citação não menciona bipolar nem mania")],
    )
    result = await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert result.anchored == 1 and result.verified == 0
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0
    assert "não implicada" in result.rejections[0]


async def test_verifier_failure_rejects_rather_than_admits(store):
    """Na dúvida, não entra. Falhar aberto encheria o banco de não verificado."""
    llm = ScriptedLLM([_extraction(_claim())], [LLMError("servidor caiu")])
    result = await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert result.verified == 0
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0
    assert "verificador indisponível" in result.rejections[0]


async def test_extraction_failure_on_one_chunk_does_not_abort_the_source(store):
    store.add_chunk(source_id=1, ord=1, text=CHUNK_TEXT + " Segunda parte.")
    llm = ScriptedLLM([LLMError("timeout"), _extraction(_claim())])
    result = await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert result.verified == 1, "o segundo chunk precisa ser processado mesmo assim"
    assert len(result.rejections) == 1


async def test_empty_extraction_is_a_valid_outcome(store):
    """Boilerplate não contém alegação; lista vazia é resposta certa, não falha."""
    llm = ScriptedLLM([_extraction()])
    result = await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert result.proposed == 0 and result.verified == 0 and result.rejections == []


async def test_entailment_check_can_be_disabled(store):
    """Útil para medir o portão 1 isoladamente e para rodadas de bulk barato."""
    llm = ScriptedLLM([_extraction(_claim())])
    result = await Extractor(store, llm, entailment_check=False, profile=PROFILE).extract_source(1)

    assert result.verified == 1 and llm.verdict_calls == 0


async def test_unknown_source_raises(store):
    with pytest.raises(ValueError, match="não existe"):
        await Extractor(store, ScriptedLLM([]), profile=PROFILE).extract_source(999)


async def test_unindexed_design_is_passed_to_the_prompt_as_such(store):
    """`design=None` precisa virar "not indexed" no prompt, não string vazia — o
    prompt instrui o modelo a julgar pelo texto nesse caso."""
    store.conn.execute("UPDATE sources SET design = NULL, sample_n = NULL")
    captured: list[str] = []

    class Capturing(ScriptedLLM):
        async def structured(self, messages, schema, **kw):
            captured.append(messages[0]["content"])
            return await super().structured(messages, schema, **kw)

    await Extractor(store, Capturing([_extraction()]),
                    profile=PROFILE).extract_source(1)
    assert "not indexed" in captured[0]
    assert "not reported" in captured[0]


async def test_persisted_claim_keeps_full_pico_fields(store):
    llm = ScriptedLLM([_extraction(_claim())])
    await Extractor(store, llm, profile=PROFILE).extract_source(1)
    row = store.conn.execute("SELECT * FROM claims").fetchone()

    assert row["population"] == "adultos com TAG"
    assert row["intervention"] == "quetiapina XR"
    assert row["comparator"] == "placebo"
    assert row["outcome"] == "HAM-A"
    assert row["direction"] == "positive"
    assert row["effect"] == "p<0.001"
    assert row["confidence"] == 0.9


async def test_empty_optional_fields_become_null_not_empty_string(store):
    """O schema força "" quando o texto não diz; no banco isso tem que virar NULL,
    senão consultas por comparador ausente ficam erradas."""
    llm = ScriptedLLM([_extraction(_claim(comparator="", effect="", population=""))])
    await Extractor(store, llm, profile=PROFILE).extract_source(1)
    row = store.conn.execute("SELECT * FROM claims").fetchone()

    assert row["comparator"] is None and row["effect"] is None and row["population"] is None


async def test_verified_claims_immediately_enter_the_scoreboard_view(store):
    """`claim_weight` só enxerga verified=1 — é o elo entre extração e placar."""
    llm = ScriptedLLM([_extraction(_claim())])
    await Extractor(store, llm, profile=PROFILE).extract_source(1)

    row = store.conn.execute("SELECT * FROM claim_weight").fetchone()
    assert row is not None
    # rct (0.85) × partial (0.60) × conf (0.9)
    assert row["weight"] == pytest.approx(0.85 * 0.60 * 0.9)


# ══════════════════════════════════════════ a aresta nasce na extração, ou nada nasce


async def test_extraction_creates_the_judgment_edge(store):
    """Separa "gravou a coluna" de "ligou a aresta".

    MUTAÇÃO: remover a escrita de `claim_directness` de `_persist` mantendo o resto. A
    claim entra verificada e SOME de `claim_weight` — o modo de falha em que o corpus
    cresce e o sistema não responde nada.
    """
    llm = ScriptedLLM([_extraction(_claim(directness=Directness.PARTIAL.value))])
    result = await Extractor(store, llm, profile=PROFILE).extract_source(1)
    (claim_id,) = result.claim_ids

    edge = store.conn.execute(
        "SELECT cd.directness, f.slug, f.target FROM claim_directness cd "
        "  JOIN focuses f ON f.id = cd.focus_id WHERE cd.claim_id = ?", (claim_id,)
    ).fetchone()
    assert edge is not None, "a claim entrou sem julgamento — ela não vale nada"
    assert (edge["directness"], edge["slug"]) == ("partial", "bipolar-tag")

    row = store.conn.execute(
        "SELECT scale_id FROM claims WHERE id = ?", (claim_id,)).fetchone()
    assert row["scale_id"] == store.active_focus()["scale_id"]

    weight = store.conn.execute(
        "SELECT weight FROM claim_weight WHERE claim_id = ?", (claim_id,)).fetchone()
    assert weight is not None
    assert weight["weight"] == pytest.approx(0.85 * 0.60 * 0.9)
    assert store.counts()["claims_unweighted"] == 0


async def test_extraction_refuses_to_write_without_an_active_focus(store):
    """Fail-closed na LEITURA, fail-LOUD na ESCRITA.

    MUTAÇÃO: remover o `raise NoActiveFocus`. MEDIDO o que acontece sem ele: `_persist`
    grava a claim (autocommit), o INSERT na aresta levanta `NOT NULL constraint failed:
    claim_directness.focus_id`, e a claim fica COMMITADA com `scale_id` NULL e sem aresta
    **para sempre** — restaurar o foco não a recupera, e o retro-preenchimento não a
    alcança porque a coluna legada já não existe. Cada retry deixa mais uma órfã.

    Segunda MUTAÇÃO: tirar o `store.tx()` de `_persist`.
    """
    from lithium.pipeline.extract import NoActiveFocus

    store.conn.execute("UPDATE meta SET value = 'abc' WHERE key = 'active_focus'")
    llm = ScriptedLLM([_extraction(_claim())])
    with pytest.raises(NoActiveFocus):
        await Extractor(store, llm, profile=PROFILE).extract_source(1)
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0, (
        "sobrou uma claim órfã: commitada, sem aresta e sem caminho de reparo"
    )


async def test_re_extracting_a_source_does_not_duplicate_its_claims(store):
    """A extração é idempotente por chunk.

    OBSERVADO no primeiro contato real: a fonte 1 terminou com 6 claims, duas idênticas
    palavra por palavra, porque a mesma fonte foi extraída duas vezes.

    O caminho até isso é curto e comum, não um acidente raro: `recover_orphans` devolve à
    fila toda tarefa que ficou `running` quando o daemon morreu no meio, e uma extração
    real leva ~2 minutos — um Ctrl-C dentro dessa janela é operação normal. Na volta, os
    chunks já processados eram extraídos de novo.

    Duplicata aqui não é cosmética: `hypothesis_scoreboard` SOMA o peso das claims
    ligadas, então a mesma evidência contada duas vezes empurra uma hipótese para cima do
    placar.

    MUTAÇÃO: remover o filtro de `done_chunks` de `extract_source`.
    """
    # UMA extração roteirizada. Se a segunda passada pedir outra, a lista acaba e o
    # `ScriptedLLM` levanta — a forma mais dura de dizer "não reprocesse".
    llm = ScriptedLLM([_extraction(_claim(statement="Quetiapina reduziu HAM-A."))])
    first = await Extractor(store, llm, profile=PROFILE).extract_source(1)
    assert first.verified == 1
    n_after_first = store.conn.execute(
        "SELECT COUNT(*) AS n FROM claims WHERE source_id = 1").fetchone()["n"]
    assert n_after_first == 1

    second = await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert second.proposed == 0, "a fonte foi reprocessada em vez de pulada"
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM claims WHERE source_id = 1"
    ).fetchone()["n"] == n_after_first, (
        "a re-extração duplicou claims; o placar de hipóteses somaria a mesma "
        "evidência duas vezes"
    )


async def test_a_source_with_a_new_chunk_still_extracts_the_new_one(store):
    """A contrapartida. Sem ela o guard poderia ser "nunca reprocessa nada" e o teste
    acima ficaria verde medindo o bug oposto — uma fonte que ganhou chunk novo (refetch
    com abstract mais completo) nunca mais seria extraída."""
    llm = ScriptedLLM([
        _extraction(_claim(statement="Primeira.")),
        _extraction(_claim(statement="Segunda.")),
    ])
    await Extractor(store, llm, profile=PROFILE).extract_source(1)
    store.add_chunk(source_id=1, ord=99, text="Texto novo que ainda não foi extraído.")

    again = await Extractor(store, llm, profile=PROFILE).extract_source(1)
    assert again.proposed == 1, "o chunk novo foi pulado junto com os antigos"


# ═══════════════════ a escapatória de julgabilidade do directness


async def test_an_unjudgeable_claim_gets_no_directness_edge(store):
    """A extração podia dizer "não sei" — e agora pode.

    Era o único caminho do sistema OBRIGADO a chutar. `DirectnessVerdict`, o juiz
    independente do relens, tem três saídas, e o docstring dele explica por quê: gravar um
    nível por falta de informação é PERMANENTE (a PK de `claim_directness` congela o valor
    e o sweep pula quem já tem aresta) e INVISÍVEL (nenhum contador distingue "julgada" de
    "defaultada"). `ExtractedClaim` tinha quatro níveis e nenhuma escapatória.

    MEDIDO no corpus real: chamando o juiz independente sobre 12 claims que a extração já
    havia julgado, ele respondeu NÃO-JULGÁVEL em 3 — onde a extração tinha gravado um
    nível com peso. (Nas 9 restantes: concordância 6, autoavaliação inflou 2, incluindo um
    `direct` -> `partial`, peso -0,40.)

    MUTAÇÃO: em `Extractor._persist`, gravar a aresta incondicionalmente de novo.
    """
    llm = ScriptedLLM([_extraction(_claim(directness_judgeable=False))])
    result = await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert result.verified == 1, "a claim foi descartada; ela deve existir, só sem peso"
    [cid] = [r["id"] for r in store.conn.execute("SELECT id FROM claims")]
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM claim_directness WHERE claim_id = ?", (cid,)
    ).fetchone()["n"] == 0, (
        "a aresta foi gravada mesmo sem a extração saber quem foi estudado"
    )
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM claim_weight WHERE claim_id = ?", (cid,)
    ).fetchone()["n"] == 0, "a claim sem julgamento carregou peso — deveria ser zero"


async def test_a_judgeable_claim_still_gets_its_edge(store):
    """A contrapartida, sem a qual a trava acima ficaria verde com a extração parando de
    julgar QUALQUER coisa — o corpus inteiro sem peso passa nos dois testes se só o
    primeiro existir.

    MUTAÇÃO: inverter a condição em `_persist` (`if not claim.directness_judgeable`).
    """
    llm = ScriptedLLM([_extraction(_claim(directness_judgeable=True,
                                          directness=Directness.PARTIAL))])
    await Extractor(store, llm, profile=PROFILE).extract_source(1)

    [row] = list(store.conn.execute("SELECT directness FROM claim_directness"))
    assert row["directness"] == Directness.PARTIAL.value


async def test_an_unjudged_claim_is_reachable_by_relens(store):
    """A escapatória ENCAMINHA, não descarta.

    Uma claim sem aresta cai em `claims_unjudged`, que é exatamente o conjunto que
    `focus --relens` varre — então ela vai para o juiz independente, que a lê em
    isolamento com o bloco de evidência inteiro. Sem esta propriedade a escapatória seria
    um buraco: claim citável, sem peso, e sem caminho de volta.

    MUTAÇÃO: fazer `_persist` gravar `out_of_scope=1` em vez de omitir a aresta — a claim
    sai de `claims_unjudged` e o relens nunca mais a vê.
    """
    llm = ScriptedLLM([_extraction(_claim(directness_judgeable=False))])
    await Extractor(store, llm, profile=PROFILE).extract_source(1)

    assert store.counts()["claims_unjudged"] == 1, (
        "a claim não julgada ficou fora de `claims_unjudged`: o relens não a alcança"
    )
