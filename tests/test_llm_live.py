"""Teste de integração contra um llama-server real. Fora do CI.

    llama-server -m base_models/gemma-4-12b-it-Q5_K_M.gguf \
        --host 127.0.0.1 --port 8080 -c 8192 --parallel 1 -ngl 99 \
        --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja \
        --reasoning-budget 0

    pytest -m live -s

`--reasoning-budget 0` não é opcional aqui. Gemma-4 é modelo de raciocínio; com o
thinking ligado ele gasta o orçamento inteiro de tokens pensando e devolve `content`
vazio, sem nunca emitir o JSON. O parâmetro equivalente por requisição é ignorado
pelo servidor.

É o critério de pronto do item 2 do plano. Mede duas coisas distintas:

* **Conformidade de schema** — a gramática deveria tornar isto 100%. Menos que
  isso indica que o schema é complexo demais para o conversor GBNF.
* **Ancoragem da citação** — `supporting_quote` é substring literal do texto?
  A gramática não pode garantir isso, e é justamente a defesa contra alucinação.
  Esta taxa é a medida honesta de quanto o modelo pode ser confiado na extração.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from lithium.llm import LLMClient
from lithium.llm.prompts import render
from lithium.llm.schemas import ClaimExtraction

pytestmark = pytest.mark.live

BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "gemma-4-12b-it"

# Passagens no estilo de abstracts reais, cobrindo as cinco frentes de busca do
# plano. Incluem de propósito dois casos difíceis: um sem nenhuma alegação
# extraível (`boilerplate`) e um com achado nulo (`buspirone`) — o modelo precisa
# devolver lista vazia no primeiro e `direction: null` no segundo, em vez de
# fabricar um efeito positivo.
PASSAGES: list[dict[str, object]] = [
    {
        "title": "Quetiapine monotherapy in generalized anxiety disorder",
        "journal": "J Clin Psychiatry", "year": 2011, "design": "rct", "sample_n": 951,
        "text": (
            "In this 8-week, double-blind, placebo-controlled trial, 951 outpatients "
            "meeting DSM-IV criteria for generalized anxiety disorder were randomized to "
            "quetiapine XR 50 mg/day, 150 mg/day, or placebo. Both quetiapine XR doses "
            "produced significantly greater reduction in HAM-A total score at week 8 than "
            "placebo (50 mg: -13.9; 150 mg: -14.3; placebo: -11.4; p<0.001). Patients with "
            "a history of bipolar disorder were excluded from participation."
        ),
    },
    {
        "title": "Antidepressant-associated manic switch in bipolar I disorder",
        "journal": "Am J Psychiatry", "year": 2016, "design": "meta_analysis", "sample_n": 3421,
        "text": (
            "We pooled 18 randomized trials comprising 3421 patients with bipolar I "
            "disorder. Antidepressant monotherapy was associated with a significantly "
            "increased risk of treatment-emergent mania compared with mood stabilizer "
            "co-treatment (OR 2.37, 95% CI 1.62-3.47). Risk was highest with tricyclic "
            "antidepressants and serotonin-norepinephrine reuptake inhibitors."
        ),
    },
    {
        "title": "Anxiety comorbidity and outcome in bipolar I disorder",
        "journal": "Bipolar Disord", "year": 2019, "design": "cohort", "sample_n": 482,
        "text": (
            "Among 482 patients with bipolar I disorder followed prospectively for 24 "
            "months, 147 (30.5%) met criteria for comorbid generalized anxiety disorder at "
            "baseline. Comorbid GAD predicted shorter time to mood episode recurrence "
            "(HR 1.64, 95% CI 1.21-2.22) and lower rates of functional recovery."
        ),
    },
    {
        "title": "Pregabalin for generalized anxiety disorder",
        "journal": "Eur Neuropsychopharmacol", "year": 2014, "design": "systematic_review",
        "sample_n": 2299,
        "text": (
            "Eight placebo-controlled trials of pregabalin in generalized anxiety disorder "
            "were identified, enrolling 2299 adults. Pregabalin 150-600 mg/day was superior "
            "to placebo on HAM-A change (standardized mean difference -0.37). Dizziness and "
            "somnolence were the most frequent adverse events. No included trial enrolled "
            "patients with bipolar disorder."
        ),
    },
    {
        "title": "Buspirone augmentation in mood-disorder patients with residual anxiety",
        "journal": "J Affect Disord", "year": 2008, "design": "rct", "sample_n": 74,
        "text": (
            "Seventy-four patients with a mood disorder and persistent anxiety symptoms "
            "were randomized to buspirone 30 mg/day or placebo for 6 weeks as adjunct to "
            "ongoing treatment. There was no statistically significant difference between "
            "groups in HAM-A change at endpoint (-6.2 vs -5.8, p=0.61). The trial was "
            "underpowered to detect small effects."
        ),
    },
    {
        "title": "Cognitive behavioural therapy for anxiety in bipolar disorder",
        "journal": "Behav Res Ther", "year": 2021, "design": "rct", "sample_n": 89,
        "text": (
            "Eighty-nine euthymic patients with bipolar I or II disorder and a comorbid "
            "anxiety disorder were randomized to 16 sessions of modified CBT or treatment as "
            "usual. The CBT group showed greater reduction in anxiety severity at "
            "post-treatment (between-group d = 0.52) with no increase in manic symptoms."
        ),
    },
    {
        "title": "Lamotrigine and anxiety symptoms: a retrospective chart review",
        "journal": "Clin Neuropharmacol", "year": 2013, "design": "case_series", "sample_n": 22,
        "text": (
            "We reviewed charts of 22 patients with bipolar I disorder started on "
            "lamotrigine who also carried an anxiety-disorder diagnosis. Eleven showed "
            "clinician-rated improvement in anxiety. Two patients discontinued because of "
            "rash. No cases of Stevens-Johnson syndrome occurred, though the sample is far "
            "too small to estimate that risk."
        ),
    },
    {
        "title": "Methods and acknowledgements",
        "journal": "Bipolar Disord", "year": 2020, "design": "cohort", "sample_n": 0,
        "text": (
            "Statistical analyses were performed using R version 4.0.2. The authors thank "
            "the study coordinators at each participating site. This work was supported by "
            "an institutional grant. Correspondence should be addressed to the first author. "
            "The authors declare no competing interests. Supplementary tables are available "
            "online."
        ),
    },
]


@pytest.fixture
async def client():
    """Escopo de função, não de módulo: pytest-asyncio dá um event loop por teste, e
    um httpx.AsyncClient criado num loop não pode ser usado em outro."""
    async with LLMClient(BASE_URL, MODEL, temperature=0.2, timeout_s=900) as c:
        if not await c.healthy():
            pytest.skip(f"nenhum llama-server em {BASE_URL}")
        yield c


async def test_server_is_up(client: LLMClient) -> None:
    assert await client.healthy()


async def test_twenty_consecutive_extractions(client: LLMClient) -> None:
    """Critério de pronto do item 2: 20 extrações seguidas com JSON válido."""
    runs = 20
    schema_ok = 0
    quotes_total = 0
    quotes_anchored = 0
    empty_returned = 0
    failures: list[str] = []
    latencies: list[float] = []

    for i in range(runs):
        passage = PASSAGES[i % len(PASSAGES)]
        prompt = render("extract_claims", **passage)
        started = time.monotonic()
        try:
            result = await client.structured(
                [{"role": "user", "content": prompt}], ClaimExtraction, max_tokens=2048
            )
        except Exception as exc:  # noqa: BLE001 — queremos a taxa, não a primeira falha
            failures.append(f"{i}: {type(exc).__name__}: {exc}")
            continue
        finally:
            latencies.append(time.monotonic() - started)

        schema_ok += 1
        if not result.claims:
            empty_returned += 1
        for claim in result.claims:
            quotes_total += 1
            if claim.supporting_quote in str(passage["text"]):
                quotes_anchored += 1
            else:
                failures.append(f"{i}: citação não literal: {claim.supporting_quote[:90]!r}")

    anchor_rate = quotes_anchored / quotes_total if quotes_total else 0.0
    print(
        f"\n  schema válido      : {schema_ok}/{runs}"
        f"\n  citações ancoradas : {quotes_anchored}/{quotes_total} ({anchor_rate:.0%})"
        f"\n  listas vazias      : {empty_returned}"
        f"\n  latência mediana   : {sorted(latencies)[len(latencies) // 2]:.1f}s"
    )
    for f in failures[:10]:
        print(f"  ! {f}")

    assert schema_ok == runs, f"{runs - schema_ok} extrações sem JSON válido"
    assert quotes_total > 0, "nenhuma claim extraída — prompt ou modelo com problema"
    # A gramática não garante ancoragem; medimos e exigimos um piso. Abaixo disso o
    # filtro determinístico descartaria tanta coisa que a extração fica inviável.
    assert anchor_rate >= 0.80, f"ancoragem de citação em {anchor_rate:.0%}"


async def test_returns_empty_list_on_boilerplate(client: LLMClient) -> None:
    """Passagem sem alegação precisa devolver lista vazia, não invenção."""
    boilerplate = PASSAGES[-1]
    results = await asyncio.gather(
        *(
            client.structured(
                [{"role": "user", "content": render("extract_claims", **boilerplate)}],
                ClaimExtraction,
                max_tokens=1024,
            )
            for _ in range(3)
        )
    )
    n_claims = [len(r.claims) for r in results]
    print(f"\n  claims extraídas de boilerplate: {n_claims}")
    assert sum(n_claims) <= 1, f"inventou alegações em texto sem conteúdo: {n_claims}"
