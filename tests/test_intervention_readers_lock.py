"""Travas do campo `intervention`: o teto é da JANELA, o marcador não é intervenção.

Cada trava aqui morreu numa mutação MEDIDA — a suíte inteira (1002 verdes) não notou
nenhuma das cinco mudanças que elas policiam, o que é a definição de fiação sem trava
neste repo.
"""

from __future__ import annotations

import json

import pytest

from lithium.db import Store
from lithium.llm.schemas import ExtractedClaim
from lithium.pipeline.extract import Extractor
from lithium.pipeline.state import MAX_COVERAGE_ROWS, build_state
from lithium.types import Directness, Grade, QuestionKind
from lithium.pipeline.question import score_priority

from conftest import _seed_claim, onco_profile

PROFILE = onco_profile()
DIM = 8


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "iv.db", embedding_dim=DIM)
    s.init_schema()
    s.upsert_source(kind="pubmed", external_id="src", raw={}, title="t", year=2020)
    yield s
    s.close()


def _heavy_filler(store, n: int) -> None:
    """`n` intervenções opacas e PESADAS, que não casam keyword de classe nenhuma.

    Pesadas porque a ordenação é por peso: elas têm de empurrar o item de teste para
    baixo do teto. Opacas porque uma keyword casada aqui mataria o cenário por acidente.
    """
    for i in range(n):
        _seed_claim(store, intervention=f"composto opaco {i:03d}",
                    grade=Grade.META_ANALYSIS, directness=Directness.DIRECT,
                    statement=f"achado {i}")


# ═════════════════════════ 1. o teto não decide o que o sistema acredita existir


def test_a_class_below_the_cap_is_not_announced_as_zero_evidence(store):
    """`untouched` é fato do CORPUS, não das 40 linhas que couberam no prompt.

    MEDIDO no banco real antes do conserto: 11 de 17 classes eram anunciadas ao gerador
    sob "## Intervention CLASSES with zero evidence gathered" e DUAS tinham claim no
    banco — 'combination or augmentation strategy' e 'deprescribing / simplification',
    abaixo do rank 40. As duas viravam 2 das 8 queries de recuperação de lacuna, então a
    agenda de pesquisa era dirigida a redescobrir o que já estava colhido.

    MUTAÇÃO QUE MATA: em `build_state`, trocar
    `seen = " ".join(c.intervention for c in coverage_all)` por `... for c in coverage`.
    """
    _heavy_filler(store, MAX_COVERAGE_ROWS + 5)
    _seed_claim(store, intervention="metronomic chlorambucil maintenance",
                grade=Grade.CASE_REPORT, directness=Directness.EXTRAPOLATED,
                statement="uma claim de peso baixo, atrás do teto")

    state = build_state(store, PROFILE)

    assert len(state.coverage) == MAX_COVERAGE_ROWS, "o teto da JANELA tem de continuar de pé"
    assert not any("metronomic" in c.intervention for c in state.coverage), (
        "o cenário exige que a linha esteja ABAIXO do corte"
    )
    assert "metronomic maintenance" not in state.untouched, (
        "classe com claim no corpus anunciada como 'zero evidence gathered'"
    )
    assert "metronomic maintenance" not in state.render()


def test_evidence_behind_the_cap_does_not_score_as_maximum_novelty(store):
    """`priority` é uma afirmação sobre o CORPUS, e é gravada uma vez e nunca recalculada.

    MEDIDO no banco real: 60 das 63 intervenções atrás do teto pontuavam exatamente
    0,6500 — `novelty=1.0` E `directness_gap=1.0` ao mesmo tempo, acima de 39 das 40
    linhas visíveis (faixa real 0,1023–0,7366). São 72 de 152 claims tratadas como
    evidência inexistente na coluna que decide as cinco vagas humanas.

    MUTAÇÃO QUE MATA: em `by_intervention`, trocar
    `rows = self.coverage_all or self.coverage` por `rows = self.coverage`.
    """
    _heavy_filler(store, MAX_COVERAGE_ROWS + 5)
    _seed_claim(store, intervention="lomustine", grade=Grade.RCT,
                directness=Directness.DIRECT, statement="atrás do teto, mas existe")

    state = build_state(store, PROFILE)
    assert not any(c.intervention == "lomustine" for c in state.coverage)

    hit = state.by_intervention("lomustine")
    assert hit.n_claims == 1, "a claim existe no corpus e o scorer tem de vê-la"
    assert hit.total_weight > 0.0
    blind = score_priority(QuestionKind.FACTUAL, state.by_intervention("nada disso existe"))
    assert score_priority(QuestionKind.FACTUAL, hit) < blind, (
        "evidência colhida pontuando como alvo nunca tocado"
    )


def test_an_exact_name_is_not_shadowed_by_a_heavier_row_containing_it(store):
    """`generate_questions.md` manda «Use the exact name from the coverage table».

    A varredura é por peso DECRESCENTE, então `needle in c.intervention` fazia a linha
    pesada sombrear a exata. MEDIDO no banco real, em 4 dos 40 nomes que o modelo lê e é
    instruído a copiar: `up+tau` resolvia para `unified protocol for emotional disorders
    (up+tau)` (peso 3,40 contra os 1,70 que a tabela mostra, prioridade 0,1023 contra
    0,1667), `anxiety disorder` para `any anxiety disorders`, `pharmacotherapy` para
    `pharmacotherapy alone or pharmacotherapy plus family intervention`, `cbt` para
    `behavioral and cognitive behavioral therapy (cbt)`.

    MUTAÇÃO QUE MATA: fundir os dois laços de `by_intervention` de volta num só,
    `if c.intervention == needle or needle in c.intervention`.
    """
    # A linha PESADA contém o nome curto; a exata é leve. Sem precedência de igualdade,
    # a varredura por peso devolve a pesada.
    for i in range(3):
        _seed_claim(store, intervention="chop protocol with cyclophosphamide",
                    grade=Grade.META_ANALYSIS, directness=Directness.DIRECT,
                    statement=f"pesado {i}")
    _seed_claim(store, intervention="chop", grade=Grade.CASE_REPORT,
                directness=Directness.EXTRAPOLATED, statement="leve")

    state = build_state(store, PROFILE)
    assert "chop" in {c.intervention for c in state.coverage}, "as duas linhas são visíveis"

    hit = state.by_intervention("chop")
    assert hit.intervention == "chop", (
        "o nome exato da tabela resolveu para outra linha — a pesada sombreou a específica"
    )
    assert hit.n_claims == 1


def test_a_name_truncated_by_the_render_still_resolves(store):
    """O substring FICA como segundo passo, e este é o motivo dele.

    `render()` corta a primeira coluna em 38 caracteres sem marcador, e o prompt manda
    copiar da tabela: um nome longo chega no scorer truncado. Sem o fallback, ele viraria
    alvo desconhecido e pontuaria novidade máxima sobre evidência que a tabela mostrou.

    MUTAÇÃO QUE MATA: apagar o segundo laço (`if needle in c.intervention`) de
    `by_intervention`.
    """
    long_name = "chlorambucil metronomic maintenance after induction"
    assert len(long_name) > 38
    _seed_claim(store, intervention=long_name, grade=Grade.RCT,
                directness=Directness.DIRECT, statement="uma claim de nome longo")

    state = build_state(store, PROFILE)
    hit = state.by_intervention(long_name[:38])
    assert hit.n_claims == 1, "nome truncado pelo próprio render não resolveu"


# ═════════════════════════ 2. o cabeçalho declara TODAS as suas exclusões


def test_the_header_declares_the_claims_that_record_no_intervention(store):
    """A quarta exclusão, que era a única sem contador.

    MEDIDO no banco real: o cabeçalho afirmava "Corpus: 50 sources, 213 weighted claims"
    sobre uma tabela cuja coluna `n` soma 80, com `n_unjudged=0` e `n_off_scale=0` — as
    61 claims sem intervenção não eram declaradas em lugar nenhum. É o modo de falha que
    o docstring de `coverage_omitted` proíbe em palavras: "um bloco truncado precisa
    declarar o que esconde, senão o modelo raciocina sobre um quadro parcial acreditando
    que é completo".

    MUTAÇÃO QUE MATA: apagar o bloco `if self.n_no_intervention:` de `render()`.
    """
    _seed_claim(store, intervention="doxorubicin", grade=Grade.RCT,
                directness=Directness.DIRECT, statement="tem intervenção")
    _seed_claim(store, intervention=None, grade=Grade.COHORT,
                directness=Directness.DIRECT, statement="coorte sem intervenção")
    _seed_claim(store, intervention="   ", grade=Grade.COHORT,
                directness=Directness.DIRECT, statement="idem, com espaços")

    state = build_state(store, PROFILE)
    rendered = state.render()

    assert state.n_no_intervention == 2
    assert sum(c.n_claims for c in state.coverage) == 1, "as duas ficam fora da tabela"
    assert "2 weighted claim(s) record no intervention" in rendered, (
        "duas claims com peso saíram da tabela sem que o cabeçalho as declarasse"
    )


def test_a_corpus_where_every_claim_names_an_intervention_declares_nothing(store):
    """Contador que aparece com zero é ruído no prompt.

    MUTAÇÃO QUE MATA: tornar o trecho do cabeçalho incondicional (renderizar
    "0 weighted claim(s) record no intervention").
    """
    _seed_claim(store, intervention="doxorubicin", grade=Grade.RCT,
                directness=Directness.DIRECT, statement="tem intervenção")
    rendered = build_state(store, PROFILE).render()
    assert "record no intervention" not in rendered


def test_every_weighted_claim_is_accounted_for_by_the_render(store):
    """A identidade que fecha o quadro: visíveis + atrás do teto + sem intervenção.

    MEDIDO no banco real depois do conserto: 80 + 72 + 61 = 213, o número que o cabeçalho
    afirma. Antes, 133 das 213 desapareciam — 72 declaradas só como contagem de
    INTERVENÇÕES (não de claims) e 61 declaradas em lugar nenhum.

    A trava exige o MESMO UNIVERSO nas três parcelas. O comentário que morreu com este
    conserto descrevia exatamente o risco: o `COUNT(DISTINCT)` que estimava
    `coverage_omitted` contava sobre `claim_weight` enquanto a tabela contava sobre o JOIN
    com `claim_directness`, e renderizar "(+N não mostradas)" com um N de outro universo é
    inventar o número que o modelo usa para saber o tamanho do que não vê.

    MUTAÇÃO QUE MATA: contar `n_no_intervention` sobre `claims` em vez de sobre
    `claim_weight` (basta apagar o `JOIN claim_weight`) — a soma passa a estourar
    `n_claims` assim que existir uma claim sem peso neste foco.
    """
    _seed_claim(store, intervention="doxorubicin", grade=Grade.RCT,
                directness=Directness.DIRECT, statement="com peso e com intervenção")
    _seed_claim(store, intervention=None, grade=Grade.COHORT,
                directness=Directness.DIRECT, statement="com peso, sem intervenção")
    # SEM aresta de directness: não tem peso neste foco, então não entra em NENHUMA das
    # três parcelas nem em `n_claims`.
    _seed_claim(store, intervention=None, grade=Grade.COHORT, judged=False,
                statement="sem julgamento neste foco")

    state = build_state(store, PROFILE)
    visible = sum(c.n_claims for c in state.coverage)
    behind_cap = sum(c.n_claims for c in state.coverage_all) - visible

    assert state.n_unjudged == 1, "o cenário precisa de uma claim FORA do universo de peso"
    assert visible + behind_cap + state.n_no_intervention == state.n_claims, (
        f"o quadro não fecha: {visible} + {behind_cap} + {state.n_no_intervention} "
        f"!= {state.n_claims} — alguma parcela mede outro universo"
    )


# ═════════════════════════ 3. o portão de escrita, o único determinístico


def _claim(**kw) -> ExtractedClaim:
    base = dict(statement="s", supporting_quote="q", population="p",
                intervention="", comparator="", outcome="o", direction="positive",
                effect="", grade=Grade.COHORT.value,
                directness_judgeable=True,
                directness=Directness.DIRECT.value, confidence=0.9)
    return ExtractedClaim.model_validate({**base, **kw})


@pytest.mark.parametrize("written", ["none", "None", "NONE", "n/a", "null", " none ",
                                     "nenhuma", "-"])
def test_a_marker_never_becomes_a_coverage_row(store, written):
    """`"none" or None` é `"none"` — o `or None` de `_persist` nunca pegou isto.

    MEDIDO no banco real: 3 das 213 claims entraram com a string literal `none`, e a
    linha `none` era a SEGUNDA por peso da tabela de cobertura (w=2,55), acima de
    `pramipexole`, que tem 10 claims. O gerador de perguntas lia `none` como a segunda
    intervenção mais evidenciada do corpus.

    MUTAÇÃO QUE MATA: em `Extractor._persist`, voltar
    `_intervention_or_none(claim.intervention)` para `claim.intervention or None`.
    """
    focus = store.conn.execute("SELECT id, scale_id FROM active_focus").fetchone()
    ext = Extractor.__new__(Extractor)
    ext.store = store
    claim_id = Extractor._persist(ext, 1, 1, _claim(intervention=written), focus)

    row = store.conn.execute("SELECT intervention FROM claims WHERE id = ?",
                             (claim_id,)).fetchone()
    assert row["intervention"] is None, f"{written!r} entrou como nome de intervenção"

    state = build_state(store, PROFILE)
    assert state.coverage == [], "um marcador virou linha da tabela de cobertura"
    assert state.n_no_intervention == 1, "e ele tem de ser DECLARADO, não sumir"


def test_a_real_name_is_written_verbatim(store):
    """O portão normaliza o MARCADOR, nunca o nome.

    O texto bruto é a única medição de que o prompt está defeituoso, e `relens` mostra
    esse texto ao juiz de directness, cujo veredito é gravado de forma durável. Uma
    normalização na escrita é irreversível: grava e o original some.

    MUTAÇÃO QUE MATA: normalizar caixa, espaço ou sufixo em `_intervention_or_none` —
    por exemplo `return text.lower()` ou cortar um sufixo ` treatment`/` therapy`.
    """
    focus = store.conn.execute("SELECT id, scale_id FROM active_focus").fetchone()
    ext = Extractor.__new__(Extractor)
    ext.store = store
    claim_id = Extractor._persist(
        ext, 1, 1, _claim(intervention="  Acupuncture Treatment  "), focus)
    row = store.conn.execute("SELECT intervention FROM claims WHERE id = ?",
                             (claim_id,)).fetchone()
    assert row["intervention"] == "Acupuncture Treatment", (
        "a grafia do extrator foi reescrita na escrita — irreversível"
    )
