"""Trocar de foco de verdade: o que atravessa, o que não pode atravessar, e quando.

O pedido literal desta fase é que o usuário troque o foco quando quiser. Isso transforma
janelas que eram teóricas em rotineiras, e a maior parte deste arquivo trava vazamentos
que só existem porque a troca passou a ser um evento normal.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from conftest import FIXTURE_FOCUSES, onco_profile, prod_profile, seed_claim
from lithium.db import Store
from lithium.pipeline.answer import Answerer
from lithium.pipeline.state import build_state
from lithium.types import Directness, Grade

PROFILE = onco_profile()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "lithium.db", embedding_dim=8)
    s.init_schema()
    s.upsert_source(kind="pubmed", external_id="1", raw={})
    yield s
    s.close()


def _focus_b(store, slug="cardio-fa"):
    scale = store.conn.execute("SELECT scale_id FROM active_focus").fetchone()["scale_id"]
    cur = store.conn.execute(
        "INSERT INTO focuses(slug, target, scale_id) VALUES(?, 'FA + DRC', ?) "
        "RETURNING id", (slug, scale))
    return int(cur.fetchone()["id"])


def _activate(store, focus_id):
    store.conn.execute("UPDATE meta SET value = ? WHERE key = 'active_focus'",
                       (str(focus_id),))


class NullEmbedder:
    async def embed(self, texts):  # pragma: no cover - não usado nestes testes
        return [[0.0] * 8 for _ in texts]


# ═════════════════════════════════ 1. o estado de conhecimento é POR FOCO


def test_the_state_does_not_show_the_old_focus_questions_or_hypotheses(store):
    """Perguntas e hipóteses do foco A não podem aparecer no prompt do foco B.

    MEDIDO E EXECUTADO no estado anterior: `build_state(store).render()` no foco novo
    listava as perguntas de psiquiatria sob "Questions already on record — do NOT repeat
    these" e a hipótese sob "Hypotheses on the board", com o sufixo "N linked claim(s)
    unjudged in this focus" — ou seja, o foco de cardiologia era instruído a não repetir
    perguntas de outro domínio e convidado a relensar claims de uma hipótese que não é
    dele. Com `max_asked=40` são até 40 linhas alheias no prompt que DECIDE a agenda.

    MUTAÇÃO: remover `WHERE focus_id = (SELECT id FROM active_focus)` de `asked`, ou
    `h.focus_id` da projeção de `hypothesis_scoreboard` e o filtro do leitor.
    """
    store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status) "
        "VALUES(1, 'Quetiapina tem RCT em TAG?', 'FACTUAL', 'OPEN')")
    store.conn.execute(
        "INSERT INTO hypotheses(focus_id, statement, tier, status) "
        "VALUES(1, 'sigma-1 agonism decouples anxiolysis', 'evidence', 'active')")

    b = _focus_b(store)
    _activate(store, b)
    rendered = build_state(store, PROFILE).render()

    assert "Quetiapina tem RCT em TAG?" not in rendered
    assert "sigma-1 agonism" not in rendered
    assert "(none yet)" in rendered

    # e o controle: no foco de origem elas continuam aparecendo
    _activate(store, 1)
    origem = build_state(store, PROFILE).render()
    assert "Quetiapina tem RCT em TAG?" in origem
    assert "sigma-1 agonism" in origem


def test_the_state_says_off_scale_instead_of_calling_it_unjudged(store):
    """O prompt distingue "não julguei" de "julguei sob outra régua".

    MEDIDO no contador único: o primeiro parágrafo de `build_state(...).render()` dizia
    "N verified claim(s) carry NO judgment for this focus" sobre uma claim que TEM
    aresta. O relens não pode mudar esse número, então o gerador de perguntas raciocina
    para sempre sobre uma ausência que não é a ausência descrita.

    MUTAÇÃO: ligar `n_unjudged` de volta a `claims_unweighted`.
    """
    outra = int(store.conn.execute(
        "INSERT INTO evidence_scales(slug) VALUES('outra') RETURNING id"
    ).fetchone()["id"])
    b = _focus_b(store)
    store.conn.execute("UPDATE focuses SET scale_id = ? WHERE id = ?", (outra, b))
    seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT)
    _activate(store, b)

    rendered = build_state(store, PROFILE).render()
    assert "DIFFERENT evidence scale" in rendered
    assert "carry NO judgment for this focus" not in rendered


# ═════════════════════════════ 2. a fila de perguntas é recurso DO FOCO


def test_the_new_focus_can_accumulate_a_research_queue_of_its_own(store):
    """`park_overflow` não pode fechar a pergunta do foco B por causa do teto do A.

    MEDIDO E EXECUTADO: com 15 perguntas OPEN no foco antigo e o teto em 15, as CINCO
    perguntas recém-geradas do foco novo viravam CLOSED, e `dispatchable` devolvia só
    as do foco antigo — o daemon gastava rodadas de pesquisa (minutos de GPU cada) em
    perguntas de um domínio contra um `claim_weight` agora vazio. O foco novo ficava
    estruturalmente incapaz de acumular fila.

    MUTAÇÃO: remover o filtro de `focus_id` de `park_overflow` e de `dispatchable`.
    """
    from lithium.pipeline.answer import AUTO_QUEUE_CAP

    for i in range(AUTO_QUEUE_CAP + 2):
        store.conn.execute(
            "INSERT INTO questions(focus_id, text, kind, status, priority) "
            "VALUES(1, ?, 'FACTUAL', 'OPEN', 0.8)", (f"velha {i}",))
    b = _focus_b(store)
    store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status, priority) "
        "VALUES(?, 'nova do foco B', 'FACTUAL', 'OPEN', 0.65)", (b,))

    _activate(store, b)
    answerer = Answerer(store, None, NullEmbedder(), profile=PROFILE)
    answerer.park_overflow()

    status = store.conn.execute(
        "SELECT status FROM questions WHERE text = 'nova do foco B'").fetchone()["status"]
    assert status == "OPEN", (
        "a pergunta do foco novo foi fechada pelo teto ocupado pelo foco velho"
    )
    assert answerer.dispatchable(4), "o foco novo não conseguiu despachar nada"
    despachadas = {
        int(r["focus_id"]) for r in store.conn.execute(
            f"SELECT focus_id FROM questions WHERE id IN "
            f"({','.join(str(i) for i in answerer.dispatchable(4))})")
    }
    assert despachadas == {b}, "despachou pergunta de outro foco"


def test_the_human_queue_limit_is_per_focus_on_both_halves(store):
    """As duas metades da mesma trava concordam.

    `QuestionEngine.escalated_count` já era escopado e `answer._escalate` não: uma via 0
    vagas livres e a outra via 5, sobre a mesma fila. Com 5 escaladas no foco A, o foco
    B nunca mais escalava nada.

    MUTAÇÃO: remover o filtro de foco do COUNT em `Answerer._escalate`.
    """
    from lithium.pipeline.answer import HUMAN_QUEUE_LIMIT

    for i in range(HUMAN_QUEUE_LIMIT):
        store.conn.execute(
            "INSERT INTO questions(focus_id, text, kind, status) "
            "VALUES(1, ?, 'FACTUAL', 'ESCALATED')", (f"escalada {i}",))
    b = _focus_b(store)
    cur = store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status) "
        "VALUES(?, 'do foco B', 'FACTUAL', 'OPEN') RETURNING id", (b,))
    qid = int(cur.fetchone()["id"])
    _activate(store, b)

    from lithium.types import StuckReason

    Answerer(store, None, NullEmbedder(), profile=PROFILE)._escalate(
        qid, StuckReason.INSUFFICIENT_EVIDENCE, "falta head-to-head")
    novo = store.conn.execute(
        "SELECT status FROM questions WHERE id = ?", (qid,)).fetchone()["status"]
    assert novo == "ESCALATED", (
        "a fila humana do foco A represou a escalação do foco B"
    )


# ══════════════════════ 3. o foco é resolvido ANTES da chamada de LLM


def test_questions_and_hypotheses_resolve_the_focus_before_the_llm_call(store):
    """Um LLM falso troca `meta['active_focus']` DURANTE a chamada.

    A pergunta tem de ser arquivada sob o foco que estava ativo no INÍCIO. As janelas
    estão medidas no PLAN: 43,8 s em `generate_questions` e 117,9 s em
    `generate_speculation`. Com o subselect dentro do INSERT, o item some de onde foi
    gerado e aparece no outro — e nenhum dado gravado permite reconstruir que a causa
    foi timing.

    MUTAÇÃO: voltar a `VALUES((SELECT id FROM active_focus), ...)`.
    """
    from lithium.llm.schemas import GeneratedQuestion, QuestionBatch
    from lithium.pipeline.question import QuestionEngine

    b = _focus_b(store)

    class TrocaDeFoco:
        async def structured(self, messages, schema, **kw):
            _activate(store, b)          # o usuário trocou no meio da chamada
            return QuestionBatch(questions=[GeneratedQuestion(
                text="Pregabalina tem RCT?", kind="FACTUAL",
                targets="pregabalin", rationale="lacuna")])

    engine = QuestionEngine(store, TrocaDeFoco(), None, profile=PROFILE)
    asyncio.run(engine.generate(max_questions=1))

    dono = store.conn.execute(
        "SELECT focus_id FROM questions WHERE text = 'Pregabalina tem RCT?'"
    ).fetchone()
    assert dono is not None and int(dono["focus_id"]) == 1, (
        "a pergunta foi arquivada sob o foco que só passou a existir DEPOIS do POST"
    )


def test_a_speculation_is_filed_under_the_focus_that_generated_it(store):
    """O mesmo, para a trilha exploratória — a janela ali é de ~118 s."""
    from lithium.llm.schemas import MechanismStep, Speculation, SpeculationBatch
    from lithium.pipeline.explore import Explorer

    b = _focus_b(store)
    item = Speculation(
        statement="metronomic dosing flattens the nadir", intervention_class="x",
        mechanism_target="y", route="metronomic oral", combination="",
        chain=[MechanismStep(claim="a", supported=False, evidence=""),
               MechanismStep(claim="b", supported=False, evidence="")],
        falsifier="se o nadir não mudar", test_proposal="t", known_risks="",
        novelty=0.9)

    class TrocaDeFoco:
        def __init__(self):
            self.n = 0

        async def structured(self, messages, schema, **kw):
            self.n += 1
            if self.n == 1:
                _activate(store, b)
                return SpeculationBatch(speculations=[item])
            from lithium.llm.schemas import SpeculationCritique
            return SpeculationCritique(
                weakest_link="w", fatal_flaw="", contradicted_by="",
                missing_risk="", survives=True)

    asyncio.run(Explorer(store, TrocaDeFoco(), profile=PROFILE).generate(max_items=1))
    dono = store.conn.execute(
        "SELECT focus_id FROM hypotheses WHERE statement = ?", (item.statement,)
    ).fetchone()
    assert dono is not None and int(dono["focus_id"]) == 1


# ══════════════════════════════ 4. a troca inteira, de ponta a ponta


def test_the_system_operates_under_a_focus_from_another_domain(store, tmp_path):
    """O ENTREGÁVEL da fase: criar um foco de outro domínio, ativar, e operar sob ele.

    Toca as quatro tabelas que eram constantes de módulo (estratégias, taxonomia,
    classes/keywords, safety), o prefixo de recuperação, e os prompts. Se qualquer uma
    delas continuasse sendo valor de módulo, alguma destas asserções acharia vocabulário
    de bipolar.
    """
    from lithium.pipeline.explore import _lessons_queries
    from lithium.pipeline.mechanism import route_block, taxonomy_block
    from lithium.pipeline.strategy import all_search_specs
    from lithium.safety import ABSENT, ruleset_from_profile, screen
    from lithium.safety.screen import NO_RULESET
    from lithium.safety.screen import render as render_alerts
    from lithium.safety.screen import Segment

    b = _focus_b(store, slug="onco-vet")
    _activate(store, b)
    profile = PROFILE

    # 1. estratégias
    specs = all_search_specs(profile)
    assert specs and all("bipolar" not in q for _, q in specs)
    assert any("canine lymphoma" in q for _, q in specs)

    # 2. taxonomia e vias
    bloco = taxonomy_block(profile) + route_block(profile)
    assert "oncolytic virotherapy" in bloco and "metronomic oral" in bloco
    assert "quetiapine" not in bloco and "sigma-1" not in bloco

    # 3. cobertura por classe (rótulo + keywords são UMA tabela)
    state = build_state(store, profile)
    assert "anthracycline" in state.untouched
    assert not any("antipsychotic" in c for c in state.untouched)

    # 4. prefixo de recuperação das lições
    assert all(q.startswith("canine multicentric lymphoma")
               for q in _lessons_queries(state, profile))

    # 5. segurança: o perfil declara `safety = false`, e o bloco DIZ isso
    ruleset = ruleset_from_profile(profile)
    assert ruleset is ABSENT or not ruleset.declared
    bloco_seg = render_alerts(screen([Segment("user", "tomo carbonato de lítio")],
                                     ruleset), ruleset_declared=ruleset.declared)
    assert NO_RULESET in bloco_seg
    assert "Lítio" not in bloco_seg, "o ruleset do foco de produção sobreviveu à troca"

    # 6. os prompts falam do alvo novo, e não do antigo
    from lithium.llm.prompts import render

    extract = render("extract_claims", **profile.prompt_blocks("extract_claims"),
                     title="t", journal="j", year=2020, design="rct", sample_n=84,
                     text="…")
    assert "canine multicentric lymphoma" in extract
    assert "bipolar" not in extract and "GAD" not in extract
    assert "tumour lysis" in extract, "o risco permanente do foco novo tem de aparecer"


def test_a_new_scale_refuses_while_another_live_focus_uses_the_current_one(store, tmp_path):
    """A RECUSA de escala nova — o item que substitui o `--regrade`.

    MEDIDO E EXECUTADO: `UPDATE claims SET scale_id = 2` leva o peso da MESMA claim de
    0,68 no foco 1 para NADA, `claim_weight` volta vazia ao reativar o foco antigo, sem
    erro e sem log. E é IRREVERSÍVEL: reconstruir exige re-derivar um `grade` que veio
    de julgamento de LLM e não é reproduzível bit a bit.

    MUTAÇÃO: remover a recusa e deixar a criação seguir com `UPDATE claims SET scale_id`.
    """
    from typer.testing import CliRunner

    from lithium import cli

    seed_claim(store, grade=Grade.RCT, directness=Directness.PARTIAL)
    peso_antes = store.conn.execute(
        "SELECT weight FROM claim_weight").fetchone()["weight"]
    assert peso_antes > 0

    focuses = tmp_path / "focuses"
    shutil.copytree(FIXTURE_FOCUSES, focuses)          # onco-vet pede `vet-onco-evidence`
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'data_dir = "{store.db_path.parent}"\n'
                   f'focuses_dir = "{focuses}"\n', encoding="utf-8")
    # O `store` da fixture já é `lithium.db`, que é o que `cfg.db_path` resolve.

    result = CliRunner().invoke(
        cli.app, ["focus", "--new", "onco-vet", "--config", str(cfg)])
    assert result.exit_code == 1, result.output
    assert "RECUSADO" in result.output
    assert "bipolar-tag" in result.output

    # e nada foi tocado
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM evidence_scales").fetchone()["n"] == 1
    assert store.conn.execute(
        "SELECT weight FROM claim_weight").fetchone()["weight"] == peso_antes


def test_a_lesson_says_when_its_qualifier_is_unavailable_in_this_focus(store):
    """A lição atravessa o foco; o QUALIFICADOR que a mantém honesta, não.

    Num foco que ainda não relensou não há linha em `claim_directness`, e o sufixo
    ficava VAZIO — a lição era renderizada com autoridade total logo abaixo de um
    cabeçalho que explica o que o sufixo significa. Ausência de sufixo lê como "não
    precisa de ressalva": fail-OPEN, exatamente onde a Fase A escolheu fail-closed.

    MUTAÇÃO: voltar `suffix = f" · {worst.value}" if worst else ""`.
    """
    from lithium.pipeline.reflect import Lesson, Reflector

    claim_id = seed_claim(store, grade=Grade.PRECLINICAL,
                          directness=Directness.EXTRAPOLATED)
    import json as _json

    lesson = Lesson(
        id=1, text="raising glutamatergic tone acutely fails", kind="pattern",
        provenance=_json.dumps({"claim_ids": [claim_id]}))
    reflector = Reflector(store, None, NullEmbedder(), profile=PROFILE)

    no_foco = reflector._pattern_line(lesson)
    assert "· extrapolated" in no_foco

    _activate(store, _focus_b(store))
    outro_foco = reflector._pattern_line(lesson)
    assert outro_foco != no_foco
    assert "não julgado neste foco" in outro_foco


def _demo(tmp_path, store):
    """Config apontando para uma cópia dos perfis, com o perfil de teste na MESMA
    escala — escala nova é recusada enquanto houver outro foco vivo nela."""
    focuses = tmp_path / "focuses"
    focuses.mkdir(exist_ok=True)
    shutil.copytree(FIXTURE_FOCUSES / "onco-vet", focuses / "onco-vet")
    f = focuses / "onco-vet" / "focus.toml"
    f.write_text(f.read_text(encoding="utf-8")
                 .replace('scale  = "vet-onco-evidence"', 'scale  = "clinical-evidence"'),
                 encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'data_dir = "{store.db_path.parent}"\n'
                   f'focuses_dir = "{focuses}"\n', encoding="utf-8")
    return cfg


def test_focus_new_then_use_reports_the_cost_before_switching(store, tmp_path):
    """O caminho REAL do CLI, de ponta a ponta.

    Existe porque um `IndexError` real escapou de tudo: `_resolve` projetava só
    `id, slug, retired_at`, e `_report_switch_cost` precisa de `scale_id`. Nenhum teste
    tocava `focus --use`, então o comando que É o entregável da fase quebrava com
    traceback na primeira execução de verdade.

    MUTAÇÃO: voltar `_resolve` para `SELECT id, slug, retired_at`.
    """
    from typer.testing import CliRunner

    from lithium import cli

    seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT,
               statement="quetiapina reduziu ansiedade")
    cfg = _demo(tmp_path, store)
    runner = CliRunner()

    criado = runner.invoke(cli.app, ["focus", "--new", "onco-vet", "--config", str(cfg)])
    assert criado.exit_code == 0, criado.output
    assert "reusada como está" in criado.output, (
        "escala existente tem de ser REUSADA, nunca reescrita"
    )

    trocado = runner.invoke(cli.app, ["focus", "--use", "onco-vet", "--config", str(cfg)])
    assert trocado.exit_code == 0, trocado.output
    assert "custo da troca" in trocado.output
    assert "1 claim(s) sem julgamento" in trocado.output
    assert "reinicie o daemon" in trocado.output
    assert "cache" in trocado.output, (
        "o aviso tem de citar a razão CERTA; escrever a errada faz alguém medir que o "
        "prompt acompanha, concluir que o aviso é supersticioso, e removê-lo"
    )

    # o placar zera no foco novo, e VOLTA intacto no antigo
    assert store.counts()["claims_unjudged"] == 1
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"] == 0
    runner.invoke(cli.app, ["focus", "--use", "bipolar-tag", "--config", str(cfg)])
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"] == 1


def test_focus_show_reports_which_half_of_the_hash_drifted(store, tmp_path):
    """As duas metades são recomputadas SEPARADAMENTE, porque pedem remédios OPOSTOS.

    Calibração divergiu -> escala NOVA + foco NOVO (peso é julgamento congelado).
    `target`/`directness_definitions` divergiram -> RELENS. Um hash composto forçaria a
    mensagem a dar o conselho errado em dois dos três casos — e faria TODO banco da
    Fase A gritar "a calibração divergiu" no primeiro comando de CLI, prescrevendo o
    remédio mais destrutivo para um evento em que nada divergiu.

    MUTAÇÃO: comparar um hash composto único e imprimir a mensagem de calibração.
    """
    from typer.testing import CliRunner

    from lithium import cli

    cfg = _demo(tmp_path, store)
    runner = CliRunner()
    runner.invoke(cli.app, ["focus", "--new", "onco-vet", "--config", str(cfg)])
    runner.invoke(cli.app, ["focus", "--use", "onco-vet", "--config", str(cfg)])

    limpo = runner.invoke(cli.app, ["focus", "--show", "--config", str(cfg)])
    assert "divergiu" not in limpo.output, limpo.output

    f = tmp_path / "focuses" / "onco-vet" / "focus.toml"
    f.write_text(f.read_text(encoding="utf-8").replace(
        'indirect     = { prose = "other canine neoplasia, or feline lymphoma" }',
        'indirect     = { prose = "outra definição qualquer" }'), encoding="utf-8")

    sujo = runner.invoke(cli.app, ["focus", "--show", "--config", str(cfg)])
    assert "o JULGAMENTO divergiu" in sujo.output
    assert "relens" in sujo.output
    assert "CALIBRAÇÃO divergiu" not in sujo.output, (
        "a calibração NÃO mudou; prescrever escala nova jogaria fora o peso do corpus"
    )
