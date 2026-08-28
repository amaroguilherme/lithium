"""Fase A: o julgamento de directness nomeia o foco, e o fail-closed é observável."""
from __future__ import annotations

import logging
import sqlite3

import pytest

from conftest import seed_claim
from lithium.db import Store
from lithium.db.store import FOCUS_TARGET
from lithium.types import Directness, Grade

from conftest import onco_profile

PROFILE = onco_profile()



@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "f.db", embedding_dim=4)
    s.init_schema()
    s.conn.execute("INSERT INTO sources(id,kind,external_id,title,raw_json) "
                   "VALUES(1,'pubmed','1','t','{}')")
    yield s
    s.close()


def _focus2(store, slug="outro", retired=False):
    sid = int(store.conn.execute("SELECT id FROM evidence_scales").fetchone()["id"])
    store.conn.execute(
        "INSERT INTO focuses(id, slug, target, scale_id, retired_at) "
        "VALUES(2, ?, 'outro alvo', ?, ?)",
        (slug, sid, "2020-01-01T00:00:00Z" if retired else None))
    return 2


# ══════════════════════════════════════════════ o seed nunca reescreve histórico

def test_the_seed_never_overwrites_a_stored_weight(tmp_path):
    """MUTAÇÃO: reverter qualquer `ON CONFLICT DO NOTHING` de `_seed_weights` ou o
    `weight` de `_seed_scale` para `DO UPDATE SET weight = excluded.weight`."""
    path = tmp_path / "w.db"
    s = Store(path, embedding_dim=4); s.init_schema()
    s.conn.execute("UPDATE scale_levels SET weight = 0.123 "
                   " WHERE axis = 'grade' AND value = 'rct'")
    s.conn.execute("UPDATE grade_weight SET weight = 0.456 WHERE grade = 'rct'")
    s.conn.execute("UPDATE directness_weight SET weight = 0.789 WHERE directness='direct'")
    s.close()

    s = Store(path, embedding_dim=4); s.init_schema()
    assert s.conn.execute("SELECT weight FROM scale_levels WHERE axis='grade' "
                          "AND value='rct'").fetchone()["weight"] == 0.123
    assert s.conn.execute("SELECT weight FROM grade_weight WHERE grade='rct'"
                          ).fetchone()["weight"] == 0.456
    assert s.conn.execute("SELECT weight FROM directness_weight WHERE directness='direct'"
                          ).fetchone()["weight"] == 0.789
    s.close()


def test_the_seed_keeps_the_rank_in_step_with_the_enum(tmp_path):
    """A outra metade: `rank` NÃO é calibração, é ordem de força derivada do enum.

    MUTAÇÃO: aplicar `DO NOTHING` também ao rank. Um nível novo entraria com rank
    duplicado e `MIN(rank)` passaria a apontar para dois níveis ao mesmo tempo.
    """
    path = tmp_path / "r.db"
    s = Store(path, embedding_dim=4); s.init_schema()
    s.conn.execute("UPDATE scale_levels SET rank = 99 WHERE axis='directness' "
                   "AND value='direct'")
    s.close()
    s = Store(path, embedding_dim=4); s.init_schema()
    assert s.conn.execute("SELECT rank FROM scale_levels WHERE axis='directness' "
                          "AND value='direct'").fetchone()["rank"] == 1
    s.close()


def test_scale_level_ranks_restart_at_one_within_each_axis(store):
    for axis, n in (("grade", len(Grade)), ("directness", len(Directness))):
        ranks = [r["rank"] for r in store.conn.execute(
            "SELECT rank FROM scale_levels WHERE axis = ? ORDER BY rank", (axis,))]
        assert ranks == list(range(1, n + 1)), axis


def test_the_active_focus_survives_a_cli_reopen(tmp_path):
    """MUTAÇÃO: `ON CONFLICT(key) DO UPDATE` no seed de `meta['active_focus']` — o
    padrão copiado de `schema_version`. O foco volta a '1' a cada comando de CLI."""
    path = tmp_path / "a.db"
    s = Store(path, embedding_dim=4); s.init_schema()
    _focus2(s)
    s.conn.execute("UPDATE meta SET value = '2' WHERE key = 'active_focus'")
    s.close()
    s = Store(path, embedding_dim=4); s.init_schema()
    assert s.conn.execute("SELECT value FROM meta WHERE key='active_focus'"
                          ).fetchone()["value"] == "2"
    assert s.active_focus()["slug"] == "outro"
    s.close()


def test_a_missing_active_focus_key_heals_on_reopen(tmp_path):
    """A quarta morte silenciosa é AUTO-CURADA, e isso é uma propriedade, não um acaso.

    Registrada aqui para que ninguém "conserte" o seed para DO UPDATE achando que está
    fechando um buraco — o que quebraria o teste acima.
    """
    path = tmp_path / "h.db"
    s = Store(path, embedding_dim=4); s.init_schema()
    s.conn.execute("DELETE FROM meta WHERE key = 'active_focus'")
    assert s.active_focus() is None
    s.close()
    s = Store(path, embedding_dim=4); s.init_schema()
    assert s.active_focus()["slug"] == "bipolar-tag"
    s.close()


def test_a_calibration_drift_warns_instead_of_freezing_in_silence(tmp_path, caplog):
    """MUTAÇÃO: remover a comparação de `profile_hash` de `_seed_focus`."""
    path = tmp_path / "d.db"
    s = Store(path, embedding_dim=4); s.init_schema()
    s.conn.execute("UPDATE focuses SET profile_hash = 'deadbeef' WHERE id = 1")
    s.close()
    with caplog.at_level(logging.WARNING):
        s = Store(path, embedding_dim=4); s.init_schema()
    assert any("escala NOVA + foco NOVO" in r.getMessage() for r in caplog.records)
    s.close()


def test_the_seeded_focus_target_comes_from_the_production_profile():
    """APOSENTADO E SUBSTITUÍDO, não consertado.

    O teste anterior fazia `assert f'"{FOCUS_TARGET}"' in extract_claims.md`, e a
    invariante que ele policiava — a MESMA string em dois lugares — DESAPARECEU quando
    o `.md` deixou de carregar a string. Ele era a QUARTA falha do `$target`, num
    arquivo que o plano da Fase B não tinha aberto; o cross-check do `.md` foi
    absorvido por `test_declared_placeholders_survive_rendering`.

    O que sobra é uma invariante REAL e nova: as duas fontes do alvo (a semente do
    banco em `store.py` e o `focus.toml` de produção) têm de concordar, senão
    `focus new` e `init_schema` criariam o mesmo foco com alvos diferentes conforme
    quem o criou — e o BANCO vence, então a divergência ficaria congelada em toda
    aresta de `claim_directness`.

    MUTAÇÃO: editar `target` no focus.toml de produção sem editar `FOCUS_TARGET`.
    """
    from conftest import prod_profile
    from lithium.db.store import FOCUS_SLUG

    profile = prod_profile()
    assert profile.slug == FOCUS_SLUG
    assert profile.target == FOCUS_TARGET


# ══════════════════════════════════════════════════════════════ fail-closed

def test_a_claim_without_a_judgment_has_no_weight(store):
    """MUTAÇÃO: trocar o `JOIN claim_directness` por `LEFT JOIN` com
    `COALESCE(dw.weight, 1.0)` — o "conserto" óbvio de quem vir o placar zerado."""
    judged = seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT)
    unjudged = seed_claim(store, grade=Grade.RCT, judged=False)
    got = {r["claim_id"] for r in store.conn.execute("SELECT claim_id FROM claim_weight")}
    assert got == {judged}, f"a claim #{unjudged} sem julgamento entrou no peso"
    assert store.counts()["claims_unweighted"] == 1


def test_a_claim_graded_on_another_scale_has_no_weight(store):
    """MUTAÇÃO: remover `AND c.scale_id = af.scale_id` do JOIN de grade."""
    store.conn.execute("INSERT INTO evidence_scales(id, slug) VALUES(2, 'outra')")
    store.conn.execute("INSERT INTO scale_levels(scale_id, axis, value, weight, rank) "
                       "SELECT 2, axis, value, weight, rank FROM scale_levels "
                       " WHERE scale_id = 1")
    ok = seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT)
    other = seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT, scale_id=2)
    got = {r["claim_id"] for r in store.conn.execute("SELECT claim_id FROM claim_weight")}
    assert got == {ok}, f"a claim #{other} foi pesada contra a escala errada"


@pytest.mark.parametrize("scenario", ["nao_numerico", "id_inexistente", "aposentado"])
def test_no_active_focus_zeroes_the_board_and_says_so_out_loud(tmp_path, caplog, scenario):
    """'chave ausente' NÃO está aqui: `_seed_focus` a cura, e isso é o certo — ver
    `test_a_missing_active_focus_key_heals_on_reopen`.

    MUTAÇÕES: remover `_warn_if_no_active_focus` (os três cenários seguem com placar
    zero, mas sem log); remover `AND f.retired_at IS NULL` (o aposentado volta a pesar).
    """
    path = tmp_path / f"{scenario}.db"
    s = Store(path, embedding_dim=4); s.init_schema()
    s.conn.execute("INSERT INTO sources(id,kind,external_id,title,raw_json) "
                   "VALUES(1,'pubmed','1','t','{}')")
    seed_claim(s, grade=Grade.RCT, directness=Directness.DIRECT)
    assert s.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"] == 1

    if scenario == "nao_numerico":
        s.conn.execute("UPDATE meta SET value='abc' WHERE key='active_focus'")
    elif scenario == "id_inexistente":
        s.conn.execute("UPDATE meta SET value='99' WHERE key='active_focus'")
    else:
        s.conn.execute("UPDATE focuses SET retired_at='2020-01-01T00:00:00Z' WHERE id=1")
    s.close()

    with caplog.at_level(logging.WARNING):
        s = Store(path, embedding_dim=4); s.init_schema()
    assert s.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"] == 0
    assert s.counts()["claims_unweighted"] == 1
    msgs = [r.getMessage() for r in caplog.records]
    assert any("NENHUM FOCO ATIVO" in m for m in msgs), msgs
    s.close()


def test_a_duplicated_judgment_cannot_double_a_claim_weight(store):
    """MUTAÇÃO: trocar a PK composta de `claim_directness` por `id INTEGER PRIMARY KEY`.
    Sem ela a claim aparece 2× em `claim_weight` e `hypothesis_scoreboard` SOMA em dobro.
    """
    cid = seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT)
    before = store.conn.execute("SELECT weight FROM claim_weight").fetchone()["weight"]
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("INSERT INTO claim_directness(claim_id, focus_id, directness) "
                           "VALUES(?, 1, 'extrapolated')", (cid,))
    rows = store.conn.execute("SELECT weight FROM claim_weight").fetchall()
    assert len(rows) == 1 and rows[0]["weight"] == before


def test_a_directness_typo_fails_on_write_not_on_read(store):
    """MUTAÇÃO: remover `REFERENCES directness_weight(directness)` da coluna. O typo
    entra calado, deixa de casar o JOIN, e o fail-closed converte erro de escrita em
    perda muda de evidência."""
    cid = seed_claim(store, judged=False)
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("INSERT INTO claim_directness(claim_id, focus_id, directness) "
                           "VALUES(?, 1, 'directt')", (cid,))


def test_a_judgment_from_another_focus_does_not_weigh_here(store):
    """A propriedade central da fase, dita numericamente: o julgamento é do FOCO."""
    _focus2(store)
    cid = seed_claim(store, grade=Grade.RCT, judged=False)
    store.conn.execute("INSERT INTO claim_directness(claim_id, focus_id, directness) "
                       "VALUES(?, 2, 'direct')", (cid,))
    assert store.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"] == 0
    store.conn.execute("UPDATE meta SET value='2' WHERE key='active_focus'")
    assert store.conn.execute("SELECT weight FROM claim_weight").fetchone()["weight"] == 0.85


# ══════════════════════════════════════════════ o placar separa as duas leituras

def test_the_scoreboard_separates_no_support_from_no_judgment(store):
    """MUTAÇÃO: remover `n_unweighted` da view, ou removê-la de `state.render()`."""
    from lithium.pipeline.state import build_state
    a = seed_claim(store, intervention="lítio", judged=False)
    b = seed_claim(store, intervention="lítio", judged=False)
    store.conn.execute("INSERT INTO hypotheses(id, focus_id, statement) VALUES(1,1,'H')")
    for cid in (a, b):
        store.conn.execute("INSERT INTO evidence_links VALUES(?,1,'support',1.0)", (cid,))
    row = store.conn.execute("SELECT * FROM hypothesis_scoreboard").fetchone()
    assert (row["n_claims"], row["support"], row["n_unweighted"]) == (2, 0, 2)
    rendered = build_state(store, PROFILE).render()
    assert "2 linked claim(s) unjudged" in rendered, rendered


def test_the_state_header_does_not_count_what_the_table_excludes(store):
    """MUTAÇÃO: reverter `n_claims` para `counts['claims_verified']`. O cabeçalho
    afirmaria "2 verified claims" acima de "(nothing extracted yet)"."""
    from lithium.pipeline.state import build_state
    seed_claim(store, intervention="lítio", judged=False)
    seed_claim(store, intervention="lítio", judged=False)
    rendered = build_state(store, PROFILE).render()
    assert "0 weighted claims" in rendered
    assert "2 verified claim(s) carry NO judgment" in rendered


# ═══════════════════════════════════ a fiação por foco, cada ponto com sua mutação




def test_a_question_from_another_focus_does_not_take_a_human_queue_slot(store):
    """MUTAÇÃO: remover o `JOIN active_focus` de `escalated_queue`, ou o filtro de
    `escalated_count`. As cinco do foco #2 ocupam as vagas e a do foco #1 fica
    represada em OPEN com `stuck_reason`, para sempre, sem log."""
    from lithium.pipeline.question import QuestionEngine
    _focus2(store)
    for i in range(5):
        store.conn.execute(
            "INSERT INTO questions(focus_id, text, kind, status) "
            "VALUES(2, ?, 'CONTEXT', 'ESCALATED')", (f"do foco 2 · {i}",))
    store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status) "
        "VALUES(1, 'do foco 1', 'CONTEXT', 'ESCALATED')")

    engine = QuestionEngine(store, None, None, profile=PROFILE)
    assert engine.escalated_count() == 1
    assert [q.text for q in engine.human_queue()] == ["do foco 1"]


def test_a_question_of_another_focus_is_not_a_duplicate(store):
    """MUTAÇÃO: remover `AND focus_id = (SELECT id FROM active_focus)` de qualquer um dos
    dois caminhos de `find_duplicate`. A pergunta do foco B que casa com uma do foco A
    recebe `return None` e um `log.debug`: nunca é criada, e nada no sistema reporta
    pergunta que não foi gerada."""
    import numpy as np

    from lithium.pipeline.question import QuestionEngine
    _focus2(store)
    vec = np.ones(4, dtype=np.float32)
    for focus, targets in ((2, "quetiapina"), (2, None)):
        store.conn.execute(
            "INSERT INTO questions(focus_id, text, kind, targets, embedding) "
            "VALUES(?, 'pergunta do outro foco', 'FACTUAL', ?, ?)",
            (focus, targets, Store.pack_embedding(vec)))

    engine = QuestionEngine(store, None, None, profile=PROFILE)
    assert engine.find_duplicate(vec, "quetiapina") is None, "caminho COM alvo vazou"
    assert engine.find_duplicate(vec, None) is None, "caminho SEM alvo (global) vazou"
    store.conn.execute("UPDATE meta SET value = '2' WHERE key = 'active_focus'")
    assert engine.find_duplicate(vec, "quetiapina") is not None, (
        "o teste é vácuo se o embedding nem colide dentro do próprio foco")


def test_the_notification_mark_and_the_query_it_covers_move_together(store):
    """MUTAÇÃO: escopar `MARK_KEY` por foco sem escopar `escalated_delta` (ou o
    contrário). As duas metades andam juntas ou a mudança é PIOR que a chave global:
    trocar de foco faria as perguntas do foco anterior voltarem a ser 'novas' — a
    tempestade que o docstring do módulo registra ter derrubado a primeira versão."""
    from lithium.notify import watch
    _focus2(store)
    store.conn.execute("INSERT INTO questions(focus_id, text, kind, status) "
                       "VALUES(1, 'do foco 1', 'CONTEXT', 'ESCALATED')")
    fresh, current = watch.escalated_delta(store, [])
    assert [f["text"] for f in fresh] == ["do foco 1"]
    watch._save(store, {"escalated": current, "dead": {}})

    store.conn.execute("UPDATE meta SET value='2' WHERE key='active_focus'")
    fresh2, _ = watch.escalated_delta(store, watch._mark(store).get("escalated", []))
    assert fresh2 == [], (
        "o foco #2 foi avisado sobre perguntas do foco #1 — a marca e a consulta "
        "deixaram de cobrir o mesmo universo")




def test_the_best_level_label_survives_a_renumbering_of_the_scale(store):
    """A decodificação de rank NÃO pode ser posicional.

    MUTAÇÃO: `best_directness=list(Directness)[rank - 1]` (o que o código fazia antes).
    Ela é indistinguível da correta enquanto `scale_levels` estiver numerado na mesma
    ordem do enum — e é exatamente isso que deixa de valer quando um nível novo entra no
    meio do enum, ou quando uma escala nova numera diferente. Aqui a renumeração é
    construída, e a expectativa vem da CLAIM (o directness que ela carrega), não da
    tabela: o rótulo tem de ser o julgamento da claim, qualquer que seja o rank.
    """
    from lithium.pipeline.state import build_state

    seed_claim(store, grade=Grade.OPINION, directness=Directness.EXTRAPOLATED,
               intervention="lítio")
    # a escala passa a numerar ao contrário — o nível mais fraco vira rank 1
    for rank, value in enumerate(reversed(list(Directness)), 1):
        store.conn.execute("UPDATE scale_levels SET rank = ? "
                           " WHERE axis = 'directness' AND value = ?", (rank, value.value))
    for rank, value in enumerate(reversed(list(Grade)), 1):
        store.conn.execute("UPDATE scale_levels SET rank = ? "
                           " WHERE axis = 'grade' AND value = ?", (rank, value.value))

    coverage = build_state(store, PROFILE).by_intervention("lítio")
    assert coverage.best_directness is Directness.EXTRAPOLATED
    assert coverage.best_grade is Grade.OPINION
