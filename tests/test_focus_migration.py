"""A migração de um banco pré-Fase-A. É o único exercício que a retro-carga inteira tem.

Sem estes testes, reverter o corpo inteiro da migração deixa os 699 verdes — todos os
outros fixtures constroem banco NOVO, onde o `CREATE TABLE` já traz tudo pronto.
"""
from __future__ import annotations

import json
import logging
import pathlib
import sqlite3

import pytest

from lithium.db import Store
from lithium.db.store import DIRECTNESS_WEIGHTS, GRADE_WEIGHTS

LEGACY_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE grade_weight (grade TEXT PRIMARY KEY, weight REAL NOT NULL, rank INTEGER NOT NULL);
CREATE TABLE directness_weight (directness TEXT PRIMARY KEY, weight REAL NOT NULL, rank INTEGER NOT NULL);
CREATE TABLE sources (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, external_id TEXT NOT NULL,
  title TEXT, year INTEGER, journal TEXT, doi TEXT, url TEXT,
  design TEXT REFERENCES grade_weight(grade), sample_n INTEGER, population_tag TEXT,
  raw_json TEXT NOT NULL, fetched_at TEXT, UNIQUE(kind, external_id));
CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES sources(id),
  ord INTEGER NOT NULL, section TEXT, text TEXT NOT NULL, n_tokens INTEGER, UNIQUE(source_id, ord));
CREATE TABLE claims (id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  chunk_ids TEXT NOT NULL, statement TEXT NOT NULL, population TEXT, intervention TEXT,
  comparator TEXT, outcome TEXT, direction TEXT, effect TEXT,
  grade TEXT NOT NULL REFERENCES grade_weight(grade),
  directness TEXT NOT NULL REFERENCES directness_weight(directness),
  confidence REAL NOT NULL DEFAULT 0.5, verified INTEGER NOT NULL DEFAULT 0,
  extracted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE INDEX idx_claims_source ON claims(source_id);
CREATE INDEX idx_claims_verified ON claims(verified);
CREATE INDEX idx_claims_interv ON claims(intervention);
CREATE VIEW claim_weight AS
SELECT c.id AS claim_id, c.intervention, c.direction,
       gw.weight * dw.weight * c.confidence AS weight
  FROM claims c JOIN grade_weight gw ON gw.grade = c.grade
                JOIN directness_weight dw ON dw.directness = c.directness
 WHERE c.verified = 1;
CREATE TABLE questions (id INTEGER PRIMARY KEY, text TEXT NOT NULL, kind TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'OPEN', priority REAL NOT NULL DEFAULT 0.5,
  parent_id INTEGER, origin TEXT NOT NULL DEFAULT 'auto', targets TEXT,
  stuck_reason TEXT, partial_work TEXT, answer TEXT, answer_origin TEXT, embedding BLOB,
  created_at TEXT, escalated_at TEXT, answered_at TEXT);
CREATE VIEW escalated_queue AS SELECT id, text, kind, priority, stuck_reason,
  partial_work, escalated_at FROM questions WHERE status = 'ESCALATED';
CREATE TABLE hypotheses (id INTEGER PRIMARY KEY, statement TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'active', tier TEXT NOT NULL DEFAULT 'evidence',
  chain_json TEXT, novelty REAL, survives_critique INTEGER,
  created_at TEXT, updated_at TEXT);
CREATE TABLE evidence_links (claim_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
  hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
  polarity TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (claim_id, hypothesis_id));
CREATE VIEW hypothesis_scoreboard AS
SELECT h.id, h.statement, h.status,
  COALESCE(SUM(CASE WHEN el.polarity='support' THEN cw.weight*el.weight END),0) AS support,
  COALESCE(SUM(CASE WHEN el.polarity='contra' THEN cw.weight*el.weight END),0) AS contra,
  COUNT(el.claim_id) AS n_claims, h.updated_at
  FROM hypotheses h LEFT JOIN evidence_links el ON el.hypothesis_id = h.id
  LEFT JOIN claim_weight cw ON cw.claim_id = el.claim_id
 WHERE h.tier='evidence' GROUP BY h.id;
CREATE TABLE memories (id INTEGER PRIMARY KEY, text TEXT NOT NULL, kind TEXT NOT NULL,
  rationale TEXT, source TEXT NOT NULL DEFAULT 'chat', provenance TEXT,
  confirmed INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
  embedding BLOB, created_at TEXT, confirmed_at TEXT, text_key TEXT);
CREATE TABLE tasks (id INTEGER PRIMARY KEY, kind TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'pending',
  priority REAL NOT NULL DEFAULT 0.5, attempts INTEGER NOT NULL DEFAULT 0,
  dedup_key TEXT UNIQUE, scheduled_at TEXT, claimed_at TEXT, finished_at TEXT,
  error TEXT, created_at TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

CLAIMS = [
    (1, "meta-análise em adultos", "adults", "meta_analysis", "direct", 1.0, 1),
    (2, "rct em adultos", "adults", "rct", "partial", 1.0, 1),
    (3, "pré-clínico", "CRIANÇAS com TAG", "preclinical", "extrapolated", 0.5, 1),
    (4, "coorte", "crianças  com tag", "cohort", "indirect", 0.8, 1),
    (5, "sem população declarada", None, "case_report", "partial", 0.4, 1),
    (6, "não verificada", "adults", "rct", "direct", 0.9, 0),
]


def _legacy(path: pathlib.Path, *, with_hypothesis: bool = True) -> dict[int, float]:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(LEGACY_SCHEMA)
    conn.executemany("INSERT INTO grade_weight VALUES(?,?,?)", GRADE_WEIGHTS)
    conn.executemany("INSERT INTO directness_weight VALUES(?,?,?)", DIRECTNESS_WEIGHTS)
    conn.execute("INSERT INTO sources(id,kind,external_id,title,raw_json) "
                 "VALUES(1,'pubmed','1','t','{}')")
    for cid, st, pop, g, d, cf, v in CLAIMS:
        conn.execute(
            "INSERT INTO claims(id, source_id, chunk_ids, statement, population, "
            "  intervention, direction, grade, directness, confidence, verified) "
            "VALUES(?,1,'[]',?,?,'lítio','positive',?,?,?,?)",
            (cid, st, pop, g, d, cf, v))
    if with_hypothesis:
        conn.execute("INSERT INTO hypotheses(id, statement) VALUES(1,'H1')")
        conn.execute("INSERT INTO evidence_links VALUES(1,1,'support',1.0)")
    conn.execute("INSERT INTO questions(id, text, kind) VALUES(1,'q?','FACTUAL')")
    conn.execute("INSERT INTO memories(text,kind,source,confirmed,active) "
                 "VALUES('lição','search_lesson','research',1,1)")
    before = {r["claim_id"]: r["weight"]
              for r in conn.execute("SELECT claim_id, weight FROM claim_weight")}
    conn.close()
    return before


def test_migrating_a_pre_phase_a_database_preserves_every_weight_and_is_idempotent(tmp_path):
    """A régua de aceite, medida em números: uma migração que muda comportamento não é
    uma migração.

    MUTAÇÕES: mover `_drop_legacy_directness_column` para antes do `executescript`
    (`OperationalError: error in view claim_weight after drop column`); remover o guard
    `'directness' in PRAGMA table_info(claims)` do backfill (a rodada 2 levanta
    `no such column: directness`); dropar só `claim_weight` antes do ALTER em vez de
    deixar o executescript recriá-la (`error in view hypothesis_scoreboard`).
    """
    path = tmp_path / "legacy.db"
    before = _legacy(path)
    assert before, "o fixture precisa produzir pesos, senão a comparação é vácua"

    rounds = []
    for _ in range(3):
        s = Store(path, embedding_dim=4)
        s.init_schema()
        rounds.append({r["claim_id"]: r["weight"]
                       for r in s.conn.execute("SELECT claim_id, weight FROM claim_weight")})
        s.close()

    assert rounds[0] == before, "a migração mudou o peso de alguma claim"
    assert rounds[0] == rounds[2], "a migração não é idempotente"

    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(claims)")}
    assert "directness" not in cols
    assert {"scale_id"} <= cols
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='claims'"
    )} >= {"idx_claims_source", "idx_claims_verified", "idx_claims_interv"}
    # e o banco continua ESCRIVÍVEL — o modo de falha do ADD COLUMN com REFERENCES
    conn.execute("INSERT INTO claims(source_id, chunk_ids, statement, grade, scale_id, "
                 "  confidence, verified) VALUES(1,'[]','nova','rct',1,0.5,1)")
    conn.execute("DELETE FROM claims WHERE statement = 'nova'")


def test_every_legacy_judgment_names_the_focus_after_the_backfill(tmp_path):
    """A propriedade que a fase existe para criar, verificada na retro-carga.

    MUTAÇÃO: `_backfill_claim_judgments` virar no-op — todas as claims ficariam sem
    peso, o que o teste acima já pega, mas aqui a falha diz POR QUÊ.
    """
    path = tmp_path / "b.db"
    _legacy(path)
    s = Store(path, embedding_dim=4); s.init_schema()
    rows = {r["claim_id"]: (r["directness"], r["slug"]) for r in s.conn.execute(
        "SELECT cd.claim_id, cd.directness, f.slug FROM claim_directness cd "
        "  JOIN focuses f ON f.id = cd.focus_id")}
    assert rows == {cid: (d, "bipolar-tag") for cid, _s, _p, _g, d, _c, _v in CLAIMS}
    # inclusive a claim sem população: o valor guardado JÁ ERA o julgamento contra
    # este alvo, e a migração não pode inventar uma exclusão que não existia.
    assert 5 in rows
    s.close()


def test_the_migration_does_not_repeat_its_log_on_every_open(tmp_path, caplog):
    """Aviso que repete a cada abertura de CLI deixa de ser lido."""
    path = tmp_path / "q.db"
    _legacy(path)
    s = Store(path, embedding_dim=4); s.init_schema(); s.close()
    caplog.clear()   # a primeira abertura PODE logar; a segunda não
    with caplog.at_level(logging.INFO):
        s = Store(path, embedding_dim=4); s.init_schema(); s.close()
    noisy = [r.getMessage() for r in caplog.records
             if r.name.startswith("lithium.db")]
    assert noisy == [], noisy


# ══════════════════════════════════════════════ as DUAS formas de `hypotheses`

def test_an_empty_legacy_hypotheses_table_is_rebuilt_so_two_focuses_can_share_a_statement(tmp_path):
    """MUTAÇÃO: `_rebuild_empty_legacy_hypotheses` virar no-op. MEDIDO: o
    `UNIQUE(statement)` legado sobrevive a qualquer índice novo e o segundo foco levanta
    `IntegrityError: UNIQUE constraint failed: hypotheses.statement` — o escopo por foco
    seria verdadeiro só em banco de teste, e o banco de produção é legado."""
    path = tmp_path / "e.db"
    _legacy(path, with_hypothesis=False)
    s = Store(path, embedding_dim=4); s.init_schema()
    assert not Store.hypotheses_are_globally_unique(s.conn)
    sid = int(s.conn.execute("SELECT id FROM evidence_scales").fetchone()["id"])
    s.conn.execute("INSERT INTO focuses(id,slug,target,scale_id) VALUES(2,'b','x',?)", (sid,))
    for focus in (1, 2):
        s.conn.execute("INSERT INTO hypotheses(focus_id, statement) VALUES(?, 'H') "
                       "ON CONFLICT(focus_id, statement) DO NOTHING", (focus,))
    assert s.conn.execute("SELECT COUNT(*) AS n FROM hypotheses").fetchone()["n"] == 2
    s.close()


def test_a_populated_legacy_hypotheses_table_keeps_working_and_says_what_it_cannot_do(tmp_path):
    """MUTAÇÃO: remover `_migrate_legacy_hypothesis_index`. MEDIDO: sem o índice ÚNICO
    composto, `ON CONFLICT(focus_id, statement)` levanta
    `OperationalError: ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE
    constraint` em TODO insert de hipótese neste banco.

    Segunda MUTAÇÃO: criar um índice COMUM em vez de ÚNICO (que é o que a versão
    anterior deste plano mandava) — mesmo OperationalError.
    """
    path = tmp_path / "p.db"
    _legacy(path, with_hypothesis=True)
    s = Store(path, embedding_dim=4); s.init_schema()
    assert Store.hypotheses_are_globally_unique(s.conn)
    # o caminho de produção FUNCIONA dentro de um foco
    s.conn.execute("INSERT INTO hypotheses(focus_id, statement) VALUES(1,'H2') "
                   "ON CONFLICT(focus_id, statement) DO NOTHING")
    s.conn.execute("INSERT INTO hypotheses(focus_id, statement) VALUES(1,'H2') "
                   "ON CONFLICT(focus_id, statement) DO NOTHING")
    assert s.conn.execute("SELECT COUNT(*) AS n FROM hypotheses WHERE statement='H2'"
                          ).fetchone()["n"] == 1
    # e a limitação é uma DECISÃO REGISTRADA, não uma surpresa
    assert s.conn.execute("SELECT value FROM meta WHERE key='hypotheses_focus_scope'"
                          ).fetchone()["value"] == "global"
    sid = int(s.conn.execute("SELECT id FROM evidence_scales").fetchone()["id"])
    s.conn.execute("INSERT INTO focuses(id,slug,target,scale_id) VALUES(2,'b','x',?)", (sid,))
    with pytest.raises(sqlite3.IntegrityError):
        s.conn.execute("INSERT INTO hypotheses(focus_id, statement) VALUES(2,'H2') "
                       "ON CONFLICT(focus_id, statement) DO NOTHING")
    s.close()
