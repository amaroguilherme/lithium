"""O goldset e sua precondição.

Item 11, segunda metade. O arquivo é dado por foco; `lithium/eval.py` é o motor e é
agnóstico — ele não sabe o que é um fármaco, só conta linhas que casem propriedades
declaradas.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lithium.db import Store
from lithium.eval import Case, check_precondition, load_goldset

RAIZ = Path(__file__).resolve().parent.parent
GOLDSET = RAIZ / "eval" / "goldset.toml"
DIM = 8


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "g.db", embedding_dim=DIM)
    s.init_schema()
    yield s
    s.close()


def _caso(**kw) -> Case:
    base = dict(id="x", question="q", expect="answerable", why="",
                requires={"min_count": 1})
    return Case(**{**base, **kw})


# ══════════════════════ a precondição, que é a razão do motor existir


def test_a_negative_control_is_skipped_when_the_corpus_grows_into_it(store):
    """O caso que este módulo existe para impedir.

    Um controle negativo diz "não há nada sobre X". No dia em que alguém colher um paper
    sobre X, ele passa a medir o CONTRÁRIO do que declara — e, sem conferência, contaria
    como acerto quando o sistema responder "não há nada", que aí seria a resposta ERRADA.

    MUTAÇÃO: fazer `check_precondition` devolver `ok=True` sempre.
    """
    caso = _caso(id="ausente_x", expect="unanswerable",
                 requires={"text_contains": "prazosin", "max_count": 0})
    assert check_precondition(store, caso).ok, "corpus vazio: o controle é válido"

    sid = store.upsert_source(kind="pubmed", external_id="1", raw={}, title="t")
    store.add_chunk(source_id=sid, ord=0, text="Prazosin reduced nightmares in the trial.")

    p = check_precondition(store, caso)
    assert not p.ok, "o corpus passou a conter o termo e o caso continuou valendo"
    assert p.found == 1
    assert "oposto" in p.detail


def test_a_presence_case_is_skipped_when_its_evidence_is_gone(store):
    """A contrapartida. Sem ela o guard poderia recusar tudo e o teste acima ficaria verde
    medindo o bug oposto.

    MUTAÇÃO: fazer `check_precondition` devolver `ok=False` sempre.
    """
    caso = _caso(id="tem_rct", requires={"design": "rct", "min_count": 1})
    assert not check_precondition(store, caso).ok, "corpus vazio: nada a achar"

    store.upsert_source(kind="pubmed", external_id="2", raw={}, title="t", design="rct")
    p = check_precondition(store, caso)
    assert p.ok and p.found == 1


def test_the_precondition_counts_sources_never_claims(store):
    """A ÂNCORA. Contar `claims` faria a precondição subir junto com a generosidade do
    extrator — o defeito auto-referencial que METRICS.md existe para evitar.

    `sources.design` vem do `PublicationTypeList` do NCBI; `claims` vêm do modelo.

    MUTAÇÃO: em `check_precondition`, trocar `FROM sources` por um JOIN com `claims`.
    """
    sid = store.upsert_source(kind="pubmed", external_id="3", raw={}, title="sobre topiramato")
    caso = _caso(requires={"title_contains": "topiramato", "min_count": 1})
    antes = check_precondition(store, caso).found

    # dez claims da MESMA fonte: se a contagem fosse por claim, saltaria para dez
    cid = store.add_chunk(source_id=sid, ord=0, text="texto")
    import json
    for i in range(10):
        store.conn.execute(
            "INSERT INTO claims(source_id, chunk_ids, statement, grade, scale_id, "
            "  confidence, verified) VALUES(?,?,?,'cohort',1,0.9,1)",
            (sid, json.dumps([cid]), f"claim {i}"))

    assert check_precondition(store, caso).found == antes == 1, (
        "a contagem se moveu com o número de claims: a precondição virou "
        "auto-referencial e sobe quando o extrator fica generoso"
    )


# ══════════════════════ a forma do arquivo


def test_a_typo_in_requires_is_refused_not_ignored(tmp_path):
    """Chave desconhecida vira precondição VAZIA, que é o mesmo que satisfeita — o caso
    pontuaria sem nunca ter sido conferido.

    MUTAÇÃO: aceitar chaves fora de `_REQUIRES_KEYS`.
    """
    p = tmp_path / "g.toml"
    p.write_text(
        'schema_version = 1\nfocus = "f"\n\n[[cases]]\nid = "a"\nquestion = "q"\n'
        'expect = "answerable"\n[cases.requires]\ntitle_contain = "x"\nmin_count = 1\n',
        encoding="utf-8")
    with pytest.raises(ValueError, match="desconhecida"):
        load_goldset(p)


def test_a_case_without_a_precondition_is_refused(tmp_path):
    """Sem precondição o caso não sabe dizer se ainda significa o que diz.

    MUTAÇÃO: aceitar `requires` ausente ou vazio.
    """
    p = tmp_path / "g.toml"
    p.write_text('schema_version = 1\nfocus = "f"\n\n[[cases]]\nid = "a"\n'
                 'question = "q"\nexpect = "answerable"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sem `requires`"):
        load_goldset(p)


def test_duplicate_ids_are_refused(tmp_path):
    """Ids são citados por teste e por relatório; duplicá-los torna o relatório ambíguo."""
    p = tmp_path / "g.toml"
    caso = ('[[cases]]\nid = "a"\nquestion = "q"\nexpect = "answerable"\n'
            '[cases.requires]\nmin_count = 1\n\n')
    p.write_text('schema_version = 1\nfocus = "f"\n\n' + caso * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicado"):
        load_goldset(p)


# ══════════════════════ o arquivo de verdade


def test_the_shipped_goldset_loads_and_has_both_directions():
    """Um goldset só de presença passa num sistema que responde "sim" a tudo; um só de
    ausência, num que responde "não" a tudo. Os dois lados são obrigatórios.

    MUTAÇÃO: apagar todos os casos `unanswerable` de `eval/goldset.toml`.
    """
    _, casos = load_goldset(GOLDSET)
    assert casos, "goldset vazio"
    lados = {c.expect for c in casos}
    assert lados == {"answerable", "unanswerable"}, (
        f"o goldset só tem casos {lados} — um lado só é satisfeito por um sistema que "
        f"responde sempre a mesma coisa"
    )


def test_every_shipped_case_still_holds_against_a_fresh_corpus():
    """Contra um banco VAZIO, todo caso de ausência tem de valer e todo caso de presença
    tem de ser pulado. Se um caso de ausência falhar aqui, ele nunca foi um controle.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        s = Store(Path(tmp) / "v.db", embedding_dim=DIM)
        s.init_schema()
        _, casos = load_goldset(GOLDSET)
        for c in casos:
            p = check_precondition(s, c)
            if c.expect == "unanswerable":
                assert p.ok, f"{c.id}: controle negativo inválido em corpus vazio"
            else:
                assert not p.ok, (
                    f"{c.id}: caso de presença passou num corpus VAZIO — a precondição "
                    f"não está exigindo nada")
        s.close()
