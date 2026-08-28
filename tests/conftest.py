"""Setup compartilhado. Em SQL CRU, de propósito.

Se `seed_claim` chamasse `Extractor._persist`, o teste deixaria de policiar a escrita:
qualquer defeito no caminho de produção apareceria igual nos dois lados e nenhuma
asserção o veria. Quem policia o caminho de produção é `test_pipeline_extract.py`.

Ponto único porque são 22 `INSERT INTO claims` em 15 arquivos. Sem ele, são 15 lugares
onde uma expectativa vira frouxa por acidente — e o subconjunto que grava a claim mas
esquece `claim_directness` produz claims INVISÍVEIS a `claim_weight`, o que deixa os
testes de contagem-zero VERDES e os de conteúdo vermelhos, convidando a "consertar" o
teste em vez do setup.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest

from lithium.focus import FocusProfile, load_profile
from lithium.types import Directness, Grade


def seed_claim(store, **kw):
    """Versão função, para os helpers de módulo. `seed_claim_fx` é o mesmo como fixture."""
    return _seed_claim(store, **kw)


@pytest.fixture
def seed_claim_fx():
    """Grava uma claim JÁ JULGADA para o foco ativo, e devolve o id.

    O julgamento é por claim (ver `claim_directness` no schema), então duas claims deste
    mesmo teste podem carregar directness diferentes sem interferência — que é o que a
    tabela de peso precisa para ser exercitável.
    """

    return _seed_claim


def _judge(store, claim_id, directness, focus_id=None):
    conn = store.conn
    if focus_id is None:
        focus_id = int(conn.execute("SELECT id FROM active_focus").fetchone()["id"])
    conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,?,?) "
        "ON CONFLICT(claim_id, focus_id) DO UPDATE SET directness = excluded.directness",
        (claim_id, focus_id, Directness(directness).value))


def _seed_claim(store, *, grade=Grade.RCT, directness=Directness.DIRECT, source_id=1,
          statement="x", intervention=None, direction="positive", confidence=1.0,
          verified=1, population=None, chunk_ids=(), claim_id=None,
          focus_id=None, scale_id=None, judged=True, extracted_at=None):
    conn = store.conn
    if focus_id is None or scale_id is None:
        focus = conn.execute("SELECT id, scale_id FROM active_focus").fetchone()
        focus_id = focus_id if focus_id is not None else int(focus["id"])
        scale_id = scale_id if scale_id is not None else int(focus["scale_id"])
    cols = ["source_id", "chunk_ids", "statement", "population", "intervention",
            "direction", "grade", "scale_id", "confidence", "verified"]
    vals = [source_id, json.dumps(list(chunk_ids)), statement, population,
            intervention, direction, Grade(grade).value, scale_id, confidence,
            verified]
    if claim_id is not None:
        cols.insert(0, "id"); vals.insert(0, claim_id)
    if extracted_at is not None:
        cols.append("extracted_at"); vals.append(extracted_at)
    cur = conn.execute(
        f"INSERT INTO claims({', '.join(cols)}) "
        f"VALUES({', '.join('?' * len(cols))}) RETURNING id", vals)
    new_id = int(cur.fetchone()["id"])
    if judged:
        conn.execute(
            "INSERT INTO claim_directness(claim_id, focus_id, directness) "
            "VALUES(?, ?, ?)",
            (new_id, focus_id, Directness(directness).value))
    return new_id


# ────────────────────────────────────────────────────────────── perfis de foco

FIXTURE_FOCUSES = Path(__file__).resolve().parent / "fixtures" / "focuses"
PROD_FOCUSES = Path(__file__).resolve().parent.parent / "focuses"


def onco_profile() -> FocusProfile:
    """O perfil de TESTE: oncologia veterinária, domínio distinto do de produção.

    Existe porque a alternativa é a suíte afirmar sobre conteúdo lendo o perfil de
    PRODUÇÃO. Isso não é tautológico por si só (o literal do teste é independente do
    TOML), mas é CEGO ao que importa: não distingue "o call site lê o perfil" de "o
    call site ainda lê a constante velha que por acaso diz a mesma coisa". Nenhuma
    string de oncologia veterinária pode vir de outro lugar do repo.
    """
    return load_profile(FIXTURE_FOCUSES / "onco-vet")


def prod_profile() -> FocusProfile:
    """O perfil de PRODUÇÃO. Só para quem afirma sobre produção: golden, contrato de
    prompt, e a trava de que a transcrição não mudou valor nenhum."""
    return load_profile(PROD_FOCUSES / "bipolar-tag")


def variant_profile(tmp_path: Path, **focus_overrides) -> FocusProfile:
    """Uma cópia de `onco-vet` com chaves de `focus.toml` trocadas.

    Copiar e sobrescrever, em vez de um terceiro diretório commitado: três perfis
    quase iguais divergem na primeira edição, e o que o teste policia é justamente a
    chave que ele troca.
    """
    dest = tmp_path / focus_overrides.get("slug", "onco-vet")
    shutil.copytree(FIXTURE_FOCUSES / "onco-vet", dest, dirs_exist_ok=True)
    raw = tomllib.loads((dest / "focus.toml").read_text(encoding="utf-8"))
    raw.update(focus_overrides)
    for key in [k for k, v in raw.items() if v is _DELETE]:
        del raw[key]
    (dest / "focus.toml").write_text(_dump_toml(raw), encoding="utf-8")
    return load_profile(dest)


_DELETE = object()


def _dump_toml(data: dict) -> str:
    """Serializador TOML mínimo. O repo não tem `tomli_w` e uma dependência nova para
    escrever fixture de teste não se paga."""
    scalars, tables = [], []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append(f"[{key}]\n" + "".join(
                f"{k} = {_lit(v)}\n" for k, v in value.items()))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            tables += [f"[[{key}]]\n" + "".join(
                f"{k} = {_lit(v)}\n" for k, v in item.items()) for item in value]
        else:
            scalars.append(f"{key} = {_lit(value)}")
    return "\n".join(scalars) + "\n\n" + "\n".join(tables)


def _lit(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_lit(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_lit(v)}" for k, v in value.items()) + "}"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


@pytest.fixture
def profile() -> FocusProfile:
    return onco_profile()


@pytest.fixture
def prod_profile_fx() -> FocusProfile:
    return prod_profile()
