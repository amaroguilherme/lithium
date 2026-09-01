"""Executor do goldset: confere a precondição de cada caso contra o corpus.

O arquivo é DADO (`eval/goldset.toml`, por foco); este módulo é o motor, e é agnóstico —
ele não sabe o que é um fármaco nem o que é um ensaio. Só sabe contar linhas em `sources`
que casem propriedades declaradas.

A PRECONDIÇÃO é a peça que justifica o módulo existir. Um caso do goldset só significa
algo se o corpus ainda estiver no estado que ele pressupõe: um controle negativo sobre um
termo ausente vira pergunta respondível assim que alguém colher um paper sobre o termo, e
a partir daí ele mede o CONTRÁRIO do que declara.

Um caso cuja precondição falhou é PULADO e ANUNCIADO, nunca contado como acerto. É a lição
que a tabela de cobertura ensinou: um agregado que deixa de contar o que diz contar, sem
reclamar, é pior que agregado nenhum.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Chaves que `requires` aceita. Fechado de propósito: uma chave desconhecida é erro de
# digitação que passaria como precondição vazia — isto é, como precondição satisfeita.
_REQUIRES_KEYS = frozenset({"title_contains", "text_contains", "design",
                            "min_count", "max_count"})


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    question: str
    expect: str            # 'answerable' | 'unanswerable'
    why: str
    requires: dict[str, Any]
    must_cite: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Precondition:
    ok: bool
    found: int
    detail: str


def load_goldset(path: Path) -> tuple[str, list[Case]]:
    """Lê o arquivo e valida a forma. Devolve `(focus_slug, casos)`."""
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    casos: list[Case] = []
    vistos: set[str] = set()
    for raw in doc.get("cases", []):
        cid = raw["id"]
        if cid in vistos:
            raise ValueError(f"goldset: id duplicado {cid!r} — os ids são citados por"
                             f" testes e por relatório, e precisam ser estáveis")
        vistos.add(cid)
        if raw["expect"] not in ("answerable", "unanswerable"):
            raise ValueError(f"{cid}: `expect` inválido {raw['expect']!r}")
        req = raw.get("requires") or {}
        desconhecidas = set(req) - _REQUIRES_KEYS
        if desconhecidas:
            raise ValueError(
                f"{cid}: chave(s) desconhecida(s) em `requires`: {sorted(desconhecidas)}."
                f" Uma chave com erro de digitação vira precondição VAZIA, que é o mesmo"
                f" que precondição satisfeita — o caso pontuaria sem nunca ter sido"
                f" conferido."
            )
        if not req:
            raise ValueError(f"{cid}: sem `requires`. Um caso sem precondição não sabe"
                             f" dizer se ainda significa o que diz.")
        casos.append(Case(
            id=cid, question=raw["question"], expect=raw["expect"],
            why=raw.get("why", ""), requires=req,
            must_cite=tuple(str(x) for x in raw.get("must_cite") or ()),
        ))
    return doc["focus"], casos


def check_precondition(store: Any, case: Case) -> Precondition:
    """Quantos `sources` casam o que o caso exige — e se isso basta.

    Conta em `sources` e `chunks`, NUNCA em `claims`: claim é produzida pelo modelo, e uma
    precondição que dependesse dela subiria junto com a generosidade do extrator. PMID,
    título e `design` vêm do indexador externo.
    """
    req = case.requires
    where, params = ["1=1"], []
    if "design" in req:
        where.append("s.design = ?")
        params.append(req["design"])
    if "title_contains" in req:
        where.append("LOWER(s.title) LIKE ?")
        params.append(f"%{req['title_contains'].lower()}%")
    if "text_contains" in req:
        # O texto vive em `chunks`; o EXISTS mantém a contagem por FONTE, para que
        # `min_count` signifique a mesma coisa nas duas formas.
        where.append(
            "EXISTS (SELECT 1 FROM chunks c WHERE c.source_id = s.id "
            "          AND LOWER(c.text) LIKE ?)")
        params.append(f"%{req['text_contains'].lower()}%")

    found = int(store.conn.execute(
        f"SELECT COUNT(*) AS n FROM sources s WHERE {' AND '.join(where)}",
        tuple(params),
    ).fetchone()["n"])

    if "max_count" in req and found > req["max_count"]:
        return Precondition(False, found, (
            f"o corpus passou a conter {found} fonte(s) que o caso pressupõe AUSENTES "
            f"(máx {req['max_count']}) — como controle negativo ele agora mede o oposto"))
    if "min_count" in req and found < req["min_count"]:
        return Precondition(False, found, (
            f"o corpus tem {found} fonte(s) e o caso exige {req['min_count']} — "
            f"a pergunta deixou de ser respondível a partir deste corpus"))
    return Precondition(True, found, "ok")
