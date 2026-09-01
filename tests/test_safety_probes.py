"""Executa `eval/safety_probes.yaml` contra o screen real.

Item 11. Um dos três critérios do gate de promoção do LoRA é **zero regressão** aqui, e
este é o único dos três avaliável sem modelo nenhum: o screen é casamento determinístico
de termo com escopo de negação, e o vocabulário vem de um TOML escrito por humano.

O arquivo é DADO e este módulo é o motor. Acrescentar cenário não exige tocar em Python —
é o que permite a um revisor clínico contribuir probe sem ler código. TOML e não YAML:
`tomllib` é biblioteca padrão, e o resto do projeto já é TOML.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from lithium.focus import load_profile
from lithium.safety.rules import ruleset_from_profile
from lithium.safety.screen import Segment, screen

PROBES = Path(__file__).resolve().parent.parent / "eval" / "safety_probes.toml"


def _load() -> dict:
    return tomllib.loads(PROBES.read_text(encoding="utf-8"))


def _ruleset(focus_slug: str):
    root = Path(__file__).resolve().parent.parent / "focuses" / focus_slug
    return ruleset_from_profile(load_profile(root))


DOC = _load()
CASES = [(p["id"], p) for p in DOC["probes"]]


@pytest.mark.parametrize("probe_id,probe", CASES, ids=[c[0] for c in CASES])
def test_safety_probe(probe_id, probe):
    """Cada probe declara o que TEM de disparar e o que NÃO pode.

    Os dois lados são obrigatórios: só `must_fire` passaria num screen que dispara tudo;
    só `must_not_fire`, num que nunca dispara.

    MUTAÇÃO (uma por regra, todas executadas): apagar qualquer `[[rules]]` de
    `focuses/bipolar-tag/safety.toml`; trocar `co_terms` de união-do-turno para
    por-segmento em `screen()`; remover `AMBIGUOUS_IN_USER_TEXT`; remover `_boundary`;
    remover o escopo por tipo de `_NEGATION`.
    """
    rs = _ruleset(DOC["focus"])
    segs = [Segment(kind=s["kind"], text=s["text"]) for s in probe["segments"]]
    disparou = {a.key for a in screen(segs, rs)}

    faltando = set(probe.get("must_fire") or []) - disparou
    assert not faltando, (
        f"{probe_id}: alerta obrigatório não disparou: {sorted(faltando)}\n"
        f"  disparou: {sorted(disparou)}\n  por quê importa: {probe['why'].strip()}"
    )
    indevidos = disparou & set(probe.get("must_not_fire") or [])
    assert not indevidos, (
        f"{probe_id}: alerta disparou indevidamente: {sorted(indevidos)}\n"
        f"  por quê importa: {probe['why'].strip()}"
    )


def test_every_rule_in_the_profile_has_at_least_one_probe():
    """Uma regra sem probe é uma regra que pode sumir sem ninguém notar.

    Esta trava é o que impede o arquivo de envelhecer: acrescentar regra ao `safety.toml`
    sem acrescentar cenário passa a quebrar a suíte, em vez de criar cobertura silenciosa
    de zero por cento.

    MUTAÇÃO: acrescentar um `[[rules]]` novo ao safety.toml sem probe correspondente.
    """
    rs = _ruleset(DOC["focus"])
    do_perfil = {r.key for r in rs.rules}
    cobertas = {k for p in DOC["probes"] for k in (p.get("must_fire") or [])}
    sem_probe = do_perfil - cobertas
    assert not sem_probe, (
        f"regras sem nenhum probe que as exercite: {sorted(sem_probe)} — "
        f"elas podem ser removidas sem que nada falhe"
    )


def test_no_probe_references_a_rule_that_does_not_exist():
    """O oposto: probe citando regra inexistente passa vazio para sempre.

    `must_fire` de uma chave que não existe nunca dispara, então o probe só passaria se a
    asserção fosse vacuamente verdadeira — que é exatamente o teste tautológico que este
    repo caça.

    MUTAÇÃO: renomear uma chave em `safety.toml` sem atualizar o probe.
    """
    rs = _ruleset(DOC["focus"])
    do_perfil = {r.key for r in rs.rules}
    citadas = {k for p in DOC["probes"]
               for k in (p.get("must_fire") or []) + (p.get("must_not_fire") or [])}
    fantasmas = citadas - do_perfil
    assert not fantasmas, f"probes citam regra inexistente: {sorted(fantasmas)}"
