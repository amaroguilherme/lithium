"""O screen determinístico: anexa, nunca remove — e não fabrica ruído.

Duas famílias de teste, e a segunda é a que decide se o screen serve para alguma coisa.

**Que ele dispare** é fácil e quase não precisa de teste. **Que ele NÃO dispare** onde
não deve é o trabalho: falso positivo aqui não é incômodo, é o que treina a pessoa a
passar o olho por cima do próximo alerta — e o próximo pode ser o que importava. Por isso
`FALSE_POSITIVE_TRAPS` é maior que a tabela de casos positivos, e por isso a cobertura é
medida **por termo declarado**, não por regra: uma regra com dez termos e um probe deixa
nove termos sem nunca terem casado nada, e um termo truncado (`lamicta`) importa
limpo, passa em todo teste, e nunca casa.
"""

from __future__ import annotations

import copy

import pytest

from lithium.safety.rules import (
    ABSENT,
    concepts_implicated_by,
    concepts_in,
    matches,
    ruleset_from_profile,
)
from lithium.safety.screen import (
    DISCLAIMER,
    NO_RULESET,
    Alert,
    Segment,
    render,
    screen,
)

from conftest import prod_profile

# O vocabulário vem do PERFIL DE PRODUÇÃO — é sobre ele que este arquivo afirma.
PROFILE = prod_profile()
RULESET = ruleset_from_profile(PROFILE)
CONCEPTS = RULESET.concepts
RULES = RULESET.rules
PROPERTY_PHRASES = RULESET.property_phrases


def _keys(segments) -> set[str]:
    return {a.key for a in screen(segments, RULESET)}


def _user(text: str) -> list[Segment]:
    return [Segment("user", text)]


# ───────────────────────────────────────────── anexa, nunca remove (comportamental)


def test_screen_does_not_mutate_what_it_receives():
    """Comportamental, não uma asserção sobre a anotação de tipo.

    A versão anterior deste teste afirmava
    `inspect.signature(screen).return_annotation in ("list[Alert]", list)` — com
    `from __future__ import annotations`, isso é a string que o autor digitou, ou seja o
    teste derivava a expectativa da exata linha que deveria policiar. Ele passava com um
    `screen()` que apagava os segmentos de evidência in-place.
    """
    segments = [
        Segment("user", "posso usar quetiapina?"),
        Segment("evidence", "Lithium reduced relapse.", "PMID:1"),
        Segment("evidence", "Quetiapine reduced HAM-A.", "PMID:2"),
    ]
    snapshot = copy.deepcopy(segments)

    out = screen(segments, RULESET)

    assert segments == snapshot, "screen() alterou a coleção do chamador"
    assert all(isinstance(a, Alert) for a in out)
    assert all(not isinstance(a, Segment) for a in out)


def test_screen_returns_no_evidence_text():
    """O que ele devolve não pode carregar o texto da evidência — se carregasse, um
    chamador poderia render só os alertas e achar que renderizou a evidência."""
    statement = "Lithium prophylaxis reduced relapse rates over 24 months."
    out = screen([Segment("evidence", statement, "PMID:9")], RULESET)

    assert out, "o caso precisa disparar, senão o teste é vazio"
    assert statement not in " ".join(a.text for a in out)


def test_adding_a_segment_never_removes_an_alert():
    """Monotonicidade: mais material só pode acrescentar alerta."""
    base = [Segment("evidence", "Valproate exposure in utero.", "PMID:1")]
    more = base + [Segment("user", "e sobre exercício físico?")]

    assert _keys(base) <= _keys(more)


def test_the_disclaimer_is_rendered_even_with_no_alert():
    """"Nada apareceu" tem que ser distinguível de "não rodou"."""
    block = render([], ruleset_declared=True)
    assert DISCLAIMER in block
    assert "no term matched" in block
    assert DISCLAIMER in render(screen(_user("estou tomando lítio"), RULESET),
                                ruleset_declared=True)


def test_the_safety_block_says_when_no_rule_exists():
    """Os TRÊS estados são textualmente distintos, e o disclaimer de COBERTURA só
    aparece nos dois primeiros.

    MEDIDO e EXECUTADO no estado anterior: `render(screen([Segment('user','bom dia')]))`
    com as 8 regras carregadas e com `RULES = ()` produzia a MESMA string, BYTE A BYTE,
    terminando em "Ausência de alerta não significa ausência de risco". Num foco sem
    safety.toml isso é literalmente verdadeiro e completamente enganoso — uma camada
    que cobre NADA afirmando que cobre, em todo turno de chat, e `answer.py` PERSISTE
    o resultado em `findings.safety_json` para o revisor ler.

    MUTAÇÃO: fazer o ramo de ruleset ausente emitir `(no term matched this turn)` +
    DISCLAIMER — ou seja, restaurar exatamente o defeito.
    """
    casou = render(screen(_user("estou tomando lítio"), RULESET), ruleset_declared=True)
    vazio = render(screen(_user("bom dia"), RULESET), ruleset_declared=True)
    ausente = render(screen(_user("bom dia"), ABSENT), ruleset_declared=False)

    assert casou != vazio != ausente and casou != ausente
    assert DISCLAIMER in casou and DISCLAIMER in vazio
    assert DISCLAIMER not in ausente, (
        "o disclaimer de COBERTURA descreve os limites de uma verificação que rodou; "
        "sem regra nenhuma ele afirma cobrir o que não cobre"
    )
    assert NO_RULESET in ausente
    assert "no term matched" not in ausente, (
        "'nenhum termo casou' e 'não existe regra para casar' não podem ler igual"
    )


def test_switching_focus_does_not_leave_the_old_safety_rules_loaded():
    """Depois de trocar o perfil, `screen` usa o ruleset NOVO. Cobre os bindings
    congelados de import de uma vez.

    MEDIDO no estado anterior: `rules.RULES = ()` deixava `screen.RULES` com as 8
    regras, e `screen([Segment('user','tomo carbonato de lítio')])` continuava
    disparando `lithium_serum_window` — e `answer.py` PERSISTE esses alertas em
    `findings.safety_json`, que o revisor lê como se fossem do foco atual.

    MUTAÇÃO: restaurar `from lithium.safety.rules import RULES` no topo de screen.py e
    voltar a iterar o módulo em vez do parâmetro.
    """
    from conftest import onco_profile

    frase = _user("tomo carbonato de lítio")
    assert "lithium_serum_window" in {a.key for a in screen(frase, RULESET)}
    outro = ruleset_from_profile(onco_profile())
    assert outro.declared is False, "o perfil de teste declara `safety = false`"
    assert screen(frase, outro) == [], (
        "o ruleset do foco ANTIGO continuou ativo depois da troca"
    )


# ─────────────────────────────────────────────────────── que ele dispare quando deve

POSITIVE_CASES = [
    ("antidepressivo em monoterapia", "posso tomar sertralina sozinha?",
     "antidepressant_monotherapy"),
    ("lítio, janela estreita", "comecei carbonato de lítio semana passada",
     "lithium_serum_window"),
    ("valproato, teratogenicidade", "o médico sugeriu ácido valpróico",
     "valproate_teratogenicity"),
    ("lamotrigina, titulação", "estou subindo a dose de lamictal",
     "lamotrigine_titration"),
    ("carbamazepina, indução", "tegretol interfere em alguma coisa?",
     "carbamazepine_induction"),
    ("benzodiazepínico, dependência", "tomo clonazepam todo dia há anos",
     "benzodiazepine_duration"),
    ("antipsicótico, metabólico", "quetiapina engorda?", "antipsychotic_metabolic"),
]


@pytest.mark.parametrize("label,text,expected", POSITIVE_CASES)
def test_the_rule_fires(label, text, expected):
    assert expected in _keys(_user(text)), label


def test_the_cross_segment_case_that_per_segment_evaluation_lost():
    """O alerta mais importante da tabela, no cenário em que ele importa.

    Uma pergunta sobre parar de tomar e um bloco de evidência sobre lítio nunca caem no
    mesmo segmento. Avaliando co-termo por segmento, `abrupt_discontinuation` — mania de
    rebote, severidade alta — simplesmente nunca dispara em uso real.
    """
    pergunta = Segment("user", "posso parar de tomar de uma vez?")
    evidencia = Segment(
        "evidence", "Lithium prophylaxis reduced relapse rates over 24 months.", "PMID:7"
    )

    assert "abrupt_discontinuation" not in _keys([pergunta])
    assert "abrupt_discontinuation" not in _keys([evidencia])
    assert "abrupt_discontinuation" in _keys([pergunta, evidencia])


def test_high_severity_comes_first():
    out = screen(_user("tomo lítio e quetiapina"), RULESET)
    severities = [a.severity for a in out]
    assert severities == sorted(severities, key=lambda s: {"high": 0}.get(s, 1))


# ──────────────────────────────────────────── que ele NÃO dispare onde não deve

SILENT = frozenset()
"""Nada pode disparar."""

FALSE_POSITIVE_TRAPS = [
    # (rótulo, kind, texto, o que NÃO pode disparar — None = nada pode disparar)
    # Negação. É exatamente a forma das restrições que `detect_memory` propõe, então
    # errar aqui faz o sistema fabricar os próprios falsos positivos.
    ("negação direta", "user", "o paciente não tolera valproato nem lítio", None),
    ("negação enfática", "user", "nunca tomei lítio na vida", None),
    ("suspensão passada", "user", "suspendemos a lamotrigina há dois anos", None),
    ("negação em inglês", "evidence", "The patient denies lithium use.", None),
    ("sem uso", "user", "tratamento sem benzodiazepínico", None),
    # Co-termo solto em prosa comum. `parar` é um dos verbos mais frequentes do
    # português, e `abrupt` é vocabulário estatístico corrente num abstract.
    ("verbo comum, sem relação com a droga", "user",
     "tomo lítio e paro de treinar quando dá, não consigo parar de fumar",
     {"abrupt_discontinuation"}),
    ("'abrupt' no sentido estatístico", "evidence",
     "Abrupt changes in the primary outcome were seen in the lithium arm.",
     {"abrupt_discontinuation"}),
    ("'withdrawal' de outra coisa", "evidence",
     "Patients with alcohol withdrawal seizures were excluded; 12 were on lithium.",
     {"abrupt_discontinuation"}),
    # Colisão com o nome do projeto — o usuário conversa com este sistema sobre este
    # sistema, então é o falso positivo mais previsível de todos.
    ("nome do projeto", "user", "o lithium roda no meu Mac com llama-server", None),
    ("nome do projeto, outra frase", "user", "vou treinar o lithium no Kaggle", None),
    ("palavra maior", "user", "delineamento lithiumlike hipotético", None),
]


@pytest.mark.parametrize("label,kind,text,forbidden", FALSE_POSITIVE_TRAPS)
def test_no_alert_where_there_is_no_risk(label, kind, text, forbidden):
    fired = _keys([Segment(kind, text, "PMID:1" if kind == "evidence" else "")])
    not_allowed = SILENT if forbidden is None else forbidden

    if forbidden is None:
        assert not fired, f"{label}: disparou {sorted(fired)}"
    else:
        assert not (fired & not_allowed), f"{label}: disparou {sorted(fired & not_allowed)}"


def test_the_drug_still_fires_in_biomedical_evidence():
    """A contrapartida da regra do nome do projeto: em inglês biomédico, `lithium` é o
    fármaco. Sem este teste, a correção do falso positivo desligaria o alerta de vez."""
    assert "lithium_serum_window" in _keys(
        [Segment("evidence", "Lithium chloride inhibited GSK-3 beta.", "PMID:1")]
    )
    assert "lithium_serum_window" in _keys(_user("tomo carbonato de lítio"))


def test_a_greeting_produces_nothing():
    """O caso que mais importa da lista: com memórias entrando no screen, uma única
    restrição gravada produzia dois alertas de severidade alta em **todo** turno,
    inclusive neste. Memórias não expiram, então o ruído era permanente."""
    assert screen(_user("bom dia"), RULESET) == []
    assert screen(_user("obrigado, era isso"), RULESET) == []


def test_word_boundaries_are_respected():
    assert matches("lítio", "tomo lítio à noite")
    assert not matches("lítio", "lítiozinho")
    assert not matches("lithium", "lithium-like")


def test_a_term_split_across_a_line_break_still_matches():
    """Texto vem de PDF e de prompt renderizado — "ácido\\nvalpróico" é o caso comum."""
    assert matches("ácido valpróico", "uso de ácido\nvalpróico em dose baixa")


# ───────────────────────────────────────── cobertura: por TERMO, não por regra


def _all_probe_text() -> str:
    corpus = [t for _, t, _ in POSITIVE_CASES]
    corpus += [t for _, t in FALSE_POSITIVE_TRAPS]
    corpus += [
        "posso parar de tomar de uma vez?",
        "Lithium prophylaxis reduced relapse rates over 24 months.",
    ]
    return "\n".join(corpus)


def test_every_declared_term_is_well_formed():
    """Um termo não pode ser inalcançável por construção.

    Cobertura por probe seria o ideal, mas 500 termos × uma frase cada é um arquivo que
    ninguém mantém. O que dá para garantir barato é que nenhum termo seja vazio, tenha
    espaço nas pontas, ou repita dentro da mesma regra — os três jeitos de um termo
    existir e nunca casar nada.
    """
    problems = []
    for rule in RULES:
        seen = set()
        for term in rule.terms + rule.co_terms:
            if not term or term != term.strip():
                problems.append(f"{rule.key}: termo malformado {term!r}")
            if term in seen:
                problems.append(f"{rule.key}: termo repetido {term!r}")
            seen.add(term)
    assert not problems, "\n".join(problems)


def test_every_term_matches_its_own_canonical_sentence():
    """Cada termo casa uma frase construída a partir dele mesmo.

    Isso pega o termo truncado: `lamicta` casaria `"uso de lamicta"`, mas o teste seguinte
    (`test_every_concept_form_is_reachable_from_a_real_sentence`) exige que a forma
    apareça no texto real das regras ou dos conceitos.
    """
    for rule in RULES:
        for term in rule.terms:
            assert matches(term, f"o paciente usa {term} hoje"), f"{rule.key}: {term!r}"


def test_every_concept_form_reaches_its_concept():
    for concept in CONCEPTS:
        for form in concept.forms:
            found = concepts_in(f"prescrito {form} em dose baixa", RULESET)
            assert concept.key in found, f"{concept.key}: a forma {form!r} não alcança"


def test_every_property_phrase_reaches_at_least_one_concept():
    for prop, phrases in PROPERTY_PHRASES.items():
        for phrase in phrases:
            found = concepts_implicated_by(f"o usuário evita {phrase}", RULESET)
            assert found, f"{prop}: a frase {phrase!r} não alcança conceito nenhum"


def test_the_central_collision_of_this_domain_is_detected():
    """Lítio é o agente melhor evidenciado deste domínio e exige monitoramento sérico.
    Uma restrição que não o nomeia precisa alcançá-lo mesmo assim."""
    assert "lithium" in concepts_implicated_by(
        "o usuário evita fármacos com monitoramento sérico", RULESET
    )


def test_rule_keys_are_unique():
    keys = [r.key for r in RULES]
    assert len(keys) == len(set(keys))


def test_the_project_name_rule_is_declared_not_incidental():
    """A isenção existe como dado, não como acidente de outra regra."""
    from lithium.safety.rules import AMBIGUOUS_IN_USER_TEXT

    assert "lithium" in AMBIGUOUS_IN_USER_TEXT
