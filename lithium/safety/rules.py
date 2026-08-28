"""Conceitos farmacológicos e regras de segurança — dados, não julgamento.

Duas tabelas, com propósitos diferentes, e a separação importa:

**`CONCEPTS`** mapeia um conceito (`lithium`) para suas formas de superfície
(`lítio`, `carbonato de lítio`, `lithium carbonate`, `Li2CO3`) e suas propriedades
(`serum_monitoring`, `teratogenic`). É a base do casamento determinístico, e é usada em
dois lugares: pela anotação de conflito no chat e pelo screen de segurança.

Por que **conceito → formas**, e não uma lista plana de termos: com lista plana, a
auditoria de omissão fica cega de um jeito específico e grave. Se uma restrição citar
`lítio` e `valproato`, e a resposta mencionar valproato mas **omitir lítio**, uma
verificação contra a lista inteira encontra "valproato" e conclui que a restrição foi
tratada. A evidência mais forte do corpus some sem que nada acuse.

**`RULES`** são as contraindicações. Determinísticas, por casamento de termo, e o ponto
é justamente esse: um casamento de termo não pode ser expulso de um top-k, não pode
decair com o tempo, e não pode ser superado por um score de importância. É a única
recuperação do sistema que **não pode** ser por similaridade — e precisa existir antes
de qualquer termo de importância entrar (Fase 4), porque depois a resposta tentadora
vira "fatos de segurança serão recuperados porque são importantes", que é falso.

**O que este módulo NÃO é.** Não é checagem de interação medicamentosa; não conhece
dose, via, função renal ou hepática; não sabe o que o paciente toma. É um conjunto
pequeno de lembretes sobre o domínio desta pesquisa. A limitação é renderizada junto
com os alertas, não escondida aqui no docstring — ver `screen.render()`.

Dict Python, não YAML: `config.py` usa `tomllib`, PyYAML não é dependência do projeto, e
uma tabela de segurança que exige uma dependência nova é uma tabela que alguém desliga.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ────────────────────────────────────────────────────────────────────── conceitos


@dataclass(frozen=True, slots=True)
class Concept:
    key: str
    label: str
    forms: tuple[str, ...]
    properties: frozenset[str] = field(default_factory=frozenset)


def _c(key: str, label: str, forms: tuple[str, ...], *props: str) -> Concept:
    return Concept(key=key, label=label, forms=forms, properties=frozenset(props))


def concept_by_key(concepts: tuple[Concept, ...]) -> dict[str, Concept]:
    return {c.key: c for c in concepts}


# ─────────────────────────────────────────────────────────────────────── regras


@dataclass(frozen=True, slots=True)
class Rule:
    key: str
    severity: str                      # 'high' | 'medium'
    terms: tuple[str, ...]
    alert: str
    co_terms: tuple[str, ...] = ()
    """Quando presente, a regra só dispara se um co-termo aparecer **em algum lugar do
    turno** — não necessariamente no mesmo segmento. Um usuário perguntando sobre
    interrupção abrupta enquanto o corpus recuperado fala de lítio é exatamente o caso
    que importa, e ele nunca cai num segmento só."""


# ─────────────────────────────────────────────────────────────────── o ruleset


@dataclass(frozen=True, slots=True)
class RuleSet:
    """O vocabulário de segurança de UM foco, ou a AUSÊNCIA declarada dele.

    `declared=False` é o TERCEIRO estado, e ele existe porque MEDI que os dois
    anteriores são BYTE A BYTE IDÊNTICOS: `render(screen([Segment('user','bom dia')]))`
    com as 8 regras carregadas e com `RULES = ()` produz a mesma string, terminando em
    "Ausência de alerta não significa ausência de risco" — uma camada que cobre NADA
    afirmando que cobre, em todo turno de chat, e `answer.py` PERSISTE isso em
    `findings.safety_json` para o revisor ler.

    Objeto passado por PARÂMETRO, nunca constante de módulo: MEDIDO que `rules.RULES =
    ()` deixava `screen.RULES` com as 8 regras (binding de import time), então trocar de
    foco sem reiniciar mantinha o screen do foco ANTIGO ativo e disparando.
    """

    concepts: tuple[Concept, ...] = ()
    rules: tuple[Rule, ...] = ()
    property_phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    declared: bool = True

    @property
    def by_key(self) -> dict[str, Concept]:
        return concept_by_key(self.concepts)


ABSENT = RuleSet(declared=False)
"""`safety = false` no perfil. AUSÊNCIA DECLARADA, não arquivo faltando."""


def ruleset_from_profile(profile) -> RuleSet:
    """Constrói o ruleset a partir do perfil. `safety = false` devolve `ABSENT`."""
    safety = profile.safety
    if safety is None:
        return ABSENT
    concepts = tuple(
        Concept(key=c.key, label=c.label, forms=tuple(c.forms),
                properties=frozenset(c.properties))
        for c in safety.concepts
    )
    by_key = concept_by_key(concepts)
    rules = tuple(
        Rule(
            key=r.key,
            severity=r.severity,
            # A INDIREÇÃO é resolvida aqui: `concepts = ["lithium"]` vira as `forms` do
            # conceito. Reescrever as formas no TOML da regra as duplicaria dentro do
            # mesmo arquivo e elas divergiriam na primeira edição.
            terms=tuple(dict.fromkeys(
                [f for key in r.concepts for f in by_key[key].forms] + list(r.terms)
            )),
            alert=r.alert,
            co_terms=tuple(r.co_terms),
        )
        for r in safety.rules
    )
    return RuleSet(
        concepts=concepts,
        rules=rules,
        property_phrases={k: tuple(v) for k, v in safety.property_phrases.items()},
    )


# ────────────────────────────────────────────────────────────── casamento de termo

_NEGATION = re.compile(
    r"\b(não|nao|nunca|jamais|sem|suspend\w*|interromp\w*|never|denies|denied|"
    r"without|no longer)\b[^.;!?]{0,40}$",
    re.I,
)
"""Contexto imediatamente ANTES do termo. "o paciente não tolera valproato" e "nunca
tomei lítio" não são menções de uso — tratá-las como menção fabrica alerta permanente
justamente sobre as restrições que o detector de memória foi feito para propor."""


def _boundary(term: str) -> re.Pattern[str]:
    # `\s+` entre palavras: o texto vem de PDF e de prompt renderizado, onde quebra de
    # linha no meio de "ácido\nvalpróico" é o caso comum, não a exceção.
    body = r"\s+".join(re.escape(w) for w in term.split())
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.I)


_TERM_CACHE: dict[str, re.Pattern[str]] = {}


def matches(term: str, text: str, *, negation_aware: bool = True) -> bool:
    r"""O termo aparece em `text`, com fronteira de palavra e (opcionalmente) fora de negação.

    **`negation_aware=False` para prosa de síntese**, e isto vem de medição. `_NEGATION`
    foi calibrada na Fase 1 para o registro de *restrição do usuário* — "o paciente não
    tolera valproato" —, onde suprimir é correto. Rodada sobre prosa de síntese, cujo
    registro dominante é hedge e verbo de descontinuação, ela perde **9 de 14** alertas.

    E o caso pior é auto-anulante: `abrupt_discontinuation`, que o `screen` chama de o
    mais importante da tabela, é desligado por `interromp\w*` — ou seja, pelo vocabulário
    que ele existe para pegar. Medido:

        "Pacientes que interromperam o lítio abruptamente tiveram mania de rebote"
            -> abruptamente NÃO casa (negado por "interromperam")
        "A parada foi feita abruptamente no braço ativo"
            -> abruptamente casa

    O parâmetro confina a mudança ao canal novo: o chat continua com supressão.
    """
    pattern = _TERM_CACHE.get(term)
    if pattern is None:
        pattern = _TERM_CACHE[term] = _boundary(term)
    for found in pattern.finditer(text):
        if not negation_aware or not _NEGATION.search(text[:found.start()]):
            return True
    return False


NEGATION_AWARE_KINDS = frozenset({"user", "memory", "evidence"})
"""Onde a supressão por negação vale — ou seja, em tudo menos `reply`.

A medição dos 9 de 14 alertas perdidos foi sobre prosa de **síntese**, e é só ali que a
regra se auto-anula. Em `evidence` o registro é outro: "the patient denies lithium use"
num abstract é história clínica, e disparar o lembrete de janela terapêutica ali é ruído
— e ruído treina o revisor a ignorar o alerta seguinte, que é o pior resultado possível
deste módulo.

Manter `evidence` com supressão é o que confina a mudança ao canal novo de verdade."""


AMBIGUOUS_IN_USER_TEXT = frozenset({"lithium"})
"""Formas que só contam fora de uma mensagem do usuário.

`lithium` é o nome deste projeto. O usuário conversa com este sistema **sobre** este
sistema — "vou rodar o lithium no Kaggle" não é uma menção a carbonato de lítio, e
disparar um lembrete de janela terapêutica ali é o falso positivo mais previsível que
existe aqui. Em texto de evidência, que é inglês biomédico, `lithium` é o fármaco e
continua valendo. `lítio` e as formas qualificadas (`lithium carbonate`, `carbonato de
lítio`) valem em qualquer lugar."""


def terms_for(terms: tuple[str, ...], kind: str) -> tuple[str, ...]:
    if kind != "user":
        return terms
    return tuple(t for t in terms if t not in AMBIGUOUS_IN_USER_TEXT)


def concepts_in(text: str, ruleset: RuleSet) -> set[str]:
    """Quais conceitos o texto nomeia diretamente."""
    return {c.key for c in ruleset.concepts if any(matches(f, text) for f in c.forms)}


def concepts_implicated_by(text: str, ruleset: RuleSet) -> set[str]:
    """Quais conceitos uma restrição em linguagem natural alcança.

    Direto (nomeia "lítio") **ou** por propriedade ("evito monitoramento sérico"
    alcança lítio, valproato e carbamazepina). O segundo caminho é o que faz a colisão
    central deste domínio ser detectável.
    """
    found = concepts_in(text, ruleset)
    for prop, phrases in ruleset.property_phrases.items():
        if any(matches(p, text) for p in phrases):
            found |= {c.key for c in ruleset.concepts if prop in c.properties}
    return found
