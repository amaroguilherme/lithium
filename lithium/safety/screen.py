"""O screen determinístico: anexa alertas, nunca remove nem reordena nada.

A propriedade portante, e ela é a razão de o módulo existir separado: um casamento de
termo **não pode** ser expulso de um top-k, não pode decair, e não pode ser superado por
um score de importância. Toda outra recuperação deste sistema é por similaridade; esta
não pode ser, porque a informação que importa aqui é justamente a que ninguém perguntou.

`screen()` recebe segmentos e devolve `Alert`s. Ele **não recebe permissão de escrita**
sobre nada: não muta a lista que recebe, não devolve segmento, e o chamador renderiza o
resultado num bloco próprio. Um screen que pudesse suprimir evidência seria pior que
nenhum screen.

**Memórias não entram.** Foi uma decisão contra o desenho inicial, e o motivo é medido:
uma única restrição gravada ("o paciente não tolera valproato nem lítio") produzia dois
alertas de severidade alta em **todo** turno subsequente, inclusive em "bom dia" —
memórias não expiram, então o ruído é permanente. E o pior é que restrições exatamente
com essa forma são as que o detector de memória foi construído para propor: o sistema
fabricava os próprios falsos positivos. Falso positivo aqui não é incômodo, é o que
treina a pessoa a ignorar o alerta seguinte, que pode ser o que importava. Memórias já
são renderizadas em `$memories`; o trabalho do screen é o material do turno.
"""

from __future__ import annotations

from dataclasses import dataclass

from lithium.safety.rules import (
    ABSENT,
    NEGATION_AWARE_KINDS,
    Rule,
    RuleSet,
    matches,
    terms_for,
)

DISCLAIMER = (
    "Lembretes determinísticos por casamento de termo. Não é checagem de interação "
    "medicamentosa, não considera dose, via, função renal ou hepática, e não sabe o que "
    "está em uso. Ausência de alerta não significa ausência de risco."
)
"""Renderizado junto com os alertas, sempre. Uma limitação que só existe no docstring é
uma limitação que o leitor do relatório nunca vê."""


@dataclass(frozen=True, slots=True)
class Segment:
    kind: str          # 'user' | 'evidence' | 'reply'
    text: str
    ref: str = ""      # PMID, id — de onde veio, para o alerta poder apontar


@dataclass(frozen=True, slots=True)
class Alert:
    key: str
    severity: str
    text: str
    where: tuple[str, ...]


def screen(segments: list[Segment], ruleset: RuleSet = ABSENT) -> list[Alert]:
    """Devolve os alertas do turno. Não toca em `segments`.

    `terms` é avaliado **por segmento** e `co_terms` sobre a **união** do turno. A
    distinção não é estilo: por-segmento, uma pergunta como "posso parar de tomar de uma
    vez?" nunca cruza com um bloco de evidência sobre lítio, e o alerta de mania de
    rebote — o mais importante da tabela — não dispara justamente no caso em que
    deveria. Avaliar o co-termo sobre o turno inteiro continua monotônico: acrescentar
    segmento só pode acrescentar alerta.
    """
    joined = "\n".join(s.text for s in segments)
    # Co-termo sobre a união do turno, e a supressão por negação vale só se TODO segmento
    # for de um kind onde ela faz sentido: um `reply` na união não pode ser silenciado
    # por uma negação que veio da mensagem do usuário.
    joined_aware = all(s.kind in NEGATION_AWARE_KINDS for s in segments)
    alerts: list[Alert] = []
    for rule in ruleset.rules:
        where = tuple(
            s.ref or s.kind
            for s in segments
            if any(
                # A exceção de descontinuação vale para os TERMOS também, não só para
                # os co-termos: em "interrompi o lítio abruptamente", quem é suprimido é
                # o próprio `lítio` — `interrompi` está a duas palavras dele e dentro da
                # janela de 40 caracteres de `_NEGATION`. Consertar só o co-termo deixava
                # a regra sem disparar do mesmo jeito. MEDIDO nas duas metades.
                matches(t, s.text,
                        negation_aware=(s.kind in NEGATION_AWARE_KINDS
                                        and not rule.discontinuation))
                for t in terms_for(rule.terms, s.kind)
            )
        )
        if not where:
            continue
        if rule.co_terms and not any(
            # Regra cujo GATILHO é a descontinuação nunca aplica supressão de negação
            # aos próprios co-termos: `interromp\w*` e `suspend\w*` estão em `_NEGATION`,
            # e sem esta exceção a regra se desliga com o vocabulário que existe para
            # caçar. MEDIDO: "interrompi o lítio abruptamente" não disparava nada.
            matches(c, joined,
                    negation_aware=joined_aware and not rule.discontinuation)
            for c in rule.co_terms
        ):
            continue
        alerts.append(
            Alert(key=rule.key, severity=rule.severity, text=rule.alert, where=where)
        )
    return _by_severity(alerts)


def _by_severity(alerts: list[Alert]) -> list[Alert]:
    order = {"high": 0, "medium": 1}
    return sorted(alerts, key=lambda a: (order.get(a.severity, 9), a.key))


NO_RULESET = (
    "Este foco NÃO declara nenhuma regra de segurança. Nada foi verificado neste "
    "turno — isto não é um resultado negativo, é a ausência da verificação."
)
"""O TERCEIRO estado. Existe porque MEDI que os outros dois são BYTE A BYTE IDÊNTICOS:
com as 8 regras carregadas e nada casando, e com `RULES = ()`, a saída era a mesma
string, terminando em "Ausência de alerta não significa ausência de risco". Num foco
sem safety.toml isso é literalmente verdadeiro e completamente enganoso — "nenhum termo
casou" e "não existe regra nenhuma para casar" renderizavam igual, em todo turno, e
`answer.py` PERSISTE o resultado em `findings.safety_json` para o revisor ler."""


def render(alerts: list[Alert], *, ruleset_declared: bool) -> str:
    """Bloco de texto para o prompt, em TRÊS estados.

    O disclaimer de COBERTURA só aparece nos dois primeiros. Ele descreve os limites de
    uma verificação que rodou; imprimi-lo quando nenhuma regra existe é uma camada que
    cobre nada afirmando que cobre.
    """
    lines = ["## Safety reminders (deterministic, term-matched)"]
    if not ruleset_declared:
        lines.append(f"  {NO_RULESET}")
        return "\n".join(lines)
    if not alerts:
        lines.append("  (no term matched this turn)")
    else:
        for a in alerts:
            lines.append(f"  [{a.severity}] {a.text}")
            if a.where:
                lines.append(f"      matched in: {', '.join(sorted(set(a.where)))}")
    lines.append(f"  ({DISCLAIMER})")
    return "\n".join(lines)


__all__ = ["Alert", "Rule", "RuleSet", "Segment", "DISCLAIMER", "NO_RULESET",
           "render", "screen"]
