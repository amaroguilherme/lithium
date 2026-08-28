"""Taxonomias abertas de mecanismo e de classe de intervenção — vindas do PERFIL.

Substituem a lista fixa de fármacos que existia antes. A diferença importa: uma lista
de nomes ("quetiapina, lamotrigina, lítio…") só torna visível a lacuna *dentro* dela —
o que não está na lista é invisível para o gerador, e o sistema fica preso à
farmacopeia conhecida.

Aqui as entradas são **classes** e **alvos mecanísticos**, não produtos. O gerador vê
"nenhuma evidência para neuromodulação" ou "ninguém olhou o eixo HPA" e fica livre
para nomear qualquer coisa dentro daquele espaço.

A ordem não é hierarquia. As entradas mais distantes da prática padrão vêm no fim de
propósito, para não sugerirem menor importância — é justamente onde a resposta menos
óbvia deve estar. **Por isso o loader não pode normalizar**: nada de `set`, nada de
ordenar por chave. TOML preserva ordem de array e é ela que chega ao prompt.

**Por PARÂMETRO, nunca por constante de módulo.** Importar qualquer submódulo de
`pipeline` executa `pipeline/__init__.py`, que importa oito módulos — as tabelas eram
construídas em IMPORT TIME, antes de existir Store, Config ou foco ativo. Fazer o
perfil ler o foco do banco aqui criaria `pipeline -> db -> pipeline`, ou pior, abriria
o SQLite como efeito colateral de `import lithium.pipeline` (e `lithium --help`
passaria a abrir o banco). Parâmetro é a única saída que não inverte a dependência.
"""

from __future__ import annotations

from lithium.focus import FocusProfile


def route_block(profile: FocusProfile) -> str:
    """A lista de vias + a parte AGNÓSTICA do argumento.

    A metade de DOMÍNIO (por que a via é mecanisticamente carregada NESTE foco) mora em
    `focus.route_rationale` e é injetada em `generate_speculation` por `$route_rationale`.
    Antes ela estava TRIPLICADA — aqui, no texto estático do mesmo prompt que recebe
    este bloco, e numa descrição de campo de `schemas.py` que vai para a gramática. As
    duas cópias no mesmo prompt podiam se contradizer dentro da mesma janela.

    A sentença agnóstica ("não é embalagem") fica separada da de domínio de propósito:
    juntá-las faria parametrizar a metade de foco matar a trava agnóstica em silêncio.
    """
    routes = "\n".join(f"  - {r}" for r in profile.taxonomy.routes)
    return (
        "### Routes of administration / delivery\n"
        f"{routes}\n\n"
        "Route is a mechanistic variable here, not packaging. Asking 'which compound?' "
        "and 'by which route?' as separate questions opens hypotheses that the single "
        "question closes."
    )


def taxonomy_block(profile: FocusProfile) -> str:
    """Bloco de texto para os prompts, na ordem literal do perfil."""
    mechanisms = "\n".join(f"  - {m}" for m in profile.taxonomy.mechanism_targets)
    classes = "\n".join(
        f"  - {c.label}" for c in profile.taxonomy.intervention_classes
    )
    return (
        "### Mechanistic targets\n"
        f"{mechanisms}\n\n"
        "### Intervention classes\n"
        f"{classes}\n\n"
        "These lists are a starting map, not a menu. Naming something outside them is "
        "welcome and often the point — say so explicitly when you do."
    )
