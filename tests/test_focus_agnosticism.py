"""O maquinário não pode nomear o domínio de nenhum foco.

O domínio vive em `focuses/<slug>/*.toml`. Prompts, código e moldes são neutros: o mesmo
sistema tem de servir a qualquer assunto.

A varredura deriva o vocabulário DO PERFIL, e isso não é elegância — é o que impede o
teste de ser tautológico. A primeira versão desta auditoria usava um padrão que eu digitei
à mão (`bipolar|lithium|quetiapin|...`), declarou os prompts limpos, e tinha deixado
`lamotrigina` passar porque eu não a incluí. Um teste cuja expectativa vem da mesma cabeça
que escreveu o código policia a memória do autor, não o código.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
PROMPTS = RAIZ / "lithium" / "llm" / "prompts"

GENERICOS = frozenset({
    "combination", "timing", "sequencing", "clinical", "phase 1", "repurposing",
    "device", "augmentation", "monotherapy", "adjunctive", "maintenance",
    # radical de inglês corrente ("repurposed, untested"), não nome de coisa
    "repurpos",
})
"""Palavras do vocabulário do perfil que também são inglês corrente.

`clinical` está aqui e é a admissão mais desconfortável da lista: ela PRESUME um domínio
de saúde. Fica isenta porque removê-la de expressões como "clinical outcome" exigiria
reescrever a explicação inteira de vários prompts, e o ganho é menor que o risco de
estragar prosa que já foi medida. É dívida declarada, não descuido."""


NOME_DO_PROJETO = frozenset({
    "lithium", "lítio", "litio", "lithium carbonate", "carbonato de lítio",
    "li2co3", "carbolitium",
})
"""O pacote SE CHAMA `lithium`, e o foco de produção tem um fármaco com o mesmo nome.

Isento porque `lithium/db/store.py`, `getLogger("lithium.x")` e o texto de ajuda
`lithium sources --spec` são o NOME DA FERRAMENTA. É a mesma colisão que
`AMBIGUOUS_IN_USER_TEXT` documenta em `safety/rules.py`.

CUSTO ACEITO E DECLARADO: enquanto o projeto se chamar assim, esta varredura é cega a um
vazamento real do fármaco lítio no código. Não há saída melhor sem renomear o pacote, e o
resto do vocabulário (7 conceitos, dezenas de formas) continua coberto."""

GENERICOS_ESTRUTURAIS = frozenset({"discontinu"})
"""Radicais do perfil que também nomeiam MECANISMO deste sistema.

`discontinu` é co-termo de `abrupt_discontinuation` no TOML e é também o nome do campo
`Rule.discontinuation`, que é maquinário agnóstico — a regra declara que verbo de parada é
o gatilho dela, e isso vale para qualquer domínio que tenha algo que se interrompe."""


def _vocabulario(slug: str) -> set[str]:
    """Os termos que ESTE foco declara. Nomes de fármaco, classes, conceitos de risco."""
    prof = RAIZ / "focuses" / slug
    vocab: set[str] = set()
    safety = tomllib.loads((prof / "safety.toml").read_text(encoding="utf-8"))
    for c in safety.get("concepts", []):
        vocab.update(f.lower() for f in c.get("forms", []) if len(f) > 3)
    for r in safety.get("rules", []):
        vocab.update(t.lower() for t in r.get("terms", []) if len(t) > 3)
    tax = tomllib.loads((prof / "taxonomy.toml").read_text(encoding="utf-8"))
    for c in tax.get("intervention_classes", []):
        vocab.update(k.lower() for k in c.get("keywords", []) if len(k) > 3)
    return vocab - GENERICOS - NOME_DO_PROJETO - GENERICOS_ESTRUTURAIS


FOCOS = sorted(p.name for p in (RAIZ / "focuses").iterdir()
               if p.is_dir() and not p.name.startswith("_"))


@pytest.mark.parametrize("slug", FOCOS)
def test_no_prompt_names_a_focus_vocabulary(slug: str) -> None:
    """Um prompt que nomeia fármaco de um foco raciocina sobre ele em todos os outros.

    O domínio chega ao prompt por `$target` e pelos blocos do perfil, nunca por literal.
    Este é o mesmo defeito que `scaffold_pending` impede no TOML, e ele sobreviveu no
    prompt até esta varredura: `classify_question.md` AFIRMAVA o domínio na primeira
    frase, sem placeholder nenhum.

    MUTAÇÃO: pôr `lamotrigina` de volta em `detect_memory.md`, ou trocar `$target` de
    `classify_question.md` pela frase antiga.
    """
    vocab = _vocabulario(slug)
    assert vocab, f"{slug}: perfil sem vocabulário — a varredura não cobriria nada"
    ofensas: dict[str, list[str]] = {}
    for f in sorted(PROMPTS.glob("*.md")):
        txt = f.read_text(encoding="utf-8").lower()
        hits = sorted({v for v in vocab if re.search(rf"\b{re.escape(v)}", txt)})
        if hits:
            ofensas[f.name] = hits
    assert not ofensas, (
        f"prompts nomeiam vocabulário do foco {slug!r}: {ofensas}\n"
        f"O domínio entra por $target e pelos blocos do perfil, nunca por literal."
    )


def _codigo_executavel(path: Path) -> str:
    """Só o que RODA: literais de string que não são docstring, mais identificadores.

    Três exclusões, cada uma por uma razão diferente:

    - **comentários** nunca entram (o AST não os vê), e é deliberado: alguns REGISTRAM
      medições cujos números só significam algo com os termos medidos — "dez restrições
      que alcançam lítio/valproato/carbamazepina ... 10.137 tokens contra 8.192". Apagar
      isso destruiria a evidência em nome da estética.
    - **docstrings** também não, pela mesma razão: são a documentação do porquê.
    - **imports** são pulados porque o pacote SE CHAMA `lithium`. Sem isso a varredura
      acusa `from lithium.db import Store` em 50 arquivos, que é ruído puro — a mesma
      colisão nome-do-projeto × fármaco que `AMBIGUOUS_IN_USER_TEXT` já documenta.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            corpo = getattr(node, "body", None)
            if corpo and isinstance(corpo[0], ast.Expr) and \
                    isinstance(corpo[0].value, ast.Constant) and \
                    isinstance(corpo[0].value.value, str):
                docstrings.add(id(corpo[0].value))
        # um `Expr` de string solto é docstring de atributo (o repo usa MUITO isso)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and \
                isinstance(node.value.value, str):
            docstrings.add(id(node.value))

    pedacos: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            pedacos.append(node.value)
        elif isinstance(node, ast.Name):
            pedacos.append(node.id)
        elif isinstance(node, ast.Attribute):
            pedacos.append(node.attr)
    return "\n".join(pedacos).lower()


@pytest.mark.parametrize("slug", FOCOS)
def test_no_module_names_a_focus_vocabulary(slug: str) -> None:
    """O mesmo para o código que EXECUTA.

    MUTAÇÃO: pôr de volta `query = "bipolar maintenance lithium guideline"` em `cli.py`,
    ou `REFERENCE_PROFILE = "bipolar-tag"`.
    """
    vocab = _vocabulario(slug)
    ofensas: dict[str, list[str]] = {}
    for f in sorted((RAIZ / "lithium").rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        codigo = _codigo_executavel(f)
        hits = sorted({v for v in vocab if re.search(rf"\b{re.escape(v)}", codigo)})
        if hits:
            ofensas[str(f.relative_to(RAIZ))] = hits
    assert not ofensas, (
        f"código executável nomeia vocabulário do foco {slug!r}: {ofensas}\n"
        f"O domínio entra pelo perfil, nunca por literal no código."
    )


def test_the_scaffold_template_is_not_a_real_focus() -> None:
    """`focus --new` copia de um molde neutro, nunca de um foco de produção.

    MUTAÇÃO: `REFERENCE_PROFILE = "bipolar-tag"`.
    """
    from lithium.cli import REFERENCE_PROFILE

    assert REFERENCE_PROFILE.startswith("_"), (
        f"o molde {REFERENCE_PROFILE!r} não tem o prefixo que o marca como não-foco"
    )
    assert REFERENCE_PROFILE not in FOCOS, (
        f"o molde {REFERENCE_PROFILE!r} é um foco real: renomeá-lo quebra `focus --new`, "
        f"e todo foco novo nasce com o domínio dele"
    )
    assert (RAIZ / "focuses" / REFERENCE_PROFILE).is_dir()
