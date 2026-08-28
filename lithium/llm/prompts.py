"""Carga e renderização de prompts.

Prompts vivem em `prompts/*.md` — arquivos de dados, não strings no código, para
poderem ser iterados sem mexer em Python e diffados de forma legível.

Renderização usa `string.Template` (`$var`) em vez de `str.format` (`{var}`), porque
os prompts contêm exemplos de JSON cheios de chaves — `format` engasgaria em todos.

**Idioma:** os prompts internos são em inglês. A literatura é em inglês e a
terminologia médica de um 12B degrada visivelmente em português. O relatório final
para o usuário é traduzido na camada de report; o pipeline interno não é.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@cache
def _load(name: str) -> Template:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(f"prompt '{name}' não existe. Disponíveis: {available}")
    return Template(path.read_text(encoding="utf-8"))


def declared_placeholders(name: str) -> frozenset[str]:
    """Os `$nomes` que o `.md` realmente usa."""
    tpl = _load(name)
    return frozenset(
        m.group("named") or m.group("braced")
        for m in tpl.pattern.finditer(tpl.template)
        if m.group("named") or m.group("braced")
    )


class UnknownPlaceholder(TypeError):
    """Um kwarg que o template não declara. Ver `render`."""


def render(name: str, /, **kwargs: object) -> str:
    """Renderiza um prompt. Placeholder faltando é erro; kwarg SOBRANDO também.

    `Template.substitute` é assimétrico: levanta `KeyError` em placeholder faltando e
    **ignora em silêncio** o kwarg que sobra. A metade silenciosa é meia-fiação — e ela
    já estava na árvore antes desta fase: `extract.py` passava `target=focus["target"]`
    para um `extract_claims.md` que não declara `$target`, a linha não fazia nada, e a
    suíte inteira passava verde. É o modo de falha que o docstring de `INJECTS`
    descreve, acontecendo na direção inversa.

    A Fase B acrescenta placeholders de perfil a oito prompts, ou seja, comete essa
    classe de erro em série. Uma linha aqui converte a classe inteira de silenciosa em
    ruidosa, e é o que torna a parametrização segura.

    Como o perfil chega por KWARG e não por composição de fragmento, `$` na prosa do
    TOML é INERTE: `substitute` varre o TEMPLATE, nunca os VALORES. Um perfil é um
    arquivo que o usuário edita à mão, e um `$500` num standing_risk não pode derrubar
    o prompt.
    """
    declared = declared_placeholders(name)
    extra = sorted(set(kwargs) - declared)
    if extra:
        raise UnknownPlaceholder(
            f"{name}.md não declara {extra} — o valor seria descartado em silêncio. "
            f"Declarados: {sorted(declared)}. Ou o `.md` perdeu o placeholder, ou o "
            f"call site passa um bloco que este prompt nunca injetou."
        )
    return _load(name).substitute(**kwargs)


CHARS_PER_TOKEN = 4
"""Mesma convenção que `ingest.py` usa para `n_tokens`. Grosseira de propósito: o
objetivo é decidir se cabe na janela, não faturar tokens."""

BUDGET_MARGIN = 256
"""Folga para o template de chat, tokens especiais e a diferença entre esta estimativa
e o tokenizador real. Errar por baixo aqui custa um 400 do llama-server."""


class PromptTooLarge(RuntimeError):
    """O prompt renderizado mais o teto de saída não cabem na janela.

    Falha **antes** do POST, de propósito. Sem isto o llama-server devolve 400, o
    cliente corretamente não repete 4xx, e o `LLMError` sobe sem handler até o
    dead-letter — que é como a trilha especulativa morria em silêncio conforme o
    corpus crescia. A mensagem carrega os números para o diagnóstico ser imediato.
    """


def estimate_tokens(rendered: str) -> int:
    return len(rendered) // CHARS_PER_TOKEN


def budget_guard(
    rendered: str,
    *,
    label: str,
    max_tokens: int,
    n_ctx: int,
    margin: int = BUDGET_MARGIN,
) -> int:
    """Aborta antes da chamada quando entrada + saída não cabem. Devolve a estimativa."""
    estimated = estimate_tokens(rendered)
    if estimated + max_tokens > n_ctx - margin:
        raise PromptTooLarge(
            f"{label}: ~{estimated} tokens de prompt + {max_tokens} de saída "
            f"excedem a janela de {n_ctx} (margem {margin}). "
            f"Algum bloco injetado cresceu sem teto."
        )
    return estimated


def system(name: str, /, **kwargs: object) -> dict[str, str]:
    return {"role": "system", "content": render(name, **kwargs)}


def user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}
