"""Contratos da camada de prompts: a invariante de domínio e golden renders.

Dois testes com propósitos diferentes, e nenhum deles trava texto literal por acidente.

**A invariante trava o conceito, não a string.** `chat.md` escreve "manic-switch"
(adjetivo composto), `extract_claims.md` escreve "manic switch" (frase nominal) e
`critique_speculation.md` escreve "switch risk". As três estão gramaticalmente corretas,
e um `assert "manic-switch" in prompt` sobre uma delas transforma edição legítima em
falha de teste — enquanto deixa passar a única mudança que importa, que é o risco de
virada **desaparecer** de um prompt que gera conteúdo substantivo.

**Os golden renders existem por causa do item 13.** Enxugar a camada de prompts é uma
reescrita de 710 linhas sem baseline: hoje não há como distinguir "reformatei um
parágrafo" de "removi uma regra". A normalização aqui colapsa espaço em branco *dentro*
do parágrafo e preserva a fronteira *entre* parágrafos — reformatar é livre, mudar
palavra falha. Sem isso, a primeira coisa que a reescrita faz é quebrar todos os testes,
e a reação natural é regenerar tudo em bloco, que é justamente perder o baseline.

Regenerar depois de uma mudança intencional:

    LITHIUM_UPDATE_GOLDEN=1 ./.venv/bin/python -m pytest tests/test_prompt_contract.py

O valor está no **diff**, não no arquivo. (A queixa antiga de que `lithium/` estava
fora do git está OBSOLETA — os 14 prompts e os 14 goldens estão rastreados. Deixá-la
aqui fazia o próximo leitor duvidar do baseline no momento em que ele mais importa.)

**A semeadura tem DOIS regimes, e a distinção é o que fez a Fase B custar 64 linhas de
baseline em vez de 458.** Placeholder de RUNTIME é semeado com sentinela — o golden
precisa mostrar de imediato se o diff está no texto estático ou no bloco injetado.
Placeholder de PERFIL é semeado com o VALOR do perfil de produção, porque os blocos de
perfil são transcrição byte a byte do que estava hard-coded: renderizá-los com o valor
real reproduz o arquivo anterior. E o golden passa a policiar algo que NADA policiava —
a fidelidade da transcrição. Editar a prosa de `partial` no focus.toml quebra
`tests/golden/prompts/extract_claims.md`.

**A invariante de domínio virou contrato de FOCO.** `REQUIRES_STANDING_RISK` mantém as
seis chaves e os motivos LITERAIS; só o RISCO vira dado, vindo do perfil ativo. Derivar
as duas metades do perfil produziria a identidade "os prompts que o perfil manda injetar
risco injetam risco", que passa com o risco em qualquer prompt e em NENHUM.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from string import Template

import pytest

from lithium.focus import PROMPTS_WITH_PROFILE_BLOCKS
from lithium.llm.prompts import PROMPTS_DIR, render

from conftest import onco_profile, prod_profile

PROD = prod_profile()
ONCO = onco_profile()

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "prompts"
UPDATE_GOLDEN = os.environ.get("LITHIUM_UPDATE_GOLDEN") == "1"

ALL_PROMPTS = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))

# ────────────────────────────────────────────────────── a invariante de domínio

REQUIRES_STANDING_RISK = {
    "answer_question": (
        "é a resposta que o revisor lê, com citação — é conteúdo substantivo tanto "
        "quanto o chat"
    ),
    "chat": "responde ao usuário sobre conduta possível",
    "extract_claims": "decide o que entra no corpus, incluindo `directness`",
    "generate_questions": "define o que o sistema vai pesquisar",
    "generate_speculation": "propõe mecanismos novos, sem precedente na literatura",
    "critique_speculation": (
        "é o portão adversarial. A linha que calibra 'nunca discute risco de virada' "
        "como lacuna e não como defeito é o que impede a crítica de rejeitar 100% das "
        "especulações — regressão já observada em operação real."
    ),
}

EXEMPT = {
    "classify_question": "roteia por tipo de pergunta; não emite conteúdo clínico",
    "judge_sufficiency": (
        "julga se a evidência recuperada responde a pergunta. Nomeia a população do "
        "foco — agora por `$target_prose` e `$sufficiency_population_rule`, não mais "
        "hard-coded — porque escopo de população é o que ele policia, mas não emite "
        "conteúdo clínico nem escreve a resposta."
    ),
    "detect_memory": "extrai preferência de uma mensagem; agnóstico ao domínio",
    "reflect": "abstrai sobre a atividade recente; agnóstico ao domínio",
    "reground_chain": "checa se elos da cadeia têm PMID; não julga mérito clínico",
    "speculation_queries": "traduz uma cadeia em queries de busca",
    "verify_pattern": (
        "julga se uma abstração segue das claims citadas. Nomeia a hierarquia de "
        "população do foco — agora por `$population_hierarchy`, e a duplicata que "
        "estava em `schemas.py` (indo para a GRAMÁTICA) saiu no mesmo commit — porque "
        "escopo de população é o que ele policia, mas não emite conteúdo clínico."
    ),
    "judge_directness": (
        "julga a aderência de UMA claim ao alvo, em isolamento. Recebe `$target` e as "
        "definições de directness DO PERFIL, então é o prompt mais dependente de foco "
        "do repo — e mesmo assim não emite conteúdo clínico nem levanta risco: ele "
        "responde um enum de quatro valores e uma frase de justificativa. Injetar "
        "risco permanente aqui seria contexto que o julgamento existe para não ter."
    ),
    "verify_citation": (
        "julga citação em isolamento, de propósito. Contexto de domínio aqui é "
        "contraindicado: o portão existe para não deixar o modelo inferir o que o "
        "resto do paper provavelmente diz."
    ),
    "recon_triage": (
        "classifica snippets de busca web num enum de quatro valores. Recebe "
        "`$target_prose` porque precisa saber sobre o que é o projeto para dizer o "
        "que é ruído, mas não emite conteúdo clínico nenhum: a saída é `index` + "
        "`kind`. Injetar risco permanente aqui seria pagar tokens de janela num passo "
        "que não escreve prosa."
    ),
    "recon_observe": (
        "resume UMA página da web aberta. O prompt PROÍBE avaliar e proíbe "
        "acrescentar conhecimento externo — o produto é 'a página diz X', nunca uma "
        "recomendação. Uma descoberta jamais vira claim, então nada daqui alcança o "
        "corpus; o que alcança o prompt de chat é a nota `recon`, e é `chat.md` (que "
        "ESTÁ em REQUIRES_STANDING_RISK) que carrega o risco permanente quando o "
        "modelo for usá-la."
    ),
}


@pytest.mark.parametrize("name", sorted(REQUIRES_STANDING_RISK))
@pytest.mark.parametrize("profile", [PROD, ONCO], ids=["producao", "onco-vet"])
def test_substantive_prompts_name_the_standing_risk(name: str, profile) -> None:
    """Metade DADO, metade LITERAL — e só uma das duas pode virar dado.

    O CONJUNTO dos seis prompts e o MOTIVO de cada um ficam literais aqui; o RISCO vem
    do perfil ativo. Derivar as duas metades do perfil produziria a identidade "os
    prompts que o perfil manda injetar risco injetam risco", que passa com o risco em
    qualquer prompt e em NENHUM — o defeito tautológico que este repo já cometeu.

    A asserção é sobre o RENDER, não sobre o `.md` cru. Depois da parametrização o
    arquivo não contém mais o risco: ele chega por `$standing_risks` (chat,
    answer_question), por `$missing_risk_line` (critique_speculation) ou dentro da
    prosa de vínculo clínico que o perfil declara (extract_claims,
    generate_questions, generate_speculation).

    RODA COM OS DOIS PERFIS de propósito. Com o de produção sozinho, dois destes seis
    ficavam verdes por TEXTO ESTÁTICO RESIDUAL — reverter a fiação não mudava o
    resultado, que é a definição de fiação-não-testada. Com o perfil de oncologia o
    texto residual não casa o pattern de lise tumoral, então só passa quem de fato lê
    o perfil.
    """
    rendered = re.sub(r"\s+", " ", _render_with_profile(name, profile))
    patterns = profile.risk_patterns()
    assert patterns, "perfil de teste precisa declarar ao menos um risco permanente"
    assert any(rx.search(rendered) for rx in patterns), (
        f"{name}.md não nomeia mais o risco permanente do foco "
        f"{profile.slug!r}. Este prompt precisa nomeá-lo porque "
        f"{REQUIRES_STANDING_RISK[name]}."
    )


def test_every_standing_risk_rejects_its_own_anti_corpus() -> None:
    """O ANTI-CORPO, e ele é obrigatório e não-vazio.

    Sucessor genérico de `test_switch_risk_pattern_rejects_unrelated_switch_wording`.
    Sem ele, `standing_risks = [{name='x', pattern='risk'}]` passa a carga, casa em 14
    de 14 prompts, e a trava dos seis vira verde cobrindo NADA — o mesmo vazio do
    bloco de segurança sem regra, transplantado para dentro do teste.

    MUTAÇÃO: permitir `must_not_match = []` em `StandingRisk`.
    """
    for profile in (PROD, ONCO):
        for risk in profile.focus.standing_risks:
            rx = re.compile(risk.pattern, re.I)
            assert risk.must_not_match, f"{risk.name}: anti-corpo vazio"
            assert rx.search(re.sub(r"\s+", " ", risk.prose)), (
                f"{risk.name}: o pattern não casa a própria prose"
            )
            for negative in risk.must_not_match:
                assert not rx.search(negative), (
                    f"{risk.name}: o pattern casa o anti-corpo {negative!r} — "
                    f"largo demais, cobriria qualquer prompt"
                )


def test_every_prompt_is_classified() -> None:
    """Um prompt novo não pode entrar sem uma decisão explícita sobre o risco de virada.

    Esta é a metade que importa: a parametrização acima só cobre o que já foi
    classificado, então sozinha ela é cega ao caso real — alguém adiciona
    `generate_protocol.md`, ele gera conteúdo substantivo, e nenhum teste nota.
    """
    classified = set(REQUIRES_STANDING_RISK) | set(EXEMPT)
    assert set(ALL_PROMPTS) == classified, (
        "classifique o prompt em REQUIRES_STANDING_RISK ou EXEMPT (com o motivo): "
        f"não classificados={sorted(set(ALL_PROMPTS) - classified)}, "
        f"classificados que não existem mais={sorted(classified - set(ALL_PROMPTS))}"
    )


# ─────────────────────────────────────────────────────────────── golden renders

WRAP_AT = 88
_PARAGRAPH = re.compile(r"\n[ \t]*\n+")


def normalize(text: str) -> str:
    """Colapsa espaço em branco dentro do parágrafo, preserva a fronteira entre eles."""
    blocks = (re.sub(r"\s+", " ", b).strip() for b in _PARAGRAPH.split(text.strip()))
    return "\n\n".join(b for b in blocks if b)


def readable(normalized: str) -> str:
    """Reflui para o disco, para o golden ser legível e diffável.

    `break_long_words` e `break_on_hyphens` desligados não são estilo: com eles ligados
    o wrap parte `manic-switch` em duas linhas, `normalize` recupera `manic- switch`, e
    a normalização deixa de ser idempotente — o arquivo passa a divergir de si mesmo.
    """
    parts = (
        textwrap.fill(b, WRAP_AT, break_long_words=False, break_on_hyphens=False)
        for b in normalized.split("\n\n")
    )
    return "\n\n".join(parts) + "\n"


def _sentinels(name: str) -> dict[str, str]:
    """Um valor visualmente distinto por placeholder.

    Sentinela em vez de dado realista de propósito: o golden precisa mostrar de imediato
    se o diff está no texto estático do prompt ou no bloco injetado.
    """
    template = Template((PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8"))
    names = {
        m.group("named") or m.group("braced")
        for m in template.pattern.finditer(template.template)
        if m.group("named") or m.group("braced")
    }
    return {n: f"«{n}»" for n in names}


def _render_with_profile(name: str, profile) -> str:
    """Sentinela para dado de RUNTIME, valor real para bloco de PERFIL.

    Esta distinção é o que faz a parametrização custar ZERO linha de baseline. Semear
    tudo com sentinela apagaria 458 das 849 linhas de golden — mas essa perda vinha da
    CONVENÇÃO de semeadura, não da parametrização: os blocos de perfil de produção são
    transcrição byte a byte do que estava hard-coded, então renderizá-los com o valor
    real reproduz o arquivo anterior.

    E o golden passa a policiar algo que NADA policiava antes: a FIDELIDADE DA
    TRANSCRIÇÃO. Editar a prosa de `partial` no focus.toml quebra
    `tests/golden/prompts/extract_claims.md` — que é exatamente o buraco pelo qual o
    significado dos quatro níveis de directness poderia mudar sem nenhum teste notar.
    """
    blocks = profile.prompt_blocks(name)
    return render(name, **{**_sentinels(name), **blocks})


def test_normalization_is_idempotent() -> None:
    """`normalize(readable(x)) == x`, senão o golden falha contra si mesmo.

    Com SENTINELA em tudo, de propósito: é a normalização que está sob teste aqui, e
    ela não pode depender do conteúdo de um perfil.
    """
    for name in ALL_PROMPTS:
        once = normalize(render(name, **_sentinels(name)))
        assert normalize(readable(once)) == once, f"round-trip perdeu conteúdo em {name}"


@pytest.mark.parametrize("name", ALL_PROMPTS)
def test_golden_render(name: str) -> None:
    rendered = normalize(_render_with_profile(name, PROD))
    path = GOLDEN_DIR / f"{name}.md"

    if UPDATE_GOLDEN or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readable(rendered), encoding="utf-8")
        if not UPDATE_GOLDEN:
            pytest.skip(f"golden criado: {path.name} — revise o diff e commite")
        return

    assert rendered == normalize(path.read_text(encoding="utf-8")), (
        f"{name}.md divergiu do golden. Se a mudança é intencional: "
        f"LITHIUM_UPDATE_GOLDEN=1 pytest tests/test_prompt_contract.py -k {name}"
    )


INJECTS = {
    "chat": {"constraint_notes", "evidence", "memories", "recon_notes", "safety",
             "standing_risks", "target_prose", "reader"},
    "classify_question": {"question"},
    "critique_speculation": {
        "chain", "falsifier", "fatal_criteria", "intervention_class", "known_risks",
        "mechanism_target", "missing_risk_line", "novelty", "statement",
    },
    "detect_memory": {"existing", "message"},
    "extract_claims": {"design", "directness_definitions", "extract_bind", "journal",
                       "sample_n", "target", "target_prose", "text", "title", "year"},
    "generate_questions": {"max_questions", "question_bind", "question_scarcity",
                           "state", "target_prose_cap"},
    "generate_speculation": {
        "existing", "lessons", "max_items", "mechanistic_question", "reader", "routes",
        "route_rationale", "speculation_problem", "state", "taxonomy",
    },
    "judge_directness": {"claim", "directness_definitions", "evidence", "target"},
    "reflect": {"activity", "lessons"},
    "reground_chain": {"chain", "evidence"},
    "speculation_queries": {
        "chain", "combination", "intervention_class", "mechanism_target",
        "route", "sources", "statement", "test_proposal",
    },
    "verify_citation": {"quote", "statement"},
    "verify_pattern": {"claims", "pattern", "population_hierarchy"},
    "answer_question": {"evidence", "question", "reader", "standing_risks"},
    "judge_sufficiency": {"evidence", "question", "reader", "sufficiency_population_rule",
                          "sufficiency_scarcity", "target_prose"},
    "recon_triage": {"query", "results", "target_prose"},
    "recon_observe": {"target_prose", "text", "title", "truncation_note", "url"},
}
"""O contrato de injeção, escrito como dado literal.

Precisa ser literal. A tentação é derivar isto do próprio `.md`, e um teste assim **não
pode falhar**: apagar `$evidence` de `chat.md` remove o placeholder *e* a expectativa
dele no mesmo movimento. `Template.substitute` levanta em kwarg faltando mas ignora
kwarg sobrando, então o turno renderizaria sem bloco de evidência — o modelo respondendo
de memória paramétrica num domínio onde toda afirmação deve ter PMID — e nada reclamaria.

Desde a Fase B `render` também RECUSA kwarg que sobra, o que fecha a direção inversa.
Este dado continua literal: é a única trava que impede um `$placeholder` novo de entrar
despercebido, e é a que falha ALTO e CEDO.
"""


RUNTIME_PLACEHOLDERS = {
    name: injected - set(PROD.prompt_blocks(name))
    for name, injected in INJECTS.items()
}
"""O complemento: o que NÃO é bloco de perfil é dado de runtime, e o call site é quem
o fornece. Derivado de `INJECTS` (literal) menos `prompt_blocks` (literal): as duas
pontas continuam independentes do `.md`."""


@pytest.mark.parametrize("name", ALL_PROMPTS)
def test_injection_contract(name: str) -> None:
    found = set(_sentinels(name))
    assert found == INJECTS[name], (
        f"{name}.md mudou os blocos que injeta. Removidos={sorted(INJECTS[name] - found)}, "
        f"novos={sorted(found - INJECTS[name])}. Se for intencional, atualize INJECTS "
        f"e confira o call site: um bloco removido não é erro em lugar nenhum."
    )


@pytest.mark.parametrize("name", ALL_PROMPTS)
def test_declared_placeholders_survive_rendering(name: str) -> None:
    """E o que `INJECTS` declara realmente aparece na saída."""
    rendered = render(name, **{p: f"«{p}»" for p in INJECTS[name]})
    for placeholder in INJECTS[name]:
        assert f"«{placeholder}»" in rendered, f"{name}.md não injeta mais ${placeholder}"


@pytest.mark.parametrize("name", ALL_PROMPTS)
def test_every_profile_placeholder_has_a_block(name: str) -> None:
    """Todo `$placeholder` de perfil declarado no `.md` é FORNECIDO por `prompt_blocks`.

    As duas pontas são independentes: o `.md` declara, o mapa literal em
    `FocusProfile.prompt_blocks` fornece. Derivar uma da outra daria um teste que não
    pode falhar — o mesmo argumento do docstring de `INJECTS`.

    Sem isto, acrescentar `$novo_campo` a um prompt e esquecer o bloco só apareceria
    como `KeyError` no primeiro POST em produção, dentro de um handler, horas depois.
    """
    declared = set(_sentinels(name))
    supplied = set(PROD.prompt_blocks(name))
    assert supplied <= declared, (
        f"prompt_blocks({name!r}) fornece {sorted(supplied - declared)}, que "
        f"{name}.md não declara — `render` recusa kwarg que sobra."
    )
    missing = declared - supplied - RUNTIME_PLACEHOLDERS[name]
    assert not missing, (
        f"{name}.md declara {sorted(missing)} e ninguém fornece. Ou é bloco de perfil "
        f"(acrescente a `FocusProfile.prompt_blocks`) ou é dado de runtime "
        f"(acrescente a RUNTIME_PLACEHOLDERS e confira o call site)."
    )
    if supplied:
        assert name in PROMPTS_WITH_PROFILE_BLOCKS
