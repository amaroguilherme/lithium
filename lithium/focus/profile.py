"""Carga e validação do perfil de foco (4 arquivos TOML num diretório).

`config.load_config` serve de PADRÃO (tomllib + pydantic + `PROJECT_ROOT`), não de
implementação, e cinco coisas mudam aqui:

1. **Sem fallback para defaults.** `load_config` cai para `Config()` quando o arquivo
   some; um `FocusProfile()` vazio seria um foco fantasma — alvo `""`, zero estratégia,
   zero taxonomia — e o sistema rodaria contra nada. Arquivo ausente é fail-loud.
2. **`extra="forbid"` em TODOS os submodelos.** Sem isso, `standing_risk = []` (typo no
   singular) é ignorado em silêncio, a chave certa fica ausente, e a regra "chave
   ausente é ERRO" não se sustenta — a diferença entre uma decisão e um esquecimento
   desaparece.
3. **Carga de DIRETÓRIO**, não de arquivo: quatro TOML com papéis diferentes.
4. **O erro cita o ARQUIVO.** `tomllib.TOMLDecodeError` sobe sem o nome do arquivo, e
   são quatro candidatos — sem o nome, o usuário abre os quatro.
5. **A ordem dos arrays é CONTEÚDO.** O docstring de `mechanism.py` diz que as entradas
   mais distantes da prática padrão vêm no fim DE PROPÓSITO. Nada aqui pode passar por
   `set` nem ordenar por chave: TOML preserva ordem de array e o loader não normaliza.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithium.types import Directness

Directness4 = tuple(d.value for d in Directness)
"""O vocabulário GLOBAL de níveis. Não é escolha de perfil — ver `_check_directness`."""


class ProfileError(RuntimeError):
    """Erro de carga de perfil. Sempre cita o arquivo: são quatro."""


class Strict(BaseModel):
    """`extra="forbid"` é a metade que faz 'chave ausente é ERRO' valer de verdade."""

    model_config = ConfigDict(extra="forbid")


# ────────────────────────────────────────────────────────────────────── focus.toml


class DirectnessLevel(Strict):
    prose: str = Field(min_length=1)
    """Só a PROSA. `weight` NÃO existe aqui, e a ausência é uma decisão medida.

    As duas portas para um peso de perfil estão fechadas: se a escala já existe, os
    pesos seriam ignorados (reescrevê-los reescreveria julgamento histórico); se a
    escala é nova, ela é semeada de `types.DIRECTNESS_WEIGHT` para que os dois eixos
    fechem. Um campo editável que não tem caminho de execução é o formato mais nocivo
    possível — o mesmo motivo pelo qual `Strategy.tags` não desceu para o disco.
    """


class StandingRisk(Strict):
    """Um risco que o foco levanta SEM ser perguntado.

    `pattern` vem junto com o risco porque sem ele o teste de contrato vira teatro:
    itera a lista, não tem com o que casar, e passa verde com os prompts já sem a
    menção. `must_not_match` é o ANTI-CORPO e é obrigatório e não-vazio: sem ele,
    `pattern = "risk"` casa em 14 de 14 prompts e cobre nada.
    """

    name: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    prose: str = Field(min_length=1)
    must_not_match: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _pattern_compiles_and_discriminates(self) -> StandingRisk:
        try:
            rx = re.compile(self.pattern, re.I)
        except re.error as exc:
            raise ValueError(f"standing_risk {self.name!r}: pattern inválido — {exc}") from exc
        if not rx.search(_flat(self.prose)):
            raise ValueError(
                f"standing_risk {self.name!r}: o pattern não casa a própria prose. "
                f"O contrato de prompt assere sobre o RENDER — um pattern que não casa "
                f"o próprio texto reprova todos os prompts que injetam este risco."
            )
        for negative in self.must_not_match:
            if rx.search(negative):
                raise ValueError(
                    f"standing_risk {self.name!r}: o pattern casa o anti-corpo "
                    f"{negative!r}. Um pattern largo demais deixa a trava dos prompts "
                    f"verde sem cobrir nada."
                )
        return self


class FocusToml(Strict):
    @model_validator(mode="before")
    @classmethod
    def _strip_prose(cls, data):
        """Espaço em volta de um bloco TOML multilinha é FORMATAÇÃO, não conteúdo.

        Sem isto, `\"\"\"\\ntexto\\n\"\"\"` injeta um `\\n` final no meio de um parágrafo do
        prompt, o que vira fronteira de parágrafo na normalização do golden — o
        arquivo passaria a divergir por causa de onde as aspas foram fechadas.
        """
        if isinstance(data, dict):
            return {
                k: (v.strip() if isinstance(v, str) else
                    [i.strip() if isinstance(i, str) else i for i in v]
                    if isinstance(v, list) else v)
                for k, v in data.items()
            }
        return data

    schema_version: Literal[1]
    slug: str = Field(min_length=1)

    scaffold_pending: bool = False
    """`focus --new` copia o perfil de REFERÊNCIA inteiro e só remove `target`.

    Isso é deliberado — um perfil em branco não ensina o formato —, mas cria um estado
    em que preencher só o `target` produz um foco que MENTE. Medido: com
    `target = "fibrilação atrial com insuficiência cardíaca"` e o resto intocado, o
    `extract_claims` renderizado diz *"The system is investigating treatment options for
    bipolar I disorder…"* e define os quatro níveis de directness em populações
    bipolares. O modelo julgaria cardiologia contra uma escada de psiquiatria, e nada
    reclamava: `focus --show` até imprime a incoerência, mas imprimir não é barrar.

    E não é só a prosa do `focus.toml`: `strategies.toml` (19 queries de psiquiatria),
    `taxonomy.toml` e `safety.toml` vêm do mesmo `copytree`. Por isso a marca é UMA e
    cobre os quatro arquivos, em vez de uma sentinela por campo — o usuário a apaga
    quando tiver revisado o perfil, e é essa a afirmação que ela carrega."""

    target: str = Field(min_length=1)
    """O alvo curto — é o que `focus new` escreve em `focuses.target`.

    `min_length=1` não é zelo: `focuses.target` é `TEXT NOT NULL` sem CHECK, o SQLite
    aceita `''`, e o prompt renderiza literalmente `matches ""`. Como o BANCO vence
    para `target`, corrigir o TOML depois não conserta — o relens já teria julgado o
    corpus inteiro contra a string vazia com instrução fail-closed.
    """

    target_prose: str = Field(min_length=1)
    reader: str = Field(min_length=1)
    scale: str = Field(min_length=1)

    mechanistic_question: str = Field(min_length=1)
    retrieval_prefix: str = Field(min_length=1)

    extract_bind: str = Field(min_length=1)
    question_bind: str = Field(min_length=1)
    question_scarcity: str = Field(min_length=1)
    speculation_problem: str = Field(min_length=1)
    sufficiency_scarcity: str = Field(min_length=1)
    sufficiency_population_rule: str = Field(min_length=1)
    population_hierarchy: str = Field(min_length=1)
    route_rationale: str = Field(min_length=1)

    fatal_criteria: list[str]
    directness: dict[str, DirectnessLevel]
    standing_risks: list[StandingRisk]
    """SEM default. Chave ausente reprova a carga; `standing_risks = []` carrega.

    A diferença entre "este foco não tem risco permanente" e "esqueci de declarar" é a
    diferença entre uma decisão e um esquecimento, e só a primeira pode passar."""

    safety: str | bool
    """Caminho do safety.toml, ou `false` para AUSÊNCIA DECLARADA.

    Arquivo faltando sem esta chave é erro de carga. `false` faz o bloco de segurança
    renderizar a ausência em vez de um disclaimer que afirma cobrir o que não cobre."""

    BIND_FIELDS: ClassVar[tuple[str, ...]] = (
        "extract_bind", "question_bind", "speculation_problem")
    """Os três blocos onde o risco permanente TEM de aparecer na prosa.

    Nesses três prompts o risco não chega por `$standing_risks` — ele é parte do
    vínculo clínico que abre o prompt, e separá-lo produziria a mesma frase duas vezes
    na mesma janela. Por isso a obrigação é verificada AQUI, na carga.

    Sem esta validação a trava de contrato dos seis prompts fica meio cega: MEDIDO que,
    com um perfil cujos binds não nomeiam o próprio risco, três dos seis reprovam só na
    suíte, e o caminho de menor resistência diante do vermelho é apagar as chaves de
    `REQUIRES_STANDING_RISK` — deixando sem menção de risco justamente o prompt que
    decide o que entra no corpus."""

    @model_validator(mode="after")
    def _binds_name_a_standing_risk(self) -> FocusToml:
        if not self.standing_risks:
            return self
        patterns = [re.compile(r.pattern, re.I) for r in self.standing_risks]
        for field_name in self.BIND_FIELDS:
            text = _flat(getattr(self, field_name))
            if not any(rx.search(text) for rx in patterns):
                raise ValueError(
                    f"`{field_name}` não nomeia nenhum risco permanente declarado "
                    f"({[r.name for r in self.standing_risks]}). Este bloco abre o "
                    f"prompt e é o único lugar em que o risco chega até ele — sem a "
                    f"menção, o extrator, o gerador de perguntas e o gerador de "
                    f"especulação rodam sem ela e nada no sistema acusa."
                )
        return self

    @model_validator(mode="after")
    def _check_directness(self) -> FocusToml:
        declared, expected = set(self.directness), set(Directness4)
        if declared != expected:
            raise ValueError(
                f"[directness] tem de declarar EXATAMENTE os {len(expected)} níveis "
                f"{sorted(expected)}. Faltando={sorted(expected - declared)}, "
                f"desconhecidos={sorted(declared - expected)}.\n"
                f"O nível é VOCABULÁRIO GLOBAL, não escolha de perfil: "
                f"`claim_directness.directness` tem FK para `directness_weight` e a "
                f"gramática espelha o mesmo conjunto fechado. Um nível a mais passa "
                f"por validação ingênua, entra em `scale_levels` (que não tem FK ali) "
                f"e só explode na PRIMEIRA ESCRITA, como `FOREIGN KEY constraint "
                f"failed` dentro de `_persist`, com o perfil já commitado. "
                f"O perfil escolhe a PROSA; o conjunto de nomes é do código."
            )
        return self


# ─────────────────────────────────────────────────────────────── strategies.toml


class StrategyToml(Strict):
    name: str = Field(min_length=1)
    expected_directness: str
    priority: float = 0.5
    limit: int = 20
    rationale: str = Field(min_length=1)
    queries: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _known_directness(self) -> StrategyToml:
        if self.expected_directness not in Directness4:
            raise ValueError(
                f"estratégia {self.name!r}: expected_directness "
                f"{self.expected_directness!r} não é um nível legal {list(Directness4)}"
            )
        return self


class StrategiesToml(Strict):
    schema_version: Literal[1]
    strategies: list[StrategyToml] = Field(min_length=1)


# ───────────────────────────────────────────────────────────────── taxonomy.toml


class InterventionClass(Strict):
    label: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    """Não-vazio é obrigatório: uma classe sem keyword apareceria como `untouched`
    PARA SEMPRE, porque nada poderia tocá-la. Rótulo e keywords são UMA tabela — hoje
    `INTERVENTION_CLASSES` e as chaves de `CLASS_KEYWORDS` são duas fontes de verdade
    da mesma taxonomia que ainda não divergiram, e nenhum teste as cruza."""


class TaxonomyToml(Strict):
    schema_version: Literal[1]
    mechanism_targets: list[str] = Field(min_length=1)
    routes: list[str] = Field(min_length=1)
    intervention_classes: list[InterventionClass] = Field(min_length=1)


# ─────────────────────────────────────────────────────────────────── safety.toml


class ConceptToml(Strict):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    forms: list[str] = Field(min_length=1)
    properties: list[str] = []


class RuleToml(Strict):
    key: str = Field(min_length=1)
    severity: Literal["high", "medium"]
    alert: str = Field(min_length=1)
    concepts: list[str] = []
    """INDIREÇÃO, não termos literais: expande para `forms` do conceito na carga.
    Reescrever as formas aqui as duplicaria dentro do MESMO arquivo — divergem na
    primeira edição."""
    terms: list[str] = []
    co_terms: list[str] = []
    # Ver `Rule.discontinuation`: verbo de parada é o GATILHO desta regra, não negação.
    discontinuation: bool = False

    @model_validator(mode="after")
    def _has_something_to_match(self) -> RuleToml:
        if not self.concepts and not self.terms:
            raise ValueError(f"regra {self.key!r}: sem `concepts` nem `terms`, nunca dispara")
        return self


class SafetyToml(Strict):
    schema_version: Literal[1]
    concepts: list[ConceptToml] = Field(min_length=1)
    property_phrases: dict[str, list[str]] = {}
    rules: list[RuleToml] = Field(min_length=1)

    @model_validator(mode="after")
    def _rules_point_at_real_concepts(self) -> SafetyToml:
        known = {c.key for c in self.concepts}
        for rule in self.rules:
            unknown = [c for c in rule.concepts if c not in known]
            if unknown:
                raise ValueError(
                    f"regra {rule.key!r} aponta para conceito(s) inexistente(s) "
                    f"{unknown}: ela nunca dispararia e nada acusaria"
                )
        return self


# ──────────────────────────────────────────────────────────────────── o agregado


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


class FocusProfile(BaseModel):
    """O perfil inteiro, já validado. Imutável na prática — ninguém escreve nele."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dir: Path
    focus: FocusToml
    strategies: StrategiesToml
    taxonomy: TaxonomyToml
    safety: SafetyToml | None
    """`None` significa AUSÊNCIA DECLARADA (`safety = false`), não arquivo faltando."""

    # ───────────────────────────────────────────── atalhos que os call sites usam

    @property
    def slug(self) -> str:
        return self.focus.slug

    @property
    def target(self) -> str:
        return self.focus.target

    @property
    def retrieval_prefix(self) -> str:
        return self.focus.retrieval_prefix

    @property
    def target_prose_cap(self) -> str:
        """`target_prose` no início de frase. Capitalizar é RENDERIZAÇÃO, não conteúdo —
        por isso mora aqui e não como um segundo campo do TOML, que divergiria."""
        prose = self.focus.target_prose
        return prose[:1].upper() + prose[1:]

    def directness_definitions(self) -> str:
        """As 4 linhas como `extract_claims.md` as traz hoje, na ordem do enum.

        Ordem do ENUM, não do TOML: rank é vocabulário global, e deixar o TOML
        reordenar faria a mesma escala ser apresentada em ordens diferentes conforme
        quem editou o arquivo.
        """
        return "\n".join(
            f"  {'`' + level + '`':<16}{self.focus.directness[level].prose}"
            for level in Directness4
        )

    def standing_risks_block(self, lead: str = "", tail: str = "") -> str:
        """`lead` + a prosa de cada risco + `tail`. VAZIA quando não há risco declarado.

        Cabeçalho e obrigação vêm DENTRO do bloco de propósito. Renderizar só o miolo
        deixaria, num foco sem risco permanente, o prompt afirmando que existe uma
        exceção permanente ("One standing exception you raise unprompted: ") seguida de
        nada — pior que não ter exceção nenhuma.
        """
        risks = self.focus.standing_risks
        if not risks:
            return ""
        body = " ".join(_sentence(_flat(r.prose)) for r in risks)
        if not lead:
            body = body[:1].upper() + body[1:]
        return f"{lead}{body}{' ' + tail if tail else ''}"

    def missing_risk_line(self) -> str:
        """A linha de `critique_speculation` que calibra risco como LACUNA, não defeito.

        PLAN.md:1177-1179 registra a regressão real: perder esta calibração custou
        3/3 → 0/3 especulações sobreviventes. Ela nomeia o risco DO FOCO — deixá-la
        hard-coded faz o portão de um foco novo cobrar a discussão de um risco que não
        existe naquele domínio, enquanto o risco real fica sem linha de gap-not-defect.
        """
        risks = self.focus.standing_risks
        if not risks:
            return ""
        names = ", ".join(r.name for r in risks)
        return (
            f"- The hypothesis never discusses {names}. Note it here; it is a gap, "
            f"not a defect."
        )

    def fatal_criteria_block(self) -> str:
        return "\n".join(self.focus.fatal_criteria)

    def risk_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(r.pattern, re.I) for r in self.focus.standing_risks]

    # ───────────────────────────────────────────── os blocos que vão aos prompts

    def prompt_blocks(self, name: str) -> dict[str, str]:
        """Os kwargs DE PERFIL de um prompt. Mapa LITERAL, um por prompt.

        Literal, e não derivado dos `$nomes` do `.md`, pelo mesmo argumento do
        docstring de `INJECTS`: derivar as duas pontas da mesma fonte produz um teste
        que não pode falhar. Aqui as duas fontes são o `.md` (que declara) e este mapa
        (que fornece), e `test_every_profile_placeholder_has_a_block` as cruza.

        É também o que permite ao golden ser renderizado com os VALORES DO PERFIL em
        vez de sentinelas — o que faz o golden do prompt policiar, de graça, a
        FIDELIDADE DA TRANSCRIÇÃO: editar a prosa de `partial` no focus.toml quebra
        `tests/golden/prompts/extract_claims.md`.
        """
        f = self.focus
        blocks: dict[str, dict[str, str]] = {
            "extract_claims": {
                "target": f.target,
                "target_prose": f.target_prose,
                "extract_bind": f.extract_bind,
                "directness_definitions": self.directness_definitions(),
            },
            "generate_questions": {
                "target_prose_cap": self.target_prose_cap,
                "question_bind": f.question_bind,
                "question_scarcity": f.question_scarcity,
            },
            "judge_sufficiency": {
                "target_prose": f.target_prose,
                "reader": f.reader,
                "sufficiency_population_rule": f.sufficiency_population_rule,
                "sufficiency_scarcity": f.sufficiency_scarcity,
            },
            "verify_pattern": {
                "population_hierarchy": f.population_hierarchy,
            },
            "critique_speculation": {
                "fatal_criteria": self.fatal_criteria_block(),
                "missing_risk_line": self.missing_risk_line(),
            },
            "generate_speculation": {
                "speculation_problem": f.speculation_problem,
                "mechanistic_question": f.mechanistic_question,
                "route_rationale": f.route_rationale,
                "reader": f.reader,
            },
            "chat": {
                "target_prose": f.target_prose,
                "reader": f.reader,
                "standing_risks": self.standing_risks_block(
                    lead=(
                        "One standing exception you raise unprompted whenever it "
                        "becomes relevant: "
                    ),
                    tail="Never let that pass unmarked.",
                ),
            },
            "answer_question": {
                "reader": f.reader,
                "standing_risks": self.standing_risks_block(
                    tail=(
                        "If your answer touches it, name that risk — it is the "
                        "standing exception of this project."
                    ),
                ),
            },
            "judge_directness": {
                "target": f.target,
                "directness_definitions": self.directness_definitions(),
            },
            # Fase C. SÓ o alvo em prosa, e nada de `standing_risks`: os dois prompts
            # do batedor não emitem conteúdo clínico — um classifica snippets num enum
            # de quatro valores, o outro resume UMA página e é explicitamente proibido
            # de avaliar. Ver a classificação em `EXEMPT`, em test_prompt_contract.py.
            "recon_triage": {"target_prose": f.target_prose},
            "recon_observe": {"target_prose": f.target_prose},
        }
        return blocks.get(name, {})


PROMPTS_WITH_PROFILE_BLOCKS = (
    "answer_question", "chat", "critique_speculation", "extract_claims",
    "generate_questions", "generate_speculation", "judge_directness",
    "judge_sufficiency", "recon_observe", "recon_triage", "verify_pattern",
)
"""Os prompts que recebem bloco de perfil. Literal, para um `$placeholder` novo num
prompt que não está aqui falhar ALTO em vez de renderizar vazio."""


# ─────────────────────────────────────────────────────────────────────── a carga


def _read(path: Path) -> dict:
    if not path.is_file():
        raise ProfileError(
            f"perfil de foco incompleto: {path} não existe. Um perfil são quatro "
            f"arquivos (focus.toml, strategies.toml, taxonomy.toml, safety.toml) e "
            f"não há fallback para default — um perfil vazio é um foco fantasma."
        )
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        # `TOMLDecodeError` sobe sem o nome do arquivo e são quatro candidatos.
        raise ProfileError(f"{path}: TOML inválido — {exc}") from exc


def _validate(model: type[BaseModel], data: dict, path: Path):
    try:
        return model.model_validate(data)
    except Exception as exc:
        raise ProfileError(f"{path}: {exc}") from exc


def profile_dir(focuses_dir: Path, slug: str) -> Path:
    return focuses_dir / slug


def load_profile(directory: Path) -> FocusProfile:
    """Lê os 4 TOML de um diretório de perfil. Erro cita o arquivo."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ProfileError(
            f"não existe perfil de foco em {directory}. Crie um com "
            f"`lithium focus new <slug>` — a linha no banco sem o perfil em disco "
            f"deixaria o sistema sem taxonomia, sem estratégia e sem definição de "
            f"directness."
        )

    focus_path = directory / "focus.toml"
    focus = _validate(FocusToml, _read(focus_path), focus_path)

    # ANTES de ler os outros três: eles vieram do mesmo `copytree` e estão igualmente
    # com o domínio de referência. Recusar aqui é o que impede um foco que mente.
    if focus.scaffold_pending:
        raise ProfileError(
            f"{focus_path}: `scaffold_pending = true` — este perfil ainda é a cópia do "
            f"foco de referência, com o domínio DELE. Revise os quatro arquivos "
            f"(focus.toml, strategies.toml, taxonomy.toml, safety.toml) e então apague "
            f"a linha. Preencher só o `target` produziria prompts que anunciam um alvo "
            f"e raciocinam sobre outro — inclusive as definições de directness, que são "
            f"a escada pela qual toda claim é pesada."
        )

    strategies_path = directory / "strategies.toml"
    strategies = _validate(StrategiesToml, _read(strategies_path), strategies_path)

    taxonomy_path = directory / "taxonomy.toml"
    taxonomy = _validate(TaxonomyToml, _read(taxonomy_path), taxonomy_path)

    safety: SafetyToml | None = None
    if focus.safety is not False:
        name = focus.safety if isinstance(focus.safety, str) else "safety.toml"
        safety_path = directory / name
        safety = _validate(SafetyToml, _read(safety_path), safety_path)

    if focus.slug != directory.name:
        raise ProfileError(
            f"{focus_path}: slug {focus.slug!r} não bate com o diretório "
            f"{directory.name!r}. O diretório é o que `focus --use` resolve."
        )

    return FocusProfile(
        dir=directory, focus=focus, strategies=strategies,
        taxonomy=taxonomy, safety=safety,
    )
