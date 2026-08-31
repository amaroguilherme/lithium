"""Schemas Pydantic para toda saída estruturada do LLM.

Estes modelos **geram** os JSON Schemas usados na decodificação restrita do
llama-server — não são só validação pós-hoc. Fonte única de verdade: mudar um campo
aqui muda a gramática que o modelo é obrigado a seguir.

Duas regras de projeto, ambas ditadas por estarmos usando um 12B e não um modelo de
fronteira:

1. **Raso e pequeno.** Aninhamento profundo degrada a aderência mesmo com gramática.
   Onde dá, campo é enum em vez de texto livre.
2. **Citação verbatim obrigatória.** Todo `ExtractedClaim` carrega `supporting_quote`,
   que precisa ser substring literal do chunk de origem. Isso transforma o teste de
   ancoragem em `quote in chunk_text` — determinístico, sem custo de LLM e sem
   possibilidade de o próprio modelo se absolver.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from lithium.types import (
    Direction,
    Directness,
    Grade,
    QuestionKind,
    StuckReason,
)


class Strict(BaseModel):
    """`extra="forbid"` vira `additionalProperties: false` no JSON Schema, que é o que
    impede a gramática de aceitar campos inventados."""

    model_config = ConfigDict(extra="forbid")


# ──────────────────────────────────────────────────────── planejamento de busca


class SearchQuery(Strict):
    source: str = Field(
        description="Slug da fonte a consultar, exatamente como listado em Available sources."
    )
    query: str = Field(description="String de busca na sintaxe nativa da fonte.")
    expected_directness: Directness = Field(
        description="Aderência esperada da população que esta busca vai retornar."
    )


class QueryPlan(Strict):
    """Uma pergunta vira várias buscas, deliberadamente em populações diferentes.

    Quando a interseção do alvo tem pouca literatura direta, buscar só ela volta vazio.
    O plano precisa cobrir as frentes adjacentes e marcar o `expected_directness` de
    cada uma — é o que permite ponderar depois.

    O alvo concreto fica FORA daqui de propósito: este docstring vai para
    `response_format`. Ver `types.Directness`.
    """

    rationale: str = Field(description="Por que estas buscas, em uma frase.")
    queries: list[SearchQuery] = Field(min_length=1, max_length=6)


# ───────────────────────────────────────────────────────────── extração de claims


class ExtractedClaim(Strict):
    """Uma claim proposta pela extração.

    `directness_judgeable` existe porque este caminho era o único do sistema OBRIGADO a
    chutar. `DirectnessVerdict` — o juiz independente do relens — tem três saídas, e o
    docstring dele explica por que: gravar um nível por falta de informação é
    **permanente** (a PK de `claim_directness` congela o valor e o sweep pula quem já tem
    aresta) e **invisível** (nenhum contador distingue "julgada" de "defaultada"). A
    extração tinha quatro níveis e nenhuma escapatória, então produzia exatamente o
    default contra o qual aquele docstring adverte.

    MEDIDO no corpus real, chamando o juiz independente sobre uma amostra de 12 claims
    que a extração já havia julgado: concordância 6/9, a autoavaliação inflou em 2/9
    (inclusive um `direct` -> `partial`, peso -0,40), e em **3 de 12** o juiz respondeu
    NÃO-JULGÁVEL onde a extração havia gravado um nível com peso. n é pequeno e a taxa de
    concordância não é estável; a existência da lacuna não depende de n.

    Sem aresta, a claim fica em `claims_unjudged` com peso ZERO — fail-closed — e o
    `focus --relens` a revisita com o juiz independente e o bloco de evidência inteiro.
    A escapatória não descarta a claim: encaminha para quem julga melhor.
    """

    statement: str = Field(description="A alegação, em uma frase autocontida.")
    supporting_quote: str = Field(
        description="Trecho VERBATIM do texto fornecido que sustenta a alegação. "
        "Copie exatamente, sem parafrasear."
    )
    population: str = Field(description="Quem foi estudado. '' se não estiver claro.")
    intervention: str
    comparator: str = Field(description="'' se não houver comparador.")
    outcome: str
    direction: Direction
    effect: str = Field(description="Tamanho de efeito como reportado. '' se ausente.")
    grade: Grade
    # ANTES de `directness`, e a ordem é o mecanismo: a decodificação é restrita por
    # gramática e o modelo emite os campos NA ORDEM DO SCHEMA. Decidir "dá para saber?"
    # depois de já ter escrito um nível é racionalizar a escolha; decidir antes é julgar.
    directness_judgeable: bool = Field(
        description="A evidência diz quem foi estudado? false quando não dá para saber."
    )
    directness: Directness
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimExtraction(Strict):
    """Lista vazia é resposta válida e esperada — a maior parte do texto de um paper
    não contém alegação extraível, e forçar extração produz invenção."""

    claims: list[ExtractedClaim] = Field(max_length=8)


class DirectnessVerdict(Strict):
    """O veredito do relens: uma claim, um foco, uma aresta.

    TRÊS resultados, não quatro níveis — e é essa diferença que decide se o relens
    repara ou destrói.

    1. **`in_scope=False`** — julgada e IRRELEVANTE. Grava a aresta com
       `out_of_scope=1`, que `claim_weight` exclui por predicado. Sem este resultado o
       relens de um foco novo importa o corpus inteiro do foco velho com peso
       POSITIVO: o piso `extrapolated` vale 0,12, não 0, então dez claims de outro
       domínio superam UMA meta-análise perfeitamente no alvo — e a tabela de
       cobertura, que ordena por peso SOMADO e corta em 40 linhas, vira do domínio
       antigo.

    2. **`judgeable=False`** — NÃO grava nada. Preserva peso zero, mantém a claim em
       `claims_unjudged` e deixa o relens revisitá-la. "Na dúvida, o nível MENOS
       aderente" NÃO vale aqui: gravar `extrapolated` por falta de informação é
       PERMANENTE (a PK congela o valor e o sweep pula quem já tem aresta) e INVISÍVEL
       (nenhum contador distinguiria "julgada" de "defaultada"). O bloco de evidência
       que o revisor lê passaria a escrever `(rct / extrapolated)` — afirmando que um
       RCT humano é "mecanismo, animal, in-vitro ou pura inferência". A regra
       fail-closed vale entre dois níveis ADJACENTES, com a população em mãos.

    3. **um dos quatro níveis** — o caso normal.

    `directness` é ENUM FECHADO de propósito: com `str` a gramática deixa de
    restringir, o modelo inventa um nível fora do vocabulário, e a FK transforma isso
    em `IntegrityError` no meio da escrita — com o retry repetindo o mesmo erro até o
    dead-letter.
    """

    in_scope: bool = Field(
        description="Esta claim fala de uma população que tem alguma relação com o "
        "alvo? false para material de outro domínio."
    )
    judgeable: bool = Field(
        description="A evidência diz quem foi estudado? false quando não dá para "
        "saber — não chute e não caia para o nível mais fraco."
    )
    directness: Directness = Field(
        description="O nível. Só é lido quando in_scope e judgeable são ambos true."
    )
    rationale: str = Field(
        description="Uma frase: a população encontrada e as palavras que decidiram."
    )


class CitationVerdict(Strict):
    """Segundo portão, depois do teste determinístico de substring: a citação existe,
    mas ela realmente *implica* a alegação?"""

    supported: bool
    reason: str


# ─────────────────────────────────────────────────────────────────── respostas


class Finding(Strict):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    claim_ids: list[int] = Field(
        description="IDs das claims fornecidas que sustentam a resposta."
    )
    caveats: str = Field(description="Limitações e incerteza residual.")


class SufficiencyVerdict(Strict):
    """Chamada SEPARADA do gerador, por construção.

    Modelo pequeno auto-avaliando na mesma chamada em que gerou diz "suficiente"
    praticamente sempre. Separar é o que dá algum sinal ao portão.
    """

    sufficient: bool = Field(
        description="A evidência dá para escrever uma nota de 4-6 frases, com citação e "
        "com os limites nomeados? NÃO é 'a questão está resolvida'. Ausência de "
        "head-to-head, amostra pequena e eficácia relativa desconhecida vão em "
        "`missing`, não aqui."
    )
    n_independent_sources: int = Field(
        description="Quantos PMIDs distintos de fato tratam da pergunta. Duas "
        "afirmações extraídas do mesmo paper são UMA fonte."
    )
    addresses_question_directly: bool = Field(
        description="A evidência responde a pergunta feita, ou só uma adjacente?"
    )
    sources_agree: bool = Field(
        description="Elas apontam na mesma direção? Discordância NÃO bloqueia a "
        "resposta — baixa a confiança dela e tem de ser relatada."
    )
    missing: str = Field(description="O que falta para fechar. '' se nada falta.")
    blocked_reason: StuckReason | None = Field(
        default=None,
        description="Preencha SÓ se mais rodadas de busca não resolveriam.",
    )


class Critique(Strict):
    """Passe adversarial: o prompt instrui a REFUTAR, não a revisar."""

    survives: bool
    strongest_objection: str
    unsupported_statements: list[str] = Field(
        default_factory=list,
        description="Trechos da resposta que vão além do que as citações sustentam.",
    )


# ────────────────────────────────────────────────────────────────── perguntas


class GeneratedQuestion(Strict):
    text: str
    kind: QuestionKind
    rationale: str = Field(description="Que lacuna esta pergunta fecha.")
    targets: str = Field(
        description="A intervenção ou tópico específico sobre o qual a pergunta é. "
        "Use o mesmo nome que aparece na tabela de cobertura, quando houver. "
        "Isto liga a pergunta à evidência já reunida — é o que permite pontuar "
        "quanto ela vale."
    )


class QuestionBatch(Strict):
    questions: list[GeneratedQuestion] = Field(max_length=8)


class QuestionClassification(Strict):
    kind: QuestionKind
    targets: str
    reason: str


# ──────────────────────────────────────────────────────── trilha especulativa
#
# A cadeia mecanística é o mecanismo que separa hipótese auditável de prosa
# plausível. Cada elo é marcado `supported` (com citação) ou `assumed`, e a fração
# ancorada vira a plausibilidade — um número, não uma impressão. `falsifier` é
# obrigatório pelo mesmo motivo: especulação que nada refuta não é hipótese.


class MechanismStep(Strict):
    claim: str = Field(description="Um elo do raciocínio, em uma frase.")
    supported: bool = Field(
        description="true SÓ se você puder citar uma fonte concreta para este elo. "
        "Na dúvida, false — marcar como assumido é honesto, inventar citação não."
    )
    evidence: str = Field(
        description="PMID, NCT ou DOI que sustenta o elo. '' quando `supported` é false."
    )


class Speculation(Strict):
    statement: str = Field(description="A hipótese, em uma frase acionável.")
    intervention_class: str
    mechanism_target: str
    route: str = Field(
        description="Via de administração ou entrega. Não é embalagem: a via define a "
        "farmacocinética, e isso pode ser parte do mecanismo. Se a via é parte do que "
        "torna a hipótese plausível, diga por quê no encadeamento."
    )
    """A justificativa MECANÍSTICA de por que a via importa NESTE foco é dado de
    perfil (`route_rationale`) e chega pelo prompt. Estava triplicada — aqui, no texto
    estático de `generate_speculation.md` e dentro de `route_block()`, que é injetado
    NO MESMO PROMPT. Uma origem só."""
    combination: str = Field(
        description="Se a hipótese é uma COMBINAÇÃO, liste os componentes e diga o "
        "que a associação faz que nenhum isolado faz. '' para agente único."
    )
    chain: list[MechanismStep] = Field(
        min_length=2, max_length=6,
        description="Do mecanismo ao desfecho clínico. Cada passo separado.",
    )
    falsifier: str = Field(
        description="Que observação concreta refutaria isto? Não pode ficar vazio."
    )
    test_proposal: str = Field(
        description="A próxima busca ou experimento que testaria a hipótese."
    )
    known_risks: str = Field(
        description="Riscos conhecidos desta abordagem na população do foco. '' se "
        "genuinamente nenhum for conhecido — o que em si é informação."
    )
    novelty: float = Field(
        ge=0.0, le=1.0,
        description="0 = prática padrão hoje; 1 = ninguém propôs isto para esta "
        "população.",
    )


class SpeculationBatch(Strict):
    speculations: list[Speculation] = Field(max_length=5)


class SourceQuery(Strict):
    source: str = Field(
        description="Slug da fonte a consultar, exatamente como listado em Available sources."
    )
    query: str = Field(description="String de busca na sintaxe nativa da fonte.")
    seeking: str = Field(description="Que elo da cadeia esta busca tentaria ancorar.")


class SpeculationQueries(Strict):
    """Converte uma hipótese em buscas concretas — é o que fecha o laço entre a
    trilha exploratória e a coleta. Sem isto a especulação é um beco sem saída: o
    modelo propõe sigma-1 e o harvest nunca consulta o PubMed sobre sigma-1."""

    queries: list[SourceQuery] = Field(min_length=1, max_length=6)


class RegroundedLink(Strict):
    index: int = Field(description="Posição do elo na cadeia, começando em 1.")
    now_supported: bool
    evidence: str = Field(
        description="PMID da claim recuperada que sustenta o elo. '' se nenhuma."
    )


class RegroundedChain(Strict):
    """Reancoragem: elos assumidos que o corpus, já maior, passou a sustentar.

    É por aqui que a plausibilidade sobe com o tempo — sem isso a cadeia congela no
    estado em que nasceu e a trilha exploratória nunca aprende.
    """

    links: list[RegroundedLink] = Field(max_length=6)


class MemoryProposal(Strict):
    """Detecção de algo durável dito na conversa.

    Propõe; nunca grava. O usuário confirma — num domínio de saúde, acumular fatos
    sobre a pessoa sem autorização explícita é diferente de acumular papers.
    """

    worth_remembering: bool = Field(
        description="true SÓ se o usuário afirmou algo durável sobre si, suas "
        "preferências, restrições ou o caso. Perguntas, especulações e conversa "
        "de passagem não contam."
    )
    text: str = Field(
        description="A memória em uma frase, na terceira pessoa e autocontida. "
        "'' se worth_remembering for false."
    )
    kind: str = Field(
        description="preference | context | constraint | fact. '' se não aplicável."
    )
    rationale: str = Field(description="Por que isto vale guardar, em uma frase.")


class ReconVerdict(Strict):
    """A triagem de UM resultado de busca web.

    **`index`, NUNCA `url`** — índice LOCAL do bloco mostrado, o idioma que
    `Reflector.remember`/`resolve_refs` já usam. Um veredito cujo índice não está no
    conjunto mostrado é DESCARTADO, como `verify_claim_ids` faz. Isso torna ESTRUTURAL
    a regra "a URL de uma descoberta vem SEMPRE da resposta da API, nunca do modelo":
    um 12B a quem se pede uma URL inventa URL.

    **Nem `url` nem `lead_id`.** O identificador de um `lead` é derivado em código pelo
    `lithium/recon/leads.py`, a partir do hit que a API devolveu. Ver o docstring de lá:
    um dígito trocado num PMID resolve para um artigo real e sem relação, e os dois
    portões de extração o aprovam.

    **Sem `reason`.** Não tem consumidor — `discoveries` não tem coluna para ele — e ele
    consumia ~40% do orçamento de saída no passo mais apertado do desenho: o esqueleto
    JSON de 10 verdicts já mede ~156 tokens, e uma frase por veredito leva a saída além
    do teto. A gramática `json_schema` strict OBRIGA os N objetos com todas as chaves,
    então o modelo não pode encerrar cedo: o corte produz JSON inválido, `complete_json`
    faz 2 reparos, `LLMInvalidOutput` sobe, `queue.fail` repete 3x — 9 chamadas de LLM
    para uma query que já foi FATURADA e produz zero descobertas. O snippet fica em
    `payload_json` para auditoria.
    """

    index: int = Field(
        description="O número do resultado no bloco mostrado, começando em 1."
    )
    kind: str = Field(
        description="lead | source | observation | skip. `lead` = um artigo específico "
        "que vale colher. `source` = um domínio que mantém um registro consultável. "
        "`observation` = algo substantivo que vale ler e resumir. `skip` = o resto, e "
        "é a resposta mais comum."
    )


class ReconTriage(Strict):
    """Um veredito por resultado mostrado. Lista vazia é resposta válida."""

    verdicts: list[ReconVerdict] = Field(max_length=10)


class ReconObservation(Strict):
    """O resumo de UMA página lida. Saída curta de propósito.

    `max_tokens=256` e não 512 porque o custo é dominado pela SAÍDA: pelo modelo do
    PLAN (`t ≈ 2,31 ms × in + 161 ms × out`), out=512 custa 82 s de decode e out=256
    custa 41 s, com a MESMA qualidade para um resumo de ~200 caracteres.
    """

    worth_reporting: bool = Field(
        description="false se a página é institucional, promocional, ou não diz nada "
        "que valha o tempo de quem vai ler."
    )
    summary: str = Field(
        description="O que a página afirma, em 1-2 frases. '' se worth_reporting for "
        "false."
    )
    why_it_matters: str = Field(
        description="Por que isto é relevante para o alvo, em uma frase."
    )


class ResearchLesson(Strict):
    text: str = Field(description="A lição em uma frase autocontida.")
    kind: str = Field(
        description="dead_end | search_lesson | source_lesson | pattern. "
        "Use `pattern` SÓ para padrão substantivo sobre o domínio — e aí "
        "`claim_ids` é obrigatório."
    )
    provenance_note: str = Field(
        description="De que evento você está generalizando: id da hipótese, a query, "
        "a fonte."
    )
    claim_ids: list[int] = Field(
        default_factory=list,
        description="OBRIGATÓRIO quando kind='pattern': ids das claims verificadas de "
        "onde o padrão foi derivado. Lista vazia nas categorias de processo.",
    )


class ResearchLessons(Strict):
    """Lista vazia é resposta válida e comum — a maioria das rodadas de pesquisa não
    produz lição generalizável, e forçar produz ruído que degrada prompts futuros."""

    lessons: list[ResearchLesson] = Field(max_length=4)


class SpeculationCritique(Strict):
    """Passe adversarial dedicado. Procura o elo mais fraco, não avalia o conjunto —
    uma cadeia é tão forte quanto seu pior elo, e avaliar 'no geral' deixa o elo
    quebrado passar escondido atrás dos bons."""

    weakest_link: str = Field(description="Qual elo da cadeia é o mais frágil, e por quê.")
    fatal_flaw: str = Field(
        description="Um erro que invalida a hipótese inteira. '' se não houver."
    )
    contradicted_by: str = Field(
        description="Evidência conhecida que contradiz isto. '' se nenhuma."
    )
    missing_risk: str = Field(
        description="Risco relevante que a hipótese omitiu. '' se nenhum."
    )
    survives: bool = Field(
        description="A hipótese merece seguir sendo investigada? false se houver "
        "falha fatal ou contradição direta."
    )


class PatternVerdict(Strict):
    """Portão de derivação para `pattern` — a única categoria auto-gravada que afirma
    algo sobre o mundo.

    **Perguntas escopadas, não um veredito global.** O repo já registra duas vezes que
    veredito global de um 12B não vale: `SufficiencyVerdict` ("modelo pequeno
    auto-avaliando na mesma chamada em que gerou diz suficiente praticamente sempre") e
    `SpeculationCritique` ("avaliar no geral deixa o elo quebrado passar escondido atrás
    dos bons"). "Isto é válido?" devolve `true`; "a abstração fala de uma população mais
    ampla que as premissas?" é checável.

    **Não reusa `verify_citation`**, de propósito: a lista de rejeição dele inclui
    *generalises past the group described*, e uma abstração é por definição mais ampla
    que cada premissa. Ele reprovaria toda abstração — ou, relaxado, viraria "parece
    seguir", que não é relação verificável.
    """

    follows_from_cited_claims_alone: bool = Field(
        description="A generalização segue das claims citadas SOZINHAS, sem "
        "conhecimento externo? Se você precisou de algo que não está nelas, false."
    )
    population_scope_exceeded: bool = Field(
        description="A generalização fala de uma população mais ampla que a das claims "
        "citadas? Qualquer alargamento — de espécie, de gravidade, de comorbidade — "
        "conta como true."
    )
    """O PAR CONCRETO de populações que define isto é dado de perfil
    (`population_hierarchy`) e chega por `verify_pattern.md`. Estava DUPLICADO: aqui
    (indo para a gramática) e no prompt. Os dois tinham de mover juntos, senão o prompt
    falaria genérico e o schema falaria do foco antigo."""
    contradicts_a_cited_claim_direction: bool = Field(
        description="A generalização contradiz a direção de efeito de alguma claim "
        "citada?"
    )
    restates_a_premise: bool = Field(
        description="É só uma paráfrase de uma claim, não uma abstração sobre várias?"
    )
    reason: str = Field(description="Uma frase. Qual pergunta decidiu, e por quê.")
