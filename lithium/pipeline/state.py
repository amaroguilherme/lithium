"""Resumo do estado do conhecimento — a entrada do gerador de perguntas.

O gerador não vê o corpus; vê este resumo. Isso é deliberado: um 12B com 30 abstracts
na janela produz perguntas sobre o que estava no texto, não sobre o que **falta**. Um
resumo agregado torna as lacunas visíveis — intervenção sem nenhuma claim, evidência
que só existe em população indireta, direções contraditórias sobre o mesmo fármaco.

Também é o mesmo resumo que a priorização consome — mas o TETO é só da janela. A tabela
renderizada é cortada em `MAX_COVERAGE_ROWS` porque o prompt não cabe; `by_intervention`
e `untouched` leem `coverage_all`, o corpus inteiro. Ler a lista cortada fazia o teto
decidir o que o sistema acredita existir, e isso foi MEDIDO: 60 de 63 intervenções atrás
do corte pontuavam novidade máxima sobre evidência já colhida, e 2 de 17 classes eram
anunciadas ao gerador como "zero evidence gathered" com claim no banco.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lithium.db import Store
from lithium.focus import FocusProfile
from lithium.types import DIRECTNESS_WEIGHT, Directness, Grade

@dataclass(slots=True)
class Coverage:
    intervention: str
    n_claims: int = 0
    total_weight: float = 0.0
    best_grade: Grade | None = None
    best_directness: Directness | None = None
    positive: int = 0
    negative: int = 0
    neutral: int = 0

    @property
    def conflict(self) -> float:
        """0 = consenso, 1 = empate perfeito entre positivo e negativo.

        Conflito é o sinal mais valioso para gerar pergunta: quando as fontes
        discordam, mais evidência do mesmo tipo não resolve — é preciso perguntar
        *por que* discordam (subpopulação, dose, desfecho diferente).
        """
        decided = self.positive + self.negative
        if decided < 2:
            return 0.0
        return 1.0 - abs(self.positive - self.negative) / decided

    @property
    def directness_gap(self) -> float:
        """Quanto falta para ter evidência na população certa. 0 = já é `direct`."""
        if self.best_directness is None:
            return 1.0
        return 1.0 - DIRECTNESS_WEIGHT[self.best_directness]


MAX_COVERAGE_ROWS = 40
"""Teto de linhas de cobertura no prompt.

`claims.intervention` é texto livre de um 12B: "quetiapine", "quetiapine XR" e
"adjunctive quetiapine" são três linhas. A cardinalidade cresce com o corpus e **não
converge**.

MEDIDO com 50 fontes: 103 grafias distintas, o teto engatado, 63 omitidas. E a atribuição
de causa acima está incompleta — a fragmentação de grafia vale ~10 dos 103 valores; 47
(46%) não são intervenção nenhuma (condição 21, exposição 19, marcador 7). Essa parcela
CONVERGE quando o prompt define o campo; a que o parágrafo acima nomeia é a que não
converge. O teto continua correto e necessário pelas duas.

**O teto é da JANELA e mais nada.** `coverage_all` existe porque derivar `untouched` e
`by_intervention` desta lista fazia o teto decidir o que o sistema acredita existir. Cada linha custa ~28 tokens, então sem teto `generate_speculation` estoura
a janela de 8192 em ~128 intervenções distintas — e uma varredura completa produz
150–400. O llama-server devolve 400, o cliente corretamente não repete 4xx, e a task
vai para dead-letter: a trilha especulativa morria em silêncio conforme o corpus
amadurecia.
"""

MAX_CONFLICT_ROWS = 10


@dataclass(slots=True)
class KnowledgeState:
    coverage: list[Coverage] = field(default_factory=list)
    untouched: list[str] = field(default_factory=list)
    """Classes de intervenção sem nenhuma evidência — não fármacos específicos."""

    conflicts: list[Coverage] = field(default_factory=list)
    asked: list[tuple[str, str]] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    n_sources: int = 0
    n_claims: int = 0
    """Claims que TÊM peso neste foco — o mesmo universo da tabela de cobertura abaixo.

    Antes do fail-closed isto era `claims_verified` e os dois universos coincidiam. Com
    ele, 400 verificadas sem julgamento renderizariam "Corpus: 400 verified claims"
    seguido de "(nothing extracted yet)": uma afirmação sobre COBERTURA quando o fato é
    ausência de JULGAMENTO. `n_unjudged` diz a outra metade, em vez de escondê-la."""

    n_unjudged: int = 0
    n_off_scale: int = 0
    """Claims COM aresta neste foco, mas graduadas sob OUTRA escala.

    Separado de `n_unjudged` porque os dois pedem remédios opostos e o relens só
    resolve o primeiro. Juntos, o prompt afirmava "N verified claim(s) carry NO
    judgment for this focus" sobre claims que TÊM julgamento — e o gerador de perguntas
    raciocinava para sempre sobre uma ausência que não é a ausência descrita."""

    coverage_all: list[Coverage] = field(default_factory=list)
    """A cobertura INTEIRA, sem o teto. Nunca renderizada — só pontuação e classes.

    O teto de `coverage` é da JANELA, não da verdade: `MAX_COVERAGE_ROWS` existe porque
    o prompt não cabe, e nada mais. Derivar `untouched` e `by_intervention` da lista
    cortada fazia o teto decidir o que o sistema ACREDITA existir: 2 de 17 classes eram
    anunciadas como "zero evidence gathered" com claim no banco, e 60 de 63
    intervenções atrás do corte pontuavam novidade MÁXIMA (0,65 — acima de 39 das 40
    linhas visíveis) sobre evidência já colhida.
    """

    focus: str | None = None
    coverage_omitted: int = 0
    """Intervenções cortadas pelo teto. Renderizado: um bloco truncado precisa
    declarar o que esconde, senão o modelo raciocina sobre um quadro parcial
    acreditando que é completo."""

    n_no_intervention: int = 0
    """Claims com peso cujo `intervention` é vazio — fora da tabela e de todo número dela.

    Declarada no cabeçalho pela mesma razão de `n_unjudged` e `n_off_scale`: o `WHERE`
    da consulta de cobertura as exclui, e o cabeçalho afirmava "213 weighted claims"
    sobre uma tabela cuja coluna `n` soma 80. Era a ÚNICA das quatro exclusões sem
    contador — exatamente o que o docstring de `coverage_omitted` proíbe em palavras.
    """

    def by_intervention(self, name: str) -> Coverage:
        """Resolve um alvo escrito pelo modelo contra a cobertura do CORPUS.

        Duas propriedades, e as duas foram medidas como defeito antes de virarem regra:

        1. **Varre `coverage_all`, não `coverage`.** O teto é da janela; a prioridade é
           uma afirmação sobre o corpus. Com a lista cortada, 60 de 63 intervenções
           escondidas pontuavam 0,65 — novidade máxima sobre evidência que existe.
        2. **Igualdade antes de substring.** A varredura é por peso decrescente, então
           `needle in c.intervention` fazia a linha PESADA sombrear a exata: `up+tau`
           resolvia para `unified protocol for emotional disorders (up+tau)` (peso 3,40
           contra 1,70 que a tabela mostra) e `anxiety disorder` para `any anxiety
           disorders`. Eram 4 dos 40 nomes que `generate_questions.md` manda copiar
           verbatim — e `priority` é gravada e nunca recalculada.

        O substring FICA como segundo passo porque `render()` corta o nome em 38 chars
        e o prompt manda copiar da tabela: sem ele, todo nome truncado vira alvo
        desconhecido.
        """
        needle = name.strip().lower()
        for c in self.coverage_all:
            if c.intervention == needle:
                return c
        for c in self.coverage_all:
            if needle in c.intervention:
                return c
        return Coverage(intervention=needle)

    def render(self) -> str:
        """Formato de texto para o prompt. Tabela compacta, não JSON — um 12B lê
        tabela alinhada melhor do que estrutura aninhada."""
        head = (
            f"Corpus: {self.n_sources} sources, {self.n_claims} weighted claims"
            f" (focus: {self.focus or 'NONE — nothing is being scored'})."
        )
        if self.n_unjudged:
            head += (
                f" {self.n_unjudged} verified claim(s) carry NO judgment for this focus "
                f"and are excluded from every number below — that is absence of "
                f"judgment, not absence of evidence."
            )
        if self.n_off_scale:
            head += (
                f" A further {self.n_off_scale} verified claim(s) were graded on a "
                f"DIFFERENT evidence scale and cannot be weighed here — re-judging "
                f"their population would not change that."
            )
        if self.n_no_intervention:
            head += (
                f" A further {self.n_no_intervention} weighted claim(s) record no "
                f"intervention and are outside the table below."
            )
        lines: list[str] = [head, "", "## Evidence coverage by intervention"]
        if self.coverage:
            lines.append(
                f"{'intervention':38} {'n':>3} {'weight':>7}  {'best grade':18} "
                f"{'best pop.':13} +/-/0"
            )
            for c in self.coverage:
                lines.append(
                    f"{c.intervention[:38]:38} {c.n_claims:3} {c.total_weight:7.2f}  "
                    f"{(c.best_grade.value if c.best_grade else '—'):18} "
                    f"{(c.best_directness.value if c.best_directness else '—'):13} "
                    f"{c.positive}/{c.negative}/{c.neutral}"
                )
            if self.coverage_omitted:
                lines.append(
                    f"  (+{self.coverage_omitted} intervenções de menor peso não "
                    f"mostradas — este quadro é parcial)"
                )
        else:
            lines.append("(nothing extracted yet)")

        if self.untouched:
            lines += ["", "## Intervention CLASSES with zero evidence gathered",
                      "  (naming something specific inside these is the point)"]
            lines += [f"  - {c}" for c in self.untouched]

        if self.conflicts:
            lines += ["", "## Contradictions (sources disagree on direction)"]
            for c in self.conflicts:
                lines.append(
                    f"  {c.intervention}: {c.positive} positive vs {c.negative} negative "
                    f"(best evidence: {c.best_grade.value if c.best_grade else '—'} / "
                    f"{c.best_directness.value if c.best_directness else '—'})"
                )

        if self.hypotheses:
            lines += ["", "## Hypotheses on the board"]
            for h in self.hypotheses:
                unjudged = (
                    f" · {h['n_unweighted']} linked claim(s) unjudged in this focus"
                    if h["n_unweighted"] else ""
                )
                lines.append(
                    f"  [{h['status']}] {h['statement']}  "
                    f"(support {h['support']:.2f} / against {h['contra']:.2f}){unjudged}"
                )

        lines += ["", "## Questions already on record — do NOT repeat these"]
        if self.asked:
            for status, text in self.asked:
                lines.append(f"  [{status}] {text}")
        else:
            lines.append("  (none yet)")

        return "\n".join(lines)


def _level_by_rank(store: Store, axis: str) -> dict[int, str]:
    """rank → value, lido da MESMA tabela que produziu o `MIN(rank)` da consulta.

    Substitui `list(Directness)[rank - 1]`. A decodificação POSICIONAL era uma bomba
    silenciosa: um nível novo no meio do enum renumera `scale_levels` e a mesma claim
    passa a ser rotulada com o nível ERRADO no prompt — ou estoura em IndexError, que
    seria a sorte grande. Aqui rank é só ordenação e nenhuma renumeração pode mentir.

    Lê `value` e `rank`, NUNCA `weight`: quem calcula peso é `claim_weight`, e
    `test_no_second_source_computes_the_weight` reprova a segunda leitura.
    """
    return {
        int(r["rank"]): r["value"]
        for r in store.conn.execute(
            "SELECT sl.value, sl.rank FROM scale_levels sl "
            "  JOIN active_focus af ON af.scale_id = sl.scale_id "
            " WHERE sl.axis = ?",
            (axis,),
        )
    }


def build_state(store: Store, profile: FocusProfile, *,
                max_asked: int = 40) -> KnowledgeState:
    rows = store.conn.execute(
        "SELECT LOWER(TRIM(c.intervention)) AS interv, "
        "       COUNT(*) AS n, SUM(cw.weight) AS total, "
        "       SUM(c.direction = 'positive') AS pos, "
        "       SUM(c.direction = 'negative') AS neg, "
        "       SUM(c.direction IN ('null', 'mixed')) AS neu, "
        "       MIN(gw.rank) AS best_grade_rank, MIN(dw.rank) AS best_dir_rank "
        "  FROM claims c "
        "  JOIN claim_weight cw ON cw.claim_id = c.id "
        "  JOIN active_focus af "
        "  JOIN claim_directness cd ON cd.claim_id = c.id AND cd.focus_id = af.id "
        "  JOIN scale_levels gw ON gw.scale_id = af.scale_id AND gw.axis = 'grade' "
        "                      AND gw.value = c.grade "
        "  JOIN scale_levels dw ON dw.scale_id = af.scale_id "
        "                      AND dw.axis = 'directness' AND dw.value = cd.directness "
        " WHERE c.intervention IS NOT NULL AND TRIM(c.intervention) <> '' "
        " GROUP BY interv ORDER BY total DESC",
    ).fetchall()

    # SEM `LIMIT` na consulta, e o corte em Python. O teto é da JANELA, e a janela é
    # `render()`; a pontuação e as classes precisam do corpus inteiro. O segundo
    # `COUNT(DISTINCT)` que estimava `coverage_omitted` MORREU com isto: ele contava
    # sobre `claim_weight` enquanto a tabela contava sobre este JOIN, e o próprio
    # comentário registrava o risco de renderizar "(+N não mostradas)" com um N de outro
    # universo. Agora N é o mesmo universo, por construção.
    omitted = max(0, len(rows) - MAX_COVERAGE_ROWS)

    grade_by_rank = _level_by_rank(store, "grade")
    dir_by_rank = _level_by_rank(store, "directness")
    coverage_all = [
        Coverage(
            intervention=r["interv"],
            n_claims=r["n"],
            total_weight=r["total"] or 0.0,
            best_grade=Grade(grade_by_rank[r["best_grade_rank"]]),
            best_directness=Directness(dir_by_rank[r["best_dir_rank"]]),
            positive=r["pos"] or 0,
            negative=r["neg"] or 0,
            neutral=r["neu"] or 0,
        )
        for r in rows
    ]
    coverage = coverage_all[:MAX_COVERAGE_ROWS]

    # Cobertura por CLASSE: a classe está tocada se alguma claim casar com uma de
    # suas palavras-chave. O que existe dentro da classe é problema do gerador.
    #
    # Rótulo e keywords vêm da MESMA entrada do perfil. Antes eram duas tabelas
    # (`INTERVENTION_CLASSES` e as chaves de `CLASS_KEYWORDS`), idênticas por acidente
    # e sem nenhum teste que as cruzasse: duas fontes de verdade da mesma taxonomia que
    # ainda não tinham divergido.
    # Do CORPUS, não das linhas mostradas. `coverage` já foi cortado em
    # `MAX_COVERAGE_ROWS`, e ler `seen` dele fazia o teto vazar para um bloco que não
    # declara truncar nada: MEDIDO no corpus real, 2 de 17 classes eram anunciadas ao
    # gerador como "zero evidence gathered" tendo claim no banco, e as duas viravam 2
    # das 8 queries de recuperação de lacuna.
    seen = " ".join(c.intervention for c in coverage_all).lower()
    untouched = [
        klass.label
        for klass in profile.taxonomy.intervention_classes
        if not any(kw.lower() in seen for kw in klass.keywords)
    ]

    # Escopadas por foco, as DUAS. Sem isto, trocar de foco faz `build_state` instruir
    # o foco novo a "não repetir" perguntas de outro domínio e lhe mostrar hipóteses
    # que não são dele — até 40 linhas de material alheio no prompt que DECIDE a
    # agenda de pesquisa.
    asked = [
        (r["status"], r["text"])
        for r in store.conn.execute(
            "SELECT status, text FROM questions "
            " WHERE focus_id = (SELECT id FROM active_focus) "
            " ORDER BY CASE status WHEN 'OPEN' THEN 0 WHEN 'RESEARCHING' THEN 1 "
            "                      WHEN 'ESCALATED' THEN 2 ELSE 3 END, id DESC LIMIT ?",
            (max_asked,),
        )
    ]

    hypotheses = [
        dict(r)
        for r in store.conn.execute(
            "SELECT statement, status, support, contra, n_unweighted "
            "  FROM hypothesis_scoreboard "
            " WHERE status = 'active' "
            "   AND focus_id = (SELECT id FROM active_focus) "
            " ORDER BY support + contra DESC LIMIT 15"
        )
    ]

    counts = store.counts()
    focus = store.active_focus()
    n_no_intervention = store.conn.execute(
        "SELECT COUNT(*) AS n FROM claims c "
        "  JOIN claim_weight cw ON cw.claim_id = c.id "
        " WHERE c.intervention IS NULL OR TRIM(c.intervention) = ''"
    ).fetchone()["n"]

    return KnowledgeState(
        coverage=coverage,
        coverage_all=coverage_all,
        n_no_intervention=n_no_intervention,
        untouched=untouched,
        conflicts=sorted(
            (c for c in coverage if c.conflict > 0.0),
            key=lambda c: c.conflict * c.total_weight, reverse=True,
        )[:MAX_CONFLICT_ROWS],
        asked=asked,
        hypotheses=hypotheses,
        n_sources=counts["sources"],
        n_claims=counts["claims_verified"] - counts["claims_unweighted"],
        n_unjudged=counts["claims_unjudged"],
        n_off_scale=counts["claims_off_scale"],
        focus=focus["slug"] if focus else None,
        coverage_omitted=omitted,
    )
