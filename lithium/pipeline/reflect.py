"""Reflexão: o sistema aprendendo sobre o próprio trabalho de pesquisa.

Por decisão do usuário, memória tem dois regimes de consentimento:

* **conversa** → propõe e espera confirmação. É sobre a pessoa.
* **pesquisa** → grava sozinho. É o sistema aprendendo a trabalhar melhor, e pedir
  permissão para cada lição seria fricção sem ganho.

Essa autonomia é segura por causa de dois limites estruturais, não de conselho no prompt.

**1. Portão de citação por categoria.** Lição de *processo* ("essa forma de query volta
vazia") não afirma nada sobre biologia e não precisa de fonte. Lição *substantiva*
(`pattern`: "aumentar tônus glutamatérgico agudamente foi o modo de falha comum em três
hipóteses") exige `claim_ids` de claims verificadas, e `verify_claim_ids` confere que
existem. Sem esse portão, uma asserção sem fonte entraria e seria injetada em todo
prompt seguinte — o sistema ensinando as próprias suposições a si mesmo, com nenhum
portão pegando. É a falha que a extração previne, entrando pela porta de trás.

**2. Recuperação por relevância, não injeção em massa.** Lições auto-gravadas compõem.
Injetar a lista inteira em todo prompt faria cada decisão futura passar pelo filtro das
crenças anteriores do sistema — um caminho silencioso para ele convergir nas próprias
opiniões e parar de olhar para fora. Claims já são recuperadas por relevância; não há
motivo para tratar lições diferente.

`provenance` guarda o evento do banco que gerou a lição, o que torna cada uma auditável
depois — e o `lithium memories --all` mostra todas, com `--forget` funcionando.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from lithium.db import Store
from lithium.db.store import normalize_memory_text
from lithium.llm import LLMClient, LLMError
from lithium.llm.prompts import BUDGET_MARGIN, budget_guard, estimate_tokens, render
from lithium.llm.schemas import PatternVerdict, ResearchLessons
from lithium.types import Directness
from lithium.worker.queue import iso, utcnow

log = logging.getLogger(__name__)

PROCESS_KINDS = frozenset({"dead_end", "search_lesson", "source_lesson"})
"""Sobre COMO pesquisar. Não afirmam nada sobre biologia, então não precisam de fonte."""

SUBSTANTIVE_KINDS = frozenset({"pattern"})
"""Padrão derivado sobre o domínio. Exige `claim_ids` de claims verificadas — mesmo
portão de citação da reancoragem, que só aceita PMID presente na evidência."""

RESEARCH_KINDS = PROCESS_KINDS | SUBSTANTIVE_KINDS

MAX_LESSONS_IN_PROMPT = 8
"""Teto do que é injetado. Lições auto-gravadas COMPÕEM: ao contrário de claims, que
são recuperadas por relevância, uma lista inteira injetada em todo prompt faria cada
decisão futura ser filtrada pelas crenças anteriores do próprio sistema — um caminho
silencioso para ele convergir nas próprias opiniões e parar de olhar para fora.

Por isso `relevant_lessons()` recupera por similaridade, e só `lessons_block()` (o
inventário) mostra tudo."""


@dataclass(slots=True)
class Lesson:
    id: int
    text: str
    kind: str
    provenance: str = ""
    """JSON cru. Carregado junto porque a renderização de uma lição depende de onde ela
    veio: um `dead_end` é renderizado a partir da hipótese que ele exclui, não do texto
    que o modelo escreveu. Ver `lessons_for_speculation`."""

    @property
    def refs(self) -> dict[int, tuple[str, int]]:
        try:
            raw = json.loads(self.provenance or "{}").get("refs") or {}
        except (TypeError, ValueError):
            return {}
        return {int(k): (v[0], int(v[1])) for k, v in raw.items() if len(v) == 2}

    @property
    def claim_ids(self) -> list[int]:
        try:
            return list(json.loads(self.provenance or "{}").get("claim_ids") or [])
        except (TypeError, ValueError):
            return []

NO_ACTIVITY = "(no research activity yet)"
"""Sentinela exata. `reflect()` compara por igualdade: um `startswith` casa com
qualquer bloco futuro que comece com a mesma palavra."""

CLAIMS_IN_ACTIVITY = 12
"""Quantas claims verificadas entram no bloco de claims. Ver a aritmética em
`_claims_block`. Sem teto o arranque produz centenas de linhas, `budget_guard`
levanta `PromptTooLarge` — que NÃO é `LLMError` e portanto não é capturado por
`reflect()` — e a tarefa de reflexão vai para o dead-letter a cada 24 h."""

SNIPPET = 60
"""Comprimento do rótulo durável que substitui `[k]` na gravação."""

_LOCAL_REF = re.compile(r"\[(\d{1,4})\]")
"""A sintaxe do índice local. Um único grupo numérico entre colchetes."""


@dataclass(frozen=True, slots=True)
class Ref:
    """O que um índice local `[k]` designa. O `kind` é o ponto todo: sem ele um
    índice que existe mas aponta para hipótese resolve COM SUCESSO para o objeto
    errado, que é pior do que não resolver."""

    kind: str            # "claim" | "hypothesis" | "search"
    id: int
    label: str = ""      # rótulo durável, para a proveniência (nunca vai a prompt)


@dataclass(slots=True)
class Activity:
    """O texto mostrado ao modelo E o conjunto exato do que foi mostrado.

    Os dois nascem do mesmo laço de renderização, de propósito: um `shown` montado
    por uma segunda query divergiria do texto no dia em que um `LIMIT` mudasse, e o
    portão passaria a autorizar id que o modelo nunca viu (ou a reprovar id que viu).
    """

    text: str
    shown: dict[int, Ref]

    @property
    def is_empty(self) -> bool:
        return self.text == NO_ACTIVITY


PATTERN_VERDICT_MAX_TOKENS = 256
"""O tempo é ~linear em tokens de SAÍDA (~6 tok/s), então este número É o custo. Quatro
booleanos mais uma frase cabem folgado; 256 é teto, não expectativa (~15 s medidos no
caso esperado). Pior caso `max_lessons=3` todas `pattern`: +1,0 chamada/dia sobre ~5,3.
"""


def pattern_is_entailed(verdict: PatternVerdict) -> bool:
    """As quatro respostas, todas consumidas.

    Escrito como função pura e testado pelo comportamento de `remember()`, não só por
    si: um portão que checa uma das quatro no call site passaria com esta função
    perfeita — foi exatamente o defeito que a auditoria construiu e viu verde.
    """
    return (
        verdict.follows_from_cited_claims_alone
        and not verdict.population_scope_exceeded
        and not verdict.contradicts_a_cited_claim_direction
        and not verdict.restates_a_premise
    )


class FabricatedReference(RuntimeError):
    """Índice citado que não estava no conjunto mostrado, ou que aponta para objeto
    de outro tipo. Exceção, não valor de retorno: um `None` devolvido é falsy e
    atravessa `if not verified` sem que ninguém note; uma exceção não capturada
    reprova a lição — falha fechada."""


SUBSTANTIVE_SLOT_CAP = 3
"""Máximo de slots para kinds auto-gerados de conteúdo (`pattern`, `dead_end`) num
bloco de 8.

Sem a cota o top-8 estacionário vira patterns + dead_ends — exatamente os dois kinds que
o próprio sistema escreveu, mantidos para sempre. E a consulta derivada de lacuna
**agrava** isso em relação ao literal genérico de antes: as lacunas são rótulos de classe
de intervenção, e só lição tópica casa com elas; `search_lesson`/`source_lesson` falam de
processo e não compartilham token nenhum com "anticonvulsant mood stabiliser". Medido num
pool tópico: sem cota, 5 `dead_end` + 2 `pattern` + 1 `search_lesson`. A cota é o
contrapeso da própria mudança, não enfeite — e ela **subiu** `pattern` de 2 para 3:
diversifica kind, não reduz monotonicamente o auto-gerado."""

CONTENT_KINDS = frozenset({"pattern", "dead_end"})
"""Os dois que afirmam (ou excluem) algo sobre o domínio."""


def _quota(lessons: Iterable[Lesson], k: int) -> list[Lesson]:
    """Aplica a cota preservando a ordem de entrada. Vale nos DOIS braços.

    Inclusive no curto-circuito fail-open: sem isso, o caminho que existe para sobreviver
    à queda do embedder seria justamente o que enche o prompt de conteúdo auto-gerado.
    """
    out: list[Lesson] = []
    content = 0
    for lesson in lessons:
        if len(out) >= k:
            break
        if lesson.kind in CONTENT_KINDS:
            if content >= SUBSTANTIVE_SLOT_CAP:
                continue
            content += 1
        out.append(lesson)
    return out


def _corpus_note(row) -> str:
    """O estado do corpus para esta intervenção, como FATO — nunca como número.

    Substitui a coluna `surprise` que o plano especificava, e a substituição é
    consequência de medição, não de preferência. Três resultados mataram o escalar:

    1. **Ele satura antes de servir.** Rodado pelo `_persist` real: com 10 claims, 100%
       das linhas ficam em `magnitude = 1.0`; com 100, 99%; com 400, 83%. Só a partir de
       ~4.000 ele começa a discriminar. Nos regimes em que este sistema vai viver, um
       limiar sobre ele dispara em tudo.
    2. **Estreia e notícia colidem no mesmo máximo.** Cinco grafias de quetiapina com o
       mesmo achado — `quetiapine`, `quetiapina`, `quetiapine XR`, `adjunctive
       quetiapine`, `Seroquel` — dão `frontier = 1.0` cada uma, valor **numericamente
       indistinguível** de um salto `extrapolated → direct`, que é a notícia mais forte
       que este sistema pode receber. Como `intervention` é texto livre de um 12B, ~93%
       dos eventos de surpresa máxima são artefato de identidade de string.
    3. **Um número no prompt mente.** Um 12B lendo `surprise: 1.0` conclui "achado
       marcante" quando o significado real é *nós nunca buscamos isto*.

    O que sobra é o que sempre foi verdade e é verificável: quantas claims existem para
    aquela intervenção, e se elas concordam. Isso é fato do corpus, some quando deixa de
    valer, e não tem escala para o modelo interpretar errado.
    """
    n_group = int(row["n_group"] or 0)
    if n_group <= 1:
        return "(no other verified claim for this intervention in the corpus)"
    if int(row["n_pos"] or 0) and int(row["n_contra"] or 0):
        return (
            f"(the corpus disagrees on this intervention: {row['n_pos']} positive, "
            f"{row['n_contra']} negative or null)"
        )
    return ""


def _lesson_of(row) -> Lesson:
    """Uma linha de `memories` vira `Lesson`. Um lugar só, para a proveniência não
    ficar de fora num call site e a lição chegar sem saber de onde veio."""
    return Lesson(
        id=row["id"], text=row["text"], kind=row["kind"],
        provenance=row["provenance"] if "provenance" in row.keys() else "",
    )


def _search_label(payload: dict) -> str:
    """Rótulo da busca, garantidamente sem id real.

    `label` chega como `spec:{hypothesis_id}` das buscas dirigidas por especulação.
    Esse inteiro é um id de hipótese na mesma faixa dos índices locais; renderizá-lo
    é oferecer ao modelo um número que resolve com sucesso para o objeto errado.
    """
    label = str(payload.get("label") or payload.get("strategy") or "?")
    return re.sub(r"\d+", "", label).rstrip(":-_") or "?"


def _plain(value: object, *, limit: int = 90) -> str:
    """Texto livre do modelo, seguro para um prompt de índices locais.

    Remove colchetes e dígitos soltos. Não é higiene genérica: o bloco de atividade usa
    `[k]` como espaço de nomes de referência, e a Fase 2 fechou exatamente a conflação em
    que um número renderizado ali resolve COM SUCESSO para o objeto errado. `seeking` é
    string livre, e "elo 3" põe um dígito na posição onde o modelo aprendeu a ler índice.
    """
    text = re.sub(r"[\[\]]", "", str(value or "")).strip()
    # Dígito ISOLADO, não qualquer dígito: `\b\d+\b` casaria o "1" de `sigma-1`
    # (o hífen é fronteira de palavra) e mutilaria justamente os nomes de alvo que
    # este campo existe para carregar — `5-HT1A`, `GABA-A`, `sigma-1`.
    text = re.sub(r"(?<![\w-])\d+(?![\w-])", "", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def resolve_refs(text: str, shown: dict[int, Ref]) -> str:
    """Troca `[k]` pelo rótulo durável do objeto. Aplicado NA GRAVAÇÃO.

    Sem isto, `[k]` é a única coisa que a lição carrega — e ele significa outra coisa
    na renderização seguinte, então a proveniência que hoje diz "hypothesis 7" viraria
    ruído. O rótulo é um trecho CITADO, não um id: um `hypothesis #41` gravado aqui
    voltaria pelo `lessons_block` para dentro do mesmo prompt que tem índices locais,
    recriando os dois namespaces que o esquema existe para eliminar.
    """
    def swap(m: re.Match[str]) -> str:
        ref = shown.get(int(m.group(1)))
        return ref.label if ref is not None and ref.label else m.group(0)

    return _LOCAL_REF.sub(swap, text)


class Reflector:
    def __init__(self, store: Store, llm: LLMClient, embedder, *,
                 profile=None, n_ctx: int = 8192) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.profile = profile
        """Opcional: só `verify_pattern` precisa dele, e só quando a reflexão de fato
        deriva um `pattern`. Exigi-lo no construtor obrigaria todo caminho de reflexão
        (que é agnóstico a foco) a carregar perfil que ele não usa."""
        self.n_ctx = n_ctx

    # ─────────────────────────────────────────────────────── o que aconteceu

    def recent_activity(
        self, *, limit: int = 12, claim_limit: int = CLAIMS_IN_ACTIVITY
    ) -> Activity:
        """Fatos do banco, não interpretação — e o conjunto exato do que foi mostrado.

        **Um único espaço de índices.** Todo objeto citável recebe `[k]` monotônico,
        e nenhum id real aparece ao lado de um `[k]`. Três namespaces coexistindo
        (índice local, id de claim, id de hipótese) é o cenário ruim: ids de claim e
        de hipótese são ambos `INTEGER PRIMARY KEY` em tabelas separadas, então uma
        conflação resolve com SUCESSO para o objeto errado. Fechar por conjunto
        mostrado só serve se houver uma coisa só para o modelo copiar.

        PMID é a exceção argumentada: fica no bloco de fontes estéreis, que **não
        tem `[k]` nenhum**. Uma fonte sem claim verificada nunca é alvo legítimo de
        `claim_ids`, então dar índice a ela criaria o risco de conflação sem
        benefício; e o PMID é o único identificador que mantém um `source_lesson`
        auditável um mês depois, quando `[3]` já designa outra coisa.
        """
        lines: list[str] = []
        shown: dict[int, Ref] = {}
        k = 0

        refuted = self.store.conn.execute(
            "SELECT id, statement, critique_json FROM hypotheses "
            " WHERE tier = 'speculative' AND survives_critique = 0 "
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if refuted:
            lines.append("### Hypotheses refuted by the critique pass")
            for r in refuted:
                critique = json.loads(r["critique_json"] or "{}")
                k += 1
                shown[k] = Ref("hypothesis", int(r["id"]),
                               f'the hypothesis "{r["statement"][:SNIPPET]}"')
                lines.append(f"  [{k}] hypothesis: {r['statement'][:130]}")
                if critique.get("fatal_flaw"):
                    lines.append(f"      fatal flaw: {critique['fatal_flaw'][:150]}")
                if critique.get("weakest_link"):
                    lines.append(f"      weakest link: {critique['weakest_link'][:150]}")

        sterile = self.store.conn.execute(
            "SELECT id, payload_json FROM tasks "
            " WHERE kind = 'harvest_query' AND status = 'done' "
            " ORDER BY id DESC LIMIT ?",
            (limit * 2,),
        ).fetchall()
        known = {
            r["external_id"]
            for r in self.store.conn.execute("SELECT external_id FROM sources")
        }
        if sterile:
            lines.append("")
            lines.append("### Recent searches (corpus now holds "
                         f"{len(known)} sources)")
            for r in sterile:
                payload = json.loads(r["payload_json"])
                query = payload["query"][:120]
                k += 1
                shown[k] = Ref("search", int(r["id"]), f'the search "{query[:SNIPPET]}"')
                # O rótulo sai do colchete e perde o sufixo numérico: `spec:3` é um id
                # REAL de hipótese, dentro da mesma faixa de inteiros pequenos dos
                # índices locais, renderizado exatamente na posição onde o modelo
                # aprendeu a ler índice local. É a conflação de pior desfecho.
                lines.append(f"  [{k}] {_search_label(payload)} {query}")
                # O elo que a busca tentava ancorar. Sem ele a lição resultante é
                # "esta query não retornou nada"; com ele, "queries visando este elo
                # não retornam nada" — que diz ONDE a cadeia está sem chão.
                seeking = _plain(payload.get("seeking"))
                if seeking:
                    lines.append(f"      aiming to anchor: {seeking}")

        barren = self.store.conn.execute(
            "SELECT s.external_id, s.journal, COUNT(c.id) AS n_claims "
            "  FROM sources s LEFT JOIN claims c ON c.source_id = s.id AND c.verified = 1 "
            " GROUP BY s.id HAVING n_claims = 0 ORDER BY s.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if barren:
            lines.append("")
            lines.append("### Sources that yielded zero verified claims "
                         "(unindexed: nothing here to cite)")
            for r in barren:
                lines.append(f"  PMID:{r['external_id']} ({r['journal'] or '—'})")

        claim_lines, shown = self._claims_block(k, shown, limit=claim_limit)
        if claim_lines:
            lines.append("")
            lines.extend(claim_lines)

        return Activity("\n".join(lines) if lines else NO_ACTIVITY, shown)

    def _claims_block(
        self, k: int, shown: dict[int, Ref], *, limit: int
    ) -> tuple[list[str], dict[int, Ref]]:
        """As claims verificadas mais fortes do corpus, com índice local.

        Existe porque sem ela `pattern` é impossível de produzir honestamente: os
        outros três blocos não mencionam claim nenhuma, então o modelo que quisesse
        registrar um padrão substantivo teria de ADIVINHAR inteiros — e o inteiro
        mais à mão era um id de hipótese.

        **Ordem: peso, com desempate alfabético.** `ORDER BY c.id DESC` seria a
        escolha natural e está errada: `claims.id` é monotônico na inserção, logo é
        proxy exato de `extracted_at` — que a TRAVA 3 proíbe justamente por ser
        anti-correlacionado com directness. A trava varre tokens de tempo e não pega
        `id`; a proibição é de ler ORDEM DE COLHEITA, e `id` é ordem de colheita com
        outro nome. O desempate por `statement` é arbitrário e por isso não enviesa.

        **LIMIT = `CLAIMS_IN_ACTIVITY` = 12, com a aritmética medida.** Medido com
        `estimate_tokens` num corpus de 400 claims / 40 lições / 40 hipóteses
        refutadas / 60 buscas / 30 fontes estéreis:

            orçamento de prompt   = 8192 (n_ctx) − 256 (margem) − 1024 (saída) = 6912
            reflect.md estático   = 1118
            três blocos antigos   = 2457   (12 hipóteses + 24 buscas + 12 fontes)
            lessons_block, 40 lições × 200 chars              = 2227
            ────────────────────────────────────────────────────────────────
            subtotal SEM o bloco de claims                    = 5803  (folga 1109)
            linha de claim medida = 43,7 tokens; 12 linhas     =  524
            total                                             = 6336  (folga 576)

        Doze é o mesmo `limit` dos outros blocos, e não por estética: 20 linhas
        deixam 226 tokens de folga — menos que a própria `BUDGET_MARGIN`, que existe
        para cobrir a diferença entre esta estimativa de 4 chars/token e o
        tokenizador real. O termo que realmente manda é `lessons_block`:
        `ResearchLesson.text` não tem `max_length`, e 40 lições de 320 caracteres
        sozinhas (3427 tokens) estouram o orçamento COM ZERO claims. Por isso o teto
        aqui é necessário mas não suficiente, e `reflect()` tem o degrau que descarta
        este bloco.
        """
        if limit <= 0:
            return [], shown
        rows = self.store.conn.execute(
            # Uma claim por intervenção, a de maior peso. Sem o GROUP BY, um corpus
            # de 400 claims devolve 12 linhas todas `meta_analysis/direct` — e como
            # elas empatam em peso, o desempate alfabético decide, então o bloco vira
            # uma fatia por letra inicial. Doze restatements da mesma coisa também
            # tornam qualquer "padrão" derivado deles circular.
            #
            # `MAX(cw.weight)` com colunas nuas: no SQLite as colunas nuas vêm da
            # linha do máximo (comportamento documentado para min/max agregado).
            "SELECT c.id, c.statement, c.grade, cd.directness, "
            "       COUNT(*) AS n_group, "
            "       SUM(CASE WHEN c.direction = 'positive' THEN 1 ELSE 0 END) AS n_pos, "
            "       SUM(CASE WHEN c.direction IN ('negative', 'null', 'no_effect') "
            "                THEN 1 ELSE 0 END) AS n_contra, "
            "       MAX(cw.weight) AS w "
            "  FROM claims c JOIN claim_weight cw ON cw.claim_id = c.id "
            "       JOIN claim_directness cd ON cd.claim_id = c.id "
            "                                AND cd.focus_id = (SELECT id FROM active_focus) "
            " GROUP BY COALESCE(NULLIF(TRIM(c.intervention), ''), 'id:' || c.id) "
            " ORDER BY w DESC, c.statement ASC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return [], shown
        out = ["### Verified claims — cite these by their index in `claim_ids`"]
        for r in rows:
            k += 1
            shown[k] = Ref("claim", int(r["id"]),
                           f'the claim "{r["statement"][:SNIPPET]}"')
            # Duas entradas em vez de uma concatenada: `f"...{r['grade']}..." + x` é um
            # `BinOp` com os nomes dos fatores dentro, e a TRAVA 2 varre BinOp por AST.
            # Ela está certa em não distinguir formatação de aritmética — a alternativa
            # seria isentar concatenação de string, e isentar é como uma trava afrouxa.
            out.append(f"  [{k}] ({r['grade']}/{r['directness']}) {r['statement'][:140]}")
            note = _corpus_note(r)
            if note:
                out.append(f"      {note}")
        return out, shown

    def lessons(self, *, limit: int = 40) -> list[Lesson]:
        rows = self.store.conn.execute(
            "SELECT id, text, kind FROM research_lessons ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Lesson(id=r["id"], text=r["text"], kind=r["kind"]) for r in rows]

    def lessons_block(self) -> str:
        """Inventário completo — para o prompt de reflexão (que precisa ver tudo para
        não repetir) e para o `lithium memories`. Prompts de trabalho usam
        `relevant_lessons`."""
        rows = self.lessons()
        if not rows:
            return "(none yet)"
        return "\n".join(f"  [{r.kind}] {r.text}" for r in rows)

    # ────────────────────────────────────────────────────────────── reflexão

    async def reflect(self, *, max_lessons: int = 3) -> list[Lesson]:
        activity = self.recent_activity()
        if activity.is_empty:
            return []

        lessons_block = self.lessons_block()
        prompt = render("reflect", lessons=lessons_block, activity=activity.text)
        if estimate_tokens(prompt) + 1024 > self.n_ctx - BUDGET_MARGIN:
            # O bloco de claims é o amortecedor: é o mais novo, o único cujo tamanho
            # controlamos linha por linha, e o único que não é a razão de a reflexão
            # existir. Sem este degrau, `budget_guard` levanta `PromptTooLarge` — que
            # não é `LLMError`, portanto não é capturado abaixo — e a tarefa vai para
            # o dead-letter a cada 24 h durante todo o arranque.
            log.warning("reflect: prompt em ~%d tokens, removendo o bloco de claims",
                        estimate_tokens(prompt))
            activity = self.recent_activity(claim_limit=0)
            prompt = render("reflect", lessons=lessons_block, activity=activity.text)

        try:
            budget_guard(prompt, label="reflect", max_tokens=1024, n_ctx=self.n_ctx)
            result = await self.llm.structured(
                [{"role": "user", "content": prompt}],
                ResearchLessons,
                max_tokens=1024,
                label="reflect",
            )
        except LLMError as exc:
            log.warning("reflexão falhou: %s", exc)
            return []

        saved: list[Lesson] = []
        for item in result.lessons[:max_lessons]:
            if item.kind not in RESEARCH_KINDS:
                # Um `kind` fora do conjunto costuma sinalizar que o modelo escorregou
                # para conclusão sobre o mundo, que é exatamente o que não pode entrar.
                log.info("lição descartada por kind inválido (%r): %s",
                         item.kind, item.text[:80])
                continue
            if not item.text.strip():
                continue
            try:
                lesson = await self.remember(
                    item.text, item.kind, item.provenance_note, item.claim_ids,
                    shown=activity.shown,
                )
            except FabricatedReference as exc:
                log.warning("lição reprovada por referência fabricada (%s): %s",
                            exc, item.text[:80])
                continue
            if lesson is not None:
                saved.append(lesson)

        if saved:
            log.info("reflexão: %d lição(ões) gravada(s) automaticamente", len(saved))
        return saved

    def verify_claim_ids(self, refs: list[int], *, shown: dict[int, Ref]) -> list[int]:
        """Traduz índices locais em ids de claim. Tudo ou nada.

        `shown` é KEYWORD OBRIGATÓRIO E SEM DEFAULT de propósito. Um default tornaria
        todo call site que não foi atualizado silenciosamente permissivo — que é
        exatamente o estado anterior desta função, onde a garantia real não era o
        projeto e sim a INANIÇÃO DE INFORMAÇÃO: o modelo não recebia id de claim
        nenhum, então tinha de adivinhar. E o inteiro mais disponível no prompt era um
        id de HIPÓTESE, no mesmo espaço numérico. Medido: `[3, 9999]` com a hipótese
        3 existindo gravava com `[3]`.

        **Tudo ou nada, não filtro.** Filtrar dá crédito parcial: a lição entra com o
        subconjunto que por acaso existia e a fabricação não deixa rastro. Um índice
        fora do conjunto mostrado, ou dentro dele mas apontando para hipótese ou
        busca, reprova a lição inteira e é registrado.
        """
        if not refs:
            return []
        resolved: list[int] = []
        for ref_id in refs:
            ref = shown.get(ref_id)
            if ref is None:
                raise FabricatedReference(
                    f"índice [{ref_id}] não estava entre os {len(shown)} mostrados"
                )
            if ref.kind != "claim":
                raise FabricatedReference(
                    f"índice [{ref_id}] é {ref.kind}, não claim"
                )
            resolved.append(ref.id)

        unique = sorted(set(resolved))
        holes = ",".join("?" for _ in unique)
        alive = {
            int(r["id"])
            for r in self.store.conn.execute(
                f"SELECT id FROM claims WHERE verified = 1 AND id IN ({holes})",
                unique,
            )
        }
        # Defesa em profundidade: o bloco já filtra por `claim_weight` (que é
        # `verified = 1`), então uma divergência aqui significa que o conjunto
        # mostrado deixou de ser o conjunto citável. Reprova, não conserta.
        missing = [i for i in unique if i not in alive]
        if missing:
            raise FabricatedReference(
                f"claim(s) {missing} mostradas mas não verificadas no banco"
            )
        return unique

    async def remember(
        self, text: str, kind: str, provenance: str, refs: list[int] | None = None,
        *, shown: dict[int, Ref] | None = None,
    ) -> Lesson | None:
        """Grava direto, sem confirmação — é o regime de consentimento da pesquisa.

        `refs` são ÍNDICES LOCAIS do bloco de atividade, não ids de claim. Citar sem
        passar `shown` é erro de programação, não do modelo, e levanta — não pode
        virar "grava sem verificar" nem "descarta em silêncio".
        """
        if refs and shown is None:
            raise ValueError(
                "remember() recebeu refs sem `shown`: não há como saber o que foi "
                "mostrado, e verificar contra o banco daria crédito a índice inventado"
            )
        verified = self.verify_claim_ids(refs or [], shown=shown or {})
        # A proveniência machine-readable é montada ANTES da resolução: depois dela os
        # `[k]` não existem mais no texto. Inclui os índices citados no `note`, não só
        # os de `claim_ids` — o elo entre um `dead_end` e a hipótese que o gerou é o
        # que hoje a nota "hypothesis 7" carrega, e ele não pode se perder na troca
        # para índice local.
        cited = list(refs or []) + [int(m) for m in _LOCAL_REF.findall(f"{provenance}\n{text}")]
        seen = {i: shown[i] for i in cited if i in (shown or {})}
        if shown:
            text = resolve_refs(text, shown)
            provenance = resolve_refs(provenance, shown)

        if kind in SUBSTANTIVE_KINDS and not verified:
            log.info(
                "padrão descartado por falta de claim verificada: %s", text[:80]
            )
            return None

        entailment: dict | None = None
        if kind in SUBSTANTIVE_KINDS and not self._window_touched_literature():
            log.info(
                "padrão descartado: a janela não contém claim verificada nova — "
                "seria o sistema generalizando sobre a própria opinião: %s", text[:70]
            )
            return None

        if kind in SUBSTANTIVE_KINDS:
            verdict = await self._pattern_entails(text, verified)
            if verdict is None or not pattern_is_entailed(verdict):
                log.info(
                    "padrão descartado pelo portão de derivação (%s): %s",
                    verdict.reason[:90] if verdict else "verificador indisponível",
                    text[:80],
                )
                return None
            entailment = verdict.model_dump()

        # Dedup na APLICAÇÃO, com o índice como defesa em profundidade — não o
        # contrário. `_migrate_indexes` degrada (não levanta) num banco que já tem
        # duplicatas, e sem esta checagem o estado degradado se autoalimenta: o daemon
        # continua gravando repetidas e o backlog a limpar cresce sozinho.
        key = normalize_memory_text(text)
        existing = self.store.conn.execute(
            "SELECT id, provenance FROM memories "
            " WHERE source = 'research' AND active = 1 AND text_key = ?",
            (key,),
        ).fetchone()
        if existing is not None:
            self._merge_provenance(existing, verified)
            log.info("lição já conhecida, não regravada: %s", text[:80])
            return None

        [vector] = await self.embedder.embed([text])
        cur = self.store.conn.execute(
            "INSERT INTO memories(text, kind, rationale, source, provenance, "
            "                     confirmed, embedding, confirmed_at, text_key) "
            "VALUES(?, ?, ?, 'research', ?, 1, ?, ?, ?) "
            "ON CONFLICT DO NOTHING RETURNING id",
            (
                text.strip(),
                kind,
                "aprendizado de pesquisa, gravado automaticamente",
                json.dumps({"note": provenance, "claim_ids": verified,
                            "refs": {str(i): [r.kind, r.id] for i, r in seen.items()},
                            "entailment": entailment}),
                Store.pack_embedding(vector),
                iso(utcnow()),
                key,
            ),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return Lesson(id=int(row["id"]), text=text.strip(), kind=kind)

    LITERATURE_MARK_PREFIX = "reflect:last_verified_claim_id"
    """Onde fica a marca d'água do último contato com a literatura, por foco."""

    @property
    def LITERATURE_MARK(self) -> str:  # noqa: N802
        """Escopada por foco. Sem isso, trocar de foco herda a marca do anterior e o
        portão de `pattern` fica fechado para claims que o novo foco nunca viu."""
        focus = self.store.active_focus()
        return f"{self.LITERATURE_MARK_PREFIX}:{focus['id'] if focus else 0}"

    def _window_touched_literature(self) -> bool:
        """A janela desta reflexão contém pelo menos uma claim verificada nova?

        Um `pattern` é o sistema afirmando algo sobre biologia. Ele não pode ser derivado
        de uma janela cujo conteúdo inteiro é opinião do próprio sistema: `_critique` não
        recebe corpus nenhum, então `fatal_flaw` é a opinião de um LLM sobre a saída de
        outro. Sem esta condição, três hipóteses refutadas por opinião viram um "padrão
        substantivo" com proveniência de aparência impecável.

        **Conta claim verificada, não fonte.** Contar fontes deixa o portão totalmente
        aberto exatamente na janela que ele existe para fechar: uma fonte ÁRIDA — zero
        claims verificadas — é o processo escrevendo sobre si mesmo, e são precisamente as
        fontes que `recent_activity()` renderiza sob "Sources that yielded zero verified
        claims". Medido: 30 fontes áridas + 40 buscas concluídas davam
        `external_delta = 30` e o `pattern` passava.
        """
        row = self.store.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (self.LITERATURE_MARK,)
        ).fetchone()
        mark = int(row["value"]) if row else 0
        # Sobre `claim_weight`, e não `MAX(id) FROM claims WHERE verified = 1`: o universo
        # global inclui claim SEM julgamento neste foco, e claim sem peso é a versão nova
        # das "30 fontes áridas" — corpus que cresce sem que este foco tenha aprendido
        # nada com ele.
        newest = self.store.conn.execute(
            "SELECT COALESCE(MAX(cw.claim_id), 0) AS n FROM claim_weight cw"
        ).fetchone()["n"]
        return int(newest) > mark

    def mark_literature_seen(self) -> None:
        """Fecha a janela. Chamado pelo handler ao FIM da passada de reflexão.

        Ao fim, não ao início: uma marca avançada antes de as lições serem escritas
        fecharia o portão para a própria passada que a colheita habilitou.
        """
        newest = self.store.conn.execute(
            "SELECT COALESCE(MAX(cw.claim_id), 0) AS n FROM claim_weight cw"
        ).fetchone()["n"]
        self.store.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (self.LITERATURE_MARK, str(int(newest))),
        )

    async def _pattern_entails(
        self, text: str, claim_ids: list[int]
    ) -> PatternVerdict | None:
        """A chamada de verificação. **Falha fechada**, como o portão 2 da extração.

        Roda DEPOIS do fechamento por conjunto-mostrado (que garante que as claims
        citadas foram de fato mostradas) e ANTES do dedup — ao custo de uma chamada numa
        lição repetida.

        A ordem inversa economizaria essa chamada e abriria um bypass concreto, que não é
        o óbvio: uma lição cujo texto colide com uma existente **não** entraria de todo
        jeito (o dedup devolve None), mas `_merge_provenance` roda no caminho de colisão
        e **amplia `claim_ids` da lição que já existe**. Com o portão depois, uma
        rederivação que o portão reprovaria inflaria o suporte declarado de um `pattern`
        já gravado — subir de "apoiado em 1 claim" para "apoiado em 3" sem que nenhuma
        das duas novas tenha passado por verificação de derivação.
        """
        holes = ",".join("?" for _ in claim_ids)
        rows = self.store.conn.execute(
            f"SELECT c.id, c.statement, c.population, c.direction, cd.directness "
            f"  FROM claims c "
            f"  JOIN claim_directness cd ON cd.claim_id = c.id "
            f"                           AND cd.focus_id = (SELECT id FROM active_focus) "
            f" WHERE c.id IN ({holes})",
            claim_ids,
        ).fetchall()
        claims = "\n".join(
            f"  - {r['statement']}\n"
            f"      population: {r['population'] or 'not stated'} · "
            f"direction: {r['direction']} · directness: {r['directness']}"
            for r in rows
        )
        try:
            return await self.llm.structured(
                [{"role": "user", "content": render(
                    "verify_pattern", pattern=text, claims=claims,
                    **_pattern_blocks(self.profile))}],
                PatternVerdict,
                max_tokens=PATTERN_VERDICT_MAX_TOKENS,
                label="verify_pattern",
            )
        except LLMError as exc:
            log.warning("verificador de padrão falhou: %s", exc)
            return None

    def _merge_provenance(self, existing, claim_ids: list[int]) -> None:
        """Acrescenta claims novas à proveniência da lição que já existe.

        Sem isto a deduplicação **subdeclara o suporte**: a primeira derivação congela
        `claim_ids`, e uma rederivação apoiada em mais claims verificadas some. Para um
        `pattern` — a única categoria com portão de claim — isso é perder exatamente o
        que o portão existe para exigir.
        """
        if not claim_ids:
            return
        try:
            prov = json.loads(existing["provenance"] or "{}")
        except (TypeError, ValueError):
            return
        known = prov.get("claim_ids") or []
        merged = sorted(set(known) | set(claim_ids))
        if merged == sorted(known):
            return
        prov["claim_ids"] = merged
        self.store.conn.execute(
            "UPDATE memories SET provenance = ? WHERE id = ?",
            (json.dumps(prov), existing["id"]),
        )
        log.info(
            "proveniência da lição #%d ampliada: %d -> %d claims",
            existing["id"], len(known), len(merged),
        )

    async def relevant_lessons(
        self, queries: Sequence[str], *, k: int = MAX_LESSONS_IN_PROMPT
    ) -> list[Lesson]:
        """Recupera por relevância a UMA LACUNA DE CADA VEZ, em rodízio.

        **`queries`, plural, e é o ponto da frente.** O único chamador passava um
        literal fixo, então o ranking era idêntico em toda rodada pela vida do sistema —
        as mesmas oito lições em todo prompt de especulação, para sempre. Não era
        recuperação por relevância, era injeção de um conjunto congelado.

        **Rodízio, não RRF.** Escolhi RRF primeiro e a auditoria mostrou que a garantia
        que eu queria não existia nele: o teste que dizia "uma lacuna cuja melhor lição é
        globalmente mediana ainda recebe seu slot" passava por causa da **cota por kind**,
        não da fusão. Trocando o `kind` da lição solitária para o das concorrentes, a
        propriedade ficava falsa — a lacuna ficava sem resposta. Rodízio (top-1 de cada
        lacuna primeiro, depois top-2, e assim por diante) torna a garantia estrutural:
        toda lacuna com alguma lição elegível recebe um slot antes de qualquer lacuna
        receber o segundo.

        Aceita `Sequence[str]` mas **recusa `str`**: `list("uma frase")` daria uma
        consulta por caractere, ranking puro ruído, sem levantar nada. É exatamente como
        o call site antigo voltaria a passar em silêncio.
        """
        if isinstance(queries, str):
            raise TypeError(
                "relevant_lessons espera uma sequência de consultas, uma por lacuna. "
                "Uma string seria iterada caractere por caractere."
            )
        rows = self.store.conn.execute(
            "SELECT m.id, m.text, m.kind, m.provenance, m.embedding FROM memories m "
            " WHERE m.source = 'research' AND m.confirmed = 1 AND m.active = 1 "
            "   AND m.embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return []

        queries = [q for q in queries if q.strip()]
        if not queries or len(rows) <= k:
            # Caminho fail-open: sem consulta útil (ou com estoque menor que o teto),
            # devolve tudo com a cota aplicada. É a única defesa contra o embedder cair.
            return _quota(( _lesson_of(r) for r in rows ), k)

        import numpy as np

        vectors = await self.embedder.embed(list(queries))
        rankings: list[list] = []
        for vector in vectors:
            target = np.asarray(vector, dtype=np.float32)
            scored = []
            for r in rows:
                vec = Store.unpack_embedding(r["embedding"])
                denom = float(np.linalg.norm(vec) * np.linalg.norm(target))
                scored.append((float(vec @ target) / denom if denom else 0.0, r["id"], r))
            # `-id` no desempate: determinístico e independente da ordem de colheita.
            scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
            rankings.append([r for _, _, r in scored])

        ordered: list = []
        seen: set[int] = set()
        for depth in range(len(rows)):
            for ranking in rankings:
                if depth < len(ranking) and ranking[depth]["id"] not in seen:
                    seen.add(ranking[depth]["id"])
                    ordered.append(ranking[depth])
        return _quota((_lesson_of(r) for r in ordered), k)

    # ─────────────────────────────────── o que a trilha de especulação recebe

    def lessons_for_speculation(self, lessons: list[Lesson]) -> str:
        """Renderiza as lições em **três seções com autoridades diferentes**.

        Duas correções, e as duas são sobre o que o bloco *causa*, não sobre formatação.

        **1. A prosa de um `dead_end` não entra.** `generate_speculation.md` instrui
        "não caia num modo de falha já nomeado aqui", então a linha injetada é uma
        **exclusão**: ela remove uma classe de mecanismo do espaço de busca. Injetar o
        texto que o modelo escreveu deixa esse poder sem portão — uma frase como "logo
        TODOS os mecanismos glutamatérgicos são becos sem saída e nunca devem ser
        propostos" atravessa como instrução, e nada no sistema reporta "hipóteses que
        nunca foram geradas": a falha não tem assinatura observável.

        Então a exclusão é renderizada a partir do **banco**: o `statement` da hipótese
        referenciada mais o `fatal_flaw` que a crítica de fato escreveu. Se o referente
        deixou de estar refutado, a lição é **omitida** — nunca há fallback para a prosa
        do modelo, porque o fallback seria justamente a porta que isto fecha.

        **2. `pattern` não pode cair na seção de notas.** É o único canal substantivo
        com portão de fonte; renderizá-lo sob um cabeçalho que diz "não afirmam nada
        sobre biologia e não devem estreitar o espaço de mecanismos" manda o gerador
        ignorar exatamente a lição melhor fundamentada que o sistema tem. Seção própria,
        com o qualificador de força e a contagem de claims.
        """
        exclusions, patterns, notes = [], [], []
        for lesson in lessons:
            if lesson.kind == "dead_end":
                line = self._exclusion_line(lesson)
                if line:
                    exclusions.append(line)
            elif lesson.kind in SUBSTANTIVE_KINDS:
                patterns.append(self._pattern_line(lesson))
            else:
                notes.append(f"  [{lesson.kind}] {lesson.text}")

        if not (exclusions or patterns or notes):
            return "(none)"

        blocks: list[str] = []
        if exclusions:
            blocks.append(
                "### Hypotheses already refuted — do not propose these again\n"
                "Each line is the refuted hypothesis and the flaw the critique found.\n"
                "A refutation of one agent is NOT a refutation of its whole class.\n"
                + "\n".join(exclusions)
            )
        if patterns:
            blocks.append(
                "### Substantive patterns, each grounded in verified claims\n"
                "The suffix is the WEAKEST directness among the claims it came from — "
                "`extrapolated` means the pattern rests on material outside this "
                "population.\n" + "\n".join(patterns)
            )
        if notes:
            blocks.append(
                "### Notes about how to search\n"
                "Process observations. They assert nothing about biology and must not "
                "narrow the mechanism space.\n" + "\n".join(notes)
            )
        return "\n\n".join(blocks)

    def _exclusion_line(self, lesson: Lesson) -> str | None:
        hypothesis_ids = [i for kind, i in lesson.refs.values() if kind == "hypothesis"]
        if not hypothesis_ids:
            return None
        row = self.store.conn.execute(
            "SELECT statement, critique_json FROM hypotheses "
            " WHERE id = ? AND survives_critique = 0",
            (hypothesis_ids[0],),
        ).fetchone()
        if row is None:
            # Referente não mais refutado (ou apagado): a exclusão perdeu a base. Sai
            # de cena em silêncio, sem virar a prosa do modelo.
            return None
        flaw = (json.loads(row["critique_json"] or "{}").get("fatal_flaw") or "").strip()
        line = f"  - {row['statement'][:130]}"
        return f"{line}\n      refuted because: {flaw[:180]}" if flaw else line

    def _pattern_line(self, lesson: Lesson) -> str:
        claim_ids = lesson.claim_ids
        worst = self.weakest_directness(claim_ids)
        # TERCEIRO estado. As lições atravessam o foco (são `memories`, sem `focus_id`
        # — é o "cérebro só" que o usuário pediu), mas o QUALIFICADOR que as mantém
        # honestas é escopado por foco. Num foco que ainda não relensou não há linha em
        # `claim_directness`, o sufixo ficava VAZIO, e a lição era renderizada com
        # autoridade total logo abaixo de um cabeçalho que explica o que o sufixo
        # significa. Ausência de sufixo lê como "não precisa de ressalva": fail-OPEN,
        # exatamente onde a Fase A escolheu fail-closed.
        if worst is not None:
            suffix = f" · {worst.value}"
        elif claim_ids:
            suffix = " · não julgado neste foco"
        else:
            suffix = ""
        n = len(claim_ids)
        support = f"  (from {n} verified claim{'s' if n != 1 else ''})" if n else ""
        return f"  [pattern{suffix}] {lesson.text}{support}"

    def weakest_directness(self, claim_ids: list[int]) -> Directness | None:
        """O pior `directness` entre as claims citadas.

        Existe porque a abstração **lava o qualificador**: três claims
        `preclinical × extrapolated` em roedor tornam-se uma linha com exatamente a
        mesma autoridade visual de um padrão derivado de `meta_analysis × direct`.

        Ordena pela posição no enum, não por `DIRECTNESS_WEIGHT` — isto é ordenação de
        uma escala, não cálculo de peso de claim, e a invariante reserva os três fatores
        à view `claim_weight`.

        Devolve `None` em DOIS casos que o chamador precisa distinguir: nenhuma claim
        citada, e claims citadas SEM julgamento neste foco. Ver `_pattern_line`.
        """
        if not claim_ids:
            return None
        holes = ",".join("?" for _ in claim_ids)
        rows = self.store.conn.execute(
            f"SELECT cd.directness FROM claims c "
            f"  JOIN claim_directness cd ON cd.claim_id = c.id "
            f"                           AND cd.focus_id = (SELECT id FROM active_focus) "
            f" WHERE c.id IN ({holes})", claim_ids
        ).fetchall()
        levels = list(Directness)
        found = [Directness(r["directness"]) for r in rows]
        return max(found, key=levels.index) if found else None


def _pattern_blocks(profile) -> dict[str, str]:
    """Os blocos de perfil de `verify_pattern`, com um default HONESTO sem perfil.

    Sem perfil o portão continua rodando, mas a frase de hierarquia de população fica
    genérica em vez de citar o par do domínio. Preencher com o par do foco ANTIGO seria
    pior que genérico: o portão passaria a policiar uma fronteira que não é a deste
    foco, e é ele que decide o que entra na memória de longo prazo.
    """
    if profile is None:
        return {"population_hierarchy": "any widening of the population studied"}
    return profile.prompt_blocks("verify_pattern")
