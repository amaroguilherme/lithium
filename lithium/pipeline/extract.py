"""Extração de claims, com dois portões de verificação antes de tocar o banco.

**Portão 1 — ancoragem, determinístico.** `supporting_quote` precisa ser um trecho
literal do chunk. Sem LLM, sem custo, e sem a possibilidade de o modelo se absolver.
Um modelo que parafraseia a citação está inventando, e isso é detectável por
comparação de string. Este portão sozinho barra a maior parte da alucinação.

**Portão 2 — implicação, por LLM.** A citação existe, mas ela realmente sustenta a
alegação? Aqui cai o erro mais comum e mais perigoso de um 12B: citar corretamente
uma frase sobre TAG puro e escrever uma alegação sobre bipolar I. Roda só nos
sobreviventes do portão 1, então o custo é proporcional ao que já é plausível.

Claim que não passa nos dois **não entra no banco**. Isso importa além da correção
imediata: `claim_weight` só enxerga `verified = 1`, então material não verificado
fica fora do placar de hipóteses, dos relatórios e do dataset de treino da LoRA.
Contaminar o treino com a própria alucinação é como um sistema desses apodrece.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from lithium.db import Store
from lithium.focus import FocusProfile, NoActiveFocus
from lithium.llm import LLMClient, LLMError
from lithium.llm.prompts import budget_guard, render
from lithium.llm.schemas import CitationVerdict, ClaimExtraction, ExtractedClaim
from lithium.types import Grade

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


_NON_INTERVENTION = frozenset({"none", "n/a", "null", "nenhum", "nenhuma", "-"})
"""O que um 12B escreve em `intervention` quando o certo era deixar vazio.

`intervention` está em `required` da gramática, então o modelo é OBRIGADO a emitir a
chave; quando não há intervenção ele devolve a palavra `none`. MEDIDO: 3 das 213 claims
entraram assim, e no corpus real a linha `none` era a **segunda por peso** da tabela de
cobertura (w=2,55, acima de `pramipexole`, que tem 10 claims) — o gerador de perguntas
lia `none` como a segunda intervenção mais evidenciada do corpus. `"none" or None` é
`"none"`, então o `or None` de `_persist` nunca pegou isso.

Este portão é o único DETERMINÍSTICO do conserto: a prosa do prompt depende de um 12B
obedecer, isto não.

Deliberadamente NÃO é o `_EMPTY_MARKERS` de `explore.py`, e a diferença é o ponto: lá o
campo é texto livre de CRÍTICA, onde `NA` não pode ser fármaco. Aqui `NA` é abreviação
padrão de noradrenalina, e uma normalização é irreversível — grava NULL e o texto
original some. Reusar o frozenset entre campos de espaços de valor diferentes é o que
torna `na` perigoso, então ele fica fora deste. (`-` fica: nenhum agente se chama `-`.)
"""


def _intervention_or_none(value: str) -> str | None:
    """`None` quando o campo não nomeia intervenção nenhuma. Ver `_NON_INTERVENTION`.

    Normaliza só o MARCADOR, nunca o nome: `Acupuncture treatment` entra verbatim. O
    texto bruto é a única medição de que o prompt está defeituoso, e `relens` mostra
    esse texto ao juiz de directness, cujo veredito é gravado de forma durável.
    """
    text = (value or "").strip()
    return None if text.lower() in _NON_INTERVENTION else (text or None)


# `NoActiveFocus` mora em `lithium.focus.resolve` desde a Fase B — a extração deixou
# de ser o único caminho que precisa dele (relens, plan_tick e explore_tick também
# resolvem foco antes de gastar GPU). Reexportado aqui porque este era o import
# público e o motivo do fail-loud está escrito no docstring de lá.
__all__ = ["Extractor", "ExtractionResult", "NoActiveFocus"]


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def quote_is_anchored(quote: str, source_text: str) -> bool:
    """Portão 1: a citação é um trecho literal do texto?

    Tolerante a espaço em branco e a caixa — o modelo normaliza quebra de linha ao
    copiar, e punir isso descartaria citação legítima. Não tolerante a nada mais:
    qualquer palavra alterada reprova, que é o ponto.
    """
    if not quote.strip():
        return False
    if quote in source_text:
        return True
    return _normalize(quote) in _normalize(source_text)


@dataclass(slots=True)
class Rejection:
    """Uma claim que NÃO entrou, e por qual portão.

    Era uma string formatada, e a formatação destruía a estrutura: `chunk_id` e `gate`
    viravam prosa dentro de uma f-string que ia para `log.debug` — e o único handler de
    log deste projeto é `RichHandler(console)`, sem arquivo. Ou seja, o NÃO de cada portão
    existia por alguns milissegundos e sumia. As taxas dos portões que motivaram este
    trabalho só puderam ser medidas porque uma sessão redirecionou stdout por acaso, e o
    arquivo estava em `/tmp` — que o macOS apaga no boot, como de fato apagou.
    """

    chunk_id: int
    gate: str          # 'anchor' | 'entailment' | 'error'
    reason: str
    statement: str = ""
    quote: str = ""

    def __str__(self) -> str:
        return f"chunk {self.chunk_id} [{self.gate}]: {self.reason}"


@dataclass(slots=True)
class ExtractionResult:
    source_id: int
    proposed: int = 0
    anchored: int = 0
    verified: int = 0
    chunks_seen: int = 0
    claim_ids: list[int] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    chunks_yielding: set[int] = field(default_factory=set)
    """Chunks que produziram ao menos uma claim. Conjunto de CHUNK id — `claim_ids` é de
    claim, e confundir os dois foi o primeiro jeito que eu escrevi isto."""

    @property
    def anchor_rate(self) -> float:
        return self.anchored / self.proposed if self.proposed else 0.0

    @property
    def chunks_annihilated(self) -> int:
        """Propôs e perdeu TUDO — o paper certo, a citação ruim. Ação: apertar o extrator."""
        propôs = {r.chunk_id for r in self.rejections if r.gate != "error"}
        return len(propôs - self.chunks_yielding)

    @property
    def chunks_sterile(self) -> int:
        """Não propôs NADA — o paper errado. Ação: trocar a frente de busca.

        A distinção de `chunks_annihilated` é a diferença entre duas ações OPOSTAS, e hoje
        as duas são a mesma ausência de linha em 76 dos 160 chunks do corpus real.
        """
        tocou = self.chunks_yielding | {r.chunk_id for r in self.rejections}
        return max(0, self.chunks_seen - len(tocou))


class Extractor:
    def __init__(self, store: Store, llm: LLMClient, *, profile: FocusProfile,
                 entailment_check: bool = True, n_ctx: int = 8192) -> None:
        self.store = store
        self.llm = llm
        self.profile = profile
        self.entailment_check = entailment_check
        self.n_ctx = n_ctx

    async def extract_source(self, source_id: int) -> ExtractionResult:
        source = self.store.conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        if source is None:
            raise ValueError(f"fonte {source_id} não existe")

        chunks = self.store.conn.execute(
            "SELECT id, text, section FROM chunks WHERE source_id = ? ORDER BY ord",
            (source_id,),
        ).fetchall()

        # Chunk que JÁ produziu claim desta fonte é pulado. A extração não era
        # idempotente, e o caminho até o dano é curto e comum: `recover_orphans` devolve
        # à fila toda tarefa que ficou `running` quando o daemon morreu no meio, e uma
        # extração leva ~2 min — um Ctrl-C no meio dela é operação normal, não acidente
        # raro. Na volta, os chunks já processados eram extraídos de novo e as claims
        # entravam DUPLICADAS.
        #
        # E duplicata aqui não é ruído cosmético: `hypothesis_scoreboard` SOMA o peso das
        # claims ligadas, então a mesma evidência contada duas vezes empurra uma hipótese
        # para cima do placar. OBSERVADO neste banco: a fonte 1 ficou com 6 claims, duas
        # delas idênticas palavra por palavra.
        #
        # Por CHUNK e não por statement: o statement varia entre execuções (temperatura
        # 0,2, não 0), então casar por texto pegaria só parte — de fato pegou 2 de 3. O
        # chunk é determinístico: ou foi processado, ou não foi.
        #
        # Limitação aceita: um chunk que produziu ZERO claims não deixa registro e será
        # reprocessado, gastando uma chamada para não gravar nada. Distinguir "não
        # processado" de "processado, estéril" exigiria uma tabela nova, e o dano que ela
        # evitaria é custo, não corrupção.
        done_chunks = {
            cid
            for (raw,) in self.store.conn.execute(
                "SELECT chunk_ids FROM claims WHERE source_id = ?", (source_id,)
            )
            for cid in json.loads(raw or "[]")
        }
        if done_chunks:
            before = len(chunks)
            chunks = [c for c in chunks if c["id"] not in done_chunks]
            log.info(
                "fonte %s: %d de %d chunk(s) já extraídos, pulando",
                source_id, before - len(chunks), before,
            )

        # Resolvido UMA vez, antes de qualquer escrita.
        focus = self.store.active_focus()
        if focus is None:
            raise NoActiveFocus(
                "nenhum foco ativo: a extração julgaria directness contra um alvo "
                "desconhecido. Escolha um com `lithium focus --use <slug>`."
            )

        result = ExtractionResult(source_id=source_id, chunks_seen=len(chunks))
        for chunk in chunks:
            await self._extract_chunk(source, chunk, result, focus)
        return result

    async def _extract_chunk(self, source, chunk, result: ExtractionResult,
                             focus) -> None:
        prompt = render(
            "extract_claims",
            # `target` do BANCO, o resto do PERFIL. A fronteira não é arbitrária: toda
            # aresta em `claim_directness` foi julgada contra o target que estava no
            # banco, então é ele que define o que o julgamento significa.
            **{**self.profile.prompt_blocks("extract_claims"), "target": focus["target"]},
            title=source["title"] or "",
            journal=source["journal"] or "",
            year=source["year"] or "",
            design=source["design"] or "not indexed",
            sample_n=source["sample_n"] if source["sample_n"] is not None else "not reported",
            text=chunk["text"],
        )
        try:
            budget_guard(prompt, label="extract_claims", max_tokens=2048, n_ctx=self.n_ctx)
            extraction = await self.llm.structured(
                [{"role": "user", "content": prompt}], ClaimExtraction,
                max_tokens=2048, label="extract_claims",
            )
        except LLMError as exc:
            log.warning("extração falhou no chunk %s: %s", chunk["id"], exc)
            result.rejections.append(
                Rejection(chunk_id=int(chunk["id"]), gate="error", reason=str(exc)))
            return

        for claim in extraction.claims:
            result.proposed += 1
            if not quote_is_anchored(claim.supporting_quote, chunk["text"]):
                result.rejections.append(Rejection(
                    chunk_id=int(chunk["id"]), gate="anchor",
                    reason="citação não literal",
                    statement=claim.statement, quote=claim.supporting_quote))
                continue
            result.anchored += 1

            if self.entailment_check:
                verdict = await self._entails(claim)
                if verdict is None or not verdict.supported:
                    reason = verdict.reason if verdict else "verificador indisponível"
                    result.rejections.append(Rejection(
                        chunk_id=int(chunk["id"]), gate="entailment", reason=reason,
                        statement=claim.statement, quote=claim.supporting_quote))
                    continue

            claim_id = self._persist(source["id"], chunk["id"], claim, focus)
            result.claim_ids.append(claim_id)
            result.chunks_yielding.add(int(chunk["id"]))
            result.verified += 1

    async def _entails(self, claim: ExtractedClaim) -> CitationVerdict | None:
        """Portão 2. Falha do verificador reprova a claim — na dúvida, não entra."""
        try:
            return await self.llm.structured(
                [
                    {
                        "role": "user",
                        "content": render(
                            "verify_citation",
                            statement=claim.statement,
                            quote=claim.supporting_quote,
                        ),
                    }
                ],
                CitationVerdict,
                max_tokens=512,
                label="verify_citation",
            )
        except LLMError as exc:
            log.warning("verificador de citação falhou: %s", exc)
            return None

    def _persist(self, source_id: int, chunk_id: int, claim: ExtractedClaim,
                 focus) -> int:
        """A claim e o julgamento entram JUNTOS ou não entram.

        A transação é o que impede a claim órfã: sem ela, a claim commita em autocommit
        e um erro na aresta deixa evidência permanentemente sem peso e sem caminho de
        reparo. Ver `NoActiveFocus`.
        """
        with self.store.tx() as conn:
            cur = conn.execute(
                "INSERT INTO claims(source_id, chunk_ids, statement, supporting_quote, "
                "  population, intervention, comparator, outcome, direction, effect, "
                "  grade, scale_id, confidence, verified) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1) RETURNING id",
                (
                    source_id,
                    json.dumps([chunk_id]),
                    claim.statement,
                    # A citação que o portão 1 ACABOU de validar. Ela já está em mãos —
                    # gravá-la custa uma coluna e é a diferença entre "rastreável ao
                    # chunk" e "rastreável à frase".
                    claim.supporting_quote,
                    claim.population or None,
                    _intervention_or_none(claim.intervention),
                    claim.comparator or None,
                    claim.outcome or None,
                    claim.direction.value,
                    claim.effect or None,
                    Grade(claim.grade).value,
                    int(focus["scale_id"]),
                    claim.confidence,
                ),
            )
            claim_id = int(cur.fetchone()["id"])
            if claim.directness_judgeable:
                conn.execute(
                    "INSERT INTO claim_directness(claim_id, focus_id, directness) "
                    "VALUES(?, ?, ?)",
                    (claim_id, int(focus["id"]), claim.directness.value),
                )
            # Sem aresta quando a extração não soube dizer quem foi estudado. A claim
            # existe, é citável e recuperável; o que ela NÃO tem é peso — `claim_weight`
            # a exclui pelo JOIN, que é fail-closed por construção.
            #
            # Isto NÃO a descarta: ela cai em `claims_unjudged`, e é exatamente esse o
            # conjunto que `focus --relens` varre. Ou seja, a escapatória encaminha a
            # claim do julgador barato (que a inventou no mesmo POST) para o juiz
            # independente, que a lê em isolamento com o bloco de evidência inteiro.
            #
            # O contrário — gravar um nível por falta de informação — é permanente (a PK
            # congela e o sweep pula quem já tem aresta) e invisível (nenhum contador
            # distingue julgada de defaultada). O docstring de `DirectnessVerdict` já
            # dizia isso; só o caminho do relens obedecia.
        return claim_id
