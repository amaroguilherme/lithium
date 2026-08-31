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
class ExtractionResult:
    source_id: int
    proposed: int = 0
    anchored: int = 0
    verified: int = 0
    claim_ids: list[int] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    @property
    def anchor_rate(self) -> float:
        return self.anchored / self.proposed if self.proposed else 0.0


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

        result = ExtractionResult(source_id=source_id)
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
            result.rejections.append(f"chunk {chunk['id']}: {exc}")
            return

        for claim in extraction.claims:
            result.proposed += 1
            if not quote_is_anchored(claim.supporting_quote, chunk["text"]):
                result.rejections.append(
                    f"chunk {chunk['id']}: citação não literal — "
                    f"{claim.supporting_quote[:80]!r}"
                )
                continue
            result.anchored += 1

            if self.entailment_check:
                verdict = await self._entails(claim)
                if verdict is None or not verdict.supported:
                    reason = verdict.reason if verdict else "verificador indisponível"
                    result.rejections.append(f"chunk {chunk['id']}: não implicada — {reason}")
                    continue

            claim_id = self._persist(source["id"], chunk["id"], claim, focus)
            result.claim_ids.append(claim_id)
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
                "INSERT INTO claims(source_id, chunk_ids, statement, population, "
                "  intervention, comparator, outcome, direction, effect, grade, "
                "  scale_id, confidence, verified) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,1) RETURNING id",
                (
                    source_id,
                    json.dumps([chunk_id]),
                    claim.statement,
                    claim.population or None,
                    claim.intervention or None,
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
            conn.execute(
                "INSERT INTO claim_directness(claim_id, focus_id, directness) "
                "VALUES(?, ?, ?)",
                (claim_id, int(focus["id"]), claim.directness.value),
            )
        return claim_id
