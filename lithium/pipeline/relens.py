"""Re-lente: julgar `claim_directness` das claims que o foco ativo ainda não julgou.

Não é só feature nova — é o ÚNICO caminho de reparo de um defeito que já existe.
`fetch_source` calcula `dedup_key=extract:{focus_id}:{sid}` no ENQUEUE e `extract_source`
resolve o foco na EXECUÇÃO: trocar de foco entre os dois grava as claims sob o foco novo
e deixa o de origem permanentemente sem aresta, sem log e sem refetch possível (a fonte
já é `known`).

**O julgamento é ISOLADO, no molde de `verify_citation`, mas com prompt PRÓPRIO.**
Reusar `verify_citation` herdaria a proibição de contexto que o EXEMPT dele declara —
ele é proibido de ver o alvo, então julgaria contra nada. O que se copia é a DISCIPLINA:
sem resumo do paper e sem as claims vizinhas "para dar contexto". Com elas o julgador
vira exatamente o que `extract_claims.md` proíbe (inferir população do enquadramento em
vez do estudo), e `claim_directness` enche de julgamentos inflacionados com a Fase A
inteira construída para dar peso a eles.

**Mas evidência primária ENTRA, e isso é diferente de contexto.** `verify_citation`
recebe o `$quote`; "julgar em isolamento" ali significa isolado do ENQUADRAMENTO, não
sem evidência. Julgar só pela paráfrase seria pior que a extração original: MEDIDO no
próprio PLAN que `claims.population` é texto livre de um 12B e que as strings mais
frequentes são genéricas (`adults`, `not specified`). Um RCT em bipolar II com pânico
gravado como "outpatients with a mood disorder" seria injulgável, e o custo do chunk é
+21% (5,40 s -> 6,54 s por claim) contra transformar o caso "torn" no caso comum.
"""

from __future__ import annotations

import json
import logging

from lithium.db import Store
from lithium.focus import FocusProfile
from lithium.llm import LLMClient, LLMError
from lithium.llm.prompts import budget_guard, render
from lithium.llm.schemas import DirectnessVerdict

log = logging.getLogger(__name__)

SECONDS_PER_CLAIM = 5.76
"""Custo medido de um julgamento (401 tok de entrada, 30 de saída, no modelo
t ~= 2,31 ms x in + 161 ms x out do PLAN). É o número que o `focus --use` usa para
dizer ao usuário quantas HORAS a troca custa antes de ele trocar."""

EVIDENCE_CHARS = 2400
"""Teto do trecho de evidência. Um chunk de abstract inteiro cabe; um paper inteiro
não, e estourar a janela aqui manda a tarefa para dead-letter com a claim ainda sem
aresta — o pior dos dois mundos, porque a `dedup_key` já teria sido gasta."""

JUDGE_MAX_TOKENS = 256


def claims_to_judge(store: Store, focus_id: int, scale_id: int) -> list[int]:
    """As claims verificadas, na escala do foco, ainda sem aresta para ele.

    Escala na cláusula: uma claim graduada sob OUTRA régua não ganha peso mesmo depois
    de julgada, então enfileirá-la gastaria GPU para mover um número de coluna. Ela
    aparece em `claims_off_scale`, cujo remédio não é este.

    SEM `ORDER BY`, e isso é a invariante do repo, não descuido: ordenar claim por `id`
    é ordenar por ORDEM DE COLHEITA, e `test_no_claim_ranking_orders_by_id` reprova.
    Aqui não existe ranking — TODA claim elegível vira tarefa, e a PK
    `(claim_id, focus_id)` faz o retomar ser grátis em qualquer ordem.
    """
    return [
        int(r["id"])
        for r in store.conn.execute(
            "SELECT c.id FROM claims c "
            " WHERE c.verified = 1 AND c.scale_id IS ? "
            "   AND NOT EXISTS (SELECT 1 FROM claim_directness cd "
            "                    WHERE cd.claim_id = c.id AND cd.focus_id = ?)",
            (scale_id, focus_id),
        )
    ]


def _evidence_for(store: Store, claim) -> str:
    """Os chunks de onde a claim saiu. `sources.raw_json` NÃO serve.

    MEDIDO contra a fixture real (PMID 30712879): `raw_json` tem 374 bytes e três
    chaves — {pmid, publication_types, mesh}. Não tem abstract, título nem ano. O texto
    está em `chunks.text` (7.875 bytes para a mesma fonte). O comentário do schema que
    prometia "reprocessar sem refetch" foi corrigido no mesmo commit.
    """
    try:
        chunk_ids = [int(i) for i in json.loads(claim["chunk_ids"] or "[]")]
    except (TypeError, ValueError):
        chunk_ids = []
    parts: list[str] = []
    source = store.conn.execute(
        "SELECT title, journal, year, design, sample_n FROM sources WHERE id = ?",
        (claim["source_id"],),
    ).fetchone()
    if source is not None:
        parts.append(
            f"Title: {source['title'] or '(untitled)'}\n"
            f"Journal: {source['journal'] or '—'} ({source['year'] or '—'})\n"
            f"Study design: {source['design'] or 'not indexed'}\n"
            f"Reported sample size: "
            f"{source['sample_n'] if source['sample_n'] is not None else 'not reported'}"
        )
    if chunk_ids:
        rows = store.conn.execute(
            "SELECT text FROM chunks WHERE id IN "
            f"({','.join('?' * len(chunk_ids))}) ORDER BY ord",
            chunk_ids,
        ).fetchall()
        text = "\n".join(r["text"] for r in rows)
        if text.strip():
            parts.append("---\n" + text[:EVIDENCE_CHARS] + "\n---")
    return "\n\n".join(parts) if parts else "(no source text available)"


def claim_block(claim) -> str:
    return (
        f"Statement: {claim['statement']}\n"
        f"Population as recorded: {claim['population'] or '(not recorded)'}\n"
        f"Intervention: {claim['intervention'] or '(not recorded)'}"
    )


async def judge_one(
    store: Store, llm: LLMClient, profile: FocusProfile, *,
    claim_id: int, focus_id: int, target: str, n_ctx: int = 8192,
) -> DirectnessVerdict | None:
    """Julga UMA claim e grava a aresta. Devolve o veredito, ou None se nada foi gravado.

    O pré-teste de existência vem ANTES do POST, e não é otimização: sem ele, uma tarefa
    recuperada por `recover_orphans` (o handler gravou a aresta e o daemon morreu antes
    do `complete`) re-julga a claim, gasta ~5,4 s de GPU, bate em `IntegrityError`, e
    repete até o dead-letter — queimando ~16 s de GPU e a `dedup_key` junto.
    """
    already = store.conn.execute(
        "SELECT 1 FROM claim_directness WHERE claim_id = ? AND focus_id = ?",
        (claim_id, focus_id),
    ).fetchone()
    if already is not None:
        return None

    claim = store.conn.execute(
        "SELECT id, source_id, chunk_ids, statement, population, intervention "
        "  FROM claims WHERE id = ?",
        (claim_id,),
    ).fetchone()
    if claim is None:
        log.info("claim %s não existe mais, nada a julgar", claim_id)
        return None

    prompt = render(
        "judge_directness",
        # `target` do BANCO, `directness_definitions` do PERFIL — e os DOIS resolvidos
        # a partir do foco do PAYLOAD, nunca de `active_focus()` relido aqui. Trocar de
        # foco no meio do lote faria o julgamento ser feito contra o alvo de um foco e
        # gravado como aresta de outro, sem log e sem nada em `judged_at` que permitisse
        # reconstruir onde foi o corte.
        **{**profile.prompt_blocks("judge_directness"), "target": target},
        claim=claim_block(claim),
        evidence=_evidence_for(store, claim),
    )
    budget_guard(prompt, label="judge_directness", max_tokens=JUDGE_MAX_TOKENS,
                 n_ctx=n_ctx)
    try:
        verdict = await llm.structured(
            [{"role": "user", "content": prompt}], DirectnessVerdict,
            max_tokens=JUDGE_MAX_TOKENS, label="judge_directness",
        )
    except LLMError as exc:
        # Sem veredito, nada é gravado: a claim continua em `claims_unjudged` e o
        # relens a revisita. Escrever um default aqui seria permanente e invisível.
        log.warning("julgamento de directness falhou na claim %s: %s", claim_id, exc)
        raise

    if not verdict.judgeable:
        # NENHUMA linha. Ver `DirectnessVerdict`: gravar o nível mais fraco por falta de
        # informação é irreversível (a PK congela, o sweep pula quem tem aresta) e
        # nenhum contador distinguiria isso de um julgamento de verdade.
        log.info("claim %s não julgável com a evidência disponível", claim_id)
        return verdict

    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness, rationale, "
        "                             out_of_scope) VALUES(?, ?, ?, ?, ?) "
        # `DO NOTHING` além do pré-teste: entre o SELECT e o INSERT cabe outro worker.
        "ON CONFLICT(claim_id, focus_id) DO NOTHING",
        (claim_id, focus_id, verdict.directness.value, verdict.rationale,
         0 if verdict.in_scope else 1),
    )
    return verdict
