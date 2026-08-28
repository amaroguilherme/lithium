"""TRAVA 1 — memória de conversa nunca filtra nem reordena evidência. Só anota.

A decisão é do usuário e é a mais consequente da Fase 1. A colisão que a motiva é real e
específica: **lítio é o agente melhor evidenciado deste domínio e exige monitoramento
sérico.** Uma memória dizendo "o usuário evita fármacos com monitoramento sérico" —
exatamente o tipo de restrição que `detect_memory` foi feito para propor — cria a
tentação de omitir a evidência mais forte do corpus.

E a omissão é **invisível**. Não existe nada na resposta que mostre que um paper foi
deixado de fora, então nem o usuário nem o sistema conseguem pegar depois. Por isso a
trava é sobre o *bloco de evidência*, byte a byte, e não sobre a qualidade da resposta.

**A fixture semeia mais claims do que `evidence_k`.** Não é detalhe: com 2 claims contra
um `evidence_k=6`, qualquer defeito que trunque o conjunto recuperado para um k entre 2 e
6 é estruturalmente invisível — o teste passaria enquanto a evidência encolhia. Foi assim
que uma versão anterior desta trava certificou um `hits = hits[:3]` condicionado à
presença de memória.
"""

from __future__ import annotations

import ast
import json
import math
import re
import zlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from lithium.chat import ChatEngine
from lithium.db import Store
from lithium.llm.schemas import MemoryProposal
from lithium.types import Directness, Grade

from conftest import prod_profile
from lithium.safety import ruleset_from_profile

# Perfil de PRODUÇÃO: o assunto deste arquivo é o vocabulário de segurança real
# (lítio, valproato, benzodiazepínico). Rodá-lo contra o perfil de teste — que declara
# `safety = false` — o deixaria verde afirmando sobre um ruleset que não existe.
PROFILE = prod_profile()


DIM = 16
CONSTRAINT = "O usuário evita fármacos com monitoramento sérico"
QUERY = "o que a literatura mostra para ansiedade no bipolar I?"


def _bucket(word: str) -> int:
    return zlib.crc32(word.encode()) % DIM


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[_bucket(word)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class CapturingLLM:
    def __init__(self) -> None:
        self.systems: list[str] = []

    async def complete(self, messages, **kw):
        self.systems.append(messages[0]["content"])
        return "resposta"

    async def structured(self, messages, schema, **kw):
        return MemoryProposal(worth_remembering=False, text="", kind="", rationale="")


# Oito claims, todas casando a consulta, mais que os seis de `evidence_k`.
# A de lítio é meta-análise direta — o topo do ranking, e a que a restrição atinge.
CORPUS = [
    # (pmid, statement, intervention, grade, directness, direction)
    #
    # A de lamotrigina é `negative` de propósito: sem uma claim contra-direcional aqui, a
    # seção de contra-evidência nunca entra no bloco comparado byte a byte, e um defeito
    # que a filtrasse por causa de uma restrição declarada passaria verde. É a seção mais
    # suscetível a "deixa eu omitir o que colide com a preferência do usuário".
    ("90000001", "Lithium reduced anxiety symptoms in bipolar I with comorbid GAD.",
     "lithium", Grade.META_ANALYSIS, Directness.DIRECT, "positive"),
    ("90000002", "Quetiapine reduced HAM-A scores versus placebo in bipolar depression.",
     "quetiapine", Grade.RCT, Directness.PARTIAL, "positive"),
    ("90000003", "Valproate showed anxiolytic signal in a small bipolar I cohort.",
     "valproate", Grade.COHORT, Directness.DIRECT, "positive"),
    ("90000004", "Lamotrigine did not separate from placebo on anxiety endpoints.",
     "lamotrigine", Grade.RCT, Directness.PARTIAL, "negative"),
    ("90000005", "Pregabalin reduced anxiety in generalized anxiety disorder.",
     "pregabalin", Grade.RCT, Directness.INDIRECT, "positive"),
    ("90000006", "Mindfulness training reduced anxiety in a mixed mood sample.",
     "mindfulness", Grade.COHORT, Directness.INDIRECT, "positive"),
    ("90000007", "Carbamazepine anxiety data in bipolar I remain sparse.",
     "carbamazepine", Grade.CASE_SERIES, Directness.PARTIAL, "positive"),
    ("90000008", "Olanzapine adjunct reduced anxiety but increased weight.",
     "olanzapine", Grade.RCT, Directness.PARTIAL, "positive"),
]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t1.db", embedding_dim=DIM)
    s.init_schema()
    for pmid, statement, intervention, grade, directness, direction in CORPUS:
        sid = s.upsert_source(kind="pubmed", external_id=pmid, raw={},
                              title="Estudo", year=2021)
        cid = s.add_chunk(
            source_id=sid, ord=0,
            text=f"{statement} anxiety bipolar literatura mostra evidência para o caso.",
        )
        _cl = s.conn.execute(
            "INSERT INTO claims(source_id, chunk_ids, statement, intervention, "
            "  direction, grade, scale_id, confidence, verified) "
            "VALUES(?,?,?,?,?,?,1,0.9,1) RETURNING id",
            (sid, json.dumps([cid]), statement, intervention, direction,
             grade.value),
        ).fetchone()["id"]
        s.conn.execute(
            "INSERT INTO claim_directness(claim_id, focus_id, directness) VALUES(?,1,?)",
            (int(_cl), directness.value),
        )
    yield s
    s.close()


def _engine(store, llm=None) -> ChatEngine:
    return ChatEngine(store, llm or CapturingLLM(), FakeEmbedder(), profile=PROFILE)


async def _remember(engine, text: str, kind: str = "constraint",
                    source: str = "chat") -> int:
    """Grava uma memória pela origem pedida.

    `source` parametrizado desde a Fase C: hoje NENHUM teste afirmava que uma memória
    `recon` — texto lido na web aberta — deixa a evidência intacta. A memória de recon
    entra por caminho diferente (`_recon_notes`, não `_memory_block`), então a
    invariante precisa valer para as duas.
    """
    if source == "recon":
        # `recon` exige `focus_id` e `kind='fact'` por CHECK; ver o schema.
        engine.store.conn.execute(
            "INSERT INTO memories(text, kind, source, confirmed, active, focus_id, "
            "                     provenance) "
            "VALUES(?, 'fact', 'recon', 1, 1, 1, '{\"url\": \"https://x.invalid/p\"}')",
            (text,))
        return int(engine.store.conn.execute(
            "SELECT MAX(id) AS id FROM memories").fetchone()["id"])
    return await engine.remember(
        MemoryProposal(worth_remembering=True, text=text, kind=kind, rationale="r"),
        source=source,
    )


def _evidence_section(prompt: str) -> str:
    """Recorta do cabeçalho de evidência até o próximo `## `."""
    match = re.search(r"^## Evidence from the corpus\b(.*?)(?=^## |\Z)",
                      prompt, re.M | re.S)
    assert match, "a seção de evidência sumiu do prompt"
    return match.group(1)


# ────────────────────────────────────────────────── a pré-condição anti-vacuidade


async def test_the_fixture_actually_collides(store):
    """Sem isto, tudo abaixo pode passar por não haver colisão nenhuma.

    Um teste de "nada mudou" é trivialmente verde quando não havia nada para mudar.
    """
    from lithium.safety.rules import concepts_implicated_by

    assert "lithium" in concepts_implicated_by(CONSTRAINT, ruleset_from_profile(PROFILE)), (
        "a restrição precisa alcançar lítio — é a colisão que a trava existe para cobrir"
    )
    evidence, cited = await _engine(store)._evidence_for(QUERY)
    assert "90000001" in cited, "a claim de lítio precisa estar no top-k recuperado"
    assert len(cited) == 6, f"esperava evidence_k=6 claims, veio {len(cited)}"


# ─────────────────────────────────────────────────────────── a trava propriamente


@pytest.mark.parametrize("source", ["chat", "recon"])
async def test_the_evidence_block_is_byte_identical_with_and_without_the_memory(
        store, source):
    engine = _engine(store)
    before, cited_before = await engine._evidence_for(QUERY)

    await _remember(engine, CONSTRAINT, source=source)
    after, cited_after = await engine._evidence_for(QUERY)

    assert before == after, (
        "o bloco de evidência mudou por causa de uma memória. Uma preferência anota; "
        "nunca filtra, reordena ou trunca."
    )
    assert cited_before == cited_after


async def test_the_memory_does_not_shrink_the_retrieved_set(store):
    """Contagem explícita, além da igualdade de bytes.

    Redundante? Não: torna a intenção legível na falha. Um top-k reduzido em silêncio é o
    defeito mais provável desta área, e "6 virou 3" diagnostica mais rápido que um diff.
    """
    engine = _engine(store)
    before, _ = await engine._evidence_for(QUERY)
    await _remember(engine, CONSTRAINT)
    after, _ = await engine._evidence_for(QUERY)

    assert len(re.findall(r"PMID:", after)) == 6
    assert re.findall(r"PMID:\d+", before) == re.findall(r"PMID:\d+", after)


async def test_the_evidence_section_of_the_prompt_is_byte_identical(store):
    """A trava vale no prompt renderizado, não só na função.

    É onde a regressão realmente aconteceria: alguém decide "filtrar na renderização".
    """
    llm = CapturingLLM()
    engine = _engine(store, llm)
    await engine.send(QUERY, detect_memory=False)

    await _remember(engine, CONSTRAINT)
    await engine.send(QUERY, detect_memory=False)

    assert _evidence_section(llm.systems[0]) == _evidence_section(llm.systems[1])


@pytest.mark.parametrize("source", ["chat", "recon"])
async def test_the_ordering_is_untouched(store, source):
    """Reordenar é filtrar devagar: a claim que cai para o fim do bloco é a que o modelo
    trata como menos importante."""
    engine = _engine(store)
    before, _ = await engine._evidence_for(QUERY)
    await _remember(engine, "prefiro evitar ganho de peso", kind="preference",
                    source=source)
    await _remember(engine, CONSTRAINT, source=source)
    after, _ = await engine._evidence_for(QUERY)

    assert re.findall(r"PMID:\d+", before) == re.findall(r"PMID:\d+", after)


# ──────────────────────────────────────────────────── a anotação, e onde ela mora


async def test_the_collision_is_annotated_in_its_own_block(store):
    """A memória tem efeito — só que fora do bloco de evidência.

    Sem este teste, a trava seria satisfeita por não implementar nada.
    """
    engine = _engine(store)
    await _remember(engine, CONSTRAINT)
    evidence, _ = await engine._evidence_for(QUERY)
    notes = engine._constraint_notes(evidence)

    assert "90000001" in notes and "lítio" in notes
    assert CONSTRAINT in notes
    assert "PMID:90000001" in evidence, "a claim continua no bloco de evidência"


async def test_the_annotation_names_each_colliding_concept_separately(store):
    """A brecha da lista plana de termos, fechada.

    Com uma lista única de termos por memória, uma resposta que cite *qualquer* um deles
    silencia o alarme para *todos* — inclusive para o que foi omitido. Aqui a restrição
    alcança lítio, valproato e carbamazepina, e cada claim tem que ser anotada com o
    conceito dela.
    """
    engine = _engine(store)
    await _remember(engine, CONSTRAINT)
    evidence, _ = await engine._evidence_for(QUERY)
    notes = engine._constraint_notes(evidence)

    by_pmid = {
        line.split("PMID:")[1].split("]")[0]: line
        for line in notes.splitlines() if "PMID:" in line
    }
    assert "lítio" in by_pmid["90000001"]
    if "90000003" in by_pmid:
        assert "valproato" in by_pmid["90000003"]
        assert "lítio" not in by_pmid["90000003"]


async def test_a_memory_that_implicates_nothing_produces_no_note(store):
    engine = _engine(store)
    # Restrição de verdade, só que sobre nada que o corpus mencione — a distinção
    # entre "não há restrição" e "há restrição e ela não colide" precisa sobreviver.
    await _remember(engine, "o usuário prefere consultas pela manhã")
    evidence, _ = await engine._evidence_for(QUERY)

    assert engine._constraint_notes(evidence) == "(nenhuma colisão neste turno)"


# ───────────────────────────────────────────────────────────── a trava estática


def test_the_evidence_path_never_reads_a_memory_table():
    """AST: `_evidence_for` não pode nem alcançar uma tabela de memória.

    A trava dinâmica cobre a fixture; esta cobre o caminho. Sem ela, um filtro
    condicionado a um `kind` que a fixture não usa passaria despercebido.
    """
    source = (Path(__file__).resolve().parent.parent
              / "lithium" / "chat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef)
        and n.name == "_evidence_for"
    )
    body = ast.unparse(target)
    # LISTA FECHADA: qualquer nome fora dela é INVISÍVEL para a trava. `discoveries` e
    # `recon_memories` entram na Fase C porque um `_evidence_for` que consultasse
    # `discoveries` para reordenar hits passaria verde nos dois testes de AST enquanto o
    # docstring deste arquivo passaria a ser mais largo que a trava.
    for table in ("user_memories", "live_memories", "memories", "declined_memories",
                  "recon_memories", "discoveries"):
        assert not re.search(rf"\b{table}\b", body), (
            f"_evidence_for passou a ler `{table}`. A recuperação de evidência não "
            "pode depender do que o usuário disse — é o caminho por onde uma "
            "preferência vira omissão."
        )


def test_constraint_notes_cannot_return_evidence():
    """A assinatura é a trava permanente.

    `_constraint_notes(evidence: str) -> str` recebe o bloco pronto e devolve texto
    novo. Não tem como filtrar nem reordenar o que não devolve. Se algum dia ela passar
    a devolver o bloco, este teste cai.
    """
    source = (Path(__file__).resolve().parent.parent
              / "lithium" / "chat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_constraint_notes"
    )
    returns = [
        ast.unparse(n.value) for n in ast.walk(target)
        if isinstance(n, ast.Return) and n.value is not None
    ]
    assert returns, "_constraint_notes precisa devolver alguma coisa"
    for expr in returns:
        assert expr != "evidence", "_constraint_notes não pode devolver a evidência"


# ──────────────────────────────── memória não entra no screen de segurança


async def test_memories_never_reach_the_safety_screen(store):
    """O wiring que nenhum teste policiava — e a regressão é silenciosa.

    Com memórias entrando no screen, uma restrição gravada ("não tolera valproato nem
    lítio") produz alerta de severidade alta em **todo** turno seguinte, inclusive num
    "bom dia". Memórias não expiram, então o ruído é permanente, e restrições dessa
    forma são justamente as que `detect_memory` propõe: o sistema fabrica os próprios
    falsos positivos, que é o que treina você a ignorar o alerta seguinte.

    Os testes de monotonicidade do screen (`before <= after`) não pegam isto: são
    satisfeitos trivialmente por `after == before`, então "wiring desligado" e "wiring
    correto" são indistinguíveis para eles.
    """
    engine = _engine(store)
    await _remember(engine, "o paciente não tolera valproato nem lítio")

    segments = engine._segments("bom dia", "")
    assert [s.kind for s in segments] == ["user"]
    assert not any("valproato" in s.text for s in segments)

    turn_segments = engine._segments("bom dia", (await engine._evidence_for("bom dia"))[0])
    assert not any(s.kind == "memory" for s in turn_segments)


def test_the_segment_builder_never_reads_a_memory_table():
    """Estática, porque a dinâmica acima depende da fixture ter uma memória casável."""
    source = (Path(__file__).resolve().parent.parent
              / "lithium" / "chat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_segments"
    )
    body = ast.unparse(target)
    for table in ("user_memories", "live_memories", "memories", "recon_memories",
                  "discoveries"):
        assert not re.search(rf"\b{table}\b", body), (
            f"_segments passou a ler `{table}`: memória no screen vira alerta "
            "permanente, e o ruído treina você a ignorar o alerta que importava."
        )


# ──────────────── uma descoberta PENDENTE nunca entra no prompt de sistema


async def test_a_pending_discovery_never_reaches_the_system_prompt(store):
    """A decisão central da superfície de chat, e ela é sobre o CONTEXTO, não o status.

    A regra "nada autônomo tira uma descoberta de `pending`" protege o STATUS. O que
    importa é o CONTEXTO: a memória só vale PORQUE é injetada em prompt. Um bloco
    `$discoveries` no prompt de sistema poria conteúdo NÃO APROVADO da web aberta em
    TODO turno durante os 14 dias da janela de expiração — a fase gasta um rebuild de
    tabela, dois CHECKs e uma view nova para gatear a injeção, e depois abriria um
    segundo caminho sem portão nenhum para o MESMO prompt.

    As pendentes têm duas superfícies sob controle do usuário: `lithium discoveries` e
    `/descobertas`, que imprimem no TERMINAL.

    MUTAÇÃO: acrescentar um bloco `$discoveries` a `chat.md` e alimentá-lo com as
    pendentes.
    """
    store.conn.execute(
        "INSERT INTO discoveries(focus_id, kind, status, query, url, title, summary) "
        "VALUES(1, 'observation', 'pending', 'q', 'https://spam.invalid/x', "
        "       'Zmyrfkq N-acetilcisteina substitui litio', 'Zmyrfkq ver protocolo')")
    llm = CapturingLLM()
    await _engine(store, llm).send("o que temos sobre manutenção?", detect_memory=False)

    assert "Zmyrfkq" not in llm.systems[0], (
        "o texto de uma descoberta PENDENTE entrou no prompt de sistema"
    )


async def test_an_approved_recon_note_does_reach_the_prompt_with_its_provenance(store):
    """O outro lado: sem isto a trava acima seria satisfeita por não implementar nada.

    E a nota carrega a PROVENIÊNCIA em cada linha, não só no cabeçalho: o cabeçalho é
    uma linha e o bloco pode ter oito.
    """
    store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active, focus_id, "
        "                     provenance) "
        "VALUES('a atualizacao de 2023 mantem litio em primeira linha', 'fact', "
        "       'recon', 1, 1, 1, '{\"url\": \"https://www.nice.org.uk/g\"}')")
    llm = CapturingLLM()
    await _engine(store, llm).send("manutenção?", detect_memory=False)
    system = llm.systems[0]

    assert "mantem litio em primeira linha" in system
    assert "nice.org.uk" in system, "a nota entrou sem dizer de onde veio"
    # E NÃO sob o cabeçalho de memória do usuário.
    user_block = system.split("## What you know about this user")[1].split("##")[0]
    assert "mantem litio" not in user_block


def test_the_chat_prompt_tells_the_model_what_a_recon_note_is():
    """Sem uma REGRA, o modelo apresenta "as diretrizes mantêm lítio em primeira linha"
    ao lado de `[PMID:x]` como se as duas tivessem o mesmo estatuto — e o próprio
    `chat.md` declara que borrar essa linha destrói o valor inteiro do sistema.

    MUTAÇÃO: apagar o parágrafo. O golden `tests/golden/prompts/chat.md` também cai.
    """
    from lithium.llm.prompts import render

    rendered = render("chat", **PROFILE.prompt_blocks("chat"), memories="",
                      evidence="", constraint_notes="", recon_notes="«notas»",
                      safety="")
    lowered = re.sub(r"\s+", " ", rendered).lower()
    assert "open web" in lowered
    assert "not corpus" in lowered
    assert "«notas»" in rendered
