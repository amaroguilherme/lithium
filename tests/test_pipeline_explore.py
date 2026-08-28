"""Trilha exploratória: especulação mecanística com salvaguardas.

O usuário pediu explicitamente criatividade sem limite, incluindo especulação sem
nenhum dado humano. Isso só é responsável se as salvaguardas forem estruturais em vez
de conselho no prompt. Estes testes travam as três:

* cadeia auditável — plausibilidade é a fração de elos ancorados, e marcar
  `supported` sem citação não conta
* falsificador obrigatório — hipótese que nada refuta não entra
* crítica adversarial — falha fatal reprova, e crítica indisponível também
"""

from __future__ import annotations

import json

import pytest

from lithium.db import Store
from lithium.llm import LLMError
from lithium.llm.schemas import Speculation, SpeculationBatch, SpeculationCritique
from lithium.pipeline.explore import Explorer, plausibility
from lithium.pipeline.mechanism import route_block, taxonomy_block

from conftest import FIXTURE_FOCUSES, onco_profile, prod_profile

PROFILE = onco_profile()



class ScriptedLLM:
    def __init__(self, batches=(), critiques=()) -> None:
        self._batches = list(batches)
        self._critiques = list(critiques)
        self.prompts: list[str] = []

    async def structured(self, messages, schema, **kw):
        self.prompts.append(messages[0]["content"])
        if schema is SpeculationBatch:
            return self._batches.pop(0)
        if schema is SpeculationCritique:
            item = self._critiques.pop(0) if self._critiques else _ok_critique()
            if isinstance(item, Exception):
                raise item
            return item
        raise AssertionError(f"schema inesperado: {schema}")


def _ok_critique(**kw) -> SpeculationCritique:
    return SpeculationCritique.model_validate({
        "weakest_link": "o elo 2 extrapola de roedor para humano",
        "fatal_flaw": "", "contradicted_by": "", "missing_risk": "", "survives": True,
        **kw,
    })


def _step(claim: str, *, supported: bool = False, evidence: str = "") -> dict:
    return {"claim": claim, "supported": supported, "evidence": evidence}


def _spec(**kw) -> Speculation:
    base = {
        "statement": "Agonismo sigma-1 dissocia ansiólise de desestabilização do humor",
        "intervention_class": "repurposed non-psychiatric drug",
        "mechanism_target": "sigma-1 receptor",
        "route": "transdermal patch",
        "combination": "",
        "chain": [
            _step("Sigma-1 modula liberação de glutamato no hipocampo",
                  supported=True, evidence="PMID:12345678"),
            _step("Essa modulação reduz ansiedade sem aumentar tônus dopaminérgico"),
        ],
        "falsifier": "Se o efeito ansiolítico sumir em knockout de sigma-1, o mecanismo cai",
        "test_proposal": "Buscar ensaios de fluvoxamina em alta dose fora de depressão",
        "known_risks": "Interação com inibidores de CYP",
        "novelty": 0.8,
    }
    return Speculation.model_validate(base | kw)


def _batch(*specs: Speculation) -> SpeculationBatch:
    return SpeculationBatch(speculations=list(specs))


FIXTURE_PMIDS = ("12345678", "777", "1", "2", "3", "9")
"""Os PMIDs que existem no corpus destas fixtures.

Lista EXPLÍCITA e curta, de propósito. A tentação é semear "todo PMID que aparecer no
teste" — e isso faria qualquer teste futuro de recusa de citação passar por acidente,
porque o PMID inventado passaria a existir. `99999999` fica deliberadamente de fora: é o
que os testes usam para exercitar a recusa.
"""


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "e.db", embedding_dim=8)
    s.init_schema()
    # O portão de citação exige que o PMID exista em `sources`. Antes deste portão a
    # suíte assumia que citação não precisava existir — 9 de 9 citações reais de
    # produção eram PMIDs verdadeiros do PubMed sem relação com a alegação.
    for pmid in FIXTURE_PMIDS:
        s.upsert_source(kind="pubmed", external_id=pmid, raw={}, title="fixture")
    yield s
    s.close()


# ─────────────────────────────────────────────── plausibilidade da cadeia


def test_plausibility_is_the_anchored_fraction():
    chain = [
        _step("a", supported=True, evidence="PMID:1"),
        _step("b", supported=True, evidence="PMID:2"),
        _step("c"),
        _step("d"),
    ]
    assert plausibility(chain) == pytest.approx(0.5)


def test_supported_without_citation_does_not_count():
    """O atalho óbvio para inflar a pontuação é marcar `supported` sem citar nada."""
    assert plausibility([_step("a", supported=True, evidence="")]) == 0.0
    assert plausibility([_step("a", supported=True, evidence="PMID:9")]) == 1.0


def test_empty_chain_is_zero_plausible():
    assert plausibility([]) == 0.0


def test_honest_short_chain_beats_padded_one():
    """Cadeia curta e ancorada tem de vencer cadeia longa e assumida — senão o
    incentivo é encompridar o raciocínio em vez de fundamentá-lo."""
    honest = [_step("a", supported=True, evidence="PMID:1"),
              _step("b", supported=True, evidence="PMID:2")]
    padded = [_step(c) for c in "abcde"] + [_step("f", supported=True, evidence="PMID:3")]
    assert plausibility(honest) > plausibility(padded)


# ──────────────────────────────────────────────────────── salvaguardas


async def test_speculation_without_falsifier_is_rejected(store):
    """Hipótese que nada refuta é prosa. O schema exige o campo, mas string vazia
    passa pela gramática — o portão está no código."""
    llm = ScriptedLLM([_batch(_spec(falsifier="   "))])
    assert await Explorer(store, llm, profile=PROFILE).generate() == []
    assert store.conn.execute("SELECT COUNT(*) AS n FROM hypotheses").fetchone()["n"] == 0


async def test_fatal_flaw_marks_the_hypothesis_refuted(store):
    llm = ScriptedLLM(
        [_batch(_spec())],
        [_ok_critique(fatal_flaw="aumenta tônus dopaminérgico; risco de virada",
                      survives=False)],
    )
    [record] = await Explorer(store, llm, profile=PROFILE).generate()

    assert record.survives is False
    row = store.conn.execute("SELECT status, survives_critique FROM hypotheses").fetchone()
    assert row["status"] == "refuted" and row["survives_critique"] == 0


async def test_unavailable_critique_fails_closed(store):
    """Sem o passe adversarial a hipótese não foi verificada; entrar como ativa
    mentiria sobre isso."""
    llm = ScriptedLLM([_batch(_spec())], [LLMError("servidor caiu")])
    [record] = await Explorer(store, llm, profile=PROFILE).generate()

    assert record.survives is False
    assert store.conn.execute(
        "SELECT status FROM hypotheses"
    ).fetchone()["status"] == "refuted"


async def test_critique_sees_the_chain_with_assumption_markers(store):
    """O crítico precisa distinguir elo citado de elo assumido para achar o mais
    fraco — sem os marcadores ele avalia prosa."""
    llm = ScriptedLLM([_batch(_spec())])
    await Explorer(store, llm, profile=PROFILE).generate()

    critique_prompt = llm.prompts[1]
    assert "[ASSUMED]" in critique_prompt
    assert "PMID:12345678" in critique_prompt
    assert "break it" in critique_prompt


async def test_surviving_hypothesis_is_stored_active_with_all_fields(store):
    llm = ScriptedLLM([_batch(_spec())])
    [record] = await Explorer(store, llm, profile=PROFILE).generate()

    row = store.conn.execute("SELECT * FROM hypotheses WHERE id = ?", (record.id,)).fetchone()
    assert row["tier"] == "speculative"
    assert row["status"] == "active"
    assert row["mechanism_target"] == "sigma-1 receptor"
    assert row["falsifier"]
    assert row["test_proposal"]
    assert row["novelty"] == 0.8
    assert len(json.loads(row["chain_json"])) == 2
    assert json.loads(row["critique_json"])["weakest_link"]


# ──────────────────────────────────────── separação entre as duas trilhas


async def test_speculation_stays_out_of_the_evidence_scoreboard(store):
    """Rankear as duas juntas ou enterra o inédito (peso ~0.01), ou tira precedência
    do estabelecido. As views precisam ficar disjuntas."""
    store.conn.execute("INSERT INTO hypotheses(focus_id, statement, tier) "
                       "VALUES(1, 'evidenciada', 'evidence')")
    llm = ScriptedLLM([_batch(_spec())])
    await Explorer(store, llm, profile=PROFILE).generate()

    evidence_board = store.conn.execute("SELECT statement FROM hypothesis_scoreboard").fetchall()
    spec_board = store.conn.execute("SELECT statement FROM speculation_board").fetchall()

    assert [r["statement"] for r in evidence_board] == ["evidenciada"]
    assert len(spec_board) == 1 and "sigma-1" in spec_board[0]["statement"]


async def test_board_ranks_by_plausibility_times_novelty(store):
    """O mais plausível que ninguém testou. Só plausibilidade devolve o óbvio; só
    ineditismo, delírio."""
    anchored = [_step("a", supported=True, evidence="PMID:1"),
                _step("b", supported=True, evidence="PMID:2")]
    assumed = [_step("a"), _step("b")]

    llm = ScriptedLLM([_batch(
        _spec(statement="plausível mas batido", chain=anchored, novelty=0.1),
        _spec(statement="inédito mas sem base", chain=assumed, novelty=1.0),
        _spec(statement="plausível e inédito", chain=anchored, novelty=0.9),
    )])
    await Explorer(store, llm, profile=PROFILE).generate(max_items=3)

    board = Explorer(store, llm, profile=PROFILE).board()
    assert board[0]["statement"] == "plausível e inédito"
    assert board[-1]["statement"] == "inédito mas sem base"


async def test_board_hides_refuted_by_default(store):
    llm = ScriptedLLM(
        [_batch(_spec(statement="sobrevive"), _spec(statement="reprovada"))],
        [_ok_critique(), _ok_critique(fatal_flaw="circular", survives=False)],
    )
    explorer = Explorer(store, llm, profile=PROFILE)
    await explorer.generate(max_items=2)

    assert [b["statement"] for b in explorer.board()] == ["sobrevive"]
    assert len(explorer.board(only_surviving=False)) == 2


async def test_duplicate_statement_is_not_reinserted(store):
    llm = ScriptedLLM([_batch(_spec()), _batch(_spec())])
    explorer = Explorer(store, llm, profile=PROFILE)
    assert len(await explorer.generate()) == 1
    assert await explorer.generate() == []


async def test_prior_speculations_are_shown_to_the_generator(store):
    """Sem isto o gerador repropõe a mesma ideia toda rodada."""
    llm = ScriptedLLM([_batch(_spec()), _batch(_spec(statement="outra ideia"))])
    explorer = Explorer(store, llm, profile=PROFILE)
    await explorer.generate()
    await explorer.generate()

    assert "sigma-1" in llm.prompts[2]
    assert "do not repeat" in llm.prompts[2].lower()


# ─────────────────────────────────────────────────────── taxonomia aberta


def test_taxonomy_is_mostly_not_drugs():
    """Contrato de FORMA do perfil de PRODUÇÃO, e ele SOBREVIVE à parametrização.

    Se a maioria das classes fosse farmacológica, a "criatividade" seria escolher outro
    comprimido. Roda contra produção de propósito: é uma afirmação sobre a calibração
    que este projeto escolheu, não sobre a mecânica de carga.
    """
    classes = [c.label for c in prod_profile().taxonomy.intervention_classes]
    drug_classes = [
        c for c in classes
        if any(t in c for t in ("stabiliser", "lithium", "antipsychotic", "anxiolytic",
                                "drug", "compound", "nutraceutical", "anti-inflammatory"))
    ]
    assert len(drug_classes) < len(classes) / 2


def test_taxonomy_covers_non_pharmacological_modalities():
    """Contra o perfil de PRODUÇÃO: é sobre o que ESTE projeto declarou.

    O teste continua sendo dado literal contra o TOML — as duas pontas são
    independentes, e apagar uma classe do taxonomy.toml reprova aqui.
    """
    joined = " ".join(c.label for c in prod_profile().taxonomy.intervention_classes)
    for expected in ("neuromodulation", "chronotherapy", "metabolic", "exercise",
                     "biofeedback", "sequencing", "deprescribing"):
        assert expected in joined


def test_mechanism_targets_go_beyond_classic_neurotransmission():
    joined = " ".join(prod_profile().taxonomy.mechanism_targets)
    for expected in ("sigma-1", "orexin", "neurosteroid", "neuroinflammation",
                     "circadian", "microbiome", "mitochondrial", "vagal"):
        assert expected in joined


def test_the_taxonomy_block_renders_the_ACTIVE_profile_not_a_constant():
    """A FIAÇÃO, e ela não pode ser verificada contra o perfil de produção.

    MUTAÇÃO que isto mata: voltar `taxonomy_block()` a ler uma constante de módulo.
    Nenhuma string de oncologia veterinária existe em `lithium/`, então o bloco só
    pode conter "oncolytic virotherapy" se ele tiver de fato lido o perfil recebido.
    """
    block = taxonomy_block(PROFILE)
    assert "oncolytic virotherapy" in block
    assert "anthracycline" in block
    assert "sigma-1" not in block, "vazou a taxonomia do perfil de produção"


def test_the_profile_preserves_declaration_order():
    """A ORDEM É CONTEÚDO: o docstring de `mechanism.py` diz que as entradas mais
    distantes da prática padrão vêm no FIM de propósito, para não sugerirem menor
    importância.

    MUTAÇÃO: passar a carga por `set` ou ordenar por chave em qualquer ponto. Com
    `sorted`, "alkylating DNA damage" continua primeiro por acaso, mas "oncolytic
    virotherapy" e "tumour microenvironment / hypoxia" sobem para o meio e o docstring
    passa a mentir sobre o que afirma proteger.
    """
    # A asserção é sobre o BLOCO RENDERIZADO, não sobre a lista carregada. A mutação
    # que importa (`sorted()` dentro de `taxonomy_block`) não toca o loader — MEDIDO:
    # com a asserção só sobre `profile.taxonomy.mechanism_targets` ela NÃO matava nada.
    linhas = [l.strip("  - ") for l in taxonomy_block(PROFILE).splitlines()
              if l.startswith("  - ")]
    alvos = linhas[:len(PROFILE.taxonomy.mechanism_targets)]
    assert alvos == list(PROFILE.taxonomy.mechanism_targets), (
        "o bloco reordenou a taxonomia: a ordem é conteúdo, não formatação"
    )
    assert alvos[-2:] == ["oncolytic virotherapy",
                          "tumour microenvironment / hypoxia"]
    assert alvos != sorted(alvos), "a ordem do TOML foi normalizada"

    # e no perfil de PRODUÇÃO, onde o docstring de mechanism.py faz a promessa
    prod_linhas = [l.strip("  - ") for l in taxonomy_block(prod_profile()).splitlines()
                   if l.startswith("  - ")]
    assert (prod_linhas.index("microbiome-gut-brain axis")
            > prod_linhas.index("GABAergic modulation")), (
        "microbioma subiu para o meio — o docstring passa a mentir sobre o que protege"
    )


def test_intervention_classes_and_their_keywords_are_one_table():
    """Rótulo e keywords vêm da MESMA entrada, e keywords vazias reprovam a CARGA.

    Antes eram duas tabelas — `INTERVENTION_CLASSES` e as chaves de `CLASS_KEYWORDS` —
    idênticas por acidente (17 e 17, diferença simétrica vazia) e sem NENHUM teste que
    as cruzasse. Uma classe sem keyword apareceria como intocada para sempre, porque
    nada poderia tocá-la.

    MUTAÇÃO: declarar uma classe sem keywords no TOML.
    """
    import pytest as _pytest
    from lithium.focus import ProfileError, load_profile

    for prof in (PROFILE, prod_profile()):
        for klass in prof.taxonomy.intervention_classes:
            assert klass.keywords, f"{klass.label} não pode ser tocada por nada"

    import tempfile, shutil, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        dest = pathlib.Path(tmp) / "onco-vet"
        shutil.copytree(FIXTURE_FOCUSES / "onco-vet", dest)
        tax = dest / "taxonomy.toml"
        tax.write_text(
            tax.read_text(encoding="utf-8")
            + '\n[[intervention_classes]]\nlabel = "muda"\nkeywords = []\n',
            encoding="utf-8")
        with _pytest.raises(ProfileError):
            load_profile(dest)


def test_taxonomy_block_declares_itself_open():
    """A lista precisa se apresentar como mapa, não como menu — senão o modelo
    trata as entradas como as únicas opções válidas."""
    block = taxonomy_block(prod_profile())
    assert "not a menu" in block
    assert "outside them is" in block


def test_sql_view_and_python_agree_on_plausibility(store):
    """Regressão: a view SQL contava `$.status = 'supported'` enquanto o schema
    Pydantic gera `supported: bool` + `evidence`. A plausibilidade em SQL era sempre
    zero, e o quadro ordenava por ineditismo puro sem que nada acusasse.

    Mesma classe do bug L2-vs-cosseno: duas implementações da mesma métrica que
    ninguém comparava. Este teste compara.
    """
    casos = [
        [],
        [_step("a")],
        [_step("a", supported=True, evidence="PMID:1")],
        [_step("a", supported=True, evidence="PMID:1"), _step("b")],
        [_step("a", supported=True, evidence=""), _step("b")],          # sem citação
        [_step("a", supported=True, evidence="PMID:1"),
         _step("b", supported=True, evidence="NCT02"), _step("c")],
    ]
    for i, chain in enumerate(casos):
        store.conn.execute(
            "INSERT INTO hypotheses(focus_id, statement, tier, chain_json, novelty) "
            "VALUES(1, ?, 'speculative', ?, 0.5)",
            (f"caso {i}", json.dumps(chain)),
        )
        row = store.conn.execute(
            "SELECT plausibility FROM speculation_board WHERE statement = ?", (f"caso {i}",)
        ).fetchone()
        assert row["plausibility"] == pytest.approx(plausibility(chain)), f"caso {i}: {chain}"


def test_views_are_recreated_on_reopen(tmp_path):
    """`CREATE VIEW IF NOT EXISTS` deixaria uma view corrigida sem chegar a bancos
    existentes. Views não têm dados; dropar e recriar é sempre seguro."""
    path = tmp_path / "v.db"
    first = Store(path, embedding_dim=8)
    first.init_schema()
    first.conn.execute("DROP VIEW speculation_board")
    first.conn.execute("CREATE VIEW speculation_board AS SELECT 1 AS obsoleta")
    first.close()

    second = Store(path, embedding_dim=8)
    second.init_schema()
    columns = {r["name"] for r in second.conn.execute("PRAGMA table_info(speculation_board)")}
    assert "plausibility" in columns
    second.close()


@pytest.mark.parametrize("raw", ["none", "None", "N/A", "-", "nenhum", "  ", "null"])
async def test_none_markers_are_normalised_to_empty(store, raw):
    """O modelo escreve "none" em vez de string vazia. Sem normalizar, a interface
    exibe "FALHA FATAL: none" — que lê como se houvesse uma falha."""
    llm = ScriptedLLM([_batch(_spec())], [_ok_critique(fatal_flaw=raw, survives=True)])
    [record] = await Explorer(store, llm, profile=PROFILE).generate()
    assert record.fatal_flaw == ""


async def test_board_exposes_the_critique(store):
    """O elo mais fraco é o que o revisor precisa checar primeiro."""
    llm = ScriptedLLM([_batch(_spec())])
    await Explorer(store, llm, profile=PROFILE).generate()
    row = store.conn.execute("SELECT critique_json FROM speculation_board").fetchone()
    assert json.loads(row["critique_json"])["weakest_link"]


# ────────────────────────── o laço: especular → buscar → ancorar (o que fechou a gaiola)


class LoopLLM(ScriptedLLM):
    """Adiciona respostas para o planejamento de buscas e a reancoragem."""

    def __init__(self, batches=(), critiques=(), queries=(), regroundings=()) -> None:
        super().__init__(batches, critiques)
        self._queries = list(queries)
        self._regroundings = list(regroundings)

    async def structured(self, messages, schema, **kw):
        from lithium.llm.schemas import RegroundedChain, SpeculationQueries

        if schema is SpeculationQueries:
            self.prompts.append(messages[0]["content"])
            return self._queries.pop(0)
        if schema is RegroundedChain:
            self.prompts.append(messages[0]["content"])
            item = self._regroundings.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return await super().structured(messages, schema, **kw)


def _queries(*pairs) -> "SpeculationQueries":  # noqa: F821
    from lithium.llm.schemas import SourceQuery, SpeculationQueries
    from lithium.types import SourceKind

    return SpeculationQueries(queries=[
        SourceQuery(source=SourceKind(src), query=q, seeking="elo 2")
        for src, q in pairs
    ])


def _regrounded(*links) -> "RegroundedChain":  # noqa: F821
    from lithium.llm.schemas import RegroundedChain, RegroundedLink

    return RegroundedChain(links=[
        RegroundedLink(index=i, now_supported=sup, evidence=ev) for i, sup, ev in links
    ])


class FakeRetriever:
    def __init__(self, hits=()) -> None:
        self._hits = list(hits)

    async def search_claims(self, query, **kw):
        return self._hits


def _hit(external_id: str, statement: str):
    from lithium.pipeline.retrieval import ClaimHit
    from lithium.types import Directness, Grade

    return ClaimHit(
        claim_id=1, statement=statement, grade=Grade.PRECLINICAL,
        directness=Directness.EXTRAPOLATED, confidence=0.8, source_id=1,
        external_id=external_id, title="Estudo", year=2020, relevance=1.0, weight=0.08,
    )


async def test_speculation_generates_directed_searches(store):
    """Sem isto a trilha exploratória é um beco sem saída: as 19 queries fixas do
    harvest não mencionam sigma-1, orexina, via transdérmica nem dispositivo, então
    o sistema propõe um alvo e nunca pergunta a nenhuma base sobre ele."""
    llm = LoopLLM([_batch(_spec())],
                  queries=[_queries(("pubmed", '"Receptors, sigma"[MeSH] AND anxiety'))])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()

    queries = await explorer.plan_queries(record.id)
    assert len(queries) == 1
    assert "sigma" in queries[0].query.lower()


async def test_query_prompt_marks_which_links_need_anchoring(store):
    """Os elos assumidos são exatamente onde uma busca teria valor."""
    llm = LoopLLM([_batch(_spec())], queries=[_queries(("pubmed", "x"))])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()
    await explorer.plan_queries(record.id)

    prompt = llm.prompts[-1]
    assert "[ASSUMED]" in prompt
    assert "transdermal patch" in prompt, "a via precisa chegar ao planejador de buscas"


async def test_queries_for_unimplemented_sources_are_dropped(store):
    """Query para fonte sem adapter viraria tarefa morta na fila."""
    llm = LoopLLM([_batch(_spec())],
                  queries=[_queries(("pubmed", "ok"), ("ctgov", "ainda não existe"))])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()
    assert [q.source.value for q in await explorer.plan_queries(record.id)] == ["pubmed"]


async def test_pursuit_is_recorded_so_it_does_not_repeat(store):
    llm = LoopLLM([_batch(_spec())], queries=[_queries(("pubmed", "x"))])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()

    assert explorer.pending_pursuit() == [record.id]
    explorer.mark_pursued(record.id)
    assert explorer.pending_pursuit() == []


async def test_refuted_hypotheses_are_never_pursued(store):
    """Gastar buscas numa hipótese com falha fatal é desperdício puro."""
    llm = LoopLLM([_batch(_spec())],
                  [_ok_critique(fatal_flaw="circular", survives=False)])
    explorer = Explorer(store, llm, profile=PROFILE)
    await explorer.generate()
    assert explorer.pending_pursuit() == []


# ─────────────────────────────────────────────────────────── reancoragem


async def test_regrounding_raises_plausibility(store):
    """É por aqui que uma hipótese especulativa ganha plausibilidade com o tempo.
    Sem isto a cadeia congela no estado em que nasceu e perseguir não teria
    consequência mensurável."""
    llm = LoopLLM([_batch(_spec())],
                  regroundings=[_regrounded((2, True, "PMID:777"))])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()
    assert record.plausibility == pytest.approx(0.5)

    anchored = await explorer.reground(
        record.id, FakeRetriever([_hit("777", "sigma-1 reduz ansiedade em roedores")])
    )
    assert anchored == 1
    row = store.conn.execute(
        "SELECT plausibility FROM speculation_board WHERE id = ?", (record.id,)
    ).fetchone()
    assert row["plausibility"] == pytest.approx(1.0)


async def test_regrounding_rejects_citations_not_in_the_evidence(store):
    """Sem esta checagem, a reancoragem vira o caminho mais fácil para inflar
    plausibilidade: basta escrever um PMID qualquer."""
    llm = LoopLLM([_batch(_spec())],
                  regroundings=[_regrounded((2, True, "PMID:99999999"))])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()

    anchored = await explorer.reground(
        record.id, FakeRetriever([_hit("777", "algo diferente")])
    )
    assert anchored == 0
    row = store.conn.execute(
        "SELECT plausibility FROM speculation_board WHERE id = ?", (record.id,)
    ).fetchone()
    assert row["plausibility"] == pytest.approx(0.5)


async def test_regrounding_never_downgrades_a_supported_link(store):
    llm = LoopLLM([_batch(_spec())],
                  regroundings=[_regrounded((1, False, ""), (2, False, ""))])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()
    await explorer.reground(record.id, FakeRetriever([_hit("777", "x")]))

    chain = json.loads(store.conn.execute(
        "SELECT chain_json FROM hypotheses WHERE id = ?", (record.id,)
    ).fetchone()["chain_json"])
    assert chain[0]["supported"] is True


async def test_regrounding_with_empty_corpus_is_a_noop(store):
    llm = LoopLLM([_batch(_spec())])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()
    assert await explorer.reground(record.id, FakeRetriever([])) == 0
    assert store.conn.execute(
        "SELECT regrounded_at FROM hypotheses WHERE id = ?", (record.id,)
    ).fetchone()["regrounded_at"] is not None


async def test_fully_anchored_chains_leave_the_reground_queue(store):
    llm = LoopLLM([_batch(_spec())], regroundings=[_regrounded((2, True, "PMID:777"))])
    explorer = Explorer(store, llm, profile=PROFILE)
    [record] = await explorer.generate()
    explorer.mark_pursued(record.id)

    assert explorer.pending_reground() == [record.id]
    await explorer.reground(record.id, FakeRetriever([_hit("777", "x")]))
    assert explorer.pending_reground() == []


# ───────────────────────────────────── via de administração e combinações


async def test_route_and_combination_are_persisted(store):
    """Via é variável mecanística, não embalagem: o risco de virada acompanha a
    velocidade da subida monoaminérgica, e a via define o Tmax."""
    llm = ScriptedLLM([_batch(_spec(
        route="subcutaneous depot / implant",
        combination="agente A + bloqueador B, que cancela a liability de A",
    ))])
    [record] = await Explorer(store, llm, profile=PROFILE).generate()

    row = store.conn.execute("SELECT * FROM speculation_board WHERE id = ?",
                             (record.id,)).fetchone()
    assert row["route"] == "subcutaneous depot / implant"
    assert "cancela a liability" in row["combination"]


def test_route_taxonomy_reaches_past_pills():
    """Duas metades, e elas ficam em SENTENÇAS separadas de propósito.

    "não é embalagem" é AGNÓSTICO e mora em `route_block`; por que a via é
    mecanisticamente carregada NESTE foco é dado de perfil (`route_rationale`).
    Parametrizar a metade de foco sem separar as sentenças mataria a trava agnóstica
    em silêncio — as duas viviam na mesma frase.
    """
    joined = " ".join(prod_profile().taxonomy.routes)
    for expected in ("transdermal", "intranasal", "implant", "inhaled",
                     "implanted device", "wearable"):
        assert expected in joined
    assert "packaging" in route_block(prod_profile()), (
        "a via precisa se apresentar como mecanismo"
    )
    assert "packaging" in route_block(PROFILE), (
        "a metade AGNÓSTICA tem de sobreviver a qualquer perfil"
    )


def test_generation_prompt_demands_compounds_combinations_and_routes():
    from lithium.llm.prompts import render
    from lithium.pipeline.mechanism import route_block, taxonomy_block

    import re

    # Normaliza espaço: as frases quebram linha no markdown, e prender o teste ao
    # ponto exato da quebra o faria falhar em toda reformatação do prompt.
    prompt = re.sub(r"\s+", " ", render(
        "generate_speculation", **prod_profile().prompt_blocks("generate_speculation"),
            taxonomy=taxonomy_block(prod_profile()),
            routes=route_block(prod_profile()),
        existing="", lessons="", state="", max_items=3,
    ))
    assert "Novel compounds and novel chemistry" in prompt
    assert "Combinations." in prompt
    assert "Non-oral routes and devices" in prompt
    assert "never been given to a patient in this population is fine" in prompt
    assert "one component *cancels the liability* of the other" in prompt
    assert "implanted and wearable devices" in prompt
