"""As travas da Fase B: perfil de foco em disco, carga fail-loud, e o relens.

Cada trava aqui vem com a MUTAÇÃO que a mata escrita no docstring, e todas foram
EXECUTADAS. Uma trava cuja mutação não mata teste não vale — é o defeito de
fiação-não-testada que este repo já cometeu seis vezes.
"""

from __future__ import annotations

import json
import shutil
import re
from pathlib import Path

import pytest

from conftest import FIXTURE_FOCUSES, onco_profile, prod_profile, seed_claim
from lithium.db import Store
from lithium.focus import ProfileError, load_profile
from lithium.llm.prompts import UnknownPlaceholder, render
from lithium.types import Directness, Grade

PROFILE = onco_profile()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "p.db", embedding_dim=8)
    s.init_schema()
    s.upsert_source(kind="pubmed", external_id="1", raw={})
    yield s
    s.close()


def _copy(tmp_path: Path, name: str = "onco-vet") -> Path:
    dest = tmp_path / name
    shutil.copytree(FIXTURE_FOCUSES / "onco-vet", dest)
    if name != "onco-vet":
        f = dest / "focus.toml"
        f.write_text(f.read_text(encoding="utf-8")
                     .replace('slug   = "onco-vet"', f'slug   = "{name}"'),
                     encoding="utf-8")
    return dest


def _edit_focus(directory: Path, old: str, new: str) -> Path:
    f = directory / "focus.toml"
    text = f.read_text(encoding="utf-8")
    assert old in text, f"o trecho a editar sumiu do fixture: {old!r}"
    f.write_text(text.replace(old, new), encoding="utf-8")
    return directory


# ══════════════════════════════════════════════════ 1. a camada de renderização


def test_render_rejects_a_kwarg_the_prompt_does_not_declare():
    """`render()` levanta quando recebe kwarg que o template não usa.

    Fecha a classe INTEIRA de meia-fiação, que era 100% silenciosa: `Template.substitute`
    levanta em kwarg FALTANDO e ignora kwarg SOBRANDO.

    MUTAÇÃO EXECUTADA: reverter `render()` para `_load(name).substitute(**kwargs)` puro.
    Antes desta trava, `extract.py` passava `target=focus["target"]` para um
    `extract_claims.md` cujos placeholders declarados eram só
    {design, journal, sample_n, text, title, year} — a linha não fazia nada e a suíte
    inteira (736/736) passava verde.
    """
    with pytest.raises(UnknownPlaceholder) as exc:
        render("verify_citation", quote="q", statement="s", target="sobrando")
    assert "target" in str(exc.value)
    # e a direção que já existia continua valendo
    with pytest.raises(KeyError):
        render("verify_citation", quote="q")


def test_the_extraction_grammar_does_not_name_any_focus():
    """A GRAMÁTICA é o segundo canal de domínio, e nenhum teste do repo olhava para ele.

    `model_json_schema()` de todo BaseModel exportado por `llm/schemas.py` vai inteiro
    para `response_format` — ou seja, chega ao modelo junto da restrição de decodificação,
    que é o que ele é OBRIGADO a seguir. Nenhum dos 14 goldens vê uma linha disto.

    MUTAÇÃO EXECUTADA: restaurar `"Aderência da população estudada ao alvo: TB-I com TAG
    comórbido."` no docstring de `types.Directness`. Reprova em 7 schemas —
    ClaimExtraction, ExtractedClaim, PatternVerdict, QueryPlan, SearchQuery, Speculation,
    SpeculationBatch — porque o docstring do enum viaja em todo schema que o referencia.

    Os termos vêm do perfil de PRODUÇÃO, não de uma lista literal: um alvo novo passaria
    a ser policiado sem ninguém editar o teste.
    """
    import re

    from pydantic import BaseModel

    import lithium.llm.schemas as schemas

    prod = prod_profile()
    termos = {"TB-I", "TAG", "GAD", "bipolar", "manic", "virada", "anxiolysis",
              "monoaminérgica", "monoaminergic"}
    termos |= {w for w in re.findall(r"[A-Za-zÀ-ÿ][\w-]{3,}", prod.target)}

    offenders: list[str] = []
    for name, obj in vars(schemas).items():
        if not (isinstance(obj, type) and issubclass(obj, BaseModel)
                and obj.__module__ == schemas.__name__):
            continue
        blob = json.dumps(obj.model_json_schema(), ensure_ascii=False)
        for termo in termos:
            if re.search(rf"(?<![A-Za-zÀ-ÿ]){re.escape(termo)}(?![A-Za-zÀ-ÿ])",
                         blob, re.I):
                offenders.append(f"{name}: {termo!r}")
    assert not offenders, (
        "o alvo do foco vazou para dentro da GRAMÁTICA (response_format):\n  "
        + "\n  ".join(offenders)
    )


# ══════════════════════════════════════════════════════ 2. a carga é fail-loud


def test_a_fifth_directness_level_is_refused_at_load(tmp_path):
    """`[directness]` tem de declarar EXATAMENTE os quatro nomes globais.

    MUTAÇÃO: remover a validação de vocabulário em `FocusToml._check_directness`. Um
    `focus.toml` com `surrogate` carregaria, entraria em `scale_levels` (que não tem FK
    nesse eixo), o foco pareceria funcionar, e o erro só apareceria na PRIMEIRA ESCRITA
    como `FOREIGN KEY constraint failed` dentro de `_persist` — com o perfil já
    commitado. Sintoma tardio, na escrita, longe da causa.
    """
    a_mais = _edit_focus(
        _copy(tmp_path, "a-mais"),
        'extrapolated = { prose = "murine models, cell lines, or pure inference" }',
        'extrapolated = { prose = "murine models, cell lines, or pure inference" }\n'
        'surrogate    = { prose = "endpoint substituto" }',
    )
    with pytest.raises(ProfileError) as exc:
        load_profile(a_mais)
    assert "surrogate" in str(exc.value)

    a_menos = _edit_focus(
        _copy(tmp_path, "a-menos"),
        'indirect     = { prose = "other canine neoplasia, or feline lymphoma" }\n', "")
    with pytest.raises(ProfileError) as exc:
        load_profile(a_menos)
    assert "indirect" in str(exc.value)


def test_profile_models_reject_unknown_keys(tmp_path):
    """`standing_risk = []` (typo no singular) REPROVA a carga.

    MUTAÇÃO: trocar `extra="forbid"` pelo default do pydantic. Nenhum modelo do repo
    usava forbid antes desta fase exceto `llm/schemas.py`, e sem esta trava a regra
    "chave ausente é ERRO" não se sustenta: o typo faz a chave certa ficar ausente, e
    se o campo tivesse default o foco rodaria SEM risco permanente sem nada acusar.
    """
    # A chave DESCONHECIDA entra sem tirar nenhuma obrigatória — senão o que reprova é
    # o campo faltando, e a trava de `forbid` fica sem ser exercida (mutação medida:
    # trocar `extra="forbid"` pelo default NÃO matava o teste anterior).
    d = _edit_focus(_copy(tmp_path, "typo"), 'reader = "veterinary oncologist"',
                    'reader = "veterinary oncologist"\nstanding_risk = []')
    with pytest.raises(ProfileError) as exc:
        load_profile(d)
    assert "standing_risk" in str(exc.value) and "extra" in str(exc.value).lower()

    # e num SUBMODELO, não só no topo
    d2 = _edit_focus(_copy(tmp_path, "typo2"), 'name    = "tumour lysis"',
                     'name    = "tumour lysis"\nseverity = "high"')
    with pytest.raises(ProfileError):
        load_profile(d2)


def test_an_empty_anti_corpus_is_refused_at_load(tmp_path):
    """`must_not_match = []` REPROVA a carga.

    Sem o anti-corpo, `pattern = "risk"` passa, casa em 14 de 14 prompts, e a trava dos
    seis prompts fica verde cobrindo NADA — o mesmo vazio do bloco de segurança sem
    regra, transplantado para dentro do teste.

    MUTAÇÃO EXECUTADA: `must_not_match: list[str] = []`. Antes desta trava a mutação
    NÃO matava nada, porque o fixture declarava o anti-corpo de qualquer jeito e o teste
    só afirmava sobre o que o fixture trazia.
    """
    vazio = _edit_focus(
        _copy(tmp_path, "sem-anticorpo"),
        'must_not_match = [\n  "the chain quietly switches between acute effects",\n'
        '  "lyse the sample before assay",\n]',
        "must_not_match = []")
    with pytest.raises(ProfileError) as exc:
        load_profile(vazio)
    assert "must_not_match" in str(exc.value)

    # e um pattern largo demais, COM anti-corpo, também reprova — é o que o anti-corpo
    # existe para pegar
    largo = _edit_focus(
        _copy(tmp_path, "largo"),
        "pattern = 'tumou?r lysis|lysis syndrome'", "pattern = 'the'")
    with pytest.raises(ProfileError) as exc:
        load_profile(largo)
    assert "anti-corpo" in str(exc.value)


def test_the_bind_prose_must_name_the_standing_risk(tmp_path):
    """Os três blocos que ABREM os prompts têm de nomear o risco permanente.

    Nesses três (`extract_bind`, `question_bind`, `speculation_problem`) o risco não
    chega por `$standing_risks` — ele é parte do vínculo clínico, e injetá-lo à parte
    diria a mesma frase duas vezes na mesma janela.

    MEDIDO sem esta validação: um perfil cujos binds não nomeiam o próprio risco deixa
    3 dos 6 prompts de `REQUIRES_STANDING_RISK` vermelhos, e o caminho de menor
    resistência diante do vermelho é apagar as chaves — removendo a obrigação de nomear
    o risco justamente do prompt que decide o que entra no corpus.

    MUTAÇÃO EXECUTADA: `return self` no topo de `_binds_name_a_standing_risk`. Antes
    desta trava ela não matava nada.
    """
    mudo = _edit_focus(
        _copy(tmp_path, "bind-mudo"),
        "alkylator protocol, but rapid\ncytoreduction in a renally impaired dog risks "
        "tumour lysis syndrome. Direct evidence is\nscarce",
        "alkylator protocol, and the dose is what it is. Direct evidence is\nscarce")
    with pytest.raises(ProfileError) as exc:
        load_profile(mudo)
    assert "extract_bind" in str(exc.value)


def test_a_missing_standing_risks_key_is_an_error_but_an_empty_list_is_not(tmp_path):
    """Chave AUSENTE reprova; `standing_risks = []` carrega e apaga o parágrafo INTEIRO.

    A diferença entre "este foco não tem risco permanente" (decisão) e "esqueci de
    declarar" (esquecimento) é a diferença que só a primeira pode passar.

    MUTAÇÃO 1: dar default `[]` ao campo no pydantic — a distinção desaparece.
    MUTAÇÃO 2: renderizar a lista vazia deixando o CABEÇALHO no prompt. Aí `chat.md`
    passa a afirmar "One standing exception you raise unprompted whenever it becomes
    relevant:" seguido de NADA — o prompt promete uma exceção permanente e não diz qual.
    """
    ausente = _copy(tmp_path, "sem-chave")
    f = ausente / "focus.toml"
    text = f.read_text(encoding="utf-8")
    f.write_text(text[:text.index("[[standing_risks]]")], encoding="utf-8")
    with pytest.raises(ProfileError) as exc:
        load_profile(ausente)
    assert "standing_risks" in str(exc.value)

    vazia = _copy(tmp_path, "sem-riscos")
    f = vazia / "focus.toml"
    text = f.read_text(encoding="utf-8")
    # Escalar de topo vem ANTES da primeira tabela — depois de `[directness]` a chave
    # cairia DENTRO dela.
    corte = text.index("[directness]")
    f.write_text(
        text[:corte] + "standing_risks = []\n\n"
        + text[corte:text.index("[[standing_risks]]")], encoding="utf-8")
    profile = load_profile(vazia)
    assert profile.focus.standing_risks == []

    rendered = render("chat", **profile.prompt_blocks("chat"),
                      memories="", evidence="", constraint_notes="",
                      recon_notes="", safety="")
    assert "standing exception" not in rendered, (
        "o cabeçalho sobreviveu à lista vazia: o prompt afirma que existe uma exceção "
        "permanente e não diz qual"
    )
    assert "Never let that pass unmarked" not in rendered
    # e o portão adversarial também não pode cobrar um risco que não existe
    assert profile.missing_risk_line() == ""


def test_a_target_that_is_blank_is_refused_at_load_and_at_the_database(tmp_path, store):
    """`target` vazio é aceito por `TEXT NOT NULL` e renderiza `matches ""` no prompt.

    E como o BANCO vence para `target`, corrigir o TOML depois NÃO conserta: o relens já
    teria julgado o corpus inteiro contra a string vazia com instrução fail-closed.

    MUTAÇÃO: remover `min_length=1` do campo, ou o `CHECK (length(trim(target)) > 0)`
    do schema. Testadas as DUAS pontas de propósito — o perfil é uma delas, mas
    `focus new` não é o único caminho até a tabela.
    """
    import sqlite3

    d = _edit_focus(_copy(tmp_path, "sem-alvo"),
                    'target = "canine multicentric lymphoma"', 'target = ""')
    with pytest.raises(ProfileError):
        load_profile(d)

    scale_id = int(store.conn.execute(
        "SELECT id FROM evidence_scales LIMIT 1").fetchone()["id"])
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO focuses(slug, target, scale_id) VALUES('vazio', '   ', ?)",
            (scale_id,))


def test_the_load_error_names_the_file(tmp_path):
    """`tomllib.TOMLDecodeError` sobe sem o nome do arquivo, e são QUATRO candidatos."""
    d = _copy(tmp_path, "quebrado")
    (d / "taxonomy.toml").write_text("isto = não é [ toml", encoding="utf-8")
    with pytest.raises(ProfileError) as exc:
        load_profile(d)
    assert "taxonomy.toml" in str(exc.value)


def test_a_missing_profile_directory_is_loud_not_a_default(tmp_path):
    """SEM fallback: `load_config` cai para defaults, e para perfil isso seria um foco
    fantasma — alvo vazio, zero estratégia, zero taxonomia, rodando contra nada."""
    with pytest.raises(ProfileError):
        load_profile(tmp_path / "nao-existe")


# ═══════════════════════════════════════════════════ 3. os três estados do peso


def _second_focus(store, *, slug="segundo", scale_id=None):
    if scale_id is None:
        scale_id = int(store.conn.execute(
            "SELECT scale_id FROM active_focus").fetchone()["scale_id"])
    cur = store.conn.execute(
        "INSERT INTO focuses(slug, target, scale_id) VALUES(?, 'outro alvo', ?) "
        "RETURNING id", (slug, scale_id))
    return int(cur.fetchone()["id"])


def test_status_separates_unjudged_from_off_scale(store):
    """Uma claim COM aresta mas em escala divergente conta em `claims_off_scale`, NÃO em
    `claims_unjudged` — e a distinção existe ANTES do relens, não depois.

    MEDIDO no contador único: a claim contava em `claims_unweighted` e a mensagem
    afirmava "sem julgamento de directness neste foco", que é FALSO. É a mensagem que
    manda o usuário rodar horas de GPU que não podem mudar o número.

    MUTAÇÃO 1: voltar ao contador único.
    MUTAÇÃO 2 (a sutil): definir `claims_off_scale` como "COM aresta E escala
    divergente". Aí, ANTES do relens, a claim de escala divergente conta como
    `unjudged`, o usuário roda o relens, e o número só TROCA DE COLUNA — `claim_weight`
    continua vazia. O passo se autodestruiria.
    """
    outra_escala = int(store.conn.execute(
        "INSERT INTO evidence_scales(slug) VALUES('outra') RETURNING id"
    ).fetchone()["id"])
    focus2 = _second_focus(store, scale_id=outra_escala)

    seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT)
    store.conn.execute("UPDATE meta SET value = ? WHERE key = 'active_focus'",
                       (str(focus2),))

    counts = store.counts()
    assert counts["claims_off_scale"] == 1, (
        "a claim está graduada em outra escala — o relens não resolve isso"
    )
    assert counts["claims_unjudged"] == 0, (
        "contar como 'sem julgamento' manda o usuário para um job que não pode ajudar"
    )
    # e AINDA assim ela não tem peso: os dois contadores decompõem, não substituem
    assert counts["claims_unweighted"] == 1


def test_the_three_counters_never_lose_a_claim(store):
    """`claims_unweighted` continua sendo o TOTAL exaustivo (derivado da própria view).

    Existe um QUARTO modo sem remédio de usuário: escala casando, aresta presente, e
    mesmo assim sem peso — `scale_levels` incompleta num dos eixos. Trocar o total
    exaustivo por uma soma de subqueries enumeradas à mão faria essa claim sumir de
    TODOS os contadores e o placar ficar zerado sem explicação.

    MUTAÇÃO: derivar `claims_unweighted` de `unjudged + off_scale + out_of_scope`.
    """
    escala_torta = int(store.conn.execute(
        "INSERT INTO evidence_scales(slug) VALUES('sem-grade') RETURNING id"
    ).fetchone()["id"])
    store.conn.executemany(
        "INSERT INTO scale_levels(scale_id, axis, value, weight, rank) VALUES(?,?,?,?,?)",
        [(escala_torta, "directness", d.value, 1.0, i)
         for i, d in enumerate(Directness, 1)],
    )
    focus2 = _second_focus(store, scale_id=escala_torta)
    seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT,
               focus_id=focus2, scale_id=escala_torta)
    store.conn.execute("UPDATE meta SET value = ? WHERE key = 'active_focus'",
                       (str(focus2),))

    c = store.counts()
    classificadas = (c["claims_unjudged"] + c["claims_off_scale"]
                     + c["claims_out_of_scope"])
    assert c["claims_unweighted"] == 1
    assert classificadas == 0, "esta claim cai no quarto modo, sem remédio de usuário"
    assert c["claims_unweighted"] >= classificadas, "o total tem de ser exaustivo"


def test_an_out_of_scope_judgment_is_not_the_same_as_no_judgment(store):
    """O TERCEIRO veredito: julgada e IRRELEVANTE, distinguível de não julgada.

    Sem ele, o relens de um foco novo importa o corpus do foco velho com peso POSITIVO:
    o piso `extrapolated` vale 0,12, não 0. MEDIDO aqui mesmo — dez claims fora de
    domínio superam UMA meta-análise perfeitamente no alvo, e como a cobertura ordena
    por peso SOMADO e corta em 40 linhas, a tabela vira do domínio antigo.

    MUTAÇÃO: remover `AND cd.out_of_scope = 0` da view `claim_weight`.
    """
    from lithium.types import evidence_weight

    dez_fora = 10 * evidence_weight(Grade.RCT, Directness.EXTRAPOLATED)
    uma_no_alvo = evidence_weight(Grade.META_ANALYSIS, Directness.DIRECT)
    assert dez_fora > uma_no_alvo, (
        "a premissa da trava: volume de material fora de domínio vence pertinência"
    )

    claim_id = seed_claim(store, grade=Grade.RCT, judged=False)
    focus_id = int(store.conn.execute("SELECT id FROM active_focus").fetchone()["id"])
    store.conn.execute(
        "INSERT INTO claim_directness(claim_id, focus_id, directness, out_of_scope) "
        "VALUES(?, ?, 'extrapolated', 1)", (claim_id, focus_id))

    pesadas = store.conn.execute("SELECT COUNT(*) AS n FROM claim_weight").fetchone()["n"]
    assert pesadas == 0, "uma claim julgada FORA DE ESCOPO não pode carregar peso"
    c = store.counts()
    assert c["claims_out_of_scope"] == 1
    assert c["claims_unjudged"] == 0, "ela FOI julgada; o relens não deve revisitá-la"


# ═══════════════════════════════════════════════════════════════ 4. o relens


def test_the_relens_does_not_enqueue_claims_it_cannot_help(store):
    """`claims_to_judge` pula a claim de escala divergente.

    Enfileirá-la gastaria ~5,8 s de GPU para mover um número de coluna: mesmo depois de
    julgada ela continua sem peso, porque `claim_weight` também casa a escala.
    """
    from lithium.pipeline.relens import claims_to_judge

    outra = int(store.conn.execute(
        "INSERT INTO evidence_scales(slug) VALUES('outra') RETURNING id"
    ).fetchone()["id"])
    focus2 = _second_focus(store, scale_id=outra)
    seed_claim(store, grade=Grade.RCT, judged=False)   # escala do foco 1
    assert claims_to_judge(store, focus2, outra) == []


def test_an_unjudgeable_claim_gets_no_row_at_all(store):
    """`judgeable=False` NÃO grava nada — nem o nível mais fraco.

    "Na dúvida, o nível MENOS aderente" só vale entre dois níveis ADJACENTES, com a
    população em mãos. Aplicá-lo à ausência de informação é permanente (a PK congela o
    valor e o sweep pula quem já tem aresta) e INVISÍVEL (nenhum contador distingue
    "julgada" de "defaultada"). O bloco de evidência que o revisor lê passaria a
    escrever `(rct / extrapolated)` sobre um RCT humano.

    MUTAÇÃO: gravar `extrapolated` quando `judgeable` é false. Aí `claims_unjudged` cai
    para 0, `lithium status` fica limpo, e não há caminho de reparo — rodar o relens de
    novo pula a claim porque ela tem aresta.
    """
    import asyncio

    from lithium.llm.schemas import DirectnessVerdict
    from lithium.pipeline.relens import judge_one

    class LLM:
        async def structured(self, messages, schema, **kw):
            return DirectnessVerdict(in_scope=True, judgeable=False,
                                     directness=Directness.EXTRAPOLATED,
                                     rationale="o texto não diz quem foi estudado")

    claim_id = seed_claim(store, grade=Grade.RCT, judged=False)
    focus_id = int(store.conn.execute("SELECT id FROM active_focus").fetchone()["id"])
    asyncio.run(judge_one(store, LLM(), PROFILE, claim_id=claim_id,
                          focus_id=focus_id, target="alvo"))

    n = store.conn.execute(
        "SELECT COUNT(*) AS n FROM claim_directness WHERE claim_id = ?", (claim_id,)
    ).fetchone()["n"]
    assert n == 0, "um julgamento sem base virou aresta permanente e invisível"
    assert store.counts()["claims_unjudged"] == 1, (
        "a claim tem de continuar visível para o relens poder revisitá-la"
    )


def test_the_judgement_prompt_carries_the_target_of_the_payload_focus(store):
    """O `$target` renderizado vem do foco do PAYLOAD, não de `active_focus()`.

    Esta é a metade que a asserção óbvia NÃO cobre: olhar o `focus_id` ESCRITO deixa o
    teste verde mesmo com o alvo vindo do foco errado, porque o id vem do payload nos
    dois casos. Trocar de foco no meio de um relens de 7,68 h faria metade do corpus ser
    julgada contra o alvo errado, sem log e sem nada em `judged_at` que permitisse
    reconstruir onde foi o corte.

    MUTAÇÃO: dentro de `judge_one`, resolver `target` por `store.active_focus()`.
    """
    import asyncio

    from lithium.llm.schemas import DirectnessVerdict
    from lithium.pipeline.relens import judge_one

    prompts: list[str] = []

    class LLM:
        async def structured(self, messages, schema, **kw):
            prompts.append(messages[0]["content"])
            return DirectnessVerdict(in_scope=True, judgeable=True,
                                     directness=Directness.PARTIAL, rationale="r")

    claim_id = seed_claim(store, grade=Grade.RCT, judged=False)
    focus2 = _second_focus(store, slug="tb2-panico")
    store.conn.execute("UPDATE focuses SET target = 'bipolar II + panic disorder' "
                       " WHERE id = ?", (focus2,))
    # o foco ATIVO continua sendo o #1 — o julgamento é do #2, que veio no payload
    asyncio.run(judge_one(store, LLM(), PROFILE, claim_id=claim_id, focus_id=focus2,
                          target="bipolar II + panic disorder"))

    assert "bipolar II + panic disorder" in prompts[0]
    assert "comorbid GAD" not in prompts[0], "vazou o alvo do foco ATIVO"
    gravado = store.conn.execute(
        "SELECT focus_id FROM claim_directness WHERE claim_id = ?", (claim_id,)
    ).fetchone()["focus_id"]
    assert int(gravado) == focus2


def test_the_judgement_sees_the_primary_evidence_not_only_the_paraphrase(store):
    """O chunk de origem ENTRA no prompt. `claims.population` é texto livre de um 12B.

    O próprio PLAN mede que as strings mais frequentes são genéricas (`adults`,
    `not specified`). Julgar só pela paráfrase daria ao relens informação ESTRITAMENTE
    PIOR que a que a extração teve, e transformaria o caso "torn" no caso comum.

    MUTAÇÃO: remover `$evidence` do prompt e o `_evidence_for` do call site.
    """
    import asyncio

    from lithium.llm.schemas import DirectnessVerdict
    from lithium.pipeline.relens import judge_one

    prompts: list[str] = []

    class LLM:
        async def structured(self, messages, schema, **kw):
            prompts.append(messages[0]["content"])
            return DirectnessVerdict(in_scope=True, judgeable=True,
                                     directness=Directness.DIRECT, rationale="r")

    source_id = store.upsert_source(kind="pubmed", external_id="30712879", raw={},
                                    title="A randomised trial")
    chunk_id = store.add_chunk(
        source_id=source_id, ord=0,
        text="We randomised 84 outpatients meeting DSM-5 criteria for bipolar II "
             "with comorbid panic disorder.")
    claim_id = seed_claim(store, source_id=source_id, chunk_ids=(chunk_id,),
                          population="outpatients with a mood disorder", judged=False)
    focus_id = int(store.conn.execute("SELECT id FROM active_focus").fetchone()["id"])
    asyncio.run(judge_one(store, LLM(), PROFILE, claim_id=claim_id,
                          focus_id=focus_id, target="alvo"))

    assert "84 outpatients" in prompts[0], (
        "sem a evidência primária o julgador só vê a paráfrase que outro 12B escreveu"
    )
    assert "A randomised trial" in prompts[0]


def test_the_extractor_renders_the_profile_it_was_given(store):
    """O prompt de extração carrega o vocabulário do perfil RECEBIDO.

    MUTAÇÃO EXECUTADA: trocar `self.profile.prompt_blocks("extract_claims")` por blocos
    do perfil de PRODUÇÃO fixos. Antes desta trava a mutação NÃO matava nada — a
    extração é o caminho mais coberto do repo, mas nenhum teste olhava QUAL perfil
    chegava ao prompt. Entregar isto errado é meia-troca de foco, e a metade que sobra é
    a que o decodificador obriga.
    """
    import asyncio

    from lithium.llm.schemas import ClaimExtraction
    from lithium.pipeline.extract import Extractor

    prompts: list[str] = []

    class Capturing:
        async def structured(self, messages, schema, **kw):
            prompts.append(messages[0]["content"])
            return ClaimExtraction(claims=[])

    # `target` vem do BANCO (é ele que define contra o quê as arestas foram julgadas);
    # todo o resto vem do PERFIL. A fronteira é disjunta e o teste checa as duas pontas.
    store.conn.execute("UPDATE focuses SET target = ? WHERE id = 1",
                       (PROFILE.target,))
    source_id = store.upsert_source(kind="pubmed", external_id="99", raw={})
    store.add_chunk(source_id=source_id, ord=0, text="algum texto de método")
    asyncio.run(Extractor(store, Capturing(), profile=PROFILE).extract_source(source_id))

    assert prompts, "nenhum prompt foi renderizado"
    # o vocabulário do PERFIL, que não existe em lithium/
    assert "dogs with multicentric lymphoma and renal impairment" in prompts[0]
    assert "tumour lysis" in prompts[0]
    assert "canine multicentric lymphoma" in prompts[0]
    # e nada do perfil de produção
    assert "bipolar" not in prompts[0]
    assert "comorbid GAD" not in prompts[0]


def test_a_recovered_relens_task_is_a_no_op_not_a_dead_letter(store):
    """A aresta já existente é detectada ANTES do POST, sem gastar GPU.

    Como isto acontece de verdade: o handler grava a aresta (autocommit,
    `isolation_level=None`) e o daemon morre antes de `queue.complete(task_id)` —
    inclusive por Ctrl-C, porque o `complete` está fora do `except CancelledError`. A
    tarefa fica `running`; no restart `recover_orphans()` a devolve para `pending` sem
    zerar `attempts`.

    Sem o pré-teste: re-julga (5,4 s de GPU), bate em `IntegrityError`, `fail`, 3
    tentativas — ~16 s de GPU queimados e um dead-letter com "UNIQUE constraint failed"
    que não diz nada ao usuário. E, pelo defeito da chave queimada, com a `dedup_key`
    perdida junto.

    MUTAÇÃO EXECUTADA: `if False:` no lugar de `if already is not None:`. Antes desta
    trava a mutação NÃO matava nada.
    """
    import asyncio

    from lithium.pipeline.relens import judge_one

    class Explode:
        async def structured(self, messages, schema, **kw):
            raise AssertionError("o LLM foi chamado para uma claim já julgada")

    claim_id = seed_claim(store, grade=Grade.RCT, directness=Directness.DIRECT)
    focus_id = int(store.conn.execute("SELECT id FROM active_focus").fetchone()["id"])
    assert asyncio.run(judge_one(store, Explode(), PROFILE, claim_id=claim_id,
                                 focus_id=focus_id, target="alvo")) is None
    # e nada foi duplicado
    n = store.conn.execute(
        "SELECT COUNT(*) AS n FROM claim_directness WHERE claim_id = ?", (claim_id,)
    ).fetchone()["n"]
    assert n == 1


# ═══════════════════════════════ 5. o esqueleto não pode virar um foco que mente


def test_a_scaffolded_profile_refuses_to_load_until_it_is_reviewed(tmp_path):
    """`focus --new` copia o perfil de REFERÊNCIA inteiro e só remove `target`.

    Encontrado rodando a troca de ponta a ponta pelo CLI real, não por leitura: criei um
    foco `cardio-af`, preenchi só o `target` com "fibrilação atrial com insuficiência
    cardíaca", e o `extract_claims` renderizado saiu dizendo *"The system is investigating
    treatment options for bipolar I disorder with comorbid generalized anxiety disorder"*
    com os quatro níveis de directness definidos em populações bipolares — sob um alvo
    cardiológico. Nada reclamou. `focus --show` até imprime a incoerência; imprimir não é
    barrar, e a escada de directness é por onde TODA claim é pesada.

    A marca é uma e cobre os quatro arquivos de propósito: `strategies.toml`,
    `taxonomy.toml` e `safety.toml` vêm do mesmo `copytree` e estão igualmente com o
    domínio de referência.

    MUTAÇÃO: `scaffold_pending: bool = False` deixar de ser lido em `load_profile`, ou o
    `focus --new` parar de escrever a linha (as duas metades têm asserção separada).
    """
    d = _edit_focus(
        _copy(tmp_path, "meia-adaptado"),
        'slug   = "meia-adaptado"',
        'slug   = "meia-adaptado"\nscaffold_pending = true',
    )
    with pytest.raises(ProfileError, match="scaffold_pending"):
        load_profile(d)

    # E a contrapartida: sem a marca, o MESMO perfil carrega. Sem isto a trava poderia
    # ser "recusa sempre" e o teste não notaria.
    f = d / "focus.toml"
    f.write_text(f.read_text(encoding="utf-8").replace("scaffold_pending = true", ""),
                 encoding="utf-8")
    assert load_profile(d).slug == "meia-adaptado"


def test_focus_new_writes_the_scaffold_mark_it_refuses_to_load(tmp_path, monkeypatch):
    """A outra metade, por COMPORTAMENTO do comando — não por leitura do fonte.

    Sem esta, remover a escrita da marca em `cli.py` deixa a suíte verde: o teste acima
    põe a linha à mão. É a classe de defeito que já apareceu seis vezes neste repo —
    trava com fiação não exercitada.
    """
    from typer.testing import CliRunner

    from lithium.cli import app

    data = tmp_path / "d"
    focuses = tmp_path / "focuses"
    focuses.mkdir()
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text(f'data_dir = "{data}"\nfocuses_dir = "{focuses}"\n',
                       encoding="utf-8")

    runner = CliRunner()
    assert runner.invoke(app, ["init", "-c", str(cfgfile)]).exit_code == 0
    r = runner.invoke(app, ["focus", "-c", str(cfgfile), "--new", "cardio-af"])
    assert r.exit_code == 0, r.output

    toml = focuses / "cardio-af" / "focus.toml"
    written = toml.read_text(encoding="utf-8")
    assert "scaffold_pending = true" in written, (
        "o esqueleto nasceu sem a marca: preencher só o `target` produziria um foco "
        "que anuncia um alvo e raciocina sobre outro"
    )

    # O usuário desatento: preenche `target` (a única chave que o esqueleto exige) e
    # para por aí. É exatamente o caminho que produzia o foco mentiroso.
    toml.write_text(written.replace('# target = "..."',
                                    'target = "fibrilação atrial com IC"'),
                    encoding="utf-8")
    with pytest.raises(ProfileError, match="scaffold_pending"):
        load_profile(focuses / "cardio-af")


def test_a_new_focus_is_born_free_of_any_other_focus_domain(tmp_path):
    """O molde de `focus --new` é NEUTRO, não o foco de produção.

    Era `focuses/bipolar-tag`, e isso acoplava o maquinário a um foco específico: o
    comando quebrava se aquele foco fosse renomeado ou aposentado, e o perfil novo nascia
    cheio do domínio DELE — de modo que preencher só o `target` produzia um perfil que
    anunciava um assunto e raciocinava sobre outro. MEDIDO na época: com alvo cardiológico
    e o resto intocado, o prompt de extração falava de transtorno bipolar.

    MUTAÇÃO: apontar `REFERENCE_PROFILE` de volta para `"bipolar-tag"`.
    """
    from typer.testing import CliRunner

    from lithium.cli import app

    focuses = tmp_path / "f"
    focuses.mkdir()
    raiz = Path(__file__).resolve().parent.parent / "focuses"
    shutil.copytree(raiz / "_reference", focuses / "_reference")
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text(f'data_dir = "{tmp_path / "d"}"\nfocuses_dir = "{focuses}"\n',
                       encoding="utf-8")
    runner = CliRunner()
    assert runner.invoke(app, ["init", "-c", str(cfgfile)]).exit_code == 0
    r = runner.invoke(app, ["focus", "-c", str(cfgfile), "--new", "ligas-metalicas"])
    assert r.exit_code == 0, r.output

    # O molde SOZINHO basta: o teste não copiou nenhum foco real para `focuses_dir`, e
    # se `focus --new` ainda dependesse de `bipolar-tag` ele teria falhado acima.
    dominio = re.compile(r"bipolar|lítio|quetiapin|anxiety|psiquiatr|mania", re.I)
    for arquivo in (focuses / "ligas-metalicas").glob("*.toml"):
        achados = dominio.findall(arquivo.read_text(encoding="utf-8"))
        assert not achados, (
            f"{arquivo.name} nasceu com vocabulário de outro foco: {sorted(set(achados))}"
        )
