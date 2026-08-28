"""As três travas: testes que impedem a evolução futura de destruir o que importa.

Ausências não são pegas por testes sobre saídas. Nenhum teste de "a resposta está boa"
nota que uma claim foi omitida, que um fator entrou na fórmula de peso, ou que a ordem
passou a favorecer o que chegou por último. Estes testes olham para o **código**, não
para o resultado.

**Trava 1** — memória de conversa nunca filtra nem reordena evidência (vive em
`test_invariant_memory_never_filters.py`, porque precisa de banco e embedder).

**Trava 2** — o peso de evidência é `grade × directness × confidence`, e tem **uma
origem só**: a view `claim_weight`.

**Trava 3** — nenhuma expressão de ranking de claim lê tempo. `claims.extracted_at`
mede ordem de colheita e é *anti-correlacionada* com directness: `pursue_speculation`
enfileira buscas com directness fraca que chegam depois da varredura fixa. Decair por
tempo de extração promoveria sistematicamente o material de população mais fraca, por
uma porta lateral que nenhum teste de saída pega.

---

**Duas brechas que uma versão anterior destas travas deixava passar**, ambas provadas
com o defeito construído — estão aqui porque um teste de invariante que não fecha o
caminho óbvio é pior que nenhum, já que passa a certificar o que não verifica:

1. Um decaimento por idade **dentro da própria view `claim_weight`**. A allowlist
   isentava a view das regras de aritmética (ela precisa multiplicar, afinal), e a
   verificação de tempo só olhava `ORDER BY` — e a view não tem `ORDER BY`. Fechado por
   igualdade exata da expressão de peso, varredura de tempo sobre o statement inteiro, e
   um teste numérico sobre as 324 combinações.

2. Desempate por colheita escrito como *decorate-sort-undecorate*: `sort()` sem `key=`
   sobre tuplas `(score, timestamp, hit)`, com o timestamp vindo de um dict lateral.
   Não tem `ORDER BY`, não tem `key=`, e `ClaimHit` não ganha campo nenhum. Fechado pela
   **fonte do dado**: um statement que amarra claims não pode SELECIONAR coluna de
   tempo. Um ranking não pode ler o que nunca foi buscado.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from pathlib import Path

import pytest

from lithium.db.store import SCHEMA_PATH
from lithium.types import DIRECTNESS_WEIGHT, GRADE_WEIGHT, Directness, Grade

PKG = Path(__file__).resolve().parent.parent / "lithium"
PY_FILES = sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)

# ───────────────────────────────────────────────────────── definições mecânicas

FACTORS = frozenset({"grade", "directness", "confidence"})
"""Os três multiplicandos da invariante central."""

TIME_TOKENS = re.compile(
    r"\b\w+_at\b|\bstrftime\s*\(|\bjulianday\s*\(|\bunixepoch\b"
    r"|\bdatetime\s*\(|\bdate\s*\(|\btime\s*\(",
    re.I,
)
"""Como se lê tempo em SQL aqui. `sources.year` **não** entra: mede a idade da
literatura, não a ordem em que o colhedor passou — um RCT de 2011 não fica menos
verdadeiro, mas um paper de 1970 é legitimamente outro tipo de evidência."""

CLAIM_TABLES = frozenset({"claims", "claim_weight", "claim_directness"})
"""`claim_directness` entra porque ganhou `judged_at` — uma coluna de TEMPO numa tabela
que participa do cálculo do peso. "Pegar o julgamento mais recente" é o desempate óbvio
que reintroduz ordem-de-colheita no eixo exato que a TRAVA 3 protege.

`scale_levels` NÃO entra: não tem coluna de tempo. O risco dela é outro — virar uma
SEGUNDA ORIGEM NUMÉRICA do peso — e acrescentar nomes a `FACTORS` não fecha esse buraco.
Quem a policia é `test_no_second_source_computes_the_weight`."""


def _py_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sql_statements(text: str) -> list[str]:
    """Quebra em statements, com comentários removidos.

    Comentários fora porque o schema explica as invariantes em prosa — e a prosa
    menciona `extracted_at` e `ORDER BY` justamente para dizer que não se usa.
    """
    body = re.sub(r"--[^\n]*", "", text)
    return [s.strip() for s in body.split(";") if s.strip()]


def _binds_claims(statement: str) -> bool:
    """O statement amarra `claims` ou `claim_weight` num FROM/JOIN/UPDATE/INTO?"""
    return any(
        re.search(rf"\b(from|join|update|into)\s+{t}\b", statement, re.I)
        for t in CLAIM_TABLES
    )


# ══════════════════════════════════════════════════════════════════════ TRAVA 2
#      o peso de evidência é o produto dos três fatores, e tem uma origem só


def _claim_weight_statement() -> str:
    for stmt in _sql_statements(SCHEMA_PATH.read_text(encoding="utf-8")):
        if re.match(r"create\s+view\s+claim_weight\b", stmt, re.I):
            return stmt
    raise AssertionError("a view claim_weight sumiu do schema")


def test_the_weight_expression_is_exactly_the_product_of_the_three_factors():
    """Igualdade exata, não continência.

    `"a * b * c" in body` passa quando alguém acrescenta um quarto fator — que é
    precisamente a mudança que a invariante existe para proibir. Só reprova a troca de
    `*` por `+`, que ninguém vai fazer por acidente.
    """
    stmt = _claim_weight_statement()
    match = re.search(r",\s*([^,]+?)\s+AS\s+weight\b", stmt, re.I | re.S)
    assert match, "não achei a expressão do peso na view claim_weight"

    expression = re.sub(r"\s+", " ", match.group(1)).strip()
    assert expression == "gw.weight * dw.weight * c.confidence", (
        f"a fórmula do peso mudou para {expression!r}. Ela é multiplicativa e tem "
        "exatamente três fatores: qualquer fator novo altera o significado de todo "
        "placar do sistema, e uma soma faria uma claim de grade máxima fora do "
        "assunto vencer uma no assunto."
    )


def test_the_weight_view_reads_no_time():
    """A brecha #1, fechada.

    A allowlist de aritmética isenta esta view — ela precisa multiplicar. Sem uma
    contrapartida explícita para tempo, um `CASE WHEN julianday('now') - ... > 30` cabe
    dentro da própria fórmula e nada reclama.
    """
    stmt = _claim_weight_statement()
    offender = TIME_TOKENS.search(stmt)
    assert offender is None, (
        f"a view claim_weight passou a ler tempo ({offender.group()!r}). "
        "`extracted_at` mede ordem de colheita e é anti-correlacionada com "
        "directness — decair por ele inverte a invariante central."
    )


@pytest.mark.parametrize("grade", list(Grade))
def test_the_sql_weight_equals_the_python_product(tmp_path, grade):
    """O teste numérico que fecha por comportamento, não por sintaxe.

    Qualquer fator extra na view aparece aqui, inclusive um que só se manifeste com
    dados velhos — as linhas são inseridas com `extracted_at` de dois anos atrás
    justamente para que um decaimento por idade não se esconda atrás de um fator 1.0.
    """
    from lithium.db import Store

    store = Store(tmp_path / f"w_{grade.value}.db", embedding_dim=4)
    store.init_schema()
    store.conn.execute(
        "INSERT INTO sources(kind, external_id, title, raw_json) "
        "VALUES('pubmed', '1', 't', '{}')"
    )

    expected: dict[int, float] = {}
    claim_id = 0
    for directness in Directness:
        for confidence in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            claim_id += 1
            store.conn.execute(
                "INSERT INTO claims(id, source_id, chunk_ids, statement, direction, "
                "  grade, scale_id, confidence, verified, extracted_at) "
                "VALUES(?, 1, '[]', 'x', 'positive', ?, 1, ?, 1, "
                "       '2023-01-01T00:00:00Z')",
                (claim_id, grade.value, confidence),
            )
            store.conn.execute(
                "INSERT INTO claim_directness(claim_id, focus_id, directness) "
                "VALUES(?, 1, ?)", (claim_id, directness.value),
            )
            expected[claim_id] = (
                GRADE_WEIGHT[grade] * DIRECTNESS_WEIGHT[directness] * confidence
            )

    got = {r["claim_id"]: r["weight"]
           for r in store.conn.execute("SELECT claim_id, weight FROM claim_weight")}
    # A cardinalidade ANTES da comparação. Sem ela, um fixture que deixe de ligar a
    # aresta faz a view devolver ZERO linhas, o laço não itera, nenhum assert roda e as
    # 9 parametrizações passam VERDES — a trava numérica da invariante central some
    # exatamente na fase que a reescreve.
    assert got.keys() == expected.keys(), (
        f"a view devolveu {len(got)} linha(s) para {len(expected)} claim(s) julgadas: "
        "faltando " + repr(sorted(set(expected) - set(got)))
    )
    for claim_id, weight in got.items():
        assert weight == pytest.approx(expected[claim_id], rel=1e-12), (
            f"claim #{claim_id} ({grade.value}): a view devolveu {weight!r}, o produto "
            f"dos três fatores é {expected[claim_id]!r}"
        )
    store.close()


class _FactorConsumption(ast.NodeVisitor):
    """Acha os três fatores usados para produzir NÚMERO ou ORDEM.

    Consumir é entrar em aritmética, comparação, chave de ordenação, ou índice das
    tabelas de peso. **Transportar não é consumir**: `SELECT`, `INSERT`, campo de
    dataclass, `Grade(row["grade"])` e formatação em texto passam — sem isso a claim
    não chega ao prompt com a etiqueta de qualidade que o revisor precisa ver.
    """

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def _names(self, node: ast.AST) -> set[str]:
        found: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in FACTORS:
                found.add(sub.id)
            elif isinstance(sub, ast.Attribute) and sub.attr in FACTORS:
                found.add(sub.attr)
            elif isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant):
                if sub.slice.value in FACTORS:
                    found.add(str(sub.slice.value))
        return found

    def visit_BinOp(self, node: ast.BinOp) -> None:
        for name in self._names(node.left) | self._names(node.right):
            self.hits.append((node.lineno, name))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        target = node.value
        base = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
        if base in {"GRADE_WEIGHT", "DIRECTNESS_WEIGHT"}:
            self.hits.append((node.lineno, base))
        self.generic_visit(node)


WEIGHT_ALLOWLIST = {
    ("lithium/types.py", "evidence_weight"),
    ("lithium/pipeline/state.py", "directness_gap"),
}
"""Por escopo QUALIFICADO (arquivo, função), nunca por arquivo.

Uma allowlist por arquivo liberaria `state.py` inteiro para calcular peso de claim, que
é exatamente a evolução que a trava tem de impedir. `evidence_weight` continua existindo
como a definição de referência dos pesos; `directness_gap` mede distância entre níveis
de directness, o que é ordenação de uma escala, não peso de claim."""


def _enclosing_function(tree: ast.AST, lineno: int) -> str:
    best = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= lineno <= end:
                best = node.name
    return best


def test_only_claim_weight_computes_the_evidence_weight():
    """Nenhuma expressão fora da allowlist transforma os três fatores em número."""
    offenders: list[str] = []
    for path in PY_FILES:
        tree = ast.parse(_py_source(path))
        visitor = _FactorConsumption()
        visitor.visit(tree)
        rel = path.relative_to(PKG.parent).as_posix()
        for lineno, what in visitor.hits:
            scope = _enclosing_function(tree, lineno)
            if (rel, scope) not in WEIGHT_ALLOWLIST:
                offenders.append(f"{rel}:{lineno} em {scope or '<módulo>'} usa {what}")
    assert not offenders, (
        "o peso de evidência tem uma origem só (a view `claim_weight`). "
        "Uma segunda implementação diverge em silêncio no dia em que os pesos "
        "mudarem, e aí o ranking do chat discorda do placar de hipóteses:\n  "
        + "\n  ".join(offenders)
    )


def test_the_weight_allowlist_has_no_dead_entries():
    """Uma permissão não pode sobreviver ao código que ela liberava — vira um buraco
    aberto esperando alguém escrever ali."""
    alive = set()
    for path in PY_FILES:
        tree = ast.parse(_py_source(path))
        rel = path.relative_to(PKG.parent).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                alive.add((rel, node.name))
    dead = {entry for entry in WEIGHT_ALLOWLIST if entry not in alive}
    assert not dead, f"entradas mortas na WEIGHT_ALLOWLIST: {sorted(dead)}"


LEVEL_TABLES = ("scale_levels", "grade_weight", "directness_weight")
"""As TRÊS tabelas de nível, não só `scale_levels`.

`grade_weight`/`directness_weight` sobrevivem como VOCABULÁRIO, mas continuam semeadas
com a calibração inteira e deixaram de ter leitor legítimo. Provado que escopar a
varredura só em `scale_levels` deixa passar uma sombra completa: um `shadow_score()` que
lê `(SELECT gwt.weight FROM grade_weight gwt ...)` e `(SELECT dwt.weight FROM
directness_weight dwt ...)` por subquery e multiplica em Python devolve exatamente o
mesmo número que `claim_weight`, e a TRAVA 2 por AST é cega para ele (os nomes estão
dentro de uma string)."""


def _level_aliases(statement: str) -> set[str]:
    """Aliases ligados a uma tabela de nível. Mesmo idioma de `_claim_aliases`.

    Por ALIAS e não por menção da tabela: o predicado ingênuo ("menciona a tabela E tem
    `weight` no SELECT") acusa `state.py`, que legitimamente faz `SUM(cw.weight)` vindo de
    `claim_weight` na mesma consulta em que dá JOIN em `scale_levels` para o rank. E o
    conserto tentador desse falso positivo — isentar statements que também mencionam
    `claim_weight` — reabriria o buraco, porque basta a sombra dar JOIN em claim_weight
    junto.
    """
    out: set[str] = set()
    for table in LEVEL_TABLES:
        for m in re.finditer(
            rf"\b(?:from|join)\s+{table}\b(?:\s+(?:as\s+)?(\w+))?", statement, re.I
        ):
            out.add(table)
            alias = m.group(1)
            if alias and alias.lower() not in {
                "on", "where", "group", "order", "join", "left", "inner", "and", "union",
            }:
                out.add(alias)
    return out


def test_no_second_source_computes_the_weight():
    """Nenhum statement fora da view `claim_weight` lê `weight` de uma tabela de nível.

    Acrescentar nomes a `FACTORS` NÃO fecha este buraco: a sombra vive dentro de uma
    string SQL, onde a AST não enxerga. Só a varredura fecha.
    """
    offenders = []
    for where, lineno, sql in _all_sql():
        if where == "lithium/db/schema.sql" and re.match(
            r"create\s+view\s+claim_weight\b", sql, re.I
        ):
            continue
        aliases = _level_aliases(sql)
        if not aliases:
            continue
        select = re.search(r"\bselect\b(.*?)\bfrom\b", sql, re.I | re.S)
        if not select:
            continue  # INSERT de semeadura: não tem lista de SELECT
        body = select.group(1)
        bare = re.search(r"(?<![\w.])weight\b", body)
        qualified = any(re.search(rf"\b{re.escape(a)}\.weight\b", body) for a in aliases)
        if bare or qualified:
            offenders.append(f"{where}:{lineno} → {' '.join(body.split())[:90]}")
    assert not offenders, (
        "uma SEGUNDA origem numérica do peso. `scale_levels` calibra e "
        "`grade_weight`/`directness_weight` são só vocabulário — quem multiplica é a "
        "view `claim_weight`, e mais nada:\n  " + "\n  ".join(offenders)
    )


def test_the_anti_shadow_scan_rejects_a_constructed_shadow():
    """Auto-verificação: a varredura precisa reprovar a sombra que já foi construída e
    que passou por TODAS as outras travas com a suíte inteira verde."""
    shadows = [
        ("subquery nas duas tabelas de vocabulário",
         "SELECT (SELECT gwt.weight FROM grade_weight gwt WHERE gwt.grade = c.grade) AS a, "
         "       (SELECT dwt.weight FROM directness_weight dwt "
         "          JOIN claim_directness cd ON cd.directness = dwt.directness "
         "         WHERE cd.claim_id = c.id) AS b, c.confidence AS k "
         "  FROM claims c WHERE c.id = ?"),
        ("leitura direta de scale_levels",
         "SELECT sl.weight FROM scale_levels sl WHERE sl.axis = 'grade' AND sl.value = ?"),
    ]
    for label, sql in shadows:
        aliases = _level_aliases(sql)
        body = re.search(r"\bselect\b(.*?)\bfrom\b", sql, re.I | re.S).group(1)
        caught = bool(re.search(r"(?<![\w.])weight\b", body)) or any(
            re.search(rf"\b{re.escape(a)}\.weight\b", body) for a in aliases
        )
        assert caught, f"a varredura deixou passar: {label}"


# ══════════════════════════════════════════════════════════════════════ TRAVA 3
#               nenhuma expressão de ranking de claim lê tempo


TIME_SELECT_ALLOWLIST = {
    # `pursued_at`/`regrounded_at` são controle de agenda: "já persegui esta hipótese?".
    # Não entram em ranking nenhum, e o predicado que os usa é `IS NULL`.
    "hypotheses",
}


def _sql_literals(path: Path) -> list[tuple[int, str]]:
    """Strings que parecem SQL, com a linha. Concatenação implícita já vem unida."""
    out: list[tuple[int, str]] = []
    tree = ast.parse(_py_source(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.search(r"\bselect\b|\bupdate\b|\binsert\s+into\b", node.value, re.I):
                out.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                v.value for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            if re.search(r"\bselect\b|\bupdate\b|\binsert\s+into\b", text, re.I):
                out.append((node.lineno, text))
    return out


def _all_sql() -> list[tuple[str, int, str]]:
    found = [
        (path.relative_to(PKG.parent).as_posix(), lineno, sql)
        for path in PY_FILES
        for lineno, sql in _sql_literals(path)
    ]
    found += [
        ("lithium/db/schema.sql", 0, stmt)
        for stmt in _sql_statements(SCHEMA_PATH.read_text(encoding="utf-8"))
    ]
    return found


def test_no_claim_ranking_orders_by_time():
    """A forma direta: `ORDER BY ... extracted_at`."""
    offenders = []
    for where, lineno, sql in _all_sql():
        if not _binds_claims(sql):
            continue
        order_by = re.search(r"\border\s+by\b(.*?)(?:\blimit\b|$)", sql, re.I | re.S)
        if order_by and TIME_TOKENS.search(order_by.group(1)):
            offenders.append(f"{where}:{lineno}")
    assert not offenders, (
        "ranking de claim ordenando por tempo: " + ", ".join(offenders)
    )


def _claim_aliases(statement: str) -> set[str]:
    """Aliases que se ligam a `claims`/`claim_weight` neste statement.

    Precisa ser por alias, não por qualquer `id`: `ORDER BY s.id DESC` na consulta de
    fontes áridas ordena **fontes** — "quais tentamos recentemente e não renderam nada" —
    e recência de tentativa de colheita é justamente o sinal certo ali. O que não pode é
    ordenar *claim* por id.
    """
    aliases = {"claims", "claim_weight"}
    for table in CLAIM_TABLES:
        for m in re.finditer(
            rf"\b(?:from|join)\s+{table}\s+(?:as\s+)?(\w+)", statement, re.I
        ):
            if m.group(1).lower() not in {"on", "where", "group", "order", "join", "left"}:
                aliases.add(m.group(1))
    return aliases


def test_no_claim_ranking_orders_by_id():
    """`claims.id` é ordem de colheita com outro nome, e `TIME_TOKENS` não o vê.

    `INTEGER PRIMARY KEY` é monotônico na inserção, então `ORDER BY c.id` é um **proxy
    exato** de `extracted_at` — a mesma anti-correlação com directness, pela mesma porta
    lateral, sem casar nenhum dos padrões temporais.

    Medido no bloco de claims da reflexão: com 20 claims `opinion × extrapolated`
    colhidas primeiro e 20 `meta_analysis × direct` depois, `ORDER BY c.id ASC` com
    `LIMIT 12` renderiza **doze linhas, todas `opinion/extrapolated`, zero
    meta-análise** — e entrega isso ao modelo sob o cabeçalho "claims verificadas, cite
    estas". A inversão completa da invariante central.
    """
    offenders = []
    for where, lineno, sql in _all_sql():
        if not _binds_claims(sql):
            continue
        order_by = re.search(r"\border\s+by\b(.*?)(?:\blimit\b|$)", sql, re.I | re.S)
        if not order_by:
            continue
        terms = order_by.group(1)
        aliases = _claim_aliases(sql)
        hit = re.search(r"\b(\w+)\.id\b", terms, re.I)
        bare = re.search(r"(?:^|,)\s*id\b", terms, re.I)
        if bare or (hit and hit.group(1) in aliases):
            offenders.append(f"{where}:{lineno} → ORDER BY {terms.strip()[:60]}")
    assert not offenders, (
        "ranking de claim ordenando por id — que é ordem de colheita:\n  "
        + "\n  ".join(offenders)
    )


def test_no_claim_query_even_selects_a_time_column():
    """A brecha #2, fechada pela FONTE do dado.

    Fechar pela forma da ordenação não basta: um desempate escrito como
    decorate-sort-undecorate (`sort()` sem `key=`, timestamp vindo de um dict lateral)
    não tem `ORDER BY`, não tem `key=`, e não põe campo nenhum em `ClaimHit`. Mas ele
    precisa **ler** o timestamp de algum lugar. Um ranking não pode ler o que nunca
    foi buscado.
    """
    offenders = []
    for where, lineno, sql in _all_sql():
        if not _binds_claims(sql):
            continue
        if any(re.search(rf"\b{t}\b", sql, re.I) for t in TIME_SELECT_ALLOWLIST):
            continue
        select = re.search(r"\bselect\b(.*?)\bfrom\b", sql, re.I | re.S)
        if select and TIME_TOKENS.search(select.group(1)):
            offenders.append(f"{where}:{lineno} → {select.group(1).strip()[:70]}")
    assert not offenders, (
        "consulta sobre claims trazendo coluna de tempo. Mesmo sem ORDER BY, o valor "
        "pode virar desempate em Python sem deixar rastro sintático:\n  "
        + "\n  ".join(offenders)
    )


def test_claim_hit_carries_no_timestamp():
    """Defesa em profundidade estrutural: sem campo, não há o que ler."""
    from lithium.pipeline.retrieval import ClaimHit, Hit

    for cls in (ClaimHit, Hit):
        temporal = [f for f in cls.__dataclass_fields__ if f.endswith("_at")]
        assert not temporal, f"{cls.__name__} ganhou campo temporal: {temporal}"


PY_SORT_KEYS = {
    # ordena claim (ou linha derivada de claim) → a checagem de tempo se aplica
    "lambda c: c.score": True,                       # retrieval: o ranking de evidência
    "lambda c: c.conflict * c.total_weight": True,   # state: tabela de conflitos
    "lambda t: (t[0], -t[1])": True,                 # reflect: lições por lacuna
    # não ordena claim
    "lambda kv: kv[1]": False,                       # retrieval: fusão RRF de chunks
    "lambda d: d['index']": False,                   # client: remonta ordem do batch
    "lambda a: (order.get(a.severity, 9), a.key)": False,   # safety: alerta, não claim
    "levels.index": True,          # reflect.weakest_directness: directness de claim
    "lambda g: _GRADE_RANK[g]": False,   # pubmed: tipo de publicação, não claim
    "<sem key=>": False,                             # ver a nota abaixo
}
"""Registro das chaves de ordenação em Python.

Só as marcadas `True` ordenam claim — ou linha derivada de claim, que conta igual: a
tabela de conflitos de `state.py` agrega peso de claims, e um termo temporal ali teria o
mesmo efeito perverso de promover o que foi colhido por último.

`max`/`min` entram na varredura a partir da Fase A: `weakest_directness` é um
`max(..., key=levels.index)` sobre directness derivado de claim, e a lógica de
comparação entre julgamentos é exatamente onde um `max(rows, key=lambda r: r["judged_at"])`
apareceria. Antes disso a varredura só via sort/sorted/nlargest/nsmallest e era cega
para os dois.

`<sem key=>` está registrado como um grupo porque as ocorrências atuais são todas
ordenações de valores escalares para exibição ou para lista de parâmetros
(`sorted(stats.items())`, `sorted(p.stem for p in ...)`, `sorted({h.source_id ...})`),
onde não há atributo para ler tempo. O que o registro garante é que uma chave **nova**
apareça como não classificada; a defesa contra desempate por colheita escrito sem
`key=` é `test_no_claim_query_even_selects_a_time_column`, que fecha pela fonte do
dado."""


def _sort_calls(path: Path):
    tree = ast.parse(_py_source(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name not in {"sort", "sorted", "nlargest", "nsmallest", "max", "min"}:
            continue
        key = next((kw.value for kw in node.keywords if kw.arg == "key"), None)
        yield node, key


def test_every_python_sort_key_over_claims_is_classified():
    """Toda chave de ordenação é conhecida — inclusive a ausência de chave.

    `sort()` sem `key=` sobre uma lista de tuplas é a forma idiomática do desempate por
    colheita, e uma varredura que só coleta `key=` é cega para ela. Aqui ela aparece
    como não classificada.
    """
    unknown = []
    for path in PY_FILES:
        rel = path.relative_to(PKG.parent).as_posix()
        for node, key in _sort_calls(path):
            expr = ast.unparse(key) if key is not None else "<sem key=>"
            if expr not in PY_SORT_KEYS:
                unknown.append(f"{rel}:{node.lineno} → {expr}")
    assert not unknown, (
        "chave de ordenação não classificada. Registre em PY_SORT_KEYS dizendo se "
        "ordena claim (True) ou não (False):\n  " + "\n  ".join(unknown)
    )


def test_the_sort_key_registry_has_no_dead_entries():
    """Uma classificação não pode sobreviver ao código que ela classificava.

    MUTAÇÃO: tirar `max`/`min` de `_sort_calls`. Sem esta checagem, a varredura
    simplesmente deixa de ver `weakest_directness` e `_pick_strongest` e ninguém nota —
    a mesma cegueira que existia antes da Fase A, restaurada em silêncio.
    """
    seen = {"<sem key=>"}
    for path in PY_FILES:
        for _node, key in _sort_calls(path):
            seen.add(ast.unparse(key) if key is not None else "<sem key=>")
    dead = set(PY_SORT_KEYS) - seen
    assert not dead, (
        f"entradas mortas em PY_SORT_KEYS: {sorted(dead)}. Ou o código sumiu, ou a "
        "varredura deixou de enxergá-lo."
    )


def test_claim_sort_keys_read_no_time():
    offenders = []
    for path in PY_FILES:
        rel = path.relative_to(PKG.parent).as_posix()
        for node, key in _sort_calls(path):
            if key is None:
                continue
            expr = ast.unparse(key)
            if not PY_SORT_KEYS.get(expr):
                continue
            if TIME_TOKENS.search(expr):
                offenders.append(f"{rel}:{node.lineno} → {expr}")
    assert not offenders, "chave de ordenação de claim lendo tempo: " + ", ".join(offenders)


# ─────────────────────────────────────── auto-verificação: as travas podem falhar

CONSTRUCTED_DEFECTS = [
    ("fator extra na fórmula",
     "gw.weight * dw.weight * c.confidence * 0.5"),
    ("decaimento por idade dentro da view",
     "gw.weight * dw.weight * c.confidence * "
     "(CASE WHEN julianday('now') - julianday(c.extracted_at) > 30 THEN 0.5 ELSE 1.0 END)"),
    ("soma no lugar do produto",
     "gw.weight + dw.weight + c.confidence"),
]


TIME_DEFECTS = [
    # Os defeitos NÃO nomeiam `claims`: é justamente por isso que `claim_directness`
    # precisa estar em CLAIM_TABLES. Um statement que só toca a aresta já ranqueia claim.
    ("desempate pelo julgamento mais recente",
     "SELECT cd.claim_id, cd.directness FROM claim_directness cd "
     " WHERE cd.focus_id = 1 ORDER BY cd.judged_at DESC"),
    ("mesma coisa, escrita sem ORDER BY (decorate-sort-undecorate)",
     "SELECT cd.claim_id, cd.judged_at FROM claim_directness cd WHERE cd.focus_id = 1"),
]


@pytest.mark.parametrize("label,sql", TIME_DEFECTS)
def test_the_time_lock_covers_the_judgment_edge(label, sql):
    """`judged_at`, no retro-preenchimento, é ordem de extração das claims — a mesma
    anti-correlação com directness que a TRAVA 3 documenta. E "pegar o julgamento mais
    recente" é o desempate óbvio no eixo exato que ela protege.

    MUTAÇÃO: remover `claim_directness` de `CLAIM_TABLES`. `_binds_claims` deixa de
    casar e os dois defeitos passam por todas as checagens.
    """
    assert _binds_claims(sql), (
        "claim_directness saiu de CLAIM_TABLES: a TRAVA 3 nem olha para este statement"
    )
    order_by = re.search(r"\border\s+by\b(.*?)(?:\blimit\b|$)", sql, re.I | re.S)
    by_time = bool(order_by and TIME_TOKENS.search(order_by.group(1)))
    select = re.search(r"\bselect\b(.*?)\bfrom\b", sql, re.I | re.S)
    selects_time = bool(select and TIME_TOKENS.search(select.group(1)))
    assert by_time or selects_time, f"a trava deixou passar: {label}"


@pytest.mark.parametrize("label,expression", CONSTRUCTED_DEFECTS)
def test_the_weight_lock_rejects_a_constructed_defect(label, expression):
    """Cada defeito é construído e submetido às mesmas checagens.

    Sem isto o módulo inteiro é uma afirmação sobre si mesmo. As duas brechas
    documentadas no topo passaram *verdes* por uma versão anterior destas travas.
    """
    original = _claim_weight_statement()
    stmt = original.replace("gw.weight * dw.weight * c.confidence", expression)
    # A auto-verificação precisa verificar A SI MESMA. MEDIDO: renomear os aliases da
    # view para `g`/`d` faz este `.replace()` virar no-op, `got` volta a ser a expressão
    # antiga, `caught` fica True pela razão errada e os TRÊS defeitos construídos passam
    # verdes sem ter injetado nada.
    assert stmt != original, (
        "o defeito não foi injetado: os aliases da view deixaram de ser `gw`/`dw`"
    )
    match = re.search(r",\s*([^,]+?)\s+AS\s+weight\b", stmt, re.I | re.S)
    got = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""

    caught = got != "gw.weight * dw.weight * c.confidence" or bool(TIME_TOKENS.search(stmt))
    assert caught, f"a trava deixou passar: {label}"
