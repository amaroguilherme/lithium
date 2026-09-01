"""O relatório periódico: o que MUDOU, e o que não dá para saber.

Agnóstico a foco. Ele não sabe o que é um fármaco nem um ensaio — só lê contadores,
arestas e as corridas de extração, e o alvo entra como texto vindo do banco.

**Delta, não total.** "213 claims" é teatro de crescimento: o número sobe com qualquer
afrouxamento de portão, e sobe sozinho com o tempo. O que responde "o que aconteceu" é a
diferença desde o último relatório, e é por isso que `reports.created_at` é a régua.

**O NÃO tem o mesmo espaço que o SIM.** Um relatório que só conta o que entrou descreve um
sistema que nunca recusa nada — e recusar é metade do trabalho deste. Ver METRICS.md.

**O que não é mensurável é DECLARADO.** Bloco que esconde tem de dizer o que esconde; a
tabela de cobertura já ensinou o custo de não fazer isso.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KIND = "periodic"


@dataclass(slots=True)
class Section:
    title: str
    lines: list[str] = field(default_factory=list)
    caveat: str = ""


@dataclass(slots=True)
class Report:
    focus: str | None
    since: str | None
    sections: list[Section] = field(default_factory=list)

    def render(self) -> str:
        alvo = self.focus or "NENHUM foco ativo — nada está sendo pontuado"
        janela = f"desde {self.since}" if self.since else "desde o início (primeiro relatório)"
        out = [f"# Relatório — {alvo}", "", f"Janela: {janela}.", ""]
        for s in self.sections:
            out.append(f"## {s.title}")
            out.extend(s.lines or ["(nada neste período)"])
            if s.caveat:
                out += ["", f"> {s.caveat}"]
            out.append("")
        return "\n".join(out).rstrip() + "\n"


def _last_report_at(store: Any) -> str | None:
    row = store.conn.execute(
        "SELECT created_at FROM reports WHERE kind = ? ORDER BY created_at DESC LIMIT 1",
        (KIND,),
    ).fetchone()
    return row["created_at"] if row else None


def _gates(store: Any, since: str | None) -> Section:
    """MF1: o portão 1 é o operador `in` do Python sobre o texto do chunk.

    Nenhuma complacência de modelo move este número — é a única linha do relatório em que
    o sistema é pontuado por uma função pura. Ver METRICS.md.
    """
    where, params = "1=1", []
    if since:
        where, params = "created_at > ?", [since]
    r = store.conn.execute(
        f"SELECT COUNT(*) AS n, COALESCE(SUM(proposed),0) AS p, "
        f"       COALESCE(SUM(anchored),0) AS a, COALESCE(SUM(verified),0) AS v, "
        f"       SUM(chunks_annihilated) AS ani, SUM(chunks_sterile) AS est, "
        f"       COALESCE(SUM(chunks_annihilated IS NULL), 0) AS sem_chunk, "
        f"       COALESCE(SUM(origin = 'log'), 0) AS segunda_mao "
        f"  FROM extraction_runs WHERE {where}", tuple(params)).fetchone()
    s = Section("Os dois portões")
    if not r["p"]:
        s.lines.append("Nenhuma extração nesta janela.")
        return s
    s.lines += [
        f"- Fontes extraídas: **{r['n']}**",
        f"- Portão 1 (citação literal): **{r['a']}/{r['p']}** "
        f"({r['a'] / r['p'] * 100:.1f}%)",
        f"- Portão 2 (a citação sustenta): **{r['v']}/{r['a']}** "
        f"({r['v'] / r['a'] * 100:.1f}%)" if r["a"] else "- Portão 2: nada chegou nele",
    ]
    if r["ani"] is not None:
        s.lines.append(
            f"- Chunks: **{r['ani']}** aniquilados (paper certo, citação ruim) · "
            f"**{r['est']}** estéreis (paper errado)")
    if r["sem_chunk"]:
        s.lines.append(
            f"- Chunks: **desconhecido** em {r['sem_chunk']} corrida(s) — retro-encaixe "
            f"de log não tem as rejeições individuais")
    if r["segunda_mao"]:
        s.caveat = (
            f"{r['segunda_mao']} destas {r['n']} corridas foram retro-encaixadas de um "
            f"arquivo de log (`origin='log'`): medição de segunda mão, e as rejeições "
            f"individuais delas não existem.")
    return s


def _corpus(store: Any, since: str | None) -> Section:
    """O que entrou. Delta quando há relatório anterior."""
    where, params = "1=1", []
    if since:
        where, params = "extracted_at > ?", [since]
    n = store.conn.execute(
        f"SELECT COUNT(*) AS n FROM claims WHERE {where}", tuple(params)).fetchone()["n"]
    s = Section("O que entrou")
    s.lines.append(f"- Claims novas: **{n}**")
    if not n:
        return s
    direcoes = {r["direction"]: r["n"] for r in store.conn.execute(
        f"SELECT direction, COUNT(*) AS n FROM claims WHERE {where} GROUP BY direction",
        tuple(params))}
    contra = direcoes.get("negative", 0) + direcoes.get("null", 0) + direcoes.get("mixed", 0)
    s.lines.append(
        f"- Direção: {direcoes.get('positive', 0)} a favor · **{contra} contra, nula ou "
        f"mista**")
    s.caveat = (
        "Contra-evidência com espaço próprio de propósito: um corpus só de achados "
        "positivos é um corpus que parou de procurar o que contradiz.")
    return s


def _unweighted(store: Any) -> Section:
    """As exclusões silenciosas. São estado, não delta — e o ponto é serem visíveis."""
    c = store.counts()
    s = Section("O que NÃO está pesando")
    s.lines += [
        f"- Sem julgamento neste foco: **{c.get('claims_unjudged', 0)}**",
        f"- Graduadas em outra escala: **{c.get('claims_off_scale', 0)}**",
        f"- Julgadas fora de escopo: **{c.get('claims_out_of_scope', 0)}**",
    ]
    s.caveat = (
        "Sem peso não é invisível: estas claims continuam recuperáveis e citáveis, "
        "marcadas como fora do foco ativo. O que elas não fazem é entrar no placar.")
    return s


def _needs_you(store: Any) -> Section:
    """O que espera decisão humana. É a única seção acionável do relatório."""
    s = Section("O que precisa de você")
    esc = list(store.conn.execute(
        "SELECT id, text, stuck_reason FROM questions WHERE status = 'ESCALATED' "
        " ORDER BY priority DESC LIMIT 10"))
    disc = store.conn.execute(
        "SELECT COUNT(*) AS n FROM discoveries WHERE status = 'pending'").fetchone()["n"]
    for q in esc:
        s.lines.append(f"- #{q['id']} [{q['stuck_reason'] or '—'}] {q['text'][:90]}")
    if disc:
        s.lines.append(f"- **{disc}** descoberta(s) da web esperando decisão "
                       f"(`lithium discoveries`)")
    return s


def build(store: Any, *, since: str | None = None) -> Report:
    """Monta o relatório. `since` explícito vence; senão, o último relatório salvo."""
    focus = store.active_focus()
    janela = since if since is not None else _last_report_at(store)
    return Report(
        focus=focus["target"] if focus else None,
        since=janela,
        sections=[
            _gates(store, janela),
            _corpus(store, janela),
            _unweighted(store),
            _needs_you(store),
        ],
    )


def save(store: Any, report: Report) -> int:
    """Persiste o corpo renderizado. É o que define a janela do PRÓXIMO relatório."""
    cur = store.conn.execute(
        "INSERT INTO reports(kind, body) VALUES(?, ?) RETURNING id",
        (KIND, report.render()))
    return int(cur.fetchone()["id"])
