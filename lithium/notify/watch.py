"""O que merece um aviso, e como não virar tempestade.

**Por estado derivado, não por gancho.** A escalação é escrita em quatro lugares e o
dead-letter em dois: pendurar um gancho em cada um são seis call sites, e wiring em N
call sites é a classe de falha que este projeto repetiu seis vezes. Um tick que compara o
estado atual contra uma marca d'água tem **um** call site, é durável e sobrevive a
restart.

**Mas estado derivado sozinho não resolve tempestade** — isso foi medido e derrubou a
primeira versão desta frente. Uma falha determinística (um prompt que estourou a janela,
digamos) manda toda tarefa daquele tipo para o dead-letter de uma vez: **380 avisos**. O
que resolve é **agrupar**, e agrupando a primeira execução deixa de ser caso especial —
vira só um delta grande, com um aviso.

**As duas marcas d'água óbvias estão quebradas**, as duas por motivos reais:

* `max(escalated_at)` — `cli.py` promove N linhas num único UPDATE com
  `strftime('now')`, então os timestamps são **idênticos por construção**. Com `>` você
  perde avisos para sempre; com `>=`, reavisa para sempre.
* contagem simples — não distingue "duas novas" de "uma nova e uma resolvida".

Então a marca é o **conjunto de ids**, que é exato e não depende de relógio.

E o laço de amplificação que nenhum dos dois desenhos evita sozinho: um `notify_tick` que
morre vira dead-letter, dead-letter é gatilho de aviso, aviso roda `notify_tick`. Fechado
em dois lugares — o delta de dead-letter exclui o próprio `notify_tick`, e o handler é
estruturalmente incapaz de morrer.

**A MARCA TEM ESCOPO, e cada categoria declara o seu.** Isto é conserto de um defeito
VIVO, e é pré-requisito da Fase C em vez de bônus. EXECUTADO antes desta mudança: com
dois focos e 3 tarefas mortas, o tick do foco A anuncia "3 tarefa(s) falharam", o
segundo tick do foco A anuncia nada (correto), e ao trocar para o foco B o MESMO
cemitério é anunciado inteiro de novo — porque a MARCA era por foco e a CONSULTA de
dead-letter é GLOBAL. Reverter o escopo por foco de `_mark_key` matava **0 de 805**
testes E eliminava o reaviso: as duas metades da Fase B nunca andaram juntas, ao
contrário do que o docstring afirmava. Construir a terceira categoria sobre uma marca
que já reavisa é construir sobre a falha.

`DELTAS` é o mapa literal que amarra as duas metades: cada categoria declara a ESCALA
que a query dela de fato tem, e `test_every_notify_category_declares_the_scope_its_query_actually_has`
compara a declaração com o texto da função.
"""

from __future__ import annotations

import json
import logging

from lithium.db import Store
from lithium.notify.send import Notice

log = logging.getLogger(__name__)

MARK_KEY_PREFIX = "notify:seen"
GLOBAL_SCOPE_KEY = f"{MARK_KEY_PREFIX}:global"

MAX_NAMED = 3
"""Quantos itens o aviso nomeia antes de virar contagem. Um toast com 380 linhas não é
um toast."""


def _mark_key(store: Store, scope: str = "focus") -> str:
    """A chave da marca, na ESCALA que a categoria declarou.

    `'focus'` → `notify:seen:<focus_id>`; `'global'` → `notify:seen:global`. Marca por
    foco com consulta global é estritamente PIOR que a chave global: trocar de foco faz
    o cemitério inteiro voltar a ser "novo".
    """
    if scope == "global":
        return GLOBAL_SCOPE_KEY
    focus = store.active_focus()
    return f"{MARK_KEY_PREFIX}:{focus['id'] if focus else 0}"


def _mark(store: Store, scope: str = "focus") -> dict:
    row = store.conn.execute(
        "SELECT value FROM meta WHERE key = ?", (_mark_key(store, scope),)
    ).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row["value"]) or {}
    except (TypeError, ValueError):
        # `meta` corrompido não pode matar o tick — ver o docstring do módulo.
        log.warning("marca de notificação ilegível, recomeçando do zero")
        return {}


def _save(store: Store, mark: dict, scope: str = "focus") -> None:
    store.conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_mark_key(store, scope), json.dumps(mark)),
    )


def seed_global_mark(store: Store) -> bool:
    """Semeia `notify:seen:global` a partir das marcas por foco já gravadas.

    Sem isto, a primeira subida DEPOIS desta mudança produz exatamente a tempestade que
    ela existe para evitar: `dead_delta` passa a ler uma chave que não existe, `seen`
    vira `{}`, `fresh` vira `current`, e o tick anuncia um cemitério inteiro que o
    usuário já viu e já ignorou. O docstring do módulo registra que 380 avisos de uma
    vez derrubaram a primeira versão desta frente.

    Máximo por `kind` sobre TODAS as marcas por foco: a contagem de dead-letter é
    global, então a marca mais avançada é a que já foi anunciada.
    """
    if store.conn.execute(
        "SELECT 1 FROM meta WHERE key = ?", (GLOBAL_SCOPE_KEY,)
    ).fetchone() is not None:
        return False
    rows = store.conn.execute(
        "SELECT value FROM meta WHERE key LIKE ? AND key <> ?",
        (MARK_KEY_PREFIX + ":%", GLOBAL_SCOPE_KEY),
    ).fetchall()
    merged: dict[str, int] = {}
    for row in rows:
        try:
            old = (json.loads(row["value"]) or {}).get("dead") or {}
        except (TypeError, ValueError):
            continue
        for kind, n in old.items():
            merged[kind] = max(merged.get(kind, 0), int(n))
    if not merged:
        return False
    _save(store, {"dead": merged}, "global")
    log.info("marca global de dead-letter semeada a partir de %d marca(s) por foco",
             len(rows))
    return True


def escalated_delta(store: Store, seen: list[int]) -> tuple[list[dict], list[int]]:
    """Perguntas escaladas que ainda não foram anunciadas. ESCOPADA POR FOCO.

    Lê o **estado**, não um evento: uma pergunta que o `Answerer` devolveu como
    `Action.ESCALATE` sem escalar de fato — o caso `row is None`, quando a pergunta foi
    apagada — não aparece aqui, porque não está em `ESCALATED` no banco.
    """
    rows = store.conn.execute(
        "SELECT id, text FROM questions WHERE status = 'ESCALATED' "
        "  AND focus_id = (SELECT id FROM active_focus) ORDER BY id"
    ).fetchall()
    current = [int(r["id"]) for r in rows]
    known = set(seen)
    fresh = [dict(r) for r in rows if int(r["id"]) not in known]
    return fresh, current


def dead_delta(store: Store, seen: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Tarefas mortas por tipo, agrupadas. GLOBAL — a query não tem `focus_id`.

    Agrupa por `kind`, não pelo texto do erro: o erro carrega traceback truncado e a
    chave é instável, enquanto `|kinds|` é limitado e estável. E exclui `notify_tick` —
    sem isso, um aviso que falha vira dead-letter, que vira gatilho de aviso.

    A escala declarada em `DELTAS` é `'global'` porque é isto que esta consulta é. A
    alternativa (escopar por foco) não existe: `tasks` não tem `focus_id` — ele mora
    dentro de `payload_json`, e nem toda tarefa o carrega.
    """
    rows = store.conn.execute(
        "SELECT kind, COUNT(*) AS n FROM tasks "
        " WHERE status = 'dead' AND kind <> 'notify_tick' GROUP BY kind"
    ).fetchall()
    current = {r["kind"]: int(r["n"]) for r in rows}
    fresh = {
        kind: n - seen.get(kind, 0)
        for kind, n in current.items()
        if n > seen.get(kind, 0)
    }
    return fresh, current


def discoveries_delta(store: Store, seen: list[int]) -> tuple[list[dict], list[int]]:
    """Descobertas PENDENTES do foco ativo. ESCOPADA POR FOCO, via `active_focus`.

    Só `pending`: é o único estado que espera você, e o único do qual se pode sair por
    decisão. Anunciar `queued`/`approved` seria avisar sobre o que já foi decidido.

    Projeta o `id` e mais nada de prosa — ver `compose`.
    """
    rows = store.conn.execute(
        "SELECT id FROM discoveries WHERE status = 'pending' "
        "  AND focus_id = (SELECT id FROM active_focus) ORDER BY id"
    ).fetchall()
    current = [int(r["id"]) for r in rows]
    known = set(seen)
    fresh = [{"id": i} for i in current if i not in known]
    return fresh, current


DELTAS: dict[str, tuple[str, object]] = {
    "escalated": ("focus", escalated_delta),
    "dead": ("global", dead_delta),
    "discoveries": ("focus", discoveries_delta),
}
"""Categoria → (escala da marca, função de delta).

O mapa é LITERAL e é a única fonte da iteração: `_notify_tick` e `reconcile` percorrem
ele, então uma categoria nova não pode ser esquecida em metade do ciclo. Antes disto,
dropar a chave `dead` do ramo entregue de `reconcile` matava 1 teste e dropar uma chave
NOVA matava 0 — o dict era literal nos dois ramos.
"""

EMPTY: dict[str, object] = {"escalated": [], "dead": {}, "discoveries": []}
"""O valor de partida de cada categoria, por TIPO. Lista de ids ou dict de contagens —
é o tipo que decide como `reconcile` recua numa entrega falhada."""


def scopes() -> tuple[str, ...]:
    return tuple(dict.fromkeys(scope for scope, _ in DELTAS.values()))


def reconcile(seen: dict, current: dict, *, delivered: bool) -> dict:
    """A marca nova, para UMA escala. Percorre o que veio, sem chave literal.

    **Só avança quando a entrega saiu.** Entre perder um aviso e repetir um, este
    sistema prefere repetir: a pergunta escalada é a coisa que ele existe para te
    contar, e um toast duplicado é irritação, não perda.

    Na falha, a marca RECUA para a interseção (listas) ou para o mínimo (contagens) —
    o `seen.get(kind, 0)` é obrigatório, senão o **primeiro** dead-letter de um tipo
    inédito combinado com entrega falhada levanta `KeyError`, e essa exceção mata o
    tick, que vira dead-letter, que é invisível ao gatilho por construção.
    """
    if delivered:
        return dict(current)
    out: dict = {}
    for key, value in current.items():
        old = seen.get(key, EMPTY.get(key))
        if isinstance(value, dict):
            base = old if isinstance(old, dict) else {}
            out[key] = {k: min(base.get(k, 0), v) for k, v in value.items()}
        else:
            base = set(old) if isinstance(old, (list, tuple, set)) else set()
            out[key] = [i for i in value if i in base]
    return out


def compose(fresh: dict) -> Notice | None:
    """Um aviso por tick, agrupado. `None` quando não há nada.

    O corpo não carrega conteúdo clínico: diz que existe algo para ler e onde. Um toast
    pode aparecer na tela de bloqueio.

    A parte de DESCOBERTAS carrega só a CONTAGEM, e o argumento é mais forte que o
    clínico: o texto de uma descoberta vem da web aberta, é escrito por um terceiro e é
    atacável. E ela vem POR ÚLTIMO na lista de partes de propósito — um dia com 12
    descobertas empurraria "N perguntas esperando você" para fora dos 240 caracteres do
    `clipped()`.

    O sufixo depende do que EXISTE no aviso: mandar `lithium questions` quando só há
    descobertas é mandar o usuário para a tela errada.
    """
    parts: list[str] = []
    where: list[str] = []

    escalated = fresh.get("escalated") or []
    if escalated:
        n = len(escalated)
        parts.append(f"{n} pergunta{'s' if n > 1 else ''} esperando você")
        where.append("`lithium questions`")

    dead = fresh.get("dead") or {}
    if dead:
        total = sum(dead.values())
        kinds = ", ".join(sorted(dead)[:MAX_NAMED])
        more = "…" if len(dead) > MAX_NAMED else ""
        parts.append(f"{total} tarefa(s) falharam ({kinds}{more})")
        where.append("`lithium status`")

    discoveries = fresh.get("discoveries") or []
    if discoveries:
        n = len(discoveries)
        parts.append(f"{n} descoberta(s) na web esperando sua decisão")
        where.append("`lithium discoveries`")

    if not parts:
        return None
    return Notice(title="lithium", body=" · ".join(parts) + " — " + " ".join(where))
