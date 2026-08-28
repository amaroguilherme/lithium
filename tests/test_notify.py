"""O aviso no SO — item 7.5.

Sem ele o daemon faz a pergunta para uma sala vazia: a escalada espera você lembrar de
rodar `lithium questions`.

Três propriedades carregam o peso, e as três vêm de medição:

1. **O texto nunca é executado.** A injeção foi confirmada *por execução*: a carga
   `" & (do shell script "echo owned > ...") & "` dentro de um `osascript -e`
   interpolado escreveu um arquivo no disco. Escapar quase resolve — e o "quase" falha
   em silêncio, com o AppleScript dando erro de sintaxe e o aviso nunca saindo.
2. **O daemon não trava.** `subprocess.run` síncrono no event loop atrasou o loop em
   609 ms; por thread, 1,9 ms.
3. **O aviso não derruba a tarefa.** Mesma lição do medidor de tokens: uma exceção no
   caminho de aviso converteria trabalho bem-sucedido em falha e retry.

Nenhum teste aqui invoca `osascript` ou `powershell`: as invariantes que importam são
propriedades **estáticas** da linha de comando montada, e por isso rodam igual numa CI
Windows e numa macOS.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from lithium.db import Store
from lithium.notify.send import (
    MacBackend,
    Notice,
    NullNotifier,
    WindowsBackend,
    backend_for,
    deliver,
)
from lithium.notify import watch

# As cargas que escreveram arquivo no disco na versão interpolada, mais as variantes
# que quebram um escape meio-feito.
PAYLOADS = [
    '" & (do shell script "echo owned > /tmp/PWNED") & "',
    '"; do shell script "touch /tmp/PWNED"; display notification "',
    'texto com \\ contrabarra e " aspas',
    "linha um\nlinha dois",
    '$(whoami)',
    '`id`',
    "'; Start-Process calc.exe; '",
    '</text><audio src="x"/><text>',
]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "n.db", embedding_dim=4)
    s.init_schema()
    yield s
    s.close()


def _question(store, text="pergunta", status="ESCALATED") -> int:
    cur = store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status) "
        "VALUES(1, ?, 'CONTEXT', ?) RETURNING id",
        (text, status),
    )
    return int(cur.fetchone()["id"])


def _dead(store, kind: str, n: int = 1) -> None:
    for _ in range(n):
        store.conn.execute(
            "INSERT INTO tasks(kind, payload_json, status, priority) "
            "VALUES(?, '{}', 'dead', 0.5)", (kind,),
        )


# ═══════════════════════════════ 1. o texto nunca vira comando


@pytest.mark.parametrize("payload", PAYLOADS)
def test_the_mac_command_never_puts_the_text_in_the_script(payload):
    """O corpo entra por argv, depois do `--`. A fonte do script é constante, então não
    existe carga que a altere — a defesa é estrutural, não um escape."""
    argv = MacBackend().command(Notice("lithium", payload).clipped())
    script = argv[argv.index("-e") + 1]

    assert payload not in script, "o texto foi parar dentro da fonte do AppleScript"
    assert argv[-1] == payload, "o texto tem que chegar VERBATIM, como argumento"
    assert "--" in argv, "sem o separador, um corpo começando com '-' vira opção"


def test_the_mac_script_source_is_constant_across_payloads():
    """A propriedade que torna a injeção impossível: a mesma fonte para toda carga."""
    scripts = {
        MacBackend().command(Notice("t", p).clipped())[2] for p in PAYLOADS
    }
    assert len(scripts) == 1


@pytest.mark.parametrize("payload", PAYLOADS)
def test_the_windows_command_never_puts_the_text_in_the_script_or_argv(payload):
    """No Windows a superfície é dupla — o shell (`-Command` recebe fonte) e o XML (o
    corpo do toast é XML). Por isso o texto vai por ENV, e o nó é montado por DOM."""
    backend = WindowsBackend()
    notice = Notice("lithium", payload).clipped()
    argv = backend.command(notice)
    script = argv[-1]

    assert payload not in script
    assert payload not in " ".join(argv), "o texto foi parar em argv, que é código"
    assert backend.env(notice)["LITHIUM_NOTIFY_BODY"] == notice.body
    # Presença de `CreateTextNode` em QUALQUER lugar não basta: a linha do título tem
    # uma, então trocar só a do corpo por `InnerText` passaria. A asserção precisa ser
    # sobre a LIGAÇÃO — toda ocorrência da variável do corpo está dentro de um
    # `CreateTextNode(...)`, que é o que impede `</text><audio src=` de injetar XML.
    import re

    uses = re.findall(r"[^;]*\$env:LITHIUM_NOTIFY_BODY[^;]*", script)
    assert uses, "o corpo não chega ao script"
    for use in uses:
        assert "CreateTextNode(" in use, (
            f"o corpo entra no XML sem passar por CreateTextNode: {use.strip()!r}"
        )


def test_the_body_is_clipped_before_it_leaves():
    """O toast pode aparecer na tela de bloqueio, e o domínio é psiquiatria."""
    from lithium.notify.send import MAX_BODY

    clipped = Notice("t", "x" * 5000).clipped()
    assert len(clipped.body) == MAX_BODY


# ══════════════════════════ 2. o daemon não trava, e a falha não propaga


async def test_a_blocking_backend_does_not_stall_the_event_loop():
    """A medição que decidiu o contrato.

    O protocolo é **síncrono** de propósito: um `async def send` com `subprocess.run`
    dentro trava o loop e o `wait_for` nunca dispara — `wait_for` só cancela corrotina
    que cede o controle. Medido na variante errada: 3,04 s de loop parado com a entrega
    voltando como bem-sucedida.

    O teste mede a fome de uma corrotina concorrente, não o tempo de retorno: só o
    heartbeat não mente, porque o retorno rápido poderia ser acidente.
    """
    class Blocking:
        def send(self, notice: Notice) -> bool:
            time.sleep(0.4)
            return True

    gaps: list[float] = []

    async def heartbeat() -> None:
        last = time.perf_counter()
        for _ in range(40):
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    assert await deliver(Blocking(), Notice("t", "b")) is True
    await beat

    assert max(gaps) < 0.15, f"o loop ficou parado {max(gaps):.3f}s"


async def test_a_backend_that_raises_never_propagates():
    """Uma exceção aqui converteria trabalho bem-sucedido em falha e retry."""
    class Broken:
        def send(self, notice: Notice) -> bool:
            raise RuntimeError("osascript sumiu")

    assert await deliver(Broken(), Notice("t", "b")) is False


async def test_a_hung_backend_gives_up():
    """E o teste LIBERA a thread no fim, de propósito.

    Uma thread bloqueada sobrevive ao `wait_for` — ele cancela o await, não o trabalho —
    e o interpretador a espera na saída. Foi medido: 8,08 s de saída de processo contra
    0,47 s. Em produção quem fecha isso é o `timeout=` do `subprocess.run`; aqui é este
    `Event`, e deixá-lo de fora custaria 30 s na suíte a cada execução.
    """
    import threading

    from lithium.notify import send as send_mod

    released = threading.Event()

    class Hung:
        def send(self, notice: Notice) -> bool:
            released.wait(30)
            return True

    original = send_mod.TIMEOUT_S
    send_mod.TIMEOUT_S = 0.05
    try:
        started = time.perf_counter()
        assert await deliver(Hung(), Notice("t", "b")) is False
        assert time.perf_counter() - started < 5.0
    finally:
        send_mod.TIMEOUT_S = original
        released.set()


def test_a_missing_binary_returns_false_instead_of_raising(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert WindowsBackend().send(Notice("t", "b")) is False


def test_the_null_notifier_is_an_object_not_none():
    """`None` no contexto reintroduz `AttributeError` dentro do handler — e um handler
    que morre por causa do aviso é o que este módulo existe para evitar."""
    assert NullNotifier().send(Notice("t", "b")) is False


@pytest.mark.parametrize("platform,expected", [
    ("darwin", "MacBackend"), ("win32", "WindowsBackend"), ("linux", "NullNotifier"),
])
def test_the_backend_of_the_other_os_is_constructible_here(platform, expected):
    """Portabilidade testável sem executar nada: se o backend do outro SO não puder ser
    construído nesta máquina, ele nunca é exercitado por CI nenhuma."""
    assert type(backend_for(platform)).__name__ == expected


# ═════════════════════════════ 3. o gatilho: delta, agrupado, sem tempestade


def test_only_new_escalations_are_announced(store):
    first = _question(store, "primeira")
    fresh, current = watch.escalated_delta(store, seen=[])
    assert [q["id"] for q in fresh] == [first]

    fresh, _ = watch.escalated_delta(store, seen=current)
    assert fresh == [], "reanunciou o que já tinha sido anunciado"

    second = _question(store, "segunda")
    fresh, _ = watch.escalated_delta(store, seen=current)
    assert [q["id"] for q in fresh] == [second]


def test_the_watermark_is_a_set_of_ids_not_a_timestamp(store):
    """As duas marcas óbvias estão quebradas.

    `max(escalated_at)` não serve: `cli.py` promove N linhas num único UPDATE com
    `strftime('now')`, então os timestamps são idênticos **por construção** — com `>`
    perde-se aviso para sempre, com `>=` reavisa-se para sempre. Contagem também não:
    não distingue "duas novas" de "uma nova e uma resolvida".
    """
    a, b = _question(store, "a"), _question(store, "b")
    store.conn.execute(
        "UPDATE questions SET escalated_at = '2024-01-01T00:00:00.000Z'"
    )
    store.conn.execute("UPDATE questions SET status = 'ANSWERED_HUMAN' WHERE id = ?", (a,))
    novo = _question(store, "c")

    fresh, _ = watch.escalated_delta(store, seen=[a, b])
    assert [q["id"] for q in fresh] == [novo]


def test_a_deterministic_failure_produces_one_notice_not_hundreds(store):
    """A tempestade medida: 380 avisos de uma única falha determinística.

    Estado derivado resolve durabilidade e um call site; quem resolve tempestade é o
    AGRUPAMENTO — e agrupando, a primeira execução deixa de ser caso especial.
    """
    _dead(store, "extract_source", n=380)
    fresh, _ = watch.dead_delta(store, seen={})
    notice = watch.compose({"dead": fresh})

    assert notice is not None
    assert "380" in notice.body
    assert len(notice.body.splitlines()) == 1


def test_the_notice_never_carries_clinical_content(store):
    _question(store, "o paciente tomou 1200 mg de lítio e teve tremor")
    fresh, _ = watch.escalated_delta(store, seen=[])
    notice = watch.compose({"escalated": fresh})

    assert "lítio" not in notice.body and "1200" not in notice.body
    assert "lithium questions" in notice.body


def test_the_dead_delta_excludes_the_notifier_itself(store):
    """O laço de amplificação: um `notify_tick` que morre vira dead-letter, dead-letter
    é gatilho de aviso, aviso roda `notify_tick`."""
    _dead(store, "notify_tick", n=5)
    fresh, current = watch.dead_delta(store, seen={})

    assert "notify_tick" not in fresh
    assert "notify_tick" not in current


def test_nothing_to_say_produces_no_notice(store):
    assert watch.compose({}) is None


# ══════════════════════════ 4. a marca só avança quando a entrega saiu


def test_the_mark_does_not_advance_on_a_failed_delivery():
    """Entre perder um aviso e repetir um, este sistema prefere repetir: a pergunta
    escalada é a coisa que ele existe para te contar."""
    seen = {"escalated": [1], "dead": {"fetch_source": 2}}
    mark = watch.reconcile(
        seen, {"escalated": [1, 2], "dead": {"fetch_source": 5}}, delivered=False
    )

    assert mark["escalated"] == [1], "a pergunta 2 seria perdida para sempre"
    assert mark["dead"] == {"fetch_source": 2}


def test_the_mark_advances_on_a_successful_delivery():
    seen = {"escalated": [1], "dead": {}}
    mark = watch.reconcile(
        seen, {"escalated": [1, 2], "dead": {"fetch_source": 5}}, delivered=True
    )

    assert mark == {"escalated": [1, 2], "dead": {"fetch_source": 5}}


def test_a_brand_new_kind_with_a_failed_delivery_does_not_raise():
    """O `KeyError` que matava o canal inteiro.

    Primeiro dead-letter de um tipo inédito + entrega falhada → a fórmula sem
    `seen.get(kind, 0)` levanta, a exceção mata o tick, o tick vira dead-letter — e o
    delta de dead-letter exclui `notify_tick` por construção, então o cadáver é
    invisível ao próprio gatilho. O canal morre em silêncio, para sempre.
    """
    mark = watch.reconcile({}, {"dead": {"extract_source": 1}}, delivered=False)
    assert mark["dead"] == {"extract_source": 0}


def test_a_corrupt_mark_restarts_instead_of_crashing(store):
    store.conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, 'isto não é json')", (watch._mark_key(store),)
    )
    assert watch._mark(store) == {}


# ═══════════════════════════════════ 5. A FIAÇÃO


def test_the_handler_is_registered():
    from lithium.worker.handlers import HANDLERS

    assert "notify_tick" in HANDLERS


def test_the_scheduler_runs_it_and_respects_the_hour_bucket():
    """`dedup_key` do scheduler usa um balde horário, então qualquer Job com intervalo
    menor que 3600 s é silenciosamente estrangulado para 1×/hora."""
    from lithium.worker.scheduler import DEFAULT_JOBS

    jobs = [j for j in DEFAULT_JOBS if j.task_kind == "notify_tick"]
    assert jobs, "sem Job, o aviso nunca sai sozinho"
    assert jobs[0].interval_s >= 3600


async def test_the_handler_delivers_and_advances_the_mark(store, tmp_path, monkeypatch):
    """Ponta a ponta pelo handler REAL. Reverter o corpo dele para `pass` tem que ficar
    vermelho — é a classe que escapou seis vezes neste projeto."""
    from lithium.config import Config
    from lithium.worker import handlers

    sent: list[Notice] = []

    class Spy:
        def send(self, notice: Notice) -> bool:
            sent.append(notice)
            return True

    monkeypatch.setattr(handlers, "backend_for", lambda *a, **k: Spy())
    qid = _question(store, "pergunta escalada")

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.config = Config(data_dir=tmp_path)

    await handlers.HANDLERS["notify_tick"]({}, Ctx())

    assert sent, "nenhum aviso saiu"
    mark = json.loads(store.conn.execute(
        "SELECT value FROM meta WHERE key = ?", (watch._mark_key(store),)
    ).fetchone()["value"])
    assert mark["escalated"] == [qid]


async def test_the_handler_survives_a_broken_database(store, tmp_path, monkeypatch):
    """O handler é estruturalmente incapaz de morrer, e isso não é zelo: o delta de
    dead-letter exclui `notify_tick`, e essa exclusão só é segura se ele nunca virar
    dead-letter."""
    from lithium.config import Config
    from lithium.worker import handlers

    def boom(*a, **k):
        raise RuntimeError("banco sumiu")

    monkeypatch.setattr(watch, "escalated_delta", boom)

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.config = Config(data_dir=tmp_path)

    await handlers.HANDLERS["notify_tick"]({}, Ctx())   # não pode levantar


# ─────────────────────────────────────────────────────────────── helpers





def test_the_subprocess_always_carries_a_timeout(monkeypatch):
    """Sem `timeout=`, um filho pendurado deixa a thread viva e o interpretador a espera
    na saída — medido, 8,08 s de saída de processo contra 0,47 s. O diálogo de permissão
    de notificação do macOS na primeira execução é exatamente esse cenário."""
    from lithium.notify import send as send_mod

    seen: dict = {}

    def spy(argv, **kw):
        seen.update(kw)
        raise OSError("não executa de verdade")

    monkeypatch.setattr(send_mod.subprocess, "run", spy)
    MacBackend().send(Notice("t", "b"))

    assert seen.get("timeout"), "o subprocess foi disparado sem teto"
    assert seen["timeout"] <= 30


def test_a_dead_letter_already_announced_is_not_announced_again(store):
    """O delta precisa ser delta. Sem isso, cada tick reavisa o cemitério inteiro —
    e o cemitério não encolhe."""
    _dead(store, "fetch_source", n=4)
    fresh, current = watch.dead_delta(store, seen={})
    assert fresh == {"fetch_source": 4}

    fresh, _ = watch.dead_delta(store, seen=current)
    assert fresh == {}, "reavisou o que já tinha sido anunciado"

    _dead(store, "fetch_source", n=2)
    fresh, _ = watch.dead_delta(store, seen=current)
    assert fresh == {"fetch_source": 2}, "só o incremento, não o total"


# ═════════════════ 6. ESCALA: cada categoria declara o que a query dela é
#
# Este bloco é PRÉ-REQUISITO da Fase C, não bônus. EXECUTADO antes dele: com dois focos
# e 3 tarefas mortas, o tick do foco A anuncia "3 tarefa(s) falharam", o segundo tick do
# foco A anuncia nada (correto), e ao trocar para o foco B o MESMO cemitério é anunciado
# inteiro de novo. Reverter o escopo por foco de `_mark_key` matava 0 de 805 testes E
# eliminava o reaviso: o escopo da Fase B era simultaneamente não-testado e a CAUSA do
# defeito, porque `dead_delta` não tem `focus_id` na query.


def _second_focus(store) -> int:
    scale_id = int(store.conn.execute(
        "SELECT id FROM evidence_scales LIMIT 1").fetchone()["id"])
    store.conn.execute(
        "INSERT INTO focuses(id, slug, target, scale_id) VALUES(2, 'outro', 'x', ?)",
        (scale_id,))
    return 2


def _use_focus(store, focus_id: int) -> None:
    store.conn.execute("UPDATE meta SET value = ? WHERE key = 'active_focus'",
                       (str(focus_id),))


def test_every_notify_category_declares_the_scope_its_query_actually_has():
    """A DECLARAÇÃO e a CONSULTA têm de andar juntas — foi a divergência entre as duas
    que produziu a tempestade.

    MUTAÇÃO: declarar `discoveries` como 'focus' e escrever a query sem `active_focus`
    (o defeito exato que `dead` tinha). A equivalência abaixo cai.
    """
    import inspect

    for name, (scope, fn) in watch.DELTAS.items():
        source = inspect.getsource(fn)
        scoped_in_sql = "active_focus" in source
        assert scoped_in_sql == (scope == "focus"), (
            f"{name}: declara escala {scope!r} e a query "
            f"{'menciona' if scoped_in_sql else 'NÃO menciona'} active_focus"
        )


def test_a_dead_letter_is_not_re_announced_when_you_switch_focus(store):
    """A tempestade, reproduzida e fechada."""
    _second_focus(store)
    _dead(store, "harvest_query", n=3)

    seen = watch._mark(store, "global")
    fresh, current = watch.dead_delta(store, seen.get("dead", {}))
    assert sum(fresh.values()) == 3
    watch._save(store, watch.reconcile(seen, {"dead": current}, delivered=True),
                "global")

    _use_focus(store, 2)
    seen = watch._mark(store, "global")
    fresh, _ = watch.dead_delta(store, seen.get("dead", {}))
    assert fresh == {}, "trocar de foco reanunciou o cemitério inteiro"


def test_an_escalated_question_is_announced_again_in_the_focus_that_owns_it(store):
    """O outro lado: o que É por foco continua por foco. Sem isto, mover `escalated`
    para a marca global faria a pergunta do foco B nunca ser anunciada depois de um tick
    no foco A."""
    _second_focus(store)
    qid = _question(store, "pergunta do foco 1")
    fresh, current = watch.escalated_delta(store, [])
    assert [q["id"] for q in fresh] == [qid]
    watch._save(store, {"escalated": current}, "focus")

    _use_focus(store, 2)
    store.conn.execute(
        "INSERT INTO questions(focus_id, text, kind, status) "
        "VALUES(2, 'pergunta do foco 2', 'CONTEXT', 'ESCALATED')")
    seen = watch._mark(store, "focus")
    fresh, _ = watch.escalated_delta(store, seen.get("escalated", []))
    assert len(fresh) == 1 and fresh[0]["text"] == "pergunta do foco 2"


def test_reconcile_carries_every_declared_category_in_both_branches():
    """MUTAÇÃO: dropar uma categoria de um dos ramos.

    MEDIDO: dropar a chave `dead` do ramo entregue matava 1 teste; dropar uma chave NOVA
    matava 0, porque o dict era LITERAL nos dois ramos. Agora `reconcile` percorre o que
    recebe, e este teste percorre `DELTAS`.
    """
    current = {key: ({"x": 1} if key == "dead" else [1, 2])
               for key in watch.DELTAS}
    for delivered in (True, False):
        mark = watch.reconcile({}, current, delivered=delivered)
        assert set(mark) == set(watch.DELTAS), (delivered, mark)


def test_the_global_mark_is_seeded_from_the_per_focus_ones(store):
    """MUTAÇÃO: não semear.

    `dead_delta` passa a ler `notify:seen:global`, que não existe → `seen = {}` →
    `fresh = current` → o primeiro tick pós-upgrade anuncia um cemitério inteiro que o
    usuário já viu e já ignorou. O docstring do módulo registra que 380 avisos de uma vez
    derrubaram a primeira versão desta frente.
    """
    _dead(store, "harvest_query", n=3)
    _dead(store, "relens_claim", n=40)
    store.conn.execute(
        "INSERT INTO meta(key, value) VALUES('notify:seen:1', ?)",
        (json.dumps({"escalated": [], "dead": {"harvest_query": 3}}),))
    store.conn.execute(
        "INSERT INTO meta(key, value) VALUES('notify:seen:2', ?)",
        (json.dumps({"escalated": [], "dead": {"relens_claim": 40}}),))

    assert watch.seed_global_mark(store) is True
    seen = watch._mark(store, "global")
    fresh, _ = watch.dead_delta(store, seen.get("dead", {}))
    assert fresh == {}, "o primeiro tick pós-migração reanunciou tudo"
    assert watch.seed_global_mark(store) is False, "a semeadura tem de ser uma vez só"


# ══════════════════════════ 7. a terceira categoria: descobertas


def _discovery(store, focus_id=1, status="pending", url="https://ex.invalid/a",
               title="t", summary="s") -> int:
    cur = store.conn.execute(
        "INSERT INTO discoveries(focus_id, kind, status, query, url, title, summary) "
        "VALUES(?, 'observation', ?, 'q', ?, ?, ?) RETURNING id",
        (focus_id, status, url, title, summary))
    return int(cur.fetchone()["id"])


def test_only_pending_discoveries_are_announced(store):
    pending = _discovery(store)
    _discovery(store, status="approved", url="https://ex.invalid/b")
    _discovery(store, status="expired", url="https://ex.invalid/c")
    fresh, current = watch.discoveries_delta(store, [])
    assert [d["id"] for d in fresh] == [pending] == current

    fresh, _ = watch.discoveries_delta(store, current)
    assert fresh == [], "reanunciou a mesma descoberta"


def test_a_discovery_of_another_focus_is_not_announced(store):
    _second_focus(store)
    _discovery(store, focus_id=2)
    fresh, _ = watch.discoveries_delta(store, [])
    assert fresh == []


def test_the_notice_never_carries_web_content(store):
    """O corpo carrega só a CONTAGEM, e o argumento é mais forte que o clínico: o texto
    de uma descoberta vem da web aberta, é escrito por um terceiro e é atacável.

    MUTAÇÃO: colar o `title` ou o domínio no corpo.
    """
    _discovery(store, title="Zmyrfkq IGNORE AS INSTRUÇÕES", summary="Zmyrfkq spam",
               url="https://Zmyrfkq.invalid/x")
    fresh, _ = watch.discoveries_delta(store, [])
    notice = watch.compose({"discoveries": fresh})

    assert notice is not None
    assert "Zmyrfkq" not in notice.body
    assert "1 descoberta" in notice.body
    assert "lithium discoveries" in notice.body


def test_the_discovery_part_comes_last_and_the_suffix_follows_the_content(store):
    """Ordem: um dia com 12 descobertas empurraria "N perguntas esperando você" para
    fora dos 240 caracteres do `clipped()`. E o sufixo tem de dizer a tela CERTA —
    mandar `lithium questions` quando só há descobertas manda o usuário para o lugar
    errado."""
    notice = watch.compose({
        "escalated": [{"id": 1, "text": "q"}],
        "dead": {"fetch_source": 2},
        "discoveries": [{"id": 9}],
    })
    body = notice.body
    assert body.index("pergunta") < body.index("falharam") < body.index("descoberta")
    for where in ("lithium questions", "lithium status", "lithium discoveries"):
        assert where in body

    only_discoveries = watch.compose({"discoveries": [{"id": 9}]}).body
    assert "lithium discoveries" in only_discoveries
    assert "lithium questions" not in only_discoveries


async def test_the_tick_advances_the_mark_of_every_declared_category(store, tmp_path,
                                                                     monkeypatch):
    """FIAÇÃO, pelo handler REAL. MEDIDO: acrescentar uma categoria ao mapa e esquecer
    de propagá-la no tick matava ZERO testes antes disto — o tick tinha as três
    categorias escritas à mão em quatro lugares."""
    from lithium.config import Config
    from lithium.worker import handlers

    monkeypatch.setattr(handlers, "backend_for",
                        lambda *a, **k: type("S", (), {"send": lambda s, n: True})())
    _question(store, "escalada")
    _dead(store, "fetch_source", n=2)
    _discovery(store)

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.config = Config(data_dir=tmp_path)

    await handlers.HANDLERS["notify_tick"]({}, Ctx())

    focus_mark = watch._mark(store, "focus")
    global_mark = watch._mark(store, "global")
    assert set(focus_mark) == {"escalated", "discoveries"}, focus_mark
    assert set(global_mark) == {"dead"}, global_mark
    assert focus_mark["discoveries"], "a categoria nova não avançou"

    # Segundo tick: nada novo, nenhum aviso.
    from lithium.notify.send import Notice

    sent: list[Notice] = []
    monkeypatch.setattr(
        handlers, "backend_for",
        lambda *a, **k: type("S", (), {
            "send": lambda s, n: sent.append(n) or True})())
    await handlers.HANDLERS["notify_tick"]({}, Ctx())
    assert sent == [], "o segundo tick reavisou o que já tinha sido anunciado"


def test_the_delta_of_discoveries_projects_only_the_id(store):
    """A trava ESTRUTURAL do aviso: se o delta não CARREGA prosa, `compose` não tem o
    que colar. Sem isto, a proteção depende de `compose` se lembrar de não usar campos
    que estão bem ali.

    MUTAÇÃO: projetar `title` no SELECT de `discoveries_delta`.
    """
    _discovery(store, title="Zmyrfkq", summary="Zmyrfkq")
    fresh, _ = watch.discoveries_delta(store, [])
    assert [set(d) for d in fresh] == [{"id"}], fresh


async def test_the_tick_seeds_the_global_mark_before_reading_it(store, tmp_path,
                                                                monkeypatch):
    """FIAÇÃO da semeadura. MUTAÇÃO: tirar `seed_global_mark` do tick.

    Chamar a função direto num teste prova que ela FUNCIONA e não que ela é CHAMADA — e
    é ser chamada, uma vez, antes da primeira leitura, que evita a tempestade do
    upgrade. Sem esta trava a remoção matava 0 de 962.
    """
    from lithium.config import Config
    from lithium.notify.send import Notice
    from lithium.worker import handlers

    _dead(store, "harvest_query", n=3)
    store.conn.execute(
        "INSERT INTO meta(key, value) VALUES('notify:seen:1', ?)",
        (json.dumps({"escalated": [], "dead": {"harvest_query": 3}}),))

    sent: list[Notice] = []
    monkeypatch.setattr(
        handlers, "backend_for",
        lambda *a, **k: type("S", (), {
            "send": lambda s, n: sent.append(n) or True})())

    class Ctx:
        def __init__(self) -> None:
            self.store = store
            self.config = Config(data_dir=tmp_path)

    await handlers.HANDLERS["notify_tick"]({}, Ctx())
    assert sent == [], (
        "o primeiro tick pós-upgrade reanunciou um cemitério que o usuário já viu"
    )
