"""Entrega de aviso no SO — e as três coisas que ela não pode fazer.

**Não pode executar o texto.** O corpo do aviso carrega texto de pergunta escrito por um
LLM, e `osascript -e` recebe AppleScript. Interpolar ali é execução remota de código, e
isto foi confirmado por execução, não por análise: a carga

    " & (do shell script "echo owned > /tmp/PWNED") & "

dentro de `f'display notification "{body}"'` escreveu o arquivo. Escapar aspas *quase*
resolve — e o "quase" é pior que nada: esquecer a contrabarra deixa o AppleScript com
erro de sintaxe e o aviso **nunca é entregue, sem ninguém notar**; um `\\x00` no texto faz
o próprio `subprocess` levantar. A defesa tem de ser estrutural, então o texto vai por
**argv** (`osascript -e SCRIPT -- título corpo`), onde nunca é parseado como script.

No Windows a superfície é dupla — o shell (`-Command` recebe fonte de script) **e** o XML
(o corpo do toast é XML, e um `<audio src=` fechando a tag injeta sem shell nenhum). Por
isso o texto vai por **variável de ambiente** e o nó é montado por DOM, nunca por
concatenação de string.

**Não pode travar o daemon.** Medido nesta máquina: `osascript` leva 221 ms na primeira
chamada e ~120 ms depois. Um `subprocess.run` síncrono no event loop atrasou o loop em
**609 ms**; por `to_thread`, 1,9 ms. O `timeout=` no subprocess é o que impede a thread
órfã — sem ele, um filho pendurado custa 8 s na saída do processo, porque a thread do
executor padrão não é cancelável e o interpretador a espera.

**Não pode derrubar a tarefa que notifica.** Mesma lição do medidor de tokens da Fase 0:
uma exceção no caminho de aviso converteria trabalho bem-sucedido em falha e retry. Tudo
aqui devolve `False`; nada levanta.

E uma limitação que muda o desenho: `osascript` devolve código 0 **apareça o toast ou
não**. Não existe recibo de entrega. Então o aviso é redundância — `lithium questions`
continua sendo a fonte de verdade.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

TIMEOUT_S = 8.0
"""Teto do subprocess. O diálogo de permissão de notificação do macOS na primeira
execução pendura `osascript` até alguém clicar — sem teto, a thread fica presa."""

MAX_BODY = 240
"""O toast pode aparecer na tela de bloqueio, e o domínio é psiquiatria. Corpo curto e
sem conteúdo clínico: o aviso diz que existe algo para ler, não o quê."""

_MAC_SCRIPT = (
    'on run argv\n'
    '  display notification (item 2 of argv) with title (item 1 of argv)\n'
    'end run'
)
"""Fonte CONSTANTE. O texto entra por `argv`, nunca por interpolação — é o que torna a
injeção impossível por construção em vez de por escape."""

_WIN_SCRIPT = (
    "$ErrorActionPreference='Stop';"
    "[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,"
    "ContentType=WindowsRuntime]>$null;"
    "$x=New-Object Windows.Data.Xml.Dom.XmlDocument;"
    "$x.LoadXml('<toast><visual><binding template=\"ToastText02\">"
    "<text id=\"1\"></text><text id=\"2\"></text></binding></visual></toast>');"
    "$n=$x.GetElementsByTagName('text');"
    "$n.Item(0).AppendChild($x.CreateTextNode($env:LITHIUM_NOTIFY_TITLE))>$null;"
    "$n.Item(1).AppendChild($x.CreateTextNode($env:LITHIUM_NOTIFY_BODY))>$null;"
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('lithium')"
    ".Show([Windows.UI.Notifications.ToastNotification]::new($x))"
)
"""Também constante, e o texto entra por ENV — não por argv, porque `-Command` recebe
fonte de script. `CreateTextNode` fecha a segunda superfície: montar o XML por
concatenação deixaria um `</text><audio src=...` injetar sem passar pelo shell."""


@dataclass(frozen=True, slots=True)
class Notice:
    title: str
    body: str

    def clipped(self) -> Notice:
        return Notice(self.title[:80], self.body[:MAX_BODY])


class Backend(Protocol):
    """**Síncrono de propósito.**

    Um protocolo `async` empurra o implementador para o antipadrão: `wait_for` só cancela
    corrotina que cede o controle, então um `subprocess.run` bloqueante dentro de um
    `async def` trava o loop e o timeout **nunca dispara** — medido, 3,04 s de loop parado
    com a entrega voltando como bem-sucedida. Sendo síncrono, o chamador é obrigado a
    confinar em thread, que é o precedente do medidor de tokens da Fase 0.
    """

    def send(self, notice: Notice) -> bool: ...


class NullNotifier:
    """Objeto nulo, não `None`.

    `None` no contexto reintroduz `AttributeError` dentro do handler — e um handler que
    morre por causa do aviso é exatamente o que este módulo existe para evitar.
    """

    def send(self, notice: Notice) -> bool:
        return False


class MacBackend:
    def command(self, notice: Notice) -> list[str]:
        """A linha de comando, exposta para poder ser INSPECIONADA por teste.

        Sem isto o teste de injeção só consegue afirmar sobre a constante do script — e
        aí ele não pode falhar, porque trocar o corpo de `send()` por uma interpolação
        não muda a constante. A propriedade que importa é sobre o que o backend
        **realmente executa**.
        """
        return ["osascript", "-e", _MAC_SCRIPT, "--", notice.title, notice.body]

    def send(self, notice: Notice) -> bool:
        return _run(self.command(notice))


class WindowsBackend:
    def command(self, notice: Notice) -> list[str]:
        exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return [exe, "-NoProfile", "-NonInteractive", "-Command", _WIN_SCRIPT]

    def env(self, notice: Notice) -> dict[str, str]:
        """O texto vai por ENV, nunca por argv: `-Command` recebe **fonte de script**,
        então um corpo em argv estaria em posição de código."""
        return {**os.environ,
                "LITHIUM_NOTIFY_TITLE": notice.title,
                "LITHIUM_NOTIFY_BODY": notice.body}

    def send(self, notice: Notice) -> bool:
        if shutil.which("pwsh") is None and shutil.which("powershell") is None:
            log.debug("nenhum PowerShell no PATH; aviso não entregue")
            return False
        return _run(self.command(notice), env=self.env(notice))


class NtfyBackend:
    """Canal externo, para quando você não está na máquina.

    Redundante por projeto, não decorativo: o toast local não tem recibo de entrega, e
    sob `launchd` a atribuição da notificação muda sem o código saber.
    """

    def __init__(self, topic_url: str) -> None:
        self.topic_url = topic_url

    def send(self, notice: Notice) -> bool:
        import httpx

        try:
            r = httpx.post(self.topic_url, content=notice.body.encode("utf-8"),
                           headers={"Title": notice.title}, timeout=TIMEOUT_S)
            return r.status_code < 400
        except Exception as exc:  # noqa: BLE001
            log.debug("ntfy falhou: %s", exc)
            return False


def _run(argv: list[str], *, env: dict[str, str] | None = None) -> bool:
    """Executa e engole tudo. `timeout` mata o filho — sem ele a thread vaza."""
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=TIMEOUT_S, env=env,
                              check=False)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # FileNotFoundError: binário ausente. ValueError: NUL no texto. TimeoutExpired:
        # diálogo de permissão pendurado — e o `subprocess.run` já matou o filho.
        log.debug("aviso não entregue (%s): %s", type(exc).__name__, exc)
        return False
    if proc.returncode != 0:
        log.debug("aviso não entregue (rc=%s): %s", proc.returncode,
                  proc.stderr[:200].decode("utf-8", "replace"))
        return False
    return True


def backend_for(platform: str, *, ntfy_url: str | None = None):
    """Escolhe o backend. `platform` é parâmetro para o backend do outro SO ser
    construível — e portanto testável — em qualquer CI."""
    if ntfy_url:
        return NtfyBackend(ntfy_url)
    if platform == "darwin":
        return MacBackend()
    if platform == "win32":
        return WindowsBackend()
    return NullNotifier()


async def deliver(backend, notice: Notice) -> bool:
    """Entrega fora do event loop, com teto. Nunca levanta.

    O `to_thread` é o que confina um backend bloqueante; o `wait_for` é a segunda camada,
    para o caso de o `timeout` do subprocess não cobrir (um backend que faça I/O de rede
    sem timeout, por exemplo).
    """
    import asyncio

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(backend.send, notice.clipped()), TIMEOUT_S + 2.0
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("entrega de aviso falhou: %s", exc)
        return False
