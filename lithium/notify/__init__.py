"""Aviso no sistema operacional. Sem ele, o daemon faz a pergunta para uma sala vazia."""

from lithium.notify.send import (
    MacBackend,
    Notice,
    NtfyBackend,
    NullNotifier,
    WindowsBackend,
    backend_for,
    deliver,
)

__all__ = [
    "MacBackend", "Notice", "NtfyBackend", "NullNotifier", "WindowsBackend",
    "backend_for", "deliver",
]
