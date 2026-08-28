"""Modo pesquisa: ligado (24/7) ou desligado (sob demanda).

**Ligado** — o comportamento padrão. O scheduler dispara harvest, plan e explore em
cadência; o daemon trabalha sozinho, consumindo CPU e GPU continuamente.

**Desligado** — o daemon segue no ar e responde, mas não pesquisa por conta própria.
Concretamente: o scheduler para de enfileirar, e os workers **só reivindicam tarefas
`on_demand`** — as que você pediu explicitamente (`lithium ask`, `harvest`, `explore`,
ou uma conversa). Trabalho agendado que já estava na fila fica parado até você religar.

Essa distinção por origem é o que faz o modo desligado ser útil em vez de inerte. Um
`stop` global também congelaria o que você acabou de pedir; e deixar tudo rodando
significaria que "desligado" não devolve a máquina, que é o ponto.

**A memória continua ativa nos dois modos.** Aprender com o que você diz é conversa,
não pesquisa — desligar a pesquisa não deve fazer o sistema esquecer o que você contou.

O estado vive na tabela `meta`, não em memória: reiniciar o daemon não pode religar a
pesquisa sozinho.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from lithium.db import Store

log = logging.getLogger(__name__)

META_KEY = "research_mode"


class ResearchMode(StrEnum):
    ON = "on"
    OFF = "off"

    @property
    def label(self) -> str:
        return "ligado (pesquisa 24/7)" if self is ResearchMode.ON else "desligado (sob demanda)"


def get_mode(store: Store) -> ResearchMode:
    """Padrão é ligado — é o comportamento que o projeto assume."""
    row = store.conn.execute("SELECT value FROM meta WHERE key = ?", (META_KEY,)).fetchone()
    if row is None:
        return ResearchMode.ON
    try:
        return ResearchMode(row["value"])
    except ValueError:
        log.warning("modo inválido no banco (%r), assumindo ligado", row["value"])
        return ResearchMode.ON


def set_mode(store: Store, mode: ResearchMode) -> ResearchMode:
    store.conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (META_KEY, mode.value),
    )
    log.info("modo pesquisa: %s", mode.label)
    return mode


def is_researching(store: Store) -> bool:
    return get_mode(store) is ResearchMode.ON


def backlog(store: Store) -> int:
    """Tarefas agendadas represadas — o que voltará a rodar ao religar."""
    return int(
        store.conn.execute(
            "SELECT COUNT(*) AS n FROM tasks "
            " WHERE status = 'pending' AND origin = 'scheduled'"
        ).fetchone()["n"]
    )
