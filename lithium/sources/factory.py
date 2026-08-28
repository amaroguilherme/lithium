"""Monta `Context.sources` a partir do registro, não de um dict literal.

Era o último elo da dívida de fiação do item 9. Enquanto a fonte fosse escolhida por
edição de `daemon.py`, "aprovar uma fonte no registro" não podia significar nada — a
decisão real morava no import, fora de qualquer política, e foi por ali que a medição do
item 9 fez o campo de contraindicação de uma bula virar claim com peso 0,408.
"""

from __future__ import annotations

import logging
from typing import Any

from lithium.sources.http import HttpSource, resolve_credential
from lithium.sources.pubmed import PubMedSource

log = logging.getLogger(__name__)


class MissingCredential(RuntimeError):
    """A fonte exige credencial e a referência não resolve."""


def build_sources(store: Any, cfg: Any, stack: Any) -> dict[str, Any]:
    """Instancia uma fonte por linha aprovada do registro.

    Fonte cuja credencial não resolve fica **de fora do dict**, com aviso nomeando a
    referência. Não tenta sem auth: um 401 dentro do worker vira retry, backoff e
    dead-letter, e o diagnóstico aponta para a fila em vez da configuração.
    """
    sources: dict[str, Any] = {}
    for row in store.active_sources():
        slug = row["slug"]
        ref = row["credential_ref"]
        key = resolve_credential(ref, cfg)
        if ref and key is None and row["adapter"] != "pubmed":
            # O PubMed é a exceção e ela é documentada: a api_key da NCBI é OPCIONAL —
            # sem ela a taxa cai de 10 para 3 req/s, e é assim que o projeto roda hoje.
            log.warning(
                "fonte %r ignorada: `credential_ref = %r` não resolve. Acrescente a "
                "chave em config.local.toml (gitignored) ou exporte a variável.",
                slug, ref,
            )
            continue
        try:
            if row["adapter"] == "pubmed":
                src: Any = PubMedSource(api_key=key, rate_per_s=float(row["rate_per_s"]))
            else:
                src = HttpSource(row, api_key=key,
                                 contact=getattr(cfg.recon, "contact", None))
        except Exception as exc:  # spec inválida não pode derrubar o daemon
            log.warning("fonte %r não pôde ser construída: %s", slug, exc)
            continue
        if stack is not None and hasattr(src, "aclose"):
            stack.push_async_callback(src.aclose)
        sources[slug] = src

    if not sources:
        log.warning(
            "nenhuma fonte ativa: a colheita não vai produzir nada. Veja "
            "`lithium sources` — ou nenhuma está aprovada, ou as credenciais não resolvem."
        )
    return sources
