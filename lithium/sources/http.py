"""Adapter genérico dirigido por configuração.

Existe para que uma fonte nova NÃO exija código novo. Antes desta fase, acrescentar uma
fonte significava escrever um módulo, importá-lo no `daemon.py` e editar um dict literal —
três lugares, um deles fora de qualquer política. O efeito era que "aprovar uma fonte" não
podia significar nada, porque a decisão real morava no import.

O que este adapter NÃO faz, de propósito: parsing irregular. O `pubmed` mantém módulo
próprio porque o XML das E-utilities tem abstract estruturado, `PublicationType` que vira
`grade`, e `MedlineDate` em formato livre ("2019 Jan-Feb") — nada disso cabe em spec
declarativa sem virar uma linguagem de programação em JSON. A fronteira é: se a resposta é
JSON com campos nomeados, este adapter serve; se exige lógica, escreva um módulo e
registre `adapter = 'pubmed'`-style.

**A credencial nunca vem do banco.** `credential_ref` guarda o NOME de uma chave em
`config.local.toml` (gitignored) ou de uma variável de ambiente. O banco é o que o item 8
sincroniza para o Hugging Face; um segredo em coluna sairia de casa junto.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from lithium.sources.base import Passage, SearchSpec, SourceRecord
from lithium.sources.rate_limit import RateLimiter

log = logging.getLogger(__name__)

TIMEOUT_S = 20.0
MAX_BYTES = 4 * 1024 * 1024
"""Teto de resposta. Uma fonte que devolve 200 MB não é um caso de uso, é um acidente —
e sem teto ele viraria um MemoryError no worker em vez de um erro nomeado."""


def _dig(data: Any, path: str) -> Any:
    """Caminha um caminho pontuado num JSON. `a.b.0.c` funciona em dict e lista.

    Devolve `None` em qualquer passo que não resolva, em vez de levantar: a spec é escrita
    à mão e um caminho errado tem de virar campo vazio com log, não uma tarefa morta.
    """
    cur = data
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class SpecError(RuntimeError):
    """A spec do registro não descreve uma busca utilizável."""


class HttpSource:
    """Uma fonte HTTP+JSON descrita por `search_spec_json` / `fetch_spec_json`.

    Forma da spec de busca:
        {"path": "works", "query_param": "search", "id_path": "results",
         "id_field": "id", "limit_param": "per-page", "params": {"select": "id"}}

    Forma da spec de fetch:
        {"path": "works/{id}", "fields": {"title": "title", "doi": "doi",
         "year": "publication_year", "passages": ["abstract"]}}
    """

    def __init__(self, row: Any, *, api_key: str | None = None,
                 contact: str | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        self.kind: str = row["slug"]
        self.base_url: str = row["base_url"].rstrip("/")
        self.search_spec: dict[str, Any] = json.loads(row["search_spec_json"] or "{}")
        self.fetch_spec: dict[str, Any] = json.loads(row["fetch_spec_json"] or "{}")
        self.api_key = api_key
        self.contact = contact
        self.limiter = RateLimiter(rate_per_s=float(row["rate_per_s"]))
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": _user_agent(contact)},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = dict(self.search_spec.get("params") or {})
        params.update(extra)
        key_param = self.search_spec.get("key_param")
        if key_param and self.api_key:
            params[key_param] = self.api_key
        return params

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        await self.limiter.acquire()
        r = await self._client.get(f"{self.base_url}/{path.lstrip('/')}", params=params)
        r.raise_for_status()
        if len(r.content) > MAX_BYTES:
            raise SpecError(
                f"{self.kind}: resposta de {len(r.content)} bytes excede o teto de "
                f"{MAX_BYTES}"
            )
        return r.json()

    async def search(self, spec: SearchSpec) -> list[str]:
        s = self.search_spec
        if not s.get("query_param"):
            raise SpecError(
                f"{self.kind}: `search_spec_json` sem `query_param` — não há como dizer "
                f"à fonte o que buscar. Corrija com `lithium sources --spec`."
            )
        extra = {s["query_param"]: spec.query}
        if s.get("limit_param"):
            extra[s["limit_param"]] = spec.limit
        data = await self._get(s.get("path", ""), self._params(extra))

        rows = _dig(data, s["id_path"]) if s.get("id_path") else data
        if not isinstance(rows, list):
            log.warning("%s: `id_path` %r não resolveu para lista", self.kind,
                        s.get("id_path"))
            return []
        field = s.get("id_field")
        ids = [str(_dig(r, field) if field else r) for r in rows]
        return [i for i in ids if i and i != "None"][: spec.limit]

    async def fetch(self, external_ids: list[str]) -> list[SourceRecord]:
        """Um GET por id. Id inexistente é tolerado — a lista volta menor."""
        f = self.fetch_spec
        if not f.get("path"):
            raise SpecError(f"{self.kind}: `fetch_spec_json` sem `path`")
        fields = f.get("fields") or {}
        out: list[SourceRecord] = []
        for external_id in external_ids:
            try:
                data = await self._get(
                    f["path"].replace("{id}", external_id), self._params({}))
            except httpx.HTTPStatusError as exc:
                log.info("%s/%s: %s", self.kind, external_id, exc.response.status_code)
                continue

            texts = [
                str(_dig(data, path))
                for path in (fields.get("passages") or [])
                if _dig(data, path)
            ]
            if not texts:
                # Mesma regra do PubMed: registro sem prosa citável não é evidência. O
                # portão de citação verbatim não teria onde ancorar.
                log.info("%s/%s sem prosa aproveitável", self.kind, external_id)
                continue

            year = _dig(data, fields["year"]) if fields.get("year") else None
            out.append(SourceRecord(
                kind=self.kind,
                external_id=external_id,
                title=str(_dig(data, fields["title"]) or "") if fields.get("title") else "",
                passages=[Passage(text=x) for x in texts],
                raw={"id": external_id, "source": self.kind},
                year=int(year) if isinstance(year, int | str) and str(year).isdigit()
                else None,
                journal=str(_dig(data, fields["journal"]) or "") or None
                if fields.get("journal") else None,
                doi=str(_dig(data, fields["doi"]) or "") or None
                if fields.get("doi") else None,
                url=str(_dig(data, fields["url"]) or "") or None
                if fields.get("url") else None,
                # `design` fica None DE PROPÓSITO: a fonte genérica não sabe traduzir um
                # tipo de publicação em `grade`, e chutar `opinion` esmagaria evidência
                # real. None significa "não indexado", e quem julga é o LLM.
                design=None,
            ))
        return out


def _user_agent(contact: str | None) -> str:
    """Acesso automatizado se identifica. Uso pessoal não relaxa ToS de terceiro."""
    base = "lithium/0.1 (biomedical evidence synthesis; personal research)"
    return f"{base} contact:{contact}" if contact else base


def resolve_credential(ref: str | None, cfg: Any) -> str | None:
    """Resolve `credential_ref` — o NOME de onde a chave está, nunca a chave.

    Duas formas: `env:VAR` lê o ambiente, e `a.b.c` navega o objeto de config (que vem de
    `config.local.toml`, gitignored). Referência que não resolve devolve `None`, e quem
    chamou decide — uma fonte sem credencial obrigatória fica INATIVA com erro nomeado, em
    vez de tentar sem auth e receber 401 dentro do worker.
    """
    if not ref:
        return None
    if ref.startswith("env:"):
        return os.environ.get(ref[4:]) or None
    cur: Any = cfg
    for part in ref.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return str(cur) or None
