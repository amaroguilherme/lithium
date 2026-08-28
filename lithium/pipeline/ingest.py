"""Ingestão: SourceRecord → chunks persistidos → embeddings indexados.

Chunking respeita as fronteiras que o documento já traz. Abstract estruturado do
PubMed vem com `<AbstractText Label="RESULTS">`, e essa divisão é semanticamente
melhor que qualquer janela de N caracteres. Só quando uma passagem estoura o teto é
que caímos para divisão por sentença, com sobreposição.

Sobreposição existe por um motivo concreto deste projeto: a citação verbatim precisa
caber inteira em um chunk. Uma frase partida ao meio entre dois chunks vira uma claim
que o verificador de ancoragem descarta — evidência real perdida por acidente de
tokenização.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from lithium.db import Store
from lithium.llm import EmbeddingClient
from lithium.sources.base import SourceRecord

log = logging.getLogger(__name__)

# ~4 caracteres por token; bge-m3 aceita 8192, mas chunk grande dilui o embedding.
MAX_CHUNK_CHARS = 1600
OVERLAP_CHARS = 200

# Seções de abstract estruturado que nunca contêm evidência. Indexá-las não é
# inofensivo: num piloto real com 163 chunks, um chunk cujo texto era literalmente
# "None." (seção `funding`) apareceu em 2º lugar numa busca — boilerplate curto tem
# embedding próximo de qualquer coisa, então ele emerge justamente quando não há
# resposta boa, que é o pior momento possível.
#
# `limitations` fica DE FORA da lista: é conteúdo, e conteúdo que importa para
# graduar a evidência.
SKIP_SECTIONS: frozenset[str] = frozenset({
    "funding",
    "declaration of interest",
    "declarations of interest",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "acknowledgement",
    "acknowledgements",
    "acknowledgments",
    "copyright",
    "disclosure",
    "disclosures",
})

MIN_CHUNK_CHARS = 40
"""Piso de tamanho. Pega o boilerplate que escapa da lista de seções — "None.",
"Not applicable.", "See above." — e que é curto demais para sustentar uma citação."""

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass(slots=True)
class IngestResult:
    source_id: int
    chunk_ids: list[int]
    embedded: int
    skipped_existing: bool = False


def split_passage(text: str, *, max_chars: int = MAX_CHUNK_CHARS,
                  overlap: int = OVERLAP_CHARS) -> list[str]:
    """Divide em fronteira de sentença, com sobreposição, sem estourar `max_chars`."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = _SENTENCE_END.split(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        # Sentença sozinha maior que o teto (tabelas, listas coladas): corta na força.
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), max_chars - overlap):
                chunks.append(sentence[start : start + max_chars].strip())
            continue

        if len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            # Recomeça com a cauda do anterior, para não partir uma citação em duas.
            current = (current[-overlap:] + " " + sentence) if overlap else sentence
        else:
            current = f"{current} {sentence}".strip()

    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


class Ingestor:
    def __init__(self, store: Store, embedder: EmbeddingClient) -> None:
        self.store = store
        self.embedder = embedder

    async def ingest(self, record: SourceRecord, *, reembed: bool = False) -> IngestResult:
        """Persiste a fonte, fatia em chunks e indexa os embeddings.

        Idempotente: reprocessar o mesmo registro reaproveita os ids e não duplica.
        """
        source_id = self.store.upsert_source(
            kind=record.kind.value,
            external_id=record.external_id,
            raw=record.raw,
            title=record.title,
            year=record.year,
            journal=record.journal,
            doi=record.doi,
            url=record.url,
            design=record.design.value if record.design else None,
            sample_n=record.sample_n,
            population_tag=record.population_tag,
        )

        chunk_ids: list[int] = []
        texts: list[str] = []
        ordinal = 0
        for passage in record.passages:
            if (passage.section or "").strip().lower() in SKIP_SECTIONS:
                continue
            for piece in split_passage(passage.text):
                if len(piece) < MIN_CHUNK_CHARS:
                    continue
                chunk_id = self.store.add_chunk(
                    source_id=source_id,
                    ord=ordinal,
                    text=piece,
                    section=passage.section,
                    n_tokens=len(piece) // 4,
                )
                chunk_ids.append(chunk_id)
                texts.append(piece)
                ordinal += 1

        if not chunk_ids:
            return IngestResult(source_id=source_id, chunk_ids=[], embedded=0)

        pending = chunk_ids if reembed else self._without_embeddings(chunk_ids)
        if not pending:
            return IngestResult(source_id, chunk_ids, embedded=0, skipped_existing=True)

        wanted = {cid: text for cid, text in zip(chunk_ids, texts, strict=True)}
        vectors = await self.embedder.embed([wanted[cid] for cid in pending])
        for chunk_id, vector in zip(pending, vectors, strict=True):
            self.store.set_chunk_embedding(chunk_id, vector)

        return IngestResult(source_id=source_id, chunk_ids=chunk_ids, embedded=len(pending))

    def _without_embeddings(self, chunk_ids: list[int]) -> list[int]:
        holes = ",".join("?" for _ in chunk_ids)
        rows = self.store.conn.execute(
            f"SELECT chunk_id FROM chunk_vec WHERE chunk_id IN ({holes})", chunk_ids
        ).fetchall()
        have = {r["chunk_id"] for r in rows}
        return [cid for cid in chunk_ids if cid not in have]
