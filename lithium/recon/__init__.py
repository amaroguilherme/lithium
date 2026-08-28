"""O canal de RECONHECIMENTO — o batedor.

Este pacote busca na web aberta, lê páginas e escreve `discoveries`. Ele **nunca**
escreve `claims`, e a impossibilidade é estrutural, em três camadas:

1. **Banco.** A única coluna de `discoveries` que o canal de evidência lê é
   `lead_external_id`, e os CHECKs de FORMA em `schema.sql` impedem que uma URL, uma
   frase ou um resumo caibam ali. Para levar prosa da web ao corpus é preciso EDITAR UM
   CHECK, embaixo do comentário que diz por que ele existe.
2. **Pacote.** `tests/test_recon_gate.py` varre por AST todo arquivo daqui e proíbe
   mencionar `SourceRecord`, `Passage`, `Ingestor`, `Extractor`, `upsert_source`,
   `add_chunk`, `chunks`, `claims` e `sources`. Sem esses nomes importáveis, o código
   que atravessaria a fronteira não tem onde ser escrito. MEDIDO por que isso importa:
   `Ingestor.ingest(SourceRecord(kind=PUBMED, passages=<HTML da nice.org.uk>))` grava
   chunks com `sources.kind='pubmed'`, e os dois portões de extração APROVAM — a
   citação é literal e a implicação é verdadeira.
3. **Conteúdo.** `test_no_web_text_reaches_the_evidence_channel` drena uma varredura
   inteira com um sentinela nas fixtures e afirma que ele não aparece em lugar nenhum
   do canal de evidência.

A PONTE — `recon_lead` — mora deliberadamente FORA daqui, em
`lithium/worker/handlers.py`, porque ela pertence ao canal de evidência. Ela carrega um
IDENTIFICADOR e nada mais, e um teste de AST proíbe que ela sequer NOMEIE as colunas de
prosa de `discoveries`.
"""

from lithium.recon.search import ReconDisabled, WebHit, WebSearcher

__all__ = ["ReconDisabled", "WebHit", "WebSearcher"]
