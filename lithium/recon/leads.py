"""O identificador de um `lead`, DERIVADO EM CÓDIGO do hit que a API devolveu.

Esta é a correção do defeito mais caro que a refutação achou no desenho. O desenho
provava, para a URL, que "um 12B a quem se pede uma URL inventa URL" e resolvia isso com
índice local — e depois pedia ao MESMO 12B que emitisse o PMID no campo que é a ÚNICA
coisa que atravessa a ponte para o canal de evidência.

O cenário concreto: a varredura acha `pubmed.ncbi.nlm.nih.gov/30712879/` com o título de
uma meta-análise; o modelo transcreve o PMID do snippet e troca um dígito. O CHECK do
banco valida FORMA (8 dígitos), não IDENTIDADE, então `30712873` passa. Você lê o TÍTULO
da meta-análise em `lithium discoveries`, aprova, e o `fetch_source` colhe um artigo real
e SEM RELAÇÃO (PMIDs abaixo de ~38.000.000 são quase todos atribuídos). Os dois portões
de extração aprovam — a citação é literal e a implicação é verdadeira, porque o artigo é
real. Nada liga a linha de `claims` de volta à descoberta que você leu: sua autorização
foi para um artigo e o corpus recebeu outro.

A correção é não perguntar. O modelo decide **o tipo** da descoberta; o identificador sai
de `re.search` sobre o texto que a API devolveu. Um hit sem identificador extraível faz o
veredito `lead` ser DESCARTADO — o mesmo tratamento que um índice fora do conjunto
mostrado.
"""

from __future__ import annotations

import re

PMID_RX = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{1,9})(?:\D|$)")
"""Só do CAMINHO de uma URL do PubMed. Um número solto num snippet ("n = 30712879"
nunca acontece, mas "PMID 30712879" no meio de prosa de terceiro, sim) não é
identidade — é texto que alguém escreveu."""

PMCID_RX = re.compile(r"pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d{4,9})")

DOI_RX = re.compile(r"\b(10\.\d{4,9}/[0-9A-Za-z./_():;+-]{1,90})")
"""O vocabulário de caractere é o MESMO do CHECK de `discoveries.lead_external_id`.

Deliberadamente sem `<`, `>`, `"` e espaço: um DOI colhido de prosa de terceiro tende a
arrastar a pontuação da frase, e o que arrasta prosa não pode caber na ponte."""

_DOI_TRAILING = ".,;:)"

DOI_FULL_RX = re.compile(r"^10\.\d{4,9}/[0-9A-Za-z._():;+-][0-9A-Za-z./_():;+-]{0,88}$")
"""Revalidação depois de podar a pontuação de fim de frase.

Um DOI colhido de prosa arrasta o ponto final, e `rstrip` pode deixar `10.1234/` — que
o CHECK do banco recusaria em runtime, dentro de uma transação, longe daqui."""


def lead_from_hit(url: str, *, haystack: str = "") -> tuple[str, str] | None:
    """`('pubmed', '30712879')`, `('doi', '10.xxxx/yyy')` ou None.

    A URL vem PRIMEIRO e é o caminho preferido: ela é o campo mais confiável da
    resposta da API. O `haystack` (título + descrição + snippets, ainda da API) é o
    fallback para o caso comum de um agregador que cita o DOI no resumo.
    """
    match = PMID_RX.search(url)
    if match:
        return "pubmed", match.group(1)

    for text in (url, haystack):
        if not text:
            continue
        found = DOI_RX.search(text)
        if found:
            doi = found.group(1).rstrip(_DOI_TRAILING)
            if DOI_FULL_RX.match(doi):
                return "doi", doi
    return None


def is_indexed_article(url: str) -> bool:
    """A página é um artigo que o canal de evidência sabe recolher sozinho?

    Usado pela triagem para ROTEAR: um hit assim vira `lead` sem gastar leitura, mesmo
    que o modelo tenha dito `observation`. É o outro lado da denylist de `read.py`.
    """
    return bool(PMID_RX.search(url) or PMCID_RX.search(url))
