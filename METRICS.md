# Plano de métricas — item E

> Este arquivo é um **plano**, não implementação. Ele diz o que medir, por que, e o que
> foi recusado. O roadmap e as decisões de arquitetura vivem em [PLAN.md](PLAN.md).

## A tese, e por que ela decide o formato

O projeto é uma prova de conceito para **como um modelo acumula conhecimento sem acumular
delírio**. Isso torna quase toda métrica óbvia inútil, porque quase toda métrica óbvia é
auto-referencial: "claims verificadas" sobe quando os portões afrouxam, "cobertura" sobe
quando o extrator fatia mais fino, "perguntas respondidas" sobe quando o juiz fica
complacente. **Uma métrica que sobe quando a qualidade cai é pior que métrica nenhuma.**

Duas medições enquadram o plano inteiro:

**Crescimento já é mensurável, e ninguém precisa instrumentar nada para isso.** Quase toda
tabela carrega o próprio timestamp — `claims.extracted_at`, `claim_directness.judged_at`,
`sources.fetched_at`, `llm_calls.created_at`, `tasks` com quatro marcos. A série
reconstrói retroativamente.

**Rejeição não é gravada em lugar nenhum.** Todo portão do sistema grava o SIM e joga o
NÃO num logger cujo único handler é `RichHandler(console)` — `cli.py:33-39`, verificado.
As taxas dos portões que medi (219/247 e 186/219) existem porque redirecionei stdout por
acaso numa sessão; o arquivo estava em `/tmp` e **já se perdeu num reboot**. Se os portões
afrouxarem, nada no banco mostra — e portões afrouxando *é* o mecanismo de acumular
delírio.

Daí o formato, que é o inverso do intuitivo: **quase nenhum coletor novo, quase tudo
persistir a rejeição.**

---

## As seis métricas

Cada uma declara uma **âncora**: a razão pela qual ela não sobe quando a qualidade cai. Sem
âncora declarada, a métrica não entra.

### MF1 — Ancoragem literal do portão 1, por lote, sempre em trio

**Âncora:** `quote_is_anchored` (`extract.py:81-92`) é o operador `in` do Python sobre o
texto do chunk, após colapso de espaço e `casefold`. Zero LLM no caminho. Nenhuma edição de
prompt de julgamento e nenhum afrouxamento do portão 2 move este número — é a única métrica
em que o modelo é pontuado por uma função pura.

**Trio obrigatório, na mesma linha:** (a) `anchored/proposed`; (b) `proposed / chunks
processados`; (c) mediana do comprimento da citação aceita.

**Queda em (a)** = o gerador começou a fabricar citação. **Subida para ~100% não é boa
notícia sozinha**: com (b) caindo é o gerador ficando calado; com (c) encurtando é o gerador
colando três palavras seguras.

**Quarto número, do mesmo lote:** o split chunk-**aniquilado** (propôs, nada sobreviveu) ×
chunk-**estéril** (não propôs nada). Hoje os dois são a mesma ausência de linha em **76 de
160 chunks** (verificado). São decisões opostas: aniquilado é "o paper certo, a citação
ruim" → apertar o extrator; estéril é "o paper errado" → trocar a frente de busca.

**Linha de base retro-encaixável:** 219/247 = 88,7% em 45 das 50 fontes, do log que
sobreviveu. As fontes 1–5 (27 claims, 12,7% do corpus) são **buraco declarado**, não
omissão silenciosa.

### MF2 — Massa de grade sem referente externo

**Âncora:** a partição não julga a claim; é a presença de um campo vindo do
`PublicationTypeList` do NCBI, resolvido por `_pick_strongest` (`pubmed.py:161`) antes de
qualquer modelo falar.

**Cálculo:** massa de `grade` particionada por `sources.design IS NOT NULL`. **Medido:
112 claims com design indexado, 101 sem → 52,6% ancorado** (verificado).

**Deliberadamente NÃO usa `claim_weight`.** `directness` é escrito pelo mesmo POST que
inventa a claim, e `confidence` é 1,0 em 180 de 213 — autoavaliação nos dois casos. Usar
peso completo aqui contaminaria a âncora externa com dois julgamentos internos.

**Queda** = mais do poder de ranqueamento repousa sobre a opinião do modelo sobre desenho de
estudo. **Cai por construção quando o adapter genérico entrar** — `http.py` fixa
`design=None` sempre — e é por isso que se lê *durante* a sonda de uma fonte nova, não
depois.

### MF3 — Massa cuja grade contradiz o desenho indexado

**Âncora:** piso zero e uma única direção de movimento.

**Medido: 2 promovidas de 112, massa 1,90** (verificado). Denominador restrito importa: **40
das 112 estão em `opinion`**, o piso da escala, onde deflação é aritmeticamente impossível.
Reportar sobre 112 seria o pecado do `build_state` — concluir "sem sinal" de uma amostra
travada na direção examinada.

As 2 são `opinion` indexado → `systematic_review` pelo modelo: promoção de sete degraus.

**Esta não cai, sobe.** Qualquer subida sustentada dispara re-lente e/ou aperto do prompt de
grade.

### MF4 — Fração endereçável: `intervention` que casa a taxonomia do perfil

**Âncora:** o matcher é código e a taxonomia é um TOML escrito por humano em disco. Nenhum
julgamento de modelo entra, e o número não se move quando o extrator muda de humor.

**Queda** = o extrator enche o banco de claim que a agenda de pesquisa não enxerga —
crescimento sem endereço. Contexto: **61 de 213 têm `intervention` NULL** (verificado), e
isso é o comportamento *correto* desde o conserto de hoje; o que a métrica vigia é a fração
das não-nulas que casa uma classe conhecida.

### MF5 — Discordância entre o juiz de suficiência e o piso de artigos

**Âncora:** os dois lados vêm de universos diferentes e só um é modelo. O piso é
`COUNT(DISTINCT article_key)` — SQL sobre uma coluna gerada. O juiz é um LLM.

**Cálculo:** entre as rodadas em que o piso foi **vinculante** (juiz aprovou, piso segurou),
a fração que o juiz aprovou.

**Subida = o juiz de suficiência está afrouxando.** É a única métrica do plano com poder de
**veto**: uma janela com MF5 em alta desqualifica qualquer leitura de autonomia
(`ANSWERED_AUTO` vs `ANSWERED_HUMAN`) e qualquer avanço do item 12.

### MF6 — Sobrevivência da lição e reinserção após retirada

**Âncora:** quem retira é a **pessoa**. `lithium memories --forget` e `/esquecer` são os
únicos escritores de retirada, e o humano não é o sistema.

**Duas leituras:** (a) taxa de sobrevivência caindo = a pessoa está desmentindo mais rápido
o que o sistema aprende sobre o próprio trabalho; (b) **reinserção acima de zero = o modelo
reescrevendo palavra por palavra a lição que a pessoa negou** — a assinatura mais direta de
delírio que este plano consegue capturar.

---

## As cinco assinaturas de delírio

O plano não mede "evolução" em abstrato. Ele mede cinco modos de falha nomeados, um por
métrica:

| Assinatura | Métrica |
|---|---|
| O gerador passa a fabricar citação | MF1 |
| O corpus ganha peso que nada externo sustenta | MF2 |
| O modelo contradiz para cima o rótulo que ele mesmo leu | MF3 |
| O extrator enche o banco de claim que a agenda não enxerga | MF4 |
| O juiz aprova onde o portão determinístico segura | MF5 |
| O reflector reescreve a lição que a pessoa negou | MF6 |

**Nenhuma pilota comportamento automático.** Métrica que pilota o sistema que ela mede vira
a coisa que o sistema otimiza.

---

## Instrumentação — só o que não reconstrói

### Primeiro passo (dá valor sozinho, sem daemon e sem coletor)

1. **`claims.supporting_quote`** — via `_ADDED_COLUMNS`. **Verificado: a coluna não existe
   no schema.** A citação está em memória em `extract.py:206`, é usada pelos dois portões, e
   é descartada ~50 linhas depois. Sem ela não há como reverificar o portão 1 retroativamente
   nem mostrar a um revisor o trecho exato em que a claim se apoia. **Não tem backfill
   possível** — é o item que mais perde por esperar.

2. **`extraction_runs` + `extraction_rejections`** — trocar o `log.info` de
   `handlers.py:225-232` por um INSERT do `ExtractionResult` inteiro, com `chunk_id` e
   `gate` na linha de rejeição.

3. **Retro-encaixe** das 45 linhas que o `drain.log` ainda tem, marcadas `origem='log'`
   para nunca se confundirem com medição de primeira mão.

### Depois, por ordem de perda

- `memories.retired_at` + `retired_by` (`human_cli` | `human_chat`) — três edições de uma
  linha. Sem isso MF6 é impossível.
- `claim_anchor(claim_id, indexed_grade, raw_publication_types_json, map_version)` —
  congela o referente externo de MF2/MF3, que hoje muda se `PUBLICATION_TYPE_TO_GRADE` for
  editado.
- `answer_rounds(question_id, round, n_hits, n_articles, judge_sufficient, floor_ok)` — sem
  isso MF5 não existe.
- `reflect_ticks(...)` — propostas, novas, repetidas, reinseridas.
- Estampa de procedência em toda linha persistida: `focus_id`, `scale_id`, versão do mapa.

### Um bug menor, achado no caminho

`usage.py:112` filtra com `created_at >= datetime('now', ?)`, que produz
`2026-08-24 21:21:23`, enquanto `created_at` grava `2026-08-28T20:30:10.385Z`. Formatos
diferentes, e `'T' > ' '` no ASCII: a janela fica sistematicamente até um dia larga demais.
Invisível hoje porque todo o dado cabe em 7 dias.

---

## Recusado

- **Métrica de "qualidade da claim" julgada por LLM** — auto-referencial por construção.
- **Contagem de claims, de fontes, de hipóteses como indicador de progresso** — sobe com
  qualquer afrouxamento. Descrevem crescimento, não evolução.
- **Taxa de aprovação do portão 2 isolada** — só é interpretável junto de MF1, porque o
  portão 2 só vê o que o portão 1 deixou passar.
- **Qualquer métrica derivada da tabela de cobertura antes do conserto de hoje** — ela
  contava 80 onde o cabeçalho afirmava 213.
- **Dashboards e séries de granularidade diária** — o corpus inteiro cabe em dois baldes de
  dia. A unidade é o **lote de extração**, não o calendário.

---

## Uma advertência metodológica, aprendida escrevendo isto

Ao conferir os números deste plano eu produzi **dois resultados falsos** com SQL que rodou
sem erro:

1. Contei chunks estéreis com `chunk_ids LIKE '%' || id || '%'` — o chunk 1 casa dentro de
   "11". Deu 54; o correto, com parsing de JSON, é **76**.
2. Escrevi `g1.axis = "grade"` com **aspas duplas**. No SQLite aspas duplas são
   identificador: a expressão resolveu para a coluna `c.grade`, o JOIN passou a comparar
   eixo com valor de grade, e devolveu **zero linhas sem erro nenhum**. Concluí que MF3 era
   zero e quase reportei o plano como errado. Com aspas simples: 112 linhas, 2 promovidas.

As duas falhas são a mesma classe que o `build_state` tinha: **um agregado que não conta o
que diz contar, e que não reclama.** Nenhuma métrica deste plano deve ser aceita sem que a
consulta seja conferida contra uma contagem independente.
