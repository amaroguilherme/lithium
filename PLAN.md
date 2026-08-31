# Plano de Arquitetura — lithium

> Documento de referência do projeto. A versão original foi aprovada em 21/08/2026;
> a seção **[Desvios e achados](#desvios-e-achados-da-implementação)** no fim registra
> o que mudou ao implementar, com o motivo. Onde o plano e o código divergirem, o
> código manda e este documento está desatualizado — abra issue.

## Contexto

**Problema clínico.** Transtorno bipolar tipo I (TB-I) com TAG comórbido é um beco
farmacológico: o tratamento padrão do TAG é antidepressivo, e antidepressivo em
monoterapia no TB-I tem risco de virada maníaca e aceleração de ciclagem. A literatura
direta sobre a interseção é escassa — justamente porque RCTs de TAG excluem bipolares
no critério de elegibilidade e RCTs de bipolar tratam ansiedade como desfecho
secundário. A escassez é estrutural, não acidental. Achar alternativa exige
**transferência de evidência entre populações adjacentes, com downgrade explícito por
indireção** (o domínio *indirectness* do GRADE).

**Por que um sistema, e não uma busca manual.** O trabalho é longo, incremental e
assíncrono: varrer fontes, extrair alegações, graduar, cruzar, detectar lacuna,
formular a próxima pergunta. É o loop que um agente persistente faz bem e uma sessão
de chat faz mal.

**O que entrega.** Um daemon local que (a) pesquisa continuamente em APIs biomédicas,
(b) mantém um placar vivo de hipóteses terapêuticas com balanço de evidência, (c) gera
suas próprias perguntas e **tenta se auto-responder**, escalando para o humano só
quando trava, e (d) publica relatórios preliminares em cadência — acessíveis de
qualquer lugar por um painel web.

**Fronteira de escopo.** Ferramenta de síntese de literatura e geração de hipóteses
**para revisão por psiquiatra**. Não emite conduta. Todo output carrega citação
rastreável, grau de evidência e incerteza residual — requisito de engenharia, não
disclaimer: alegação sem fonte é descartada por um passe verificador antes de entrar
no banco.

## Decisões tomadas

| Decisão | Escolha |
|---|---|
| Aprendizado | Híbrido: RAG agora, destilação LoRA depois |
| Perguntas | Auto-resposta primeiro; escala para humano só quando trava |
| Fontes | PubMed, Europe PMC, ClinicalTrials.gov, openFDA |
| Isolamento | Do `qyra-labs` usa só `base_models/gemma-4-12b-it-Q5_K_M.gguf`. Zero import de `qyra` |
| Estado | Sempre local. Sem banco na nuvem |
| Estação | Mac hoje, Windows depois. Portabilidade é requisito de primeira classe |
| Interface | HF Space (Gradio, CPU grátis) |
| Treino | Kaggle (T4, CUDA) |
| Otimização para M1 | Fora de escopo — produção não roda em Mac |
| Criatividade | Trilha exploratória com especulação mecanística **sem limite**, separada da trilha de evidência |
| Conversa | `lithium chat` no CLI agora, Space no item 8 |
| Modo pesquisa | Ligado (24/7) ou desligado (sob demanda); memória ativa nos dois |
| Memória de conversa | O sistema **propõe** e só grava com confirmação |
| Memória de pesquisa | Grava sozinho, com portão de citação para lição substantiva |

---

## Topologia — três nós, um barramento

```
┌─ NÓ 1 · ESTAÇÃO (Mac → Windows) ────────── fonte de verdade
│  llama-server (GGUF, Metal → CUDA)
│  SQLite local  ·  worker pool asyncio
│  harvest → extract → grade → synthesize → critique → perguntas
└──────────┬───────────────────────────────┬──────────────────
        publish│ (snapshot)         pull│ (respostas, adapter)
               ▼                         │
┌─ HF HUB · barramento ───────────────────┴──────────────────┐
│  datasets/<user>/lithium-corpus  (privado)                 │
│    snapshot/     placar, findings, perguntas abertas       │
│    answers/      Q-0042.json ← escrito pelo Space          │
│    training/     pares jsonl para a LoRA                   │
│  models/<user>/lithium-gemma4-lora  → adapter .gguf        │
└──────┬──────────────────────────────────────┬──────────────┘
       ▼                                      ▼
┌─ NÓ 2 · HF SPACE (Gradio, CPU grátis) ─┐  ┌─ NÓ 3 · KAGGLE (T4) ─┐
│  painel: hipóteses, relatórios, claims │  │  Unsloth QLoRA       │
│  fila de perguntas + campo de resposta │  │  → adapter GGUF      │
│  SEM inferência                        │  │  30h/semana grátis   │
└────────────────────────────────────────┘  └──────────────────────┘
```

O banco **nunca sai** da estação. O que sobe é um snapshot derivado, somente-leitura;
o que desce são respostas suas e adapters. Nenhum dos três nós precisa dos outros no
ar para funcionar.

**Por que o Space não faz inferência.** Space CPU grátis é 2 vCPU / 16 GB — Gemma-4-12B
ali roda em segundos por token, e o daemon precisa do LLM em volume contínuo. Manter o
Space como camada de apresentação pura é o que permite ele ser grátis e trivialmente
portável.

**Latência do ciclo humano.** Você responde no Space → commit em `answers/` → o próximo
tick da estação (padrão: 10 min) faz o pull e retoma. Para um loop de cadência diária,
minutos de latência não custam nada, e o preço é zero serviço extra rodando.

---

## Contrato de portabilidade (Mac → Windows)

Restrição de projeto verificada em CI desde o dia 1 — migração não pode virar
refatoração.

| Risco | Regra | Verificado em |
|---|---|---|
| Backend de inferência | LLM sempre atrás de **base URL + modelo em config**. `llama-server` é subprocesso, não biblioteca | `test_worker.py` |
| Caminhos | `pathlib` em tudo. Relativos resolvem contra a raiz do projeto, não o CWD | `test_config.py` |
| APIs POSIX | Sem `fcntl`, `os.fork`, `os.waitpid`, `signal.SIGKILL` — lint por AST | `test_config.py` |
| Subprocessos | `create_subprocess_exec` com lista de args, sem shell | `test_worker.py` |
| Supervisão | `lithium serve` é processo comum. launchd / Task Scheduler ficam **fora** do código | README |
| asyncio | Windows usa `ProactorEventLoop`; espera de processo sempre pelo caminho asyncio | — |
| SQLite | WAL. `sqlite-vec` com fallback numpy se `enable_load_extension` estiver desabilitado | `test_store.py` |
| Isolamento | Nenhum import de `qyra` | `test_config.py` |
| CI | Matriz `macos-latest` + `windows-latest` | `.github/workflows/lithium-ci.yml` |

Metal → CUDA é troca da build do llama.cpp instalada. Nenhuma linha de código muda.

---

## Stack

| Camada | Escolha | Motivo |
|---|---|---|
| Inferência | `llama-server` como processo separado, OpenAI-compatible | Concorrência por slots; binding in-process segura a GIL e mataria o worker pool |
| Saída estruturada | `response_format: json_schema` (GBNF por baixo) | **Não-negociável.** Um 12B sem gramática não produz JSON confiável em volume |
| Embeddings | Segunda instância `llama-server --embedding` com `bge-m3` | PT+EN no mesmo índice, mesma stack, sem torch na estação |
| Persistência | SQLite + `sqlite-vec` + FTS5 | Zero ops, arquivo único, portável. Vetor e keyword na mesma transação |
| Fila | Tabela `tasks` + pool asyncio | Durável, restartável, sem Redis/Celery |
| HTTP | `httpx` async + token bucket por fonte | NCBI limita a 3 req/s (10 com key) |
| Validação | `pydantic` v2 | Os modelos **geram** os JSON Schemas da decodificação restrita |
| Sync | `huggingface_hub` | Barramento sem serviço extra |
| UI | Gradio no Space | Roda em CPU grátis; bom em mobile |
| Treino | Unsloth + PEFT no Kaggle | T4 tem CUDA |

---

## Modelo de dados

```
sources          id, kind, external_id, title, year, journal, doi, url,
                 design, sample_n, population_tag, raw_json, fetched_at
chunks           id, source_id, ord, section, text, n_tokens
                 + chunk_vec (vec0)  + chunk_fts (fts5, external content)
claims           id, source_id, chunk_ids, statement, population, intervention,
                 comparator, outcome, direction, effect, grade, directness,
                 confidence, verified, extracted_at
questions        id, text, kind, status, priority, parent_id, origin, rounds,
                 stuck_reason, partial_work, answer, answer_origin, embedding
hypotheses       id, statement, status, created_at, updated_at
evidence_links   claim_id, hypothesis_id, polarity, weight
findings         id, question_id, text, confidence, citations_json,
                 critique_json, safety_json
tasks            id, kind, payload_json, status, priority, attempts,
                 dedup_key, scheduled_at, claimed_at, finished_at, error
reports          id, kind, body, created_at
training_examples id, kind, question_id, messages_json, weight, verified
sync_state       key, last_local_rev, last_remote_rev, synced_at
```

Views: `claim_weight` (peso pronto, só `verified = 1`), `hypothesis_scoreboard`
(placar por polaridade), `escalated_queue` (fila do Space).

### A invariante central

O peso de uma evidência é `grade × directness × confidence` — **multiplicativo**.

```
cohort + direct          = 0.55 × 1.00 = 0.550   ← vence
meta_analysis + indirect = 1.00 × 0.30 = 0.300
rct + indirect           = 0.85 × 0.30 = 0.255
```

Um coorte em bipolar I com TAG vale mais que uma meta-análise em unipolar. É o
julgamento que um revisor humano faria, e é a propriedade que torna o sistema útil num
domínio onde a evidência direta quase não existe. Travada por teste em três camadas:
`test_types.py` (Python), `test_store.py` (SQL) e `test_pipeline_retrieval.py`
(ponta a ponta no ranking).

`directness` — aderência da população ao alvo "TB-I + TAG":

| Valor | Significa |
|---|---|
| `direct` | TB-I com TAG comórbido |
| `partial` | TB-I com ansiedade secundária, ou TAG puro |
| `indirect` | Bipolar II, pânico, unipolar |
| `extrapolated` | Mecanismo, pré-clínico, inferência |

**`directness` não é uma propriedade da claim — é uma relação entre a população do estudo e um
alvo**, e repare que a tabela acima só faz sentido porque o alvo está nomeado na linha anterior a
ela. Hoje o banco guarda o valor e **não guarda contra qual alvo ele foi julgado**, o que torna
todo `claims.directness` silenciosamente dependente de uma constante em `types.py:29`.

A Fase A ✅ moveu isso para **`claim_directness(claim_id, focus_id, directness)`** e a Fase B o
torna rejulgável. **A invariante sobrevive em forma, não em implementação:** `claim_weight`
continua sendo origem única e produto de três fatores; o que muda é que dois dos três passam a
ser resolvidos pela lente do foco ativo, via JOIN. Claim graduada em outra escala e claim sem
julgamento neste foco caem do JOIN — as duas significam **sem peso**, fail-closed, nunca um
valor médio de conveniência.

> **A aresta é por CLAIM, não por população** — o plano dizia população e a medição derrubou.
> Ver "Fase A — o que a medição mudou" abaixo.

---

## Estrutura

```
lithium/
  pyproject.toml  config.toml  README.md  PLAN.md
  lithium/
    config.py  types.py  daemon.py  cli.py
    db/         schema.sql  store.py
    llm/        client.py  server.py  schemas.py  prompts.py  prompts/*.md
    sources/    base.py  pubmed.py  rate_limit.py
                europepmc.py  clinicaltrials.py  openfda.py     ⏳
    pipeline/   ingest.py  retrieval.py  extract.py  strategy.py
                synthesize.py  critique.py  hypothesis.py
                question.py  state.py
                report.py                                       ⏳
    safety/     contraindications.yaml  check.py                 ⏳
    worker/     queue.py  runner.py  scheduler.py  handlers.py
    sync/       publish.py  pull_answers.py  adapters.py         ⏳
    notify/     base.py  macos.py  windows.py                    ⏳
  space/        app.py  requirements.txt                         ⏳
  kaggle/       train_qlora.ipynb  build_dataset.py
                evaluate.py  promote.py                          ⏳
  eval/         goldset.yaml  safety_probes.yaml                 ⏳
  tests/        fixtures/ (XML real do NCBI)
  data/         (gitignored) lithium.db  cache/  adapters/
```

---

## Componentes

### `sources/` — adaptadores

Protocolo comum: `search(spec) -> list[str]` e `fetch(ids) -> list[SourceRecord]`.

| Adapter | Endpoint | Nota |
|---|---|---|
| `pubmed` ✅ | E-utilities `esearch`/`efetch` | `PublicationType` → grade. Abstract estruturado já vem seccionado |
| `europepmc` ⏳ | `ebi.ac.uk/europepmc/webservices/rest` | Full text de OA — muito melhor que abstract para extração |
| `clinicaltrials` ⏳ | `clinicaltrials.gov/api/v2/studies` | Estudos em andamento e **negativos não publicados** (viés de publicação) |
| `openfda` ⏳ | `api.fda.gov/drug/label.json` | Contraindicação e interação estruturadas |

**`design = None` não é `opinion`.** O NCBI indexa confiavelmente só as categorias
fortes. Para o resto o campo diz "Journal Article" e nada mais — aí devolvemos `None` e
o prompt manda o LLM julgar pelo texto. Fixar `opinion` como piso esmagaria coorte e
caso-controle legítimos, que é justamente a evidência que sobra aqui.

### `pipeline/strategy.py` — as cinco frentes de busca

Buscar só a interseção volta vazio. Cada frente carrega um `expected_directness`
atribuído **antes** de ver o resultado — é isso que permite raciocinar "meta-análise
forte, mas em unipolar; vale menos do que o tamanho sugere".

1. **`intersecao_direta`** — a interseção real. Raríssima, altíssimo valor.
2. **`bipolar_ansiedade_secundaria`** — ansiedade como desfecho secundário em ensaios de
   bipolar. Corpo grande e subexplorado: o dado existe, só não está no título.
3. **`nao_antidepressivo_em_tag`** — quetiapina é o caso canônico (tem RCT em TAG *e* é
   estabilizador aprovado no TB-I); também pregabalina, buspirona, hidroxizina,
   lamotrigina.
4. **`nao_farmacologico`** — TCC, ACT, MBCT, IPSRT, exercício. Sem risco de virada, e
   combinável com qualquer regime.
5. **`risco_e_ancoras`** — quanto risco o antidepressivo traz de fato no TB-I, e o que
   as diretrizes (CANMAT/ISBD, NICE, ABP) já resolveram.

### `pipeline/` — o loop de conhecimento

```
harvest → ingest (chunk + embed + index) → extract (2 portões) → grade
       → hypothesis (balanço) → synthesize → critique
```

**Chunking respeita fronteiras do documento.** Abstract estruturado do PubMed vem com
`<AbstractText Label="RESULTS">`, melhor que qualquer janela de N caracteres. Há
sobreposição por um motivo concreto: a citação verbatim precisa caber inteira em um
chunk — frase partida no limite vira claim que o verificador descarta, evidência real
perdida por acidente de tokenização.

**Retrieval híbrido.** Vetorial + BM25 fundidos por RRF. Híbrido porque nomes de fármaco
e identificadores (`NCT01236411`, `HAM-A`) são o que embedding dilui e BM25 acerta; e
porque "alternativa a antidepressivo" precisa casar com "mood stabilizer monotherapy",
que não compartilha um termo. RRF em vez de soma ponderada porque distância de cosseno
e score BM25 vivem em escalas incomparáveis.

### Os dois portões de verificação

Esta é a fronteira mais importante do sistema.

**Portão 1 — ancoragem, determinístico.** `supporting_quote` precisa ser trecho literal
do chunk. Sem LLM, sem custo, e sem possibilidade de o modelo se absolver. Um modelo que
parafraseia a citação está inventando, e isso é detectável por comparação de string.

**Portão 2 — implicação, por LLM.** A citação existe, mas ela sustenta a alegação? Aqui
cai o erro mais perigoso de um 12B: citar corretamente uma frase sobre TAG puro e
escrever uma alegação sobre bipolar I. Roda só nos sobreviventes do portão 1.

Claim que não passa nos dois **não entra no banco**. Isso importa além da correção
imediata: `claim_weight` só enxerga `verified = 1`, então material não verificado fica
fora do placar, dos relatórios e do dataset de treino. Contaminar o treino com a própria
alucinação é como um sistema desses apodrece.

Medido no Gemma-4-12B Q5_K_M: **97% de ancoragem** (32/33), 20/20 de conformidade de
schema, 0 alegações inventadas em texto sem conteúdo.

### `pipeline/question.py` — o coração ✅

**Taxonomia** (define roteamento, não é cosmética):

| Tipo | Rota |
|---|---|
| `FACTUAL` | auto |
| `SYNTHESIS` | auto + crítica adversarial |
| `PREFERENCE` | **sempre humano** — trade-off de valores |
| `CONTEXT` | **sempre humano** — o modelo não tem como saber |
| `METHODOLOGICAL` | humano, prioridade baixa |

`PREFERENCE` e `CONTEXT` são estruturalmente inauto-respondíveis. Mandá-las ao loop de
pesquisa gasta ciclo e produz resposta inventada.

**Loop de auto-resposta:**

```
while round < MAX_ROUNDS (3):
    queries  = LLM(question → plano de busca por fonte)
    harvest → ingest → extract
    evid     = retrieval híbrido (rerank por grade × directness)
    rascunho = LLM.answer(q, evid)
    veredito = LLM.judge_sufficiency(q, rascunho, evid)   # CHAMADA SEPARADA

    if veredito.suficiente and citações_verificadas:
        crítica = LLM.critique(rascunho)                  # instruído a REFUTAR
        if sobrevive: ANSWERED_AUTO; atualiza hipóteses; break

    if veredito.bloqueio in (NEEDS_CONTEXT, NEEDS_VALUE_JUDGMENT,
                             IRRECONCILABLE_CONFLICT): break
    round += 1

escalate(q)   # com trabalho parcial + motivo exato do travamento
```

O juiz é chamada **separada** do gerador. Modelo pequeno auto-avaliando na mesma chamada
em que gerou diz "suficiente" praticamente sempre.

**Escalonamento com trabalho parcial.** A pergunta que chega até você não vem nua: vem
com o que já achou, o que descartou e por quê, e a frase exata do impasse. Transforma um
pedido de pesquisa numa escolha de 30 segundos — é o detalhe que decide entre o sistema
ser usado ou abandonado.

**Teto de 5 perguntas abertas.** Sistema que gera 40 perguntas por dia não é usado duas
vezes. O excedente fica represado com o diagnóstico salvo e sobe quando abre vaga.

**Dedup é por alvo, não por texto.** Ver os achados abaixo — similaridade textual
sozinha não funciona neste domínio.

**Priorização** — proxy explicável sobre os três sinais que o estado expõe:
`0.45 × novidade + 0.35 × conflito + 0.20 × lacuna de directness`, dividido por um
custo de oportunidade por tipo. Explicável importa: a prioridade decide quem ocupa as
cinco vagas, e o Space mostra a justificativa junto da pergunta.

### `safety/` — checagem determinística ⏳

`contraindications.yaml` aplicado a todo finding e relatório antes de publicar. É código,
não prompt — LLM não é o guardião:

- antidepressivo em monoterapia no TB-I → risco de virada/ciclagem (obrigatório)
- combinação serotoninérgica → síndrome serotoninérgica
- lamotrigina → titulação lenta, SJS/NET
- lítio → interação AINE / IECA / tiazídico, janela estreita
- valproato → teratogenicidade em pessoas com potencial gestacional

Violação não bloqueia — **anexa o alerta**. Nenhuma hipótese é apresentada sem o risco
correspondente colado nela.

### `worker/` — assincronia

Fila durável no próprio SQLite, com claim atômico:

```sql
BEGIN IMMEDIATE;
UPDATE tasks SET status='running', claimed_at=?, attempts = attempts + 1
 WHERE id = (SELECT id FROM tasks
              WHERE status='pending' AND scheduled_at <= ?
              ORDER BY priority DESC, id LIMIT 1)
RETURNING ...;
```

`BEGIN IMMEDIATE` toma o lock de escrita antes do SELECT — sem ele, dois workers pegam a
mesma linha. Verificado com 10 threads reais e barreira para forçar colisão.

Handlers picados de propósito: cada um faz um pedaço e enfileira o próximo. Um harvest
que buscasse, baixasse, indexasse e extraísse numa única tarefa perderia horas a cada
falha de rede.

Cadência interna (não cron — cron é POSIX): harvest diário, plan tick diário, relatório
semanal, sync a cada 10 min. O último disparo vive no banco, não em memória: reiniciar o
daemon não deve ressetar o relógio.

### `sync/` — barramento HF ⏳

- **`publish.py`** — projeta o SQLite num snapshot derivado e sobe ao dataset privado.
  **Não sobe** o `.db` nem chunks brutos.
  **Restrição das Fases A–D:** com um cérebro só, o banco passa a agregar memória de conversa
  **através de focos** — o snapshot exclui `memories` de origem `chat` e `recon`, e o dataset
  continua privado não por convenção mas porque agora carrega mais sobre a pessoa do que antes.
  A constraint de vazamento para LoRA (§"Vazamento para o LoRA" do plano de memória) cobre
  metade disto e passa a valer aqui também.
- **`pull_answers.py`** — baixa `answers/`, aplica ao banco, reabre as perguntas.
  Arquivos `Q-0042.json` append-only → **sem conflito de merge por construção**.
- **`adapters.py`** — baixa adapter GGUF novo, registra versão, oferece ao gate.

### `space/app.py` — painel Gradio ⏳

Quatro abas, CPU grátis, sem inferência: **Hipóteses** (placar com suporte × contra),
**Perguntas** (fila escalada com trabalho parcial e campo de resposta), **Relatórios**,
**Evidência** (busca filtrável por grade/directness). Privado por padrão — o corpus e as
hipóteses são de saúde.

Depois das Fases A–D a aba Evidência precisa nomear o **foco ativo**: `directness` deixa de ser
uma coluna de `claims` e passa a depender da lente, então um filtro "directness = direct" sem o
foco à vista é uma pergunta sem resposta. E claim fora da escala do foco aparece marcada e sem
peso, em vez de sumir.

---

## As duas trilhas ✅

A trilha de evidência responde *"o que a literatura sustenta?"* e ordena por
`grade × directness`. Por construção, ela **suprime o inédito**: um candidato novo tem
evidência `preclinical` (0.10) ou `extrapolated` (0.12), e multiplicados dão ~0.01 —
invisível ao lado de qualquer RCT. Correto lá, e inútil para achar o que ninguém testou.

A trilha exploratória responde *"o que ninguém testou, e por quê?"* e ordena por
**plausibilidade × ineditismo**. As duas vivem na mesma tabela `hypotheses`, separadas
por `tier`, com views disjuntas — o que permite a promoção `speculative → evidence` ser
transição de estado quando a evidência aparecer, sem migrar dado. Mas os rankings nunca
se misturam num relatório: juntos, ou o inédito some, ou o estabelecido perde
precedência indevidamente.

### Taxonomia aberta, não catálogo de fármacos

A cobertura é medida por **classe de intervenção** e por **alvo mecanístico**, não por
nome de remédio. A lista fixa que existia antes (`quetiapina, lamotrigina, lítio…`) era
uma gaiola: só tornava visível a lacuna dentro dela própria. Hoje o gerador vê "ninguém
olhou neuromodulação" ou "o eixo HPA está intocado" e fica livre para nomear qualquer
coisa naquele espaço — inclusive algo que não existe como tratamento psiquiátrico. Mais
da metade das classes rastreadas não é farmacológica (`test_pipeline_explore.py` trava
essa proporção).

A pergunta que orienta a trilha não é "o fármaco X funciona?", é **"o que dissocia
ansiólise de desestabilização do humor?"** — e a partir do mecanismo, qualquer coisa que
o atinja: reposicionamento, fase 1-2, neuromodulação, cronobiologia, metabólico,
autonômico, ou a estratégia de timing como intervenção em si.

### O laço: especular → buscar → ancorar

A trilha exploratória sozinha era um **beco sem saída**. Ela propunha sigma-1, via
transdérmica, um dispositivo — e o `harvest_sweep` continuava iterando sobre as mesmas
19 queries fixas, nenhuma das quais menciona sigma-1, orexina, cronoterapia ou
reposicionamento. O sistema inventava candidatos e nunca perguntava a base alguma sobre
eles. A gaiola tinha saído da geração de hipóteses e continuado na coleta.

```
explore_tick     → propõe hipóteses mecanísticas, critica
pursue_speculation → converte cada hipótese em buscas DIRIGIDAS aos elos assumidos
harvest_query    → agora aceita query solta, não só as estratégias fixas
  ↓ (coleta, extrai, verifica — o pipeline normal)
reground_speculations → reavalia a cadeia contra o corpus já maior
```

**A reancoragem é o que faz o laço ter consequência.** Elos marcados `assumed` que a
literatura recém-coletada passou a sustentar viram `supported` com PMID, e a
plausibilidade sobe. Sem ela a cadeia congelaria no estado em que nasceu e perseguir
uma hipótese não mudaria nada de mensurável.

Uma checagem impede o atalho óbvio: a reancoragem só aceita PMID que esteja na evidência
efetivamente recuperada. Sem isso, inflar plausibilidade seria escrever um número
qualquer.

### Além da farmacopeia: composto, combinação e via

Três direções que o prompt de geração agora exige explicitamente:

**Compostos novos.** Não só medicamentos comercializados — compostos em fase 1-2 para
outras indicações, ferramentas de pesquisa pré-clínica, metabólitos, pró-fármacos,
estereoisômeros, análogos estruturais onde o composto-mãe tem o mecanismo certo e o
perfil de efeitos errado. Nomear algo que nunca foi dado a um paciente bipolar é o
ponto da trilha, não um desvio.

**Combinações.** O caso interessante não é "somar um segundo fármaco para mais efeito"
— é o par em que um componente **cancela a liability do outro**. Se o antidepressivo
desestabiliza por uma propriedade específica, o que coadministrado bloqueia aquela
propriedade preservando a ansiólise?

**Via de administração, como variável mecanística.** O risco de virada acompanha a
*velocidade* da subida monoaminérgica, e a via define o Tmax: um adesivo transdérmico ou
um depot subcutâneo achatam a curva que a mesma molécula faz picar por via oral. A mesma
substância pode ser inviável oral e viável transdérmica — e isso é uma hipótese de
verdade. A via também alcança o que não é molécula: dispositivo implantado, vestível,
neuromodulação em malha fechada, hardware de fototerapia, biofeedback.

### As três salvaguardas

Especulação sem limite só é responsável se as salvaguardas forem estruturais. Todas são
requisito de schema, não conselho de prompt:

1. **Cadeia mecanística auditável.** Cada elo de "A → B → C → desfecho" é marcado
   `supported` (com PMID) ou `assumed`. `plausibility` = fração ancorada — um número,
   não uma impressão. Marcar `supported` sem citação não conta, que é o atalho óbvio
   para inflar a pontuação.
2. **Falsificador obrigatório.** Toda hipótese declara o que a refutaria. Sem isso é
   prosa, e não entra.
3. **Crítica adversarial dedicada**, que procura o **elo mais fraco** em vez de avaliar
   o conjunto — uma cadeia vale o seu pior elo, e a média o esconde atrás dos bons.
   Crítica indisponível reprova: sem o passe, a hipótese não foi verificada.

---

## Modo pesquisa: ligado / desligado ✅

**Ligado** (padrão) — o scheduler dispara harvest, plan, explore, pursue e reground em
cadência. O daemon trabalha sozinho.

**Desligado** — o daemon segue no ar e atende, mas não pesquisa por conta própria.
Concretamente: o scheduler para de enfileirar, e os workers **só reivindicam tarefas
`on_demand`** — as que você pediu (`chat`, `ask`, `harvest`, `explore`). Trabalho
agendado que já estava na fila fica represado até religar.

A distinção por origem (`tasks.origin`) é o que faz "desligado" ser útil em vez de
inerte. Um stop global congelaria também o pedido que você acabou de fazer; deixar tudo
rodando não devolveria a máquina, que é o ponto.

Duas propriedades que valem destacar:

- **A memória continua ativa nos dois modos.** Aprender com o que você diz é conversa,
  não pesquisa — desligar a pesquisa não deve fazer o sistema esquecer o que você contou.
- **O relógio dos jobs não avança enquanto desligado**, então religar não provoca uma
  enxurrada de disparos atrasados.

O estado vive na tabela `meta`, não em memória: reiniciar o daemon não religa a pesquisa
sozinho.

```bash
lithium mode          # mostra o estado e o backlog represado
lithium mode off      # sob demanda
lithium mode on       # 24/7
```

---

## Memória: dois regimes de consentimento ✅

Por decisão explícita do usuário, a origem determina se há confirmação:

| origem | regime | por quê |
|---|---|---|
| **conversa** | propõe e espera confirmação | é sobre você; acumular fatos sobre a pessoa sem autorização é diferente de acumular papers |
| **pesquisa** | grava sozinho | é o sistema aprendendo a trabalhar melhor; pedir permissão para "aprendi que MeSH rende mais precisão" treinaria o usuário a clicar sem ler — e aí a confirmação que importa perderia valor |

A autonomia da memória de pesquisa é sustentada por **dois limites estruturais**, não
por conselho no prompt.

### 1. Portão de citação por categoria

| categoria | exemplo | portão |
|---|---|---|
| `search_lesson` / `dead_end` / `source_lesson` | "Queries pareando sigma-1 com desfecho clínico voltam vazias; a literatura pré-clínica usa outro vocabulário" | nenhum — não afirma nada sobre biologia |
| `pattern` | "Três hipóteses refutadas falharam por elevar tônus glutamatérgico agudamente" | **exige `claim_ids` de claims verificadas**, e a escrita confere que existem |

A primeira versão proibia lição substantiva por completo. Estava errado: o padrão
derivado costuma ser a coisa mais útil a registrar, e uma proibição chapada cortava
justamente o que tornaria a rodada seguinte mais inteligente. O portão de citação é a
mesma disciplina da reancoragem — que só aceita PMID presente na evidência recuperada.

Sem esse portão, `"sigma-1 não funciona em bipolar"` entraria sem fonte e seria
injetada em todo prompt seguinte: o sistema ensinando as próprias suposições a si mesmo,
com nenhum portão a jusante pegando. É a falha que os dois portões da extração previnem,
entrando pela porta de trás.

### 2. Recuperação por relevância, não injeção em massa

Memória auto-gravada **compõe**. Claims são recuperadas por relevância — só as
pertinentes entram no prompt. Se lições fossem injetadas inteiras, com 200 acumuladas
todo prompt carregaria 200 linhas das opiniões do sistema sobre si mesmo, e cada decisão
futura passaria por esse filtro. É um caminho silencioso para ele convergir nas próprias
crenças e parar de olhar para fora.

O consentimento humano é o que limita o crescimento das memórias de conversa. Removido
para pesquisa, precisava de outro limite: `relevant_lessons()` recupera por similaridade
com teto de 8, e só o inventário (`lithium memories`) mostra tudo.

`provenance` guarda o evento do banco que gerou cada lição, e `lithium memories` mostra
a origem de cada uma. Auto-gravada não significa inauditável — `--forget` funciona em
qualquer memória.

---

## Conversa e memória ✅

O daemon pesquisa sozinho; `lithium chat` é o canal para falar com ele. Duas coisas o
separam de um chat qualquer:

**Ele enxerga o corpus.** Cada turno recupera claims verificadas e as injeta com PMID.
O prompt exige separar *"a literatura diz"* de *"eu acho"* — é a única coisa valiosa
que este sistema tem, e uma frase confiante sem fonte destrói isso. Uma exceção
permanente: risco de virada com antidepressivo em TB-I é levantado sem ser perguntado.

**Ele propõe memórias, nunca grava sozinho.** Ao detectar algo durável — preferência,
restrição, contexto do caso — ele pergunta antes de guardar. Num domínio de saúde,
acumular fatos sobre a pessoa sem autorização é diferente de acumular papers.

Memórias ficam em tabela separada de `claims`, de propósito: claim é sobre o mundo e
tem citação; memória é sobre você e não tem. Misturá-las contaminaria o peso de
evidência com material não-publicado.

**A memória fecha um ciclo já projetado.** Quando o loop trava numa pergunta `CONTEXT`
("o que já foi tentado?"), ele consulta as memórias antes de escalar. O que você contou
uma vez não precisa ser perguntado de novo.

---

## Fase 2 — Destilação LoRA no Kaggle ⏳

**Gatilho:** ≥300 exemplos verificados. Não antes.

**O que a LoRA realmente aprende.** Formato, vocabulário do domínio e estilo de
raciocínio — **não fatos**. Fato continua vindo do RAG com citação. Um 12B tunado em
centenas de pares tende a alucinar *mais* se você esperar que memorize literatura. O
gate existe para provar isso empiricamente em vez de assumir.

**Dataset:**

| Tipo | Peso |
|---|---|
| pergunta → finding verificado com citações | 1.0 |
| pergunta escalada → **resposta humana** | 3.0 — sinal mais escasso e valioso |
| chunk → claim extraída | 1.0 — ensina o formato |
| probes de segurança | 2.0 |

Split hold-out **por pergunta**, nunca por exemplo — senão paráfrases vazam entre treino
e teste.

**Treino.** Unsloth sobre `unsloth/gemma-4-12b-it-bnb-4bit`. T4 é Turing: **fp16, não
bf16, sem flash-attn 2**. `r=16`, attn+MLP, `adamw_8bit`, gradient checkpointing,
`seq_len=2048`, `bs=1` com `grad_accum=8`. ~30–90 min por rodada.

**Empacotamento — adapter GGUF, sem merge.** `convert_lora_to_gguf.py` produz um adapter
de ~200 MB carregado em runtime com `llama-server --lora`. Evita baixar 24 GB de pesos
fp16, evita requantizar, e torna champion/challenger um **A/B de uma flag**
(`LlamaServer.lora_path` já existe). Plano B se a qualidade sobre a base Q5_K_M
decepcionar: merge fp16 + requantização, feito inteiramente no Kaggle.

**Gate de promoção.** Substitui o champion só se **as três** valerem no hold-out:

1. Acurácia ≥ base+RAG no `eval/goldset.yaml`
2. **Fidelidade de citação não piorou**
3. **Zero regressão** em `eval/safety_probes.yaml`

---

## Roadmap

| # | Entrega | Situação |
|---|---|---|
| 1 | Esqueleto, config, schema, store, CI mac+windows | ✅ |
| 2 | Cliente LLM, schemas Pydantic, decodificação restrita | ✅ |
| 3 | Adapter PubMed, ingest, retrieval híbrido | ✅ |
| 4 | Extração, grade, verificador de citação | ✅ |
| 5 | Fila, worker pool, `serve`, estratégias de busca | ✅ |
| 6 | `question.py`: geração, taxonomia, priorização | ✅ |
| 6.5 | Trilha exploratória: especulação mecanística + crítica | ✅ |
| 6.6 | `lithium chat` + memória com confirmação | ✅ |
| 6.7 | Laço especular → buscar → reancorar; composto/combinação/via | ✅ |
| 6.8 | Modo pesquisa ligado/desligado | ✅ |
| 6.9 | Memória de pesquisa: reflexão com portão de citação | ✅ |
| 7 | Loop de auto-resposta, juiz de suficiência, crítica | ✅ |
| 7.5 | **Notificação** — sem ela, "ele faz a pergunta" não tem valor prático | ✅ |
| 8 | `sync/publish` + Space — **privado**, e o export exclui memória de `chat`/`recon` | ⏳ último |
| 9 | Demais adapters — **RECUSADOS por medição**; entregou 4 bugs vivos | ✅ |
| 10 | `hypothesis` + `report` + scheduler completo | ⏳ |
| 11 | `safety/` + goldset + safety probes | ⏳ |
| 12 | Fase 2 — dataset, QLoRA Kaggle, adapter GGUF, gate | ⏳ |
| F0 | **Instrumentação + os dois tetos estourados** | ✅ |
| F1 | **Consentimento, as três travas, precedência e `safety/`** | ✅ |
| F2 | **O portão de `pattern`/`dead_end`** | ✅ |
| F3 | **Recuperação não-constante, bound em memórias, contra-evidência** | ✅ |
| F4 | **Notícia do corpus como fato + portão de contato com a literatura** | ✅ |
| F5 | Reflexão de profundidade 1 | ⛔ bloqueada: exige uma semana de operação real |
| F6 | **Portão de citação + `seeking` persistido** (aresta hipótese→hipótese RECUSADA) | ✅ |
| 13 | Enxugar a camada de prompts | ➡️ **absorvido pela Fase B** |
| 14 | Objetivo configurável | ➡️ **resolvido e substituído pelas Fases A–D** |
| **A** | **O objeto foco** — `directness` vira aresta `(claim, foco)`; `grade` ganha escala | ✅ |
| **B** | **Trocar de foco** — perfil em disco + re-lente; absorveu o item 13 | ✅ |
| **C** | **Reconhecimento web** — ele pesquisa, te conta, você autoriza a memorizar | ✅ |
| **D** | **Registro de fontes + adapter genérico** — pagou a dívida de fiação do item 9 | ✅ |
| **E** | **Plano de métricas** — o que medir para acompanhar a evolução do modelo | ⏳ plano |

As Fases A–D vêm de um plano aprovado, com decisões, custos e mutações por fase:
`~/.claude/plans/estou-pensando-numa-forma-iridescent-melody.md`. A tese em uma linha: **um
banco, um corpus, uma memória — o foco é uma lente sobre esse cérebro, não uma partição dele.**

---

## Primeiro contato real — o que só apareceu rodando

Até aqui o sistema tinha 999 testes e **zero linhas de dado real**. A primeira operação de
verdade — uma frente de busca, 50 papers do PubMed — encontrou dois defeitos em menos de
duas horas, e nenhum dos dois era alcançável por teste com dublê.

### `lithium run` nunca drenou nada

O comando documentado para execução manual montava `stop = asyncio.Event(); stop.set()` e
passava para `Daemon.run(stop=...)`. Como `Runner._worker` é `while not stop.is_set()`,
nenhum worker reivindicava uma única tarefa: ele imprimia *"daemon no ar: 1 workers"* e
saía. Sem erro, sem log, sem tarefa executada.

`Runner.drain(max_tasks=...)` **já existia**, com um docstring afirmando *"usada em testes e
no `lithium run`"*. A afirmação era falsa desde que foi escrita, e `--max-tasks` era um
parâmetro aceito que não chegava a lugar nenhum — o "botão que não configura nada" que este
repo recusa em toda revisão de código, sobrevivendo no caminho de execução manual.

Foi o primeiro obstáculo do primeiro contato: enfileirei uma varredura e nada aconteceu.

### A extração não era idempotente, e o desligamento provou

`recover_orphans` devolve à fila toda tarefa que ficou `running` quando o daemon morreu no
meio. Uma extração real leva ~3 min. Nessa janela, qualquer interrupção — Ctrl-C, queda,
sleep da máquina — fazia os chunks já processados serem extraídos DE NOVO na volta, com as
claims entrando duplicadas.

Duplicata aqui não é cosmética: `hypothesis_scoreboard` **soma** o peso das claims ligadas,
então a mesma evidência contada duas vezes empurra uma hipótese para cima do placar.

OBSERVADO duas vezes, e a segunda foi acidental e melhor que qualquer teste: **o computador
desligou** no meio da extração da fonte 5, depois de gravar 1 claim de 12 chunks. Ao
retomar, o log mostrou a correção operando contra uma órfã real —
`fonte 5: 1 de 12 chunk(s) já extraídos, pulando` — e as 9 claims novas entraram sem
duplicar a antiga.

A guarda é por CHUNK e não por statement, e a razão foi medida: o statement varia entre
execuções (temperatura 0,2, não 0), então casar por texto pegou 2 de 3 duplicatas reais. O
chunk é determinístico.

### O que o custo medido diz

O modelo de custo do repo previa 6,21 tok/s de decode. Medido em operação: **6,68 tok/s**,
7% de erro sobre uma estimativa derivada de três medições antigas — o modelo se sustenta.

A unidade de custo é o **chunk, não a fonte**: a primeira extração levou 117s com 2 chunks,
o ritmo estável ficou em ~3 min com fontes de 8 a 12 chunks. Claims por fonte variam de 1 a
10. Qualquer métrica de custo do item E tem de normalizar por chunk.

### Os dois portões discriminam, e de formas diferentes

Nas primeiras 4 fontes o portão 1 aprovou 19 de 19, o que levantou a dúvida certa: ele mede
alguma coisa, ou a busca da citação é frouxa? A fonte 5 respondeu — rejeitou 1 de 10. A
citação não estava literalmente no chunk e a claim caiu ali, antes de custar uma chamada de
verificação.

O portão 2 morde mais e de forma desigual: 100%, 83%, 33%, 100%, 88% por fonte. Um juiz que
lê caso a caso se comporta assim; um carimbo, não.

### Um defeito NÃO corrigido, e por quê

O campo `intervention` recebeu `"generalized anxiety disorder"` e `"GAD"` — a **condição**,
não uma intervenção. O paper era uma coorte epidemiológica sobre mortalidade por suicídio,
onde não há intervenção, e o extrator preencheu o campo com o que tinha à mão.

Importa porque `build_state` agrupa a tabela de cobertura por `intervention` e
`CLASS_KEYWORDS` mapeia intervenção para classe: condições ali dentro transformam a
cobertura em ruído. O conserto certo é o prompt admitir ausência de intervenção.

Ficou para depois da drenagem **de propósito**: mudar o prompt no meio faria as fontes
restantes serem extraídas sob regras diferentes das primeiras, e um corpus inconsistente é
pior que um corpus pequeno — a primeira série de métricas mediria duas coisas misturadas.

---

## Item 7 — o loop que consome a fila

622 testes, 9 mutações mortas. A lacuna mais visível do sistema: perguntas eram geradas,
classificadas e priorizadas, e ficavam em `OPEN` para sempre. Todo o andaime já existia
morto — `RESEARCHING`, `ANSWERED_AUTO`, `SufficiencyVerdict`, a tabela `findings`, e a
coluna `questions.rounds`.

### O padrão de falha que definiu o teste

Três desenhos independentes foram auditados e **os três entregaram a peça sem produtor**:
depois de aplicar qualquer um deles, `HANDLERS` não ganhava entrada, o scheduler não
ganhava `Job`, e o buraco continuava aberto byte por byte — com os testes novos passando,
porque chamavam o seam direto. Sexta ocorrência da classe neste projeto.

Pior: o ramo que **aprova** uma resposta — o único motivo do item existir — não era
exercitado por teste nenhum em nenhum dos três, e nos três ele explodia em produção. As
perguntas cuja evidência bastava eram as únicas que quebravam.

Por isso a última seção da suíte testa a fiação, e as quatro mutações correspondentes
morrem: sem entrada em `HANDLERS`, sem `Job`, handler de rodada como no-op, despachante
devolvendo lista vazia.

### As quatro decisões, cada uma de uma medição

**O juiz vem primeiro e não vê rascunho.** Julgar a evidência antes de sintetizar custa
23 s numa pergunta irrespondível; sintetizar antes gasta 44 s a mais por rodada, jogados
fora. Medido nos prompts reais: a versão ingênua gasta 198 s numa pergunta que nunca
fecha, esta gasta 91 s. E o repo já registra duas vezes que modelo pequeno auto-avaliando
o que acabou de escrever diz "suficiente" quase sempre.

**Rodada sem material novo não é rodada.** `search_claims` é função pura de (consulta,
corpus): medido, a rodada N+1 sobre um corpus imutável devolve a mesma lista, byte a byte,
com os mesmos scores. A impressão digital da evidência aborta a rodada **sem gastar
orçamento** — gastar converteria o teto de rodadas num relógio de parede, e a simulação de
30 dias dessa variante fechou **132 de 150 perguntas em silêncio**, sem ninguém ter lido
nenhuma.

**O gargalo é o despachante, não o LLM.** `vazão = em_voo × varreduras/dia ÷ rodadas`. Com
`em_voo=2` e varredura de 6 h dá 2,7 perguntas/dia contra as ~5/dia do `plan_tick`: a fila
satura. Com 4, ela esvazia — 16 rodadas/dia, ~38 min, 2,6% do dia.

**O teto da fila é 15, não 40.** O bloco "não repita" de `build_state` degrada muito antes,
e na ordem inversa da intuição: o `ORDER BY` põe `OPEN` primeiro, então pergunta aberta
expulsa primeiro a **já respondida**. Com 18 abertas o histórico já é cortado; com 35 o
gerador não vê nenhuma resposta e passa a repropor o que já resolveu.

Modelo de custo derivado das três medições do próprio PLAN, por mínimos quadrados:
`t ≈ 2,31 ms × tokens_entrada + 161 ms × tokens_saída` (prefill 433 tok/s, decode
6,21 tok/s, resíduo máximo 6,6%). O decode bate com o "~6 tok/s" medido independentemente.

### Dois bugs de reconciliação

`TaskQueue.recover_orphans()` **não** cobre o encalhe que importa: ele mexe em `tasks`, e a
pergunta é um segundo estado. Uma tarefa que chega ao dead-letter deixa a pergunta em
`RESEARCHING` para sempre e nada mais olha para ela.

E a subconsulta precisa do `IS NOT NULL`: `NOT IN` com um único NULL no conjunto devolve
NULL para **toda** linha, então uma tarefa sem `question_id` no payload desligaria a
reconciliação inteira, em silêncio, para todas as perguntas encalhadas.

### O que eu NÃO construí: a exportação para treino

`training_examples` tem zero escritores **e zero leitores**, e o LoRA é o item 12. Os três
desenhos vazaram material não-verificado para lá por três caminhos diferentes, e os três
foram medidos:

* o divisor de sentenças tinha teto de 8 fragmentos e **truncava em silêncio** — o nono,
  uma instrução de dose que nenhuma claim sustenta, entrava verbatim em `findings.text`
  com `all_sentences_supported = True`;
* `caveats` nunca passava pelo entailment por sentença e era exportado com `verified=1`;
* o divisor **fundia** proposições em prosa de dosagem (`mg`, `kg` na lista de
  abreviações), que é a prosa mais comum deste domínio — duas proposições compartilhando
  um veredito só.

Uma lição tem `--forget`; uma atualização de pesos não. A exportação entra quando tiver
consumidor e um portão medido — e as três constraints continuam escritas para esse dia.

### Ainda em aberto neste item

A **crítica adversarial do achado** não entrou. O precedente existe (`SpeculationCritique`)
e o risco de calibração também: na trilha especulativa a crítica rejeitou 100% das
hipóteses até eu separar "falha fatal" de "lacuna". Um crítico que rejeita tudo é
indistinguível, no verde do teste, de um que não roda. Fica como item próprio.

---

## Item 7.6 — o juiz destravado, e a crítica que não foi construída

677 testes, 7 mutações mortas.

### O loop que eu declarei fechado gravava zero achados

Rodado contra o modelo real (gemma-4-12b, servidor vivo), o juiz de suficiência devolveu
`sufficient: false` em **29 de 29** chamadas — sempre com `addresses_question_directly:
true` e 2 a 8 fontes independentes, `blocked_reason` vazio. Cinco perguntas, três
conjuntos de evidência, incluindo o head-to-head que o próprio juiz pediu na rodada
anterior. Com o head-to-head presente, ele pediu... um head-to-head.

Numa pergunta de puro fato, o próprio campo `missing` **reconhecia a resposta**: *"While
the current evidence provides specific HAM-A reduction points (6.2 and 4.8 points), there
is a lack of head-to-head comparison…"*.

**Causa raiz de uma linha.** `sufficient` era o único campo de `SufficiencyVerdict` sem
`Field(description=...)` e o único ausente da seção "What each field means" do prompt — e
é o campo que gateia o loop inteiro. É a mesma falha que este repo já pagou e documentou
em `critique_speculation.md`: *"se você exige prova de segurança antes de uma hipótese
sobreviver, só tratamento estabelecido passa"*. Duas vezes, o mesmo erro, em dois portões
diferentes.

A correção nomeia o que é ausência e o que é fraqueza. Refusa por **ausência**
(intervenção, desfecho ou população diferente; uma fonte só; número que não existe na
evidência) e nunca por fraqueza — "sem head-to-head", "amostra pequena", "eficácia
relativa desconhecida" vão em `missing` e são compatíveis com `sufficient: true`. E diz
por quê: a escassez nesta interseção é **estrutural** (RCT de TAG exclui bipolar, RCT de
bipolar trata ansiedade como desfecho secundário), então uma barra que exige o
head-to-head recusa tudo para sempre e o leitor recebe nada em vez de uma nota honesta com
os limites nomeados.

**Mas não com o texto que a auditoria propôs.** Ela sugeria "espere que a maioria das
perguntas seja suficiente" e "uma taxa de recusa perto de 100% significa que a barra está
errada" — dar a um juiz **por instância** um prior sobre a distribuição agregada. Com
isso ele deixa de julgar a evidência e passa a cumprir cota, e o mutante de carimbo
explorou exatamente essa brecha. Travado por teste: nenhuma frase de taxa-base no prompt.

E a calibração é testada **nos dois lados**, porque um probe que só verifica "deixou de
recusar tudo" não distingue "funciona" de "carimba tudo".

### Um bug que o teste de calibração achou

Com corpus vazio e um juiz complacente, `_answer` gravava um achado com
`citations_json = []` e marcava a pergunta como respondida. **Uma resposta sem fonte
chegando ao psiquiatra** é o pior resultado que este sistema pode produzir, e não pode
depender de o modelo ter respondido bem a uma pergunta sobre si mesmo. Agora há
`MIN_CITATIONS = 2`, determinístico.

### A crítica do achado: NÃO construída

Ela **realmente** pergunta algo que o juiz não pergunta, e isso foi provado, não suposto:
sobre a mesma evidência, o juiz disse `addresses_question_directly: true` (2/2) e a
crítica disse `answers_a_different_question: true` (2/2) sobre o texto escrito a partir
dela. Objetos diferentes — o juiz julga a evidência, a crítica julga o texto.

**Mas a taxa base é zero.** Em 10 achados gerados pelo prompt `answer_question` real, em
quatro regimes de evidência, a crítica marcou **0**. O gerador não comete o defeito: o
modelo usou espontaneamente o hedge do próprio prompt — *"Nothing in the corpus covers
whether these findings are generalizable… mechanistically I'd expect…"* — e sobre
evidência marginal escreveu *"The evidence does not provide a direct answer regarding…"*.
O `answer_question` atual já faz o trabalho.

E o desenho de **veredito global** rejeita 100%, incluindo 5 de 5 controles honestos, com
objeções **fabricadas**: *"fails to mention the risk of manic-switch associated with
antidepressant monotherapy"* num achado sobre quetiapina onde nenhum antidepressivo
aparece. A forma escopada (perguntas independentes) discrimina — 24/32, honesto 8/8 limpo
—, o que confirma o precedente do `PatternVerdict`: perguntas escopadas, nunca veredito
global.

Fica registrado como pronto para construir **se** a taxa base subir. O gatilho é
observável: um achado marcado pelo revisor como exagerado.

### O screen de segurança no achado, e a regra que se auto-anulava

`findings.safety_json` tinha zero escritores: o alerta determinístico aparecia no chat e
**não** no achado, que é o que o psiquiatra lê.

Ligar expôs um defeito na regra de negação da Fase 1. `_NEGATION` foi calibrada para o
registro de *restrição do usuário* ("o paciente não tolera valproato"), onde suprimir é
correto. Sobre prosa de **síntese** ela perde **9 de 14** alertas, e o pior caso é
auto-anulante:

```
"Pacientes que interromperam o lítio abruptamente tiveram mania de rebote"
    -> abrupt_discontinuation NÃO dispara  (negado por "interromperam")
"A parada foi feita abruptamente no braço ativo"
    -> dispara
```

O alerta que o próprio módulo chama de o mais importante da tabela era desligado **pelo
vocabulário que ele existe para pegar**.

Corrigido por `kind`: a supressão vale em `user`, `memory` e `evidence`, e não em
`reply`. Manter `evidence` com supressão é o que confina a mudança de verdade — *"the
patient denies lithium use"* num abstract é história clínica, e disparar ali é ruído, que
treina o revisor a ignorar o alerta seguinte.

E a lista vazia **não** é renderizada como atestado de limpeza: o screen é um casamento de
termo pequeno, não checagem de interação medicamentosa, e um "nada casou" afirmativo seria
pior que silêncio.

---

## Item 7.5 — Notificação

665 testes, 13 mutações mortas. `lithium/notify/` com backend macOS, Windows e ntfy.

### Injeção confirmada por execução, não por análise

O corpo do aviso carrega texto de pergunta escrito por um LLM, e `osascript -e` recebe
AppleScript. A carga

```
" & (do shell script "echo owned > /tmp/PWNED") & "
```

dentro de `f'display notification "{body}"'` **escreveu o arquivo no disco**, rc=0,
stderr vazio. É execução remota de código pelo caminho mais inocente possível.

Escapar aspas bloqueou as oito cargas — e escapar é uma linha de distância de não
funcionar: esquecer a contrabarra dá erro de sintaxe no AppleScript e o aviso **nunca é
entregue, sem ninguém notar**; um `\x00` no texto faz o próprio `subprocess` levantar. A
defesa tem de ser estrutural, então o texto vai por **argv** (`osascript -e SCRIPT --
título corpo`), onde nunca é parseado como script. Verificado: `-l JavaScript` no corpo
chega verbatim como argumento, não vira opção.

No Windows a superfície é **dupla** — o shell (`-Command` recebe fonte de script) **e** o
XML (o corpo do toast é XML, e um `</text><audio src=` injeta sem shell nenhum). Por isso
o texto vai por variável de ambiente e o nó é montado por `CreateTextNode`, nunca por
concatenação.

### O contrato é síncrono, e isso é a decisão de projeto

Um protocolo `async` empurra o implementador para o antipadrão: `wait_for` só cancela
corrotina que **cede o controle**, então um `subprocess.run` bloqueante dentro de um
`async def` trava o loop e o timeout nunca dispara — medido, 3,04 s de loop parado com a
entrega voltando como *bem-sucedida*. Sendo síncrono, o chamador é obrigado a confinar em
thread.

```
atraso máximo do event loop, 5 avisos
  create_subprocess_exec        0,78 ms
  to_thread(subprocess.run)     1,90 ms
  subprocess.run no loop      609,41 ms
```

E o `timeout=` no subprocess não é zelo: sem ele um filho pendurado custa **8,08 s** na
saída do processo contra 0,47 s, porque a thread do executor padrão não é cancelável e o
interpretador a espera. O diálogo de permissão de notificação do macOS na primeira
execução é exatamente esse cenário.

### Gatilho por estado derivado, e agrupado

A escalação é escrita em quatro lugares e o dead-letter em dois: pendurar um gancho em
cada um são seis call sites, e wiring em N call sites é a classe que este projeto repetiu
seis vezes. Um tick que compara o estado contra uma marca d'água tem **um** call site.

**Mas estado derivado sozinho não resolve tempestade** — medido: uma falha determinística
única produz **380 avisos**. O que resolve é agrupar, e agrupando a primeira execução
deixa de ser caso especial: vira só um delta grande, com um aviso.

**As duas marcas d'água óbvias estão quebradas.** `max(escalated_at)` não serve porque
`cli.py` promove N linhas num único UPDATE com `strftime('now')` — timestamps idênticos
**por construção**; com `>` perde-se aviso para sempre, com `>=` reavisa-se para sempre.
Contagem também não distingue "duas novas" de "uma nova e uma resolvida". A marca é o
**conjunto de ids**.

**Entre perder um aviso e repetir um, este sistema repete:** a marca só avança quando a
entrega saiu. A pergunta escalada é a coisa que o sistema existe para contar; um toast
duplicado é irritação, não perda.

**A frequência não é escolha livre:** o `dedup_key` do scheduler usa um balde de uma hora
(`iso(now)[:13]`), então qualquer Job com intervalo menor que 3600 s é silenciosamente
estrangulado para 1×/hora — o segundo disparo colide na chave, `enqueue` devolve `None`, e
o relógio avança como se tivesse rodado.

### O laço de amplificação, fechado em dois lugares

Um `notify_tick` que morre vira dead-letter; dead-letter é gatilho de aviso; aviso roda
`notify_tick`. O delta exclui `notify_tick` — e essa exclusão só é segura porque o handler
é estruturalmente incapaz de morrer (corpo inteiro em `try/except` com `log.exception`).
Sem isso, um `KeyError` no primeiro dead-letter de um tipo inédito com entrega falhada
mataria o canal **para sempre, em silêncio**, e o cadáver seria invisível ao próprio
gatilho.

### Três testes meus que não podiam falhar

A bateria de mutação pegou: meus testes de injeção liam a **constante** `_MAC_SCRIPT` e
montavam o argv num helper de teste, então trocar o corpo de `send()` por uma
interpolação — o RCE confirmado — passava verde. Mesma coisa no Windows. E o
`assert "CreateTextNode" in script` casava a linha do **título**, então a do corpo podia
virar `InnerText` sem ninguém ver.

Corrigido expondo `command()`/`env()` no backend: o teste passou a afirmar sobre o que o
backend **realmente executa**, e a asserção do XML virou sobre a *ligação* (toda
ocorrência da variável do corpo está dentro de um `CreateTextNode`), não sobre presença.

### Um bug do item 7 que esta auditoria achou

`Answerer._escalate` gravava `ESCALATED` **sem consultar vaga**, enquanto
`QuestionEngine.escalate` respeita o teto de 5. O loop de resposta triplica a taxa de
escalação, então a fila humana encheria — e fila cheia esconde as que importam, que é a
razão de o teto existir. Corrigido: sem vaga a pergunta fica represada e
`promote_escalations` a sobe.

### Uma limitação que muda o desenho

`osascript` devolve rc=0 **apareça o toast ou não**: não existe recibo de entrega. E sob
`launchd` a atribuição da notificação muda sem o código saber. Então o aviso é
redundância — `lithium questions` continua sendo a fonte de verdade, e o `ntfy` existe
por isso, não por conveniência.

---

## Item 9 — as quatro fontes recusadas, e os quatro bugs que elas expuseram

699 testes, 4 mutações mortas. **Nenhuma das quatro fontes foi construída.** Cada recusa
tem medição contra a API real.

### Europe PMC: adiciona ZERO literatura revisada

Com a mesma semântica de query (`[tiab]` no PubMed vs `TITLE_ABS:` no EPMC), paginando
até o fim nos dois lados:

```
pubmed=405  epmcMED=380  exclusivos=0
       264          250            0
        69           66            0
       129          117            0
soma:  867          813            0   (0,0%)
```

O EPMC-MED é **subconjunto estrito**. E o "40× de volume" que parecia justificar a fonte é
100% ruído de menção em full text: com os termos escopados em título/abstract, o EPMC
devolve **menos** que o PubMed (0,31× a 0,90×).

O que ele tem de exclusivo neste domínio são **resumos de congresso**: 315 de 320
registros sem PMID são `pubType: Abstract` — sem método, sem revisão, ~1,8 k caracteres.
`Abstract` não está no mapa de grade, então `design=None` e o LLM grada pelo texto — e o
texto de um resumo de congresso **lê como coorte**.

E o parser de query devolve **lixo silencioso, HTTP 200**, exatamente na forma da query de
prioridade 0,95 do sistema: `MESH:"bipolar disorder" OR TITLE_ABS:"bipolar I"` devolve
575.148 resultados, com operandos de 3.422 e 4.421. É dependente de ordem e não dá erro. O
agravante é que o EPMC **aceita a sintaxe `[MeSH]` do PubMed como alias**, então as queries
parecem portáveis, rodam sem erro, e estão erradas.

### ClinicalTrials.gov: 75% não tem resultado, e a prosa passa o portão 1

75,1% dos registros deste domínio não têm resultado postado. E o argumento decisivo não é
o peso: **a prosa de um registro é INTENÇÃO e ela atravessa o portão de citação literal** —
`quote_is_anchored` foi rodado contra texto real da API e aprovou, porque a citação *é*
literal. O portão não distingue intenção de resultado.

O registro mais direto do domínio inteiro tem n=3 e o próprio patrocinador o declara
inconclusivo — e entraria com `grade='rct'` = 0,85, superando um coorte publicado (0,55).

E o "canal de lacuna" que eu propus como alternativa foi medido e é caro: 12 sondagens (um
tick de `pursue`) encaminham 99 PMIDs → 392 chunks reais = **+43% do corpus inteiro**,
5,5 h de servidor de 1 slot, a cada 48 h, com 8% de precisão on-target — pior que os 16%
da varredura em massa que o próprio desenho recusava.

### openFDA: minha hipótese estava errada

Eu propus que o openFDA **populasse** a tabela de segurança. Refutado: o campo de
contraindicação **não gera uma única das 8 regras existentes**. Gerar dali derrubaria em
silêncio a teratogenicidade do valproato para esta indicação — um defeito de geração
relevante para segurança, não de qualidade. E `antidepressant_monotherapy`, que o próprio
módulo chama de "a razão de existir desta pesquisa", é ingerável.

O texto clinicamente útil é `boxed_warning`, que é **4,5× a 31,9× longo demais** para um
toast. E o resultado mais severo: passar texto de bula pelo `screen()` **satura a tabela de
alertas** — exatamente a falha que o módulo existe para prevenir.

### Base química: sem consumidor

Não é evidência sobre tratamento, e não existe consumidor no código hoje.

---

### Os quatro bugs vivos que só apareceram porque alguém tentou adicionar uma fonte

**1. `canonical_pmid` convertia PMCID no PMID de outro artigo.** `PMC7738613` →
`7738613`, que é um paper real de 1995 sobre linfoma não-Hodgkin no *J Clin Oncol*
(confirmado ao vivo no NCBI). Se esse número existisse no corpus, `gate_citations`
**aprovaria** e o quadro mostraria `[supported: PMID:7738613]` ao psiquiatra. É
exatamente o modo de falha que a Fase 6 fechou — "alucinar um identificador plausível e
real é pior que inventar um" — produzido **deterministicamente pelo meu próprio regex**, e
o modelo escreve PMCID com frequência. Fechado com fronteira à esquerda.

**2. `fetch_source` aceitava qualquer `kind`.** `AVAILABLE_SOURCES` não está nesse caminho
— ela filtra query planejada e edita uma dica de prompt. Medido: **uma linha** em
`daemon.py` mais um `fetch_source {"kind": "fda"}` fez o campo de contraindicação de uma
bula, cujo conteúdo literal é *"None with olanzapine monotherapy…"*, virar uma claim com
`grade='rct'`, `directness='partial'` e peso **0,408** na view `claim_weight`. Agora há
`EVIDENCE_KINDS` e um portão de runtime no handler que ingere — a única checagem que
sobrevive a um adapter chegando por CLI, por handler novo ou por `Context.sources`.

**3. O mesmo artigo por dois `kind` conta como duas fontes independentes.** 100% dos PMIDs
que o PubMed colhe neste domínio também estão no EPMC, e `sources` tem
`UNIQUE(kind, external_id)` — então o mesmo paper entraria duas vezes, seria extraído duas
vezes, e chegaria ao juiz como **duas fontes concordando**, satisfazendo `MIN_CITATIONS`
com um artigo só. Latente hoje (fonte única), travado por teste.

**4. `Strategy.sources` é botão de configuração que não configura nada.** Declarado desde
o dia 1 e nunca lido: `harvest_query` fixa `'pubmed'` quatro vezes. Documentado no próprio
campo, porque um botão inerte é pior que ausência — alguém o ajusta e conclui que ajustou.

### E uma trava que a auditoria mostrou ser satisfeita por refactor

O guard proposto contava ocorrências do literal `'pubmed'` no fonte. Medido: **mover o
literal para uma constante** deixa o guard verde com zero mudança de comportamento, e o
adapter novo continua sendo código morto. A trava aqui é comportamental e declara a dívida
em vez de fingir que a pagou.

---

## Dívidas técnicas registradas

**A fila automática não tem teto.** A fila humana limita a 5 com represamento; a
automática cresce sem limite, ~5/dia pelo `plan_tick`. Não é só consequência do item 7
estar pendente — **degrada sozinha**: passando de 40 perguntas, o `build_state` trunca a
lista de "não repita" e o gerador perde visibilidade do que já perguntou, sobrando só o
dedup por alvo como guarda. Quanto mais tempo roda nesse estado, pior fica.

**`lithium ask` é uma armadilha hoje.** Responde `✓ pergunta enfileirada` e diz que o
daemon vai rotear — o que é verdade e dá impressão errada. Uma pergunta `FACTUAL` entra
na fila que nada consome; uma `PREFERENCE` é escalada para a fila humana, ou seja, volta
para você mesmo responder. Some com o item 7; até lá, deveria avisar.

---

## Fase 0 (do plano de memória) — concluída

Instrumentação de custo (`llm_calls`, uma linha por **POST HTTP**), `budget_guard` que aborta
antes da chamada, e os dois tetos que estavam estourados. `lithium tokens` expõe o resultado.

### O que a primeira medição real corrigiu

A faixa de "45–70 s por chamada", usada em todas as estimativas anteriores deste documento,
**não existe**:

| etapa | entrada | saída | tempo |
|---|---|---|---|
| `generate_questions` | 1.091 | 274 | 43,8 s |
| `critique_speculation` | 1.293 | 229 | 41,0 s |
| `generate_speculation` | 2.219 | 695 | **117,9 s** |

O tempo é ~linear em tokens de **saída**, a ~6 tok/s no M1 Pro. **O botão de custo é `max_tokens`,
não contagem de chamadas** — o que reordena qualquer otimização futura: reduzir `max_tokens` de uma
etapa verbosa vale mais que eliminar uma chamada curta.

### O teto que matava a trilha especulativa

`build_state()` não tinha `LIMIT`, e `claims.intervention` é texto livre de um 12B — "quetiapine",
"quetiapine XR" e "adjunctive quetiapine" são três linhas, então a cardinalidade cresce com o corpus
e **não converge**. Medido com 200 intervenções distintas:

```
teto disponível para o prompt : 4.864 tokens
antes (sem LIMIT)             : 6.769 tokens  -> ESTOURA
agora (LIMIT 40)              : 3.148 tokens  -> cabe
```

O cruzamento fica em ~103 intervenções distintas; uma varredura completa produz 150–400. O
llama-server devolvia 400, o cliente corretamente não repete 4xx, e a task ia para dead-letter:
**a trilha especulativa morria em silêncio conforme o corpus amadurecia**, e nenhum teste pegava
porque o sintoma só aparece com corpus grande. `test_phase0_budget.py` agora falha antes da
mudança e passa depois.

Bloco truncado passou a **declarar o que esconde** ("+160 intervenções de menor peso não mostradas
— este quadro é parcial"), que é a resposta direta à falha de fragmento-incompleto relatada no
artigo: o modelo precisa *saber* que o quadro é parcial.

### Contratos de prompt: a invariante, não a string

`tests/test_prompt_contract.py`. Dois mecanismos, e a distinção entre eles é o ponto.

**A invariante trava o conceito.** `chat.md` escreve "manic-switch", `extract_claims.md`
escreve "manic switch", `critique_speculation.md` escreve "switch risk" — as três estão
gramaticalmente corretas, e o `assert "manic-switch" in prompt` que existia punia edição
legítima enquanto era cego à única mudança que importa: o risco de virada **sair** de um
prompt que gera conteúdo substantivo. Agora são cinco prompts obrigados a nomeá-lo, cada
um com o motivo escrito, e `test_every_prompt_is_classified` faz um prompt novo **falhar
até ser classificado** — sem isso a lista só cobriria o que já foi lembrado.

`critique_speculation` entrou na lista obrigatória porque a regressão já aconteceu: a
linha que calibra "nunca discute risco de virada" como *lacuna, não defeito* é o que
separa 3/3 especulações sobreviventes de 0/3.

**Os golden renders travam o texto.** Placeholders viram sentinelas (`«evidence»`), e a
normalização colapsa espaço em branco *dentro* do parágrafo preservando a fronteira
*entre* parágrafos. Verificado nos dois sentidos: refluir todos os parágrafos de
`chat.md` a 55 colunas **passa**; trocar `must never be blurred` por `must sometimes`
**falha**. Isso é pré-requisito do item 13 — sem ele, a primeira coisa que a reescrita de
710 linhas faz é quebrar tudo, e a reação natural é regenerar em bloco, que é perder o
baseline.

**Um teste que eu escrevi tautológico e troquei.** A primeira versão derivava o conjunto
de placeholders esperados do próprio `.md`, então apagar `$evidence` de `chat.md` removia
o placeholder *e* a expectativa dele: o teste **não podia falhar**. `Template.substitute`
levanta em kwarg faltando mas ignora kwarg sobrando, então o turno renderizaria sem bloco
de evidência — o modelo respondendo de memória paramétrica num domínio onde toda
afirmação precisa de PMID. Agora o contrato de injeção é dado literal (`INJECTS`), e os
três testes pegam o caso.

**Pendência registrada:** `lithium/` está untracked (não ignorado). Enquanto estiver, o
golden não tem baseline versionado e `LITHIUM_UPDATE_GOLDEN=1` não deixa rastro.

### Os outros três consertos

- **`chat._history()` sem budget.** 24 mensagens × `max_tokens=1536` chegavam a ~18.400 tokens
  contra uma janela de 8192, e o REPL não tinha `try/except`: a sessão morria com traceback. Agora
  corta por **tamanho** (12 k caracteres), nunca descarta o turno atual, e `history_turns` caiu de
  12 para 6.
- **`dedup_key` instável.** `str.__hash__` é salgado por processo e o repo não fixa
  `PYTHONHASHSEED`, então a chave mudava a cada restart do daemon — "a mesma query não roda duas
  vezes no dia" era **falso**. Trocado por `blake2s`, com teste que roda em três subprocessos.
- **O medidor não pode causar gasto.** O hook de uso bufferiza em memória, faz flush por
  `asyncio.to_thread` e engole toda exceção. Handlers rodam awaited na thread do event loop; uma
  exceção no caminho de medição converteria uma resposta de LLM bem-sucedida em falha de task e
  retry — o contador de tokens gastando tokens. Testado com um hook que levanta e um store sem
  schema.

---

## Fase 1 — Consentimento, travas de invariante, precedência e `safety/`

470 testes (eram 370). O desenho passou por auditoria adversarial em seis frentes antes
da implementação, e **o que foi implementado é a auditoria, não a proposta original** —
oito defeitos críticos, seis deles da mesma classe.

### A classe de defeito que dominou: teste que não pode falhar

Seis dos oito críticos eram testes tautológicos, escritos **depois** de o briefing
avisar explicitamente sobre a armadilha. Os dois piores:

- **`test_screen_never_removes_content`** cujo corpo inteiro era
  `assert inspect.signature(screen).return_annotation in ("list[Alert]", list)`. Com
  `from __future__ import annotations`, isso é a string que o autor digitou — o teste
  derivava a expectativa da exata linha que deveria policiar. Passava com um `screen()`
  que apagava os segmentos de evidência in-place.
- **O contrato de injeção de prompt** derivava os placeholders esperados do próprio
  `.md`: apagar `$evidence` removia o placeholder *e* a expectativa dele.

Por isso toda trava desta fase foi validada por **mutação**: o defeito é construído, e o
teste tem que ficar vermelho. **36 mutações, 36 mortas.** Um teste que não demonstrou
falhar não conta — ele certifica o que não verifica, que é pior que não existir.

### O decaimento temporal que passava verde no ponto mais central

A auditoria construiu um `CASE WHEN julianday('now') - julianday(c.extracted_at) > 30
THEN 0.5 ELSE 1.0 END` **dentro da view `claim_weight`** e a suíte inteira passou. Três
causas somadas: a allowlist isentava a view das regras de aritmética (ela precisa
multiplicar); o teste de fórmula usava **continência** de substring, então acrescentar
fator passava; e a checagem de tempo só olhava `ORDER BY`, que a view não tem.

Fechado por três frentes: igualdade **exata** da expressão de peso, varredura de tempo
sobre o statement inteiro, e um teste numérico sobre as **324 combinações**
(9 grades × 4 directness × 9 confidences) com `extracted_at` de dois anos atrás — para
que um decaimento por idade não se esconda atrás de um fator 1.0 nas fixtures.

### O desempate por colheita sem rastro sintático

A outra brecha: `sort()` **sem** `key=` sobre tuplas `(score, timestamp, hit)`, com o
timestamp vindo de um dict lateral. Não tem `ORDER BY`, não tem `key=`, e `ClaimHit` não
ganha campo nenhum — invisível para qualquer trava baseada na *forma* da ordenação.

Fechado pela **fonte do dado**: nenhum statement que amarre `claims` pode SELECIONAR
coluna de tempo. Um ranking não pode ler o que nunca foi buscado.

E `retrieval.py` deixou de calcular o peso: agora dá JOIN em `claim_weight`. A invariante
não é "chame a função certa", é **o peso de claim tem uma origem só** — com duas
implementações vivas, o ranking do chat discorda do placar de hipóteses no dia em que os
pesos mudarem, sem nada reclamar.

### A colisão central do domínio, e onde a anotação mora

**Lítio é o agente melhor evidenciado deste domínio e exige monitoramento sérico.** Uma
memória dizendo "o usuário evita fármacos com monitoramento sérico" — exatamente o que
`detect_memory` foi feito para propor — cria a tentação de omitir a evidência mais forte
do corpus. E a omissão é **invisível**: não há nada na resposta que mostre que um paper
foi deixado de fora.

`chat.md` ganhou a regra de precedência (*uma preferência governa o que você recomenda
considerar; nunca governa o que você relata que a literatura diz*), e a anotação vive em
**bloco próprio**, depois de `$evidence` — nunca colada na linha da claim, porque para o
modelo uma etiqueta grudada é indistinguível de rebaixamento: seria o gesto proibido,
cometido com outra sintaxe. `_constraint_notes(evidence: str) -> str` recebe o bloco
pronto e devolve texto novo; não tem como filtrar o que não devolve.

**A fixture semeia 8 claims contra `evidence_k=6`, e isso não é detalhe.** Com 2 claims,
qualquer defeito que trunque o top-k para um k entre 2 e 6 é estruturalmente invisível —
a auditoria provou injetando um `hits = hits[:3]` condicionado à presença de memória e
obtendo 389 testes verdes.

**O buraco cross-fármaco:** com lista plana de termos por memória, uma resposta que cite
*qualquer* um deles silencia o alarme para *todos*, inclusive para o omitido. Por isso o
casamento é por **conceito → formas de superfície**, e cada claim é anotada com o
conceito dela.

### `safety/`: o que ele não dispara importa mais

Determinístico por casamento de termo, porque um casamento de termo não pode ser expulso
de um top-k, não pode decair, e não pode ser superado por importância — e precisa
existir **antes** da Fase 4, senão a resposta tentadora vira "fatos de segurança serão
recuperados porque são importantes".

Três correções contra a proposta original, todas por falso positivo medido:

- **Memórias saem do screen.** Uma restrição gravada produzia dois alertas de severidade
  alta em **todo** turno, inclusive em "bom dia" — memórias não expiram, então o ruído é
  permanente, e o sistema fabricava os próprios falsos positivos.
- **Co-termos por frase, não por token.** `parar` é um dos verbos mais comuns do
  português e `abrupt` é vocabulário estatístico corrente: "Abrupt changes in the primary
  outcome ... lithium arm" disparava mania de rebote, severidade alta.
- **Co-termo avaliado sobre a união do turno.** Por segmento, "posso parar de tomar de
  uma vez?" nunca cruza com um bloco de evidência sobre lítio, e o alerta mais importante
  da tabela nunca dispara no caso em que deveria.

Mais a colisão com o nome do projeto: `lithium` numa mensagem sua é o software; num bloco
de evidência em inglês biomédico é o fármaco. `lítio` vale nos dois.

### Consentimento

`decline()` era **inerte**: gravava `confirmed = 0` e o detector lia uma view filtrada por
`confirmed = 1`. Você recusava e era perguntado de novo no turno seguinte. Corrigido com
filtro em Python **depois** da resposta do LLM — as duas alternativas custam caro:
alargar a view poria texto rejeitado dentro de "o que você sabe sobre este usuário", e
listar as recusas no prompt custa **+3,2 k tokens com 150 recusas (9× o prompt estático,
39% da janela), todo turno**. Mais `--allow`, porque recusa permanente é esquecimento sem
aviso.

`ON CONFLICT DO NOTHING` era no-op (não havia UNIQUE). O índice é **parcial** e mora
**fora do `executescript`**: dentro dele, um IntegrityError num banco com duplicatas
abortaria o script inteiro — inclusive as views seguintes — e derrubaria
`lithium memories`, a ferramenta para resolver as duplicatas. Degradação avisada +
`--forget-duplicates`, porque sem saída praticável o estado degradado se autoalimenta
enquanto o daemon pesquisa.

Detalhes que a auditoria corrigiu: **NFC, não NFKC** (NFKC colapsaria `10²` com `102` e
`Li₂CO₃` com `Li2CO3`); **merge de proveniência** na rederivação, senão deduplicar
subdeclara o suporte de um `pattern`; e o termo `source` saiu da chave do índice, porque
o índice é parcial em `source='research'` — quem separa os regimes é o **predicado**, e
escrever o contrário seria um comentário falso permanente numa migração.

### Dois bugs vivos, achados de raspão

- **`question.add()` chamava `embed()` antes de qualquer INSERT, sem guarda.** Embedder
  fora do ar significava pergunta nunca gravada e nunca escalada — ausência, que ninguém
  nota. Agora degrada o dedup e persiste: uma paráfrase repetida na fila é visível e
  reversível; uma pergunta perdida não.
- **`answer_origin='human'` tem dois escritores, não um.** Eu afirmei "só o CLI" num
  docstring e o teste estático me desmentiu (`QuestionEngine.answer_from_human`). O
  conjunto agora é registro literal: `training_examples` dá peso 3.0 a material humano, e
  uma síntese de LLM rotulada assim não tem `--forget`.

### O que ficou de fora, de propósito

`memories.embedding` continua sem leitor — ganha um na Fase 3 como dedup semântico.
Registrado como ativo latente, não bug.

---

## Fase 2 — O portão de `pattern` e `dead_end`

520 testes. Desbloqueia as Fases 4, 5 e 6.

### A loteria, reproduzida antes de consertar

O plano descrevia o risco; medi o fato. Com claims 1–3 verificadas e uma hipótese id=3:

```
verify_claim_ids([3, 9999]) -> [3]
  o '3' veio de [hypothesis 3] no prompt; claim 3 existe e é verified=1
  gravado? True -> Lesson(id=1, text='padrão inventado', kind='pattern')
```

Três coisas somadas: `recent_activity()` **nunca imprimia id de claim** (o modelo tinha
de adivinhar), ids de hipótese e de claim vivem no mesmo espaço numérico, e
`verify_claim_ids` **filtrava** em vez de reprovar — resultado não-vazio bastava. O
portão segurava por **inanição de informação, não por projeto**, e o teste que o cobria
usava 9999/8888: verificava só onde adivinhar falha.

Agora, medido:

```
claim que existe, verified=1, mas NÃO estava em shown -> rejeitado, 0 lições
[real, 9999]                                         -> rejeitado, 0 lições
id vindo de [hypothesis 3]                           -> "índice [3] é hypothesis, não claim"
citação honesta                                      -> gravada
```

### O que a auditoria adversarial mudou no desenho

Quatro frentes, 12 defeitos críticos. Três valem registro porque mudaram decisões:

**A trava da Trava 3 estava a um caractere de vazia.** O teste de ordenação inseria a
claim mais forte primeiro, então só pegava `ORDER BY c.id DESC`. Com `ORDER BY c.id ASC`
— ordem de colheita pura — a suíte inteira passava, e o bloco entregava **doze linhas,
todas `opinion/extrapolated`, zero meta-análise**, sob o cabeçalho "claims verificadas,
cite estas". `claims.id` é `INTEGER PRIMARY KEY`, logo proxy exato de `extracted_at`, e
`TIME_TOKENS` não o vê. Fechado por trava nova sobre `ORDER BY ... id` em statement que
amarra claim — por **alias**, porque `ORDER BY s.id` na consulta de fontes áridas ordena
*fontes*, e ali recência de tentativa de colheita é o sinal certo.

**Mostrar 12 índices válidos piora o problema se o portão só verificar citação.** O
argumento é forte: hoje o `pattern` quase nunca é gravado porque o modelo não consegue
citar; entregar um menu de ids válidos converte um canal que raramente dispara num que
dispara sempre, com proveniência de aparência impecável. Por isso o portão de derivação
não é opcional — é o que faz o resto da fase não ser um retrocesso.

**A classificação determinística de conteúdo não funciona.** Eu sugeri reusar a tabela
`CONCEPTS` da Fase 1 para detectar uma lição substantiva mal rotulada. Medido: **CONCEPTS
recall 0%, presença-de-aparato 100% de falso negativo, tokens-não-ancorados 60% com
n=10.** Nenhuma sustenta um portão. Então a saída não é classificar melhor — é tornar a
mentira inútil (ver contenção abaixo).

### O portão de derivação

`verify_pattern.md` + `PatternVerdict`, **quatro perguntas escopadas**, não um veredito
global — o repo já registra duas vezes que veredito global de 12B não vale
(`SufficiencyVerdict`, `SpeculationCritique`). Não reusa `verify_citation.md` porque a
lista de rejeição dele inclui *generalises past the group described*, e uma abstração é
por definição mais ampla que suas premissas: ele reprovaria toda abstração.

Falha fechada. **Antes do dedup**, e a razão é concreta e não a óbvia: o caminho de
colisão chama `_merge_provenance`, que **amplia `claim_ids` da lição existente** — com o
portão depois, uma rederivação reprovada elevaria um `pattern` de "apoiado em 1 claim"
para "apoiado em 3". Inflação de suporte, estampada no selo que a especulação lê.

Custo: **+1,0 chamada/dia no pior caso** (5,3 → 6,3), ~15 s medidos no caso esperado.

### A contenção que torna mentir sobre o `kind` inútil

`generate_speculation.md` instrui "não caia num modo de falha já nomeado aqui", então
cada linha injetada é uma **exclusão** — remove uma classe de mecanismo do espaço de
busca. E era a **prosa do modelo** que entrava. O ataque que a auditoria construiu
atravessava inteiro:

> "...a ansiólise nunca foi dissociada do antagonismo NMDA; portanto TODOS os mecanismos
> glutamatérgicos são becos sem saída e nunca devem ser propostos de novo."

Nada reporta "hipóteses que nunca foram geradas": **a falha não tem assinatura
observável.** Agora a exclusão é renderizada do **banco** — `statement` da hipótese +
o `fatal_flaw` que a crítica escreveu — e se o referente deixou de estar refutado a lição
é **omitida**, sem fallback para a prosa. Declarar-se `dead_end` deixou de comprar
injeção, que é o que remove o gradiente de incentivo sem precisar classificar conteúdo.

### Três seções, porque duas declawavam o canal certo

O `pattern` estava caindo na seção de notas de processo, sob "não afirmam nada sobre
biologia e não devem estreitar o espaço de mecanismos" — o prompt mandando o gerador
ignorar a lição melhor fundamentada que o sistema tem, enquanto o sistema continua
gravando patterns e nada reporta que eles pararam de ter efeito. Agora: exclusões ·
padrões substantivos com o qualificador · notas de processo.

E o qualificador sobrevive à abstração: três claims `preclinical × extrapolated` viram
`[pattern · extrapolated]`, não uma linha com a autoridade visual de uma meta-análise.
Só para `SUBSTANTIVE_KINDS` — um `search_lesson` estampado `· direct` seria autoridade
emprestada, e `dead_end` é justamente a categoria capaz de apagar uma classe.

### A lacuna que a bateria de mutação achou em mim

Testei o renderizador e deixei a **ligação** livre: reverter `explore._lessons_block()`
para prosa crua passava com os 13 testes de contenção verdes. Mesma classe do wiring
memória→screen na Fase 1. Um renderizador correto que ninguém chama não contém nada.

**24 mutações nesta fase, 24 mortas** — incluindo o defeito exato que a auditoria
construiu (call site checando uma das quatro respostas com `pattern_is_entailed`
perfeita) e as duas que passavam verdes no desenho original.

---

## Fase 3 — Recuperação não-constante, bound em memórias, contra-evidência

546 testes. 15 mutações, 15 mortas.

### O ranking de lições estava congelado desde o dia 1

`relevant_lessons` tinha um único chamador e ele passava um literal fixo, então o ranking
era **idêntico em toda rodada pela vida do sistema**: as mesmas oito lições em todo prompt
de especulação, para sempre. Não era recuperação por relevância — era injeção de um
conjunto congelado, e cada rodada nova passava pelo filtro das conclusões das anteriores.

Agora a consulta é derivada do estado: uma por lacuna (`untouched` + `conflicts`),
`untouched` ordenado **antes** de fatiar, e nunca de um digest sintetizado por LLM — isso
poria a conclusão em cache do sistema no comando de selecionar quais das próprias crenças
ele enxerga.

**Rodízio, não RRF.** Escolhi RRF e a auditoria mostrou que a garantia que eu queria dele
não existia: o teste que dizia "uma lacuna cuja melhor lição é globalmente mediana ainda
recebe seu slot" passava por causa da **cota por kind**, não da fusão. Bastava dar às
concorrentes o mesmo `kind` da lição solitária para a lacuna ficar sem resposta nenhuma.
Rodízio (top-1 de cada lacuna antes de qualquer segunda) torna a garantia estrutural.

E um bug do meu próprio primeiro desenho, achado por teste: escrevi `untouched` antes de
`conflicts` e cortei em 8. Num corpus jovem há muitos rótulos vazios, então eles enchiam
os oito slots e **nenhum conflito entrava nunca** — perda silenciosa por ordem de lista.
Conflito é o sinal mais informativo dos dois: significa que há dado e ele discorda.

### O furo de wiring, pela terceira vez

Reverter o *call site* (`lessons="(none)"`, ou pegar só a primeira linha do bloco) passava
com a suíte inteira verde: as consultas eram embedadas, a cota aplicada, o renderizador
rodava — e o resultado ia para o lixo. Aconteceu na Fase 1 (memória→screen), na Fase 2
(contenção de `dead_end`) e aqui. Agora os testes afirmam sobre `llm.prompts[0]`, o texto
que de fato chegou ao modelo, e as três formas do defeito ficam vermelhas.

### A coerção que enterrava restrições por construção

`_detect_memory` coagia `kind` inválido para `'fact'`. Parecia conservador — "perder a
memória inteira por um rótulo errado seria desproporcional" — e virou o contrário quando o
bloco de memórias passou a ser ordenado por tipo: `'fact'` é o **último** tier, então uma
restrição rotulada errado ia para o fim da fila *por construção*. A coerção e a ordenação
se combinavam justamente contra a categoria mais crítica, sem deixar rastro.

Agora rejeita e loga. Perder uma proposta é recuperável e visível; enterrar uma restrição
é invisível.

### Tetos em caracteres, não em contagem

`_memory_block` não tinha WHERE, LIMIT nem ORDER BY. A primeira correção usou teto por
**contagem** — o mesmo erro que `HISTORY_CHAR_BUDGET` já havia corrigido uma vez — e a
auditoria mediu a consequência: o bloco de colisões embute o texto inteiro da restrição
uma vez por claim que colide, então com dez restrições de 600 letras o turno mede 10.137
tokens contra uma janela de 8.192. Com 2.000 letras o `budget_guard` levanta em **todo**
`send()`, e `/novo` não pode ajudar porque o prompt de sistema sozinho já não cabe: chat
permanentemente morto.

### Contra-evidência: `evidence_links` estava morta, e o assento tem limite declarado

Zero referências Python no repo — nem escritor nem leitor — então `support`/`contra` no
`hypothesis_scoreboard` eram permanentemente 0 e o canal de polaridade nunca funcionou.

O assento reservado corrige duas armadilhas medidas: sem piso, ele foi preenchido por
*"anticoagulação oral não reduz mortalidade em fibrilação atrial"* **deslocando um RCT no
assunto**, apresentado sob o cabeçalho de contra-evidência; e truncando pela cauda, o canal
**expulsava contra-evidência** — 8% das buscas trocavam uma contra-claim por outra, ganho
líquido zero.

**O critério de aceite foi entregue depois, e a dívida está paga.** Na Fase 3 eu não
consegui: o plano pedia "negativa em 14ª chega ao top-5", e a repescagem era por janela de
posição. A razão medida era que `_minmax` espalha a relevância linearmente pelo conjunto de
candidatos, então o último recebe sempre exatamente `MINMAX_FLOOR` — a negativa no assunto
em 14ª e uma negativa de cardiologia dão as duas `0,0500`. **`relevance` é derivada de
posição, não de pertinência.**

O sinal que faltava estava disponível o tempo todo e era descartado: `search_chunks` calcula
um `vector_ranking` já filtrado por `MIN_COSINE` e **joga fora, na fusão RRF, de qual
ranking cada candidato veio**. Esse booleano é o único sinal do sistema que não deriva de
posição na lista final. Medido:

```
negativa no assunto (14ª de 14)   cosseno 0,600   passa MIN_COSINE
negativa de cardiologia           cosseno 0,000   nunca entra no ranking vetorial
```

A de cardiologia só chegava ao páreo por sobreposição de token no BM25 — que é exatamente
o que a doutrina de `MIN_COSINE` diz que o BM25 faz. Com o portão semântico o alcance passa
a ser a cauda inteira: profundidade deixa de importar, porque o que qualifica é o
julgamento do ranker semântico e não a posição numa lista ordenada por peso.

Duas coisas que a bateria de mutação achou nesse caminho, e as duas eram minhas:

* **A fixture da Fase 3 nunca semeava embedding.** `add_chunk` não escreve vetor, e os
  testes não chamavam `set_chunk_embedding` — então o ranking vetorial estava sempre vazio
  e toda aquela suíte exercitava BM25 puro. O único sinal semântico do sistema nunca era
  testado, o que explica por que só a posição parecia disponível.
* **O teste do portão não podia falhar.** Com bag-of-words, léxico e semântico são quase
  colineares, então a claim fora do assunto nem chegava ao páreo e desligar o portão
  passava verde. O caso que importa em produção — BM25 traz o que o embedding não
  endossaria — só é construível com vetores ditados.

**Limite que permanece:** com bge-m3 os cossenos começam em ~0,6 e `MIN_COSINE` é 0,45,
então em produção o portão admite mais do que nesta medição. É estritamente melhor que a
janela de posição — julgamento semântico em vez de posição — mas não é um filtro de tópico
calibrado, e `MIN_COSINE` foi calibrado errado duas vezes antes justamente por tentar ser.

### O que eu NÃO construí

O portão global de truncamento (`_checked` recusando qualquer `str` multilinha). A
auditoria provou em produção que ele converte artefato de espaço em branco em falha
determinística na trilha de **maior volume**: um `<Journal><Title>` de XML formatado tem
`\n` interno, `UndeclaredTruncation` não é `LLMError`, escapa do handler, e a task queima
as tentativas até o dead-letter. A regra "todo bloco truncado declara o que esconde" fica
— aplicada bloco a bloco, como nas Fases 0 e 2 — sem o portão que a impõe pelo tipo.

---

## Fase 4 — a medição mudou o que a fase é

558 testes, 7 mutações mortas. **O escalar `surprise` e o gatilho híbrido NÃO foram
construídos**, e isso é resultado de medição, não de escopo cortado.

### Por que o gatilho por soma de surpresa não entra

Medido com o `Scheduler` real, na vazão que a aritmética deste próprio documento deriva
(160 claims/dia), 30 dias, ticks de 30 min:

```
T=6    -> 29 disparos, mediana 24,0 h
T=24   -> 29 disparos, mediana 24,0 h    <- IDÊNTICO a T=6
T=96   -> 14 disparos, mediana 52,0 h
T=1000 ->  9 disparos, mediana 72,0 h    (= o cronograma fixo)

piso 0 h -> 57 disparos | 12 h -> 53 | 24 h -> 29 | 48 h -> 14   (T=24 fixo)
```

**O limiar é inerte: quem decide a cadência é o piso de intervalo.** O que seria entregue
é literalmente "encurtar o cronograma fixo para 24 h" com limiar, acumulador, ledger e
simulador em volta — e o desenho rejeitava a opção 24 h fixo por "pagar 3× para sempre".

E o espelho, que ninguém estava olhando: o gatilho é **auto-extintor**. Com T calibrado no
corpus jovem, o intervalo deriva para 207 h (8,6 dias) quando o corpus amadurece, porque
`frontier` só pode cair — uma vez que o grupo tem uma claim `direct`, é 0 para sempre — e
`opposition` cai conforme o balanço estabiliza. A reflexão pararia em silêncio.

```
                          jovem    maduro
exato (sem normalizar)     61 h ->  154 h
dobra Python (a spec)      72 h ->  207 h
```

### Por que o escalar `surprise` não entra

**Satura antes de servir**, medido pelo `_persist` real: N=10 → 100% das linhas em
`magnitude = 1.0`; N=100 → 99%; N=400 → 83%. Só a partir de ~4.000 começa a discriminar.
Nos regimes em que este sistema vai viver, um limiar sobre ele dispara em tudo.

**Estreia e notícia colidem no mesmo máximo.** Cinco grafias de quetiapina com o mesmo
achado — `quetiapine`, `quetiapina`, `quetiapine XR`, `adjunctive quetiapine`, `Seroquel`
— dão `frontier = 1.0` cada uma, valor **numericamente indistinguível** de um salto
`extrapolated → direct`, que é a notícia mais forte que este sistema pode receber. Como
`intervention` é texto livre de um 12B, ~93% dos eventos de surpresa máxima são artefato de
identidade de string.

E a premissa do meu próprio briefing estava errada por um fator de três: eu disse que
normalizar a intervenção era "o detalhe que decide tudo". `LOWER(TRIM())` move o intervalo
de disparo **1,17×**, não 3×. O fator 3× existe, mas é maturidade do corpus com a mesma
política (2,87×). Também medi `CONCEPTS` para este uso (recall 21,2%, e 100% dentro dos 7
conceitos que ele cobre) e ele fica de fora por outro motivo: **7,1% de fusão errada, e o
dano é direcional** — `'cetamina plus lithium'` casa `lithium`, herda o prior maduro do
lítio, e a estreia da combinação sai com surpresa ~0. Zero de 47 estreias de combinação
reconhecidas. Silenciaria exatamente o tipo de notícia que a trilha especulativa existe
para achar.

### O que entrou

**A notícia do corpus como fato, não como número.** O bloco de claims da reflexão passa a
anotar duas coisas verificáveis: `(no other verified claim for this intervention in the
corpus)` e `(the corpus disagrees on this intervention: 3 positive, 1 negative or null)`.
Silêncio quando não há notícia. **Nenhum score decimal chega ao prompt** — travado por
teste, porque um 12B lendo `surprise: 1.0` conclui "achado marcante" quando o significado
real é *nós nunca buscamos isto*.

**O portão de contato com a literatura.** Um `pattern` só pode nascer de uma janela que
contém pelo menos uma claim verificada nova. Sem isso, três hipóteses refutadas por
opinião — `_critique` não recebe corpus nenhum, então `fatal_flaw` é a opinião de um LLM
sobre a saída de outro — viravam um "padrão substantivo" com proveniência de aparência
impecável.

O portão conta **claim verificada, não fonte**, e a diferença foi medida: contar fontes
deixava o portão totalmente aberto na janela estéril que ele existe para fechar. Uma fonte
árida (zero claims verificadas) é o processo escrevendo sobre si mesmo — são precisamente
as fontes que `recent_activity()` lista sob "yielded zero verified claims". Com 30 delas o
`pattern` passava.

E ele roda **antes** do portão de derivação, então reprovar por janela não gasta chamada
de LLM.

### O furo de wiring, quarta ocorrência

Reverter `reflector.mark_literature_seen()` do handler deixava o portão **permanentemente
aberto** e a suíte verde — um portão que nunca fecha é indistinguível, no verde do teste,
de um portão que não existe. Já aconteceu na Fase 1 (memória→screen), na Fase 2 (contenção
de `dead_end`), na Fase 3 (bloco de lições) e aqui. Agora há teste que roda o handler real.

### Uma trava minha que deu falso positivo, e não afrouxei

A TRAVA 2 varre `BinOp` por AST procurando os três fatores. `f"...{r['grade']}..." + nota`
é um `BinOp` com os nomes dentro, então ela reprovou uma concatenação de string. Ela está
certa em não distinguir formatação de aritmética — a alternativa seria isentar concatenação,
e isentar é como uma trava afrouxa. Troquei por duas entradas na lista.

---

## Fase 6 — o portão de citação, e a aresta que não foi construída

587 testes, 11 mutações mortas.

### O achado mais grave do projeto

Fui ao banco de produção real e consultei o PubMed. Nas três hipóteses especulativas
geradas em operação com gemma-4-12b, **9 de 12 elos vieram marcados `supported`, cada um
com um PMID verdadeiro do PubMed que não tem relação nenhuma com a alegação**:

```
28544150  matriz dérmica acelular humana em feridas crônicas
25115112  microarray de retrovírus endógeno suíno
23731151  clado sul-africano de peixes-cachimbo costeiros
24141515  doenças priônicas
30333051  luz do dia e comunidades bacterianas em poeira doméstica
21441130  osteopenia em homens com litíase renal
28155044  amiloidose cardíaca por transtirretina
28214153  displasia arritmogênica de ventrículo direito
26346444  composto antimalárico
```

Nenhum sobre sigma-1, cetose, ansiedade ou bipolar. E `plausibility` reportava **0,75**
para as três — o número que o quadro mostra ao psiquiatra, ao lado de
`[supported: PMID:xxx]`.

**Alucinar um identificador plausível e real é pior que inventar um**: ele sobrevive a
qualquer checagem de formato e só cai contra o corpus.

### As três decisões que a medição impôs

| checagem | pega | consequência |
|---|---|---|
| formato tolerante | 0 de 9 | **no-op** |
| formato estrito `^PMID:\d+$` | 9 de 9 | **apagão** — o modelo escreve `PMID: 123` com espaço |
| existência no corpus | 9 de 9 | é o portão |

O portão **rebaixa o elo, não rejeita a hipótese**: rejeitar mataria 3 de 3 das hipóteses
reais, e os portões que julgam mérito já existem (falsificador obrigatório, crítica
adversarial). Este corrige um **número**, não julga a ideia. Custo: uma query indexada por
hipótese, p50 10,5 µs — 1e-7 de um tick de especulação, Δ LLM exatamente 0.

E o que foi recusado fica registrado (`citation_refused`): sumir em silêncio trocaria uma
mentira por outra.

**Consequência declarada:** a plausibilidade das três hipóteses reais cai de 0,75 para
0,00. O número anterior era inflado; o novo é honesto. Com plausibilidade zerada,
`plausibility × novelty` empata e o desempate fica arbitrário — a saída é o reground, não
um termo novo no `ORDER BY`.

### A trava de ordenação que não existia

A trilha é ordenada por `plausibility × novelty`, e nada policiava isso. A auditoria
construiu o defeito natural — um guard anti-thrash como **fator multiplicativo, em
Python**, com o literal SQL byte-idêntico — e ele passou verde. Uma trava que lê o texto
da query não veria nada.

Agora são travas de **comportamento** sobre `pending_pursuit` (a fila que gasta chamadas
de LLM) e sobre `board()`. E a fixture teve de ser corrigida para discriminar: com a
terceira hipótese valendo 0,075, o rebaixamento de 0,85 para 0,255 não invertia nada e o
teste passava com o defeito presente.

### `seeking`: um campo gerado e descartado num único ponto

O modelo já nomeava qual elo cada busca tentava ancorar; `plan_queries` devolvia o campo
intacto; o dict do payload em `pursue_speculation` o ignorava. Uma linha. A lição que sai
de uma busca estéril deixa de ser *"esta query não retornou nada"* e vira *"queries
visando sigma-1 → ansiólise em humanos não retornam nada"* — a primeira é verdadeira e
inútil, a segunda diz onde a cadeia está sem chão.

Com higiene: colchetes e dígitos **isolados** saem antes de ir ao prompt, porque o bloco
de atividade usa `[k]` como espaço de nomes e a Fase 2 fechou a conflação em que um
número ali resolve com sucesso para o objeto errado. Dígito isolado, não qualquer dígito:
`\b\d+\b` mutilaria `sigma-1`, `5-HT1A` e `GABA-A` — exatamente os nomes que este campo
existe para carregar.

### O que eu NÃO construí: a aresta hipótese→hipótese

**Não existe produtor, e ele não é adicionável sem mudar prompt e gramática ao mesmo
tempo.** O único INSERT em `hypotheses` é alimentado por `Speculation`, que é `Strict` —
`additionalProperties: false` na gramática — então o modelo é *literalmente impedido* de
emitir um pai. E o canal que um refinamento exigiria não existe: a crítica de uma
hipótese **sobrevivente** nunca chega ao gerador (`_existing_block` traz só `statement` e
`mechanism_target`; os blocos de exclusão filtram `survives_critique = 0`).

Medido com o modelo real, em 7 ticks: o único par relacionado que apareceu foi uma
**duplicata degradada**, não um refinamento — a mesma hipótese com o nome do alvo
abreviado, plausibilidade caindo de 0,50 para 0,25 e um PMID diferente para o mesmo elo.
Gravar `parent_id` ali registraria ruído como linhagem.

E o produtor determinístico de reserva também falha: com o limiar do próprio repo
(`dedup_threshold = 0.78`, calibrado para bge-m3), **43 dos 45 pares** ficam acima do
limiar, inclusive hipóteses genuinamente diferentes a 0,930.

`questions.parent_id` está no schema desde o dia 1 com zero referências Python. Não vou
criar a segunda.

---

## Fase A — o objeto foco ✅ (e o que a medição mudou)

736 testes (699 + 37), 9 mutações construídas e mortas, banco de produção migrado e idempotente
em três aberturas seguidas. **Uma única expectativa mudou em todo o repo.**

### A decomposição por população foi RECUSADA por medição

O plano aprovado dizia que a unidade de rejulgamento era a **população distinta**, e daí tirava
o argumento de custo da Fase B (~5,8 s × ~80 populações ≈ 8 min por troca de foco). Isso está
errado, por duas razões, e a segunda é decisiva:

1. **`claims.population` é texto livre de um 12B.** As strings mais frequentes são genéricas —
   `adults`, `not specified`. Chavear o julgamento por `text_key` faz um estudo qualquer
   REBAIXAR retroativamente todo paper que por acaso escreveu a mesma string. A regra "vence o
   mais conservador", que o plano trazia para resolver conflitos, é o mecanismo do dano.
2. **Não preserva comportamento, que é a régua da fase.** Hoje cada claim tem seu próprio
   `directness`. Duas claims com a mesma string de população e valores divergentes obrigam a
   escolher um — e qualquer escolha muda o peso de pelo menos uma claim. Uma migração que muda
   comportamento não é uma migração.

Também foi rejeitada a saída intermediária (chavear por `(source_id, text_key)`): estreita o
raio e mantém a classe do defeito.

**O que ficou:** `claim_directness(claim_id, focus_id, directness)`, PK composta. Preserva a
semântica bit a bit (medido) e entrega inteiro o que a fase promete — o julgamento nomeia o
alvo, a mesma claim pode valer diferente sob focos diferentes, e as exclusões acontecem por
JOIN sem filtro escrito.

**Consequência para a Fase B, e ela é real:** o argumento de custo da troca de foco caiu junto.
Canonicalizar população exige julgamento de LLM, não igualdade de string — é fase própria. A
saída provável é usar a população normalizada como **cache das chamadas** (forward-only, uma
linha por claim), o que preserva a economia sem o defeito de reescrita retroativa; mas a razão
claims→populações continua **não medida**, e agora se sabe que as strings são texto livre.
Nenhum número de custo de troca de foco vale enquanto isso não for medido em corpus real.

### Três defeitos vivos que a fase expôs

1. **A trava numérica de 324 combinações passava vazia.** `test_the_sql_weight_equals_the_python_product`
   itera `SELECT ... FROM claim_weight` e afirma DENTRO do laço: com a view devolvendo zero
   linhas, o laço não roda e o teste passa. Reproduzido esvaziando a view. Como esta fase
   reescreve exatamente essa view com quatro JOINs, o modo de falha mais provável da migração
   era invisível para a trava que existe para pegá-lo. Corrigido com asserção de cardinalidade
   antes do laço.
2. **A `api_key` da NCBI vazava no log.** `PubMedSource._params` a injeta como query param e o
   `httpx` loga a URL completa em INFO. Reproduzido. Corrigido baixando o logger do httpx, com
   `test_the_api_key_never_reaches_the_log`.
3. **`state.py` decodificava rank por POSIÇÃO** (`list(Directness)[rank - 1]`). Um nível novo no
   meio do enum devolveria o nível ERRADO no prompt. Passa a ler `scale_levels`.

### Dois comentários do repo que estavam errados

- **`executescript` NÃO é atômico** com `isolation_level=None` (`store.py:325` e
  `schema.sql:454-457` afirmavam o contrário). Num script com erro no meio, o que veio antes
  SOBREVIVE. Por isso `meta` subiu para o topo do `schema.sql`: `active_focus` faz subselect
  nela, e um erro no meio deixaria todo consumidor de peso com "no such table: main.meta".
- **`CREATE VIEW` não valida nada** no SQLite 3.51.2 — nem tabela ausente, nem coluna ausente.
  O comentário de `init_schema` dizia que falharia com "no such column". Não falha: cria e
  quebra só na consulta.

E um fato de ordenação contra-intuitivo: **`ALTER TABLE DROP COLUMN` revalida o schema inteiro**,
então a sequência óbvia DROP VIEW → DROP COLUMN → CREATE VIEW é **impossível** aqui — dropar
`claim_weight` quebra `hypothesis_scoreboard`, que também bloqueia o DROP. O drop tem de ser o
ÚLTIMO passo de DDL, depois que o `executescript` recriou todas as views na forma nova.

### Deixado FORA da Fase A, com a razão

`populations`/canonicalização (precisa de LLM); `$target` no prompt de extração (trocar o alvo
hard-coded por variável reprova três testes e faz o golden perder o baseline daquele parágrafo,
por zero ganho com um foco só — em vez disso `test_the_seeded_focus_target_matches_the_extraction_prompt`
compara as duas fontes); `scale_levels.definition` e `focuses.path` (sem leitor — a mesma regra
que adia `evidence_links.focus_id`); rebuild de `hypotheses` preservando linhas (construído e
verificado, deliberadamente não entregue: escrever a operação mais perigosa da migração para uma
tabela vazia em produção é a definição de defeito deste repo); recalibração do limiar 0,78 de
dedup (escopar por foco muda o CONJUNTO DE CANDIDATOS, não o limiar — precisa de medição nova).

E um NUNCA: `user_memories`/`live_memories`/`declined_memories` não podem ser escopadas por foco.
Memória de conversa é sobre a PESSOA — "já teve rash com lamotrigina" vale em todo foco, e
escopar por simetria esconderia risco clínico exatamente ao trocar de foco.

### Dívida declarada, não paga

`directness_gap` continua lendo `DIRECTNESS_WEIGHT` de `types.py`. Depois desta fase os pesos
vivem em `scale_levels`, então a justificativa na `WEIGHT_ALLOWLIST` fica mais frágil. A saída
certa é derivar o gap do rank, e ela **muda os valores** (0.0/0.4/0.7/0.88 → 0.0/0.333/0.667/1.0),
o que muda `questions.priority` em produção — vetado pela régua desta fase. O que foi feito é
tornar a divergência detectável: `focuses.profile_hash` congela a calibração e `init_schema`
avisa quando `types.py` se afasta dela.

---

## Fase B — o foco vira um diretório, e a troca opera ✅

805 testes (736 → 803 na construção, +2 na correção abaixo), 42 mutações executadas e
mortas. Seis delas só mataram **depois de consertar o teste, não a mutação**.

Um foco é `focuses/<slug>/` com quatro TOML — `focus` (alvo, definições de directness,
`standing_risks`, blocos de prosa por prompt), `strategies`, `taxonomy`, `safety`. Todo o
pipeline lê o perfil. O alvo saiu de 8 prompts **e dos 7 schemas Pydantic** que o levavam
para dentro da gramática do decodificador — canal que nenhum teste enxergava.

`lithium focus --new/--use/--show/--relens`. A re-lente julga cada claim contra o alvo
novo, isolada, com três saídas: um nível, `out_of_scope`, ou não julgável (**nenhuma
linha** — que é diferente de um valor fraco: nenhuma linha é sem peso, valor fraco é peso
pequeno). O `--use` diz ANTES de trocar quantas claims o destino não julgou, quanto custa
em GPU, e quantas estão em escala divergente.

### Dois defeitos vivos que a fase expôs

1. **`render()` aceitava kwarg que o prompt não declarava** — e `extract.py` já passava
   `target=` para um `extract_claims.md` sem `$target`. Meia-fiação viva, com 736/736
   verdes. A trava nova pegou na hora.
2. **O esqueleto de `focus --new` produzia um foco que mente.** Achado rodando a troca
   pelo CLI real: com `target = "fibrilação atrial com insuficiência cardíaca"` e o resto
   intocado, o `extract_claims` renderizado anuncia *"investigating treatment options for
   bipolar I disorder…"* e define os quatro níveis de directness em populações bipolares.
   `focus --show` imprime a incoerência, mas imprimir não é barrar. Fechado com
   `scaffold_pending`, uma marca só cobrindo os quatro arquivos, travada nas duas metades
   (a recusa na carga E o comando escrever a marca).

### `--regrade` RECUSADO, e a razão vira trabalho futuro

`claims.scale_id` é **uma** coluna: re-graduar para o foco B destruiria o peso do foco A,
irreversivelmente. **A saída correta é `grade` virar aresta, espelhando o que a Fase A fez
com `directness`** — e isso é fase própria. No lugar ficou a recusa fundamentada em
`focus --new` (quando existe foco vivo na escala das claims) e o contador
`claims_off_scale`, que torna o problema legível em vez de silencioso.

### Outras recusas por medição

- **Lote de julgamentos na re-lente** — o lote *economiza* ~11,5% em K=8 (o sinal do
  dossiê estava invertido), mas perde o resume grátis da PK (uma falha re-julga 8) e
  contamina o único julgamento cujo princípio inteiro é isolamento por claim.
- **Composição `_domain.md`** — as sentinelas do golden e a trava de risco leem o `.md`
  **cru**, contornando o loader; com composição as duas ficariam cegas ao que o fragmento
  injeta. O perfil chega por **kwarg**, o que também torna `$` na prosa do TOML inerte —
  importa num arquivo que o usuário edita à mão.
- **Cortar o `rationale`** — cairia de 5,76 s para 1,73 s por claim, mas é o único
  artefato que torna um julgamento fail-closed auditável depois. A re-lente é `scheduled`
  e de baixa prioridade: as horas são preenchimento de ocioso, não bloqueio.
- **Cancelar tarefas pendentes no `--use`** — o comando REPORTA quantas vão recusar
  (`FocusDrift`), mas não as apaga. Apagar trabalho enfileirado num comando de troca é
  destruição silenciosa.

### Preço consciente do golden

29 linhas regeneradas em 8 arquivos; **6 dos 14 goldens ficaram byte a byte idênticos**.
A previsão do desenho era 458 linhas — o golden semeado pelos valores do perfil cortou
15×, e é o que mantém "transcrição, não recalibração" verificável em vez de declarada.

### Adiado com ressalva registrada

`detect_memory`, `reflect`, `reground_chain` e `classify_question` não foram
parametrizados: o domínio neles está em **exemplos ilustrativos** e a regra sobrevive a
exemplo trocado. A ressalva é real — `classify_question` ainda tem ~7 marcas de domínio,
incluindo o par de exemplos que calibra a fronteira PREFERENCE/FACTUAL, e num 12B exemplo
pesa mais que regra abstrata. O risco é **roteamento degradado**, não corpus corrompido.

---

## Fase C — o batedor da web ✅

983 testes (805 → 981 na construção, +2 na trava abaixo), 57 mutações executadas. Seis
mataram ZERO na primeira passada e cada uma virou trava nova; uma delas provou que um
`DROP VIEW` do desenho era **código morto**, e o conserto foi remover a fiação, não
escrever um teste para ela.

O ciclo: `recon_sweep → recon_query → recon_triage → recon_read`, mais a ponte
`recon_lead` no canal de evidência. Uma vez por dia, sobre as perguntas que **o sistema**
gerou e não conseguiu responder — nunca as que você digitou, porque a query vai para um
terceiro. Três tipos (`lead`/`source`/`observation`), três verbos, expiração em 14 dias.
Desligado por padrão.

### O portão, verificado por ATAQUE

A regra é que uma descoberta nunca vira claim. Isso precisa de trava estrutural porque
está **medido** que a disciplina não basta: ingerir 21 linhas de HTML real como
`SourceRecord(kind='pubmed')` grava 21 chunks e **os dois portões de extração aprovam** —
a citação é literal e a implicação é verdadeira. `EVIDENCE_KINDS` não pega: ele checa
`payload['kind']`, uma string de payload, enquanto `SourceRecord.kind` é preenchido pelo
adapter e ninguém verifica.

Quatro ataques da construção morrem. **Um quinto, meu, passou:** escrevi em
`worker/handlers.py` um handler chamado `harvest_from_discovery` — sem o prefixo `recon_`
— que faz `SELECT title, summary FROM discoveries` e ingere. **981/981 verdes.** A trava
existente escopava por NOME de função, e nome de função é convenção.

Fechado com uma varredura por **comportamento**, no pacote inteiro: nenhuma função, em
nenhum arquivo, com nenhum nome, pode nomear `discoveries` + uma coluna de prosa +
a superfície de ingestão. Com auto-verificação (o corpo do ataque tem de ser pego, e a
ponte legítima tem de passar).

### Quatro defeitos vivos que a fase consertou — um deles enorme

1. **`Scheduler.due()` nunca disparava job com `run_on_start=False`.** `if last is None:
   return job.run_on_start` mais `_mark_run` só depois do disparo formam um laço fechado:
   a chave `sched:<nome>` nunca nasce, então `due()` devolve False **para sempre**.
   Reproduzido de forma independente com o `Scheduler` e o `DEFAULT_JOBS` reais, 60 dias
   simulados: `pursue`, `reground`, `reflect` e `purge` dispararam **ZERO** vezes.
   **A trilha especulativa, a reancoragem e toda a máquina de reflexão e memória — as
   Fases 0–6 do plano anterior — nunca foram invocadas pelo agendador.** Sem log, sem
   dead-letter. Conserto: semear o relógio na primeira vez que o job é visto.
2. A `api_key` da NCBI ia para `tasks.error` e para o log do daemon num 429.
3. O turno de chat **já estourava a janela** (8.265 tok contra 7.936 utilizáveis), com o
   `except PromptTooLarge` inalcançável e sugerindo um `/novo` que não existe.
4. A marca do notify reanunciava o cemitério inteiro ao trocar de foco.

### Como ligar

Desligado por padrão; `lithium recon` imprime a receita. Chave da Brave Search API em
`config.local.toml` (gitignored, nunca no banco), com `contact` **obrigatório** — o
`BraveSearch.__init__` recusa sem ele, porque acesso automatizado se identifica (§7).
Tetos no BANCO, não na memória do processo: reiniciar o daemon não zera cota, e
`max_calls_per_day = 0` barra até a primeira chamada. `lithium mode off` fecha a torneira,
inclusive para `--now` — é a única coisa aqui que gasta dinheiro.

**Armadilha de config encontrada:** `load_config` devolve o PRIMEIRO arquivo que existe,
**sem merge**. Nesta máquina `config.local.toml` existe, então `config.toml` nunca é lido.
Agora uma seção com nome errado emite WARNING nomeando o arquivo, e uma chave com nome
errado levanta em vez de virar `None`.

---

## Fase D — o registro de fontes ✅

999 testes, 8 mutações executadas. **Três não mataram nada na primeira passada**, e cada
uma virou trava nova — as três eram a mesma classe, "fiação-não-testada", que este repo já
cometeu seis vezes:

- reverter o piso de citações para `len(hits)` deixava 996 verdes. O teste que havia
  exercitava `distinct_articles` **isolada** — provava que a função conta certo, não que o
  loop a usa.
- apagar `"source": q.source` do payload de `pursue_speculation` deixava tudo verde. Havia
  teste de que `plan_queries` devolve a fonte; nenhum de que ela **chega** ao payload.
- o daemon voltar ao dict literal passava, porque o teste checava `"build_sources" in src`
  e o **import** sobrevivia à remoção da chamada. Passou a afirmar sobre a CHAMADA, por AST.

### `EVIDENCE_KINDS` e `SourceKind` saíram

O portão não desapareceu: continua em `fetch_source`, onde ingere, e passou a **consultar**
`sources_registry.yields_evidence`. A pergunta é a mesma ("isto pode virar claim?"); mudou
quem responde. Uma allowlist de duas APIs em código expressa uma decisão de política como se
fosse um fato sobre o mundo; a coluna expressa o contrato — publica estudo com prosa citável
verbatim e desenho graduável na escala do foco.

`SourceKind` era `StrEnum` de quatro valores espelhado num `CHECK (kind IN (...))`. Duas
consequências: uma quinta fonte não conseguia ser **nomeada** (o INSERT era rejeitado antes
de qualquer política), e três dos quatro membros nunca tiveram adapter — o mesmo "botão que
não configura nada" que o repo recusa em `Strategy.tags`. `sources.kind` virou FK para o
registro, aplicada na escrita.

**As medições do item 9 continuam válidas e continuam registradas.** Elas dizem que aquelas
três fontes não devem produzir evidência neste foco, e é por isso que só o PubMed nasce
aprovado. O que elas não justificavam era congelar a lista para sempre.

### A dívida de fiação, paga por inteiro

`harvest_query` fixava `'pubmed'` quatro vezes; `pursue_speculation` montava o payload sem
`q.source`; `Strategy.sources` era código morto documentado como tal; o daemon montava um
dict literal. O efeito somado: **"o modelo escolhe onde buscar" era verdade como estrutura
de dados e falso como comportamento** — o campo atravessava o schema e morria em dois
lugares antes da busca. Agora a fonte vem do payload, o filtro lê o registro, e o prompt
recebe as fontes **com descrição** em vez de valores crus de enum.

### Duas decisões que ficaram separadas de propósito

`--approve` e `--evidence` são flags distintas. "Consulte esta fonte" e "o que ela devolve
pode virar evidência graduável" são afirmações diferentes: um registro de ensaios serve para
descobrir o que existe e não é desenho de estudo. Juntá-las faria a segunda pegar carona na
primeira — que é exactamente como uma bula virou claim com `grade='rct'` e peso 0,408 na
medição do item 9.

E aprovar a **descoberta** não ativa a **fonte**: a proposta nasce fora de `active_sources`,
logo fora do daemon e fora do portão. Ativar exige descrever a busca e dizer que se confia
nela.

### `article_key`, e a duplicata que virou real

Com uma fonte só, `(kind, external_id)` bastava. Com duas, o mesmo paper entra como duas
linhas e chega ao juiz como **duas fontes independentes concordando** — satisfazendo o piso
de citações com um artigo só. `article_key` é coluna GENERATED (DOI normalizado, ou
`kind:external_id`), e o piso passou a contar artigos distintos.

Uma armadilha que o repo já documentava e na qual eu caí: coluna `GENERATED VIRTUAL` **não
aparece** em `PRAGMA table_info`. O guard sempre caía no fallback. `table_xinfo` a vê.

### O portão da Fase C funcionando

Escrevi o INSERT de `sources_registry` dentro de `lithium/recon/verbs.py` e a varredura de
AST reprovou: o pacote do batedor não pode nomear `sources`. A trava está certa — o SQL do
registro foi para `store.py`, e o batedor fala com ele por indireção. Mesma razão pela qual
a ponte `recon_lead` mora em `worker/handlers.py`.

### Fora da Fase D

`--regrade` e `grade` como aresta (fase própria, já registrada); parser dedicado por fonte
além do PubMed (o genérico cobre JSON com campos nomeados; XML irregular exige módulo);
remoção de `sources.population_tag` (exige rebuild de `sources` com dados).

---

## Item 14 — Objetivo configurável → **RESOLVIDO: Fases A–D**

> **Este item foi decidido e substituído.** O desenho aprovado está em
> `~/.claude/plans/estou-pensando-numa-forma-iridescent-melody.md`, e o que ele decide está
> resumido em "A decisão" no fim desta seção. O inventário abaixo **continua válido** — é a
> lista do que precisa ser parametrizado, e foi confirmado por varredura completa em 2026-08.
> O que mudou foi a escolha entre as três opções, e o item deixou de ser o último da lista.

Hoje o sistema tem **um** objetivo, cravado no código: encontrar alternativa de tratamento para
TB-I com TAG comórbido. A meta é poder definir um objetivo novo ao entrar em modo pesquisa.

**O domínio não está só nos prompts, e é isso que torna o item grande.** Inventário do que hoje
assume o alvo:

| Onde | O que assume |
|---|---|
| 8 de 11 prompts | a moldura clínica, o impasse do antidepressivo, o risco de virada |
| `types.Directness` | **os valores do enum são definidos em relação ao alvo** — `direct` = "TB-I com TAG comórbido", `indirect` = "bipolar II, pânico, unipolar" |
| `types.DIRECTNESS_WEIGHT` | a calibração 1.0 / 0.6 / 0.3 / 0.12 foi raciocinada para *este* domínio |
| `strategy.py` | 5 frentes com 19 queries de psiquiatria |
| `mechanism.py` | 23 alvos mecanísticos + 17 classes de intervenção, psiquiatria-específicos |
| `state.py` | `CLASS_KEYWORDS` — 17 classes com vocabulário de psicofarmacologia |
| `safety/rules.py` (Fase 1) | contraindicações de lítio, lamotrigina, valproato |
| `chat.md` | a exceção permanente de risco de virada |
| `eval/goldset.yaml` | perguntas de resposta conhecida, específicas do alvo |

### A dificuldade central, e ela não é de engenharia

**`directness` não é um campo de texto — é uma escada semântica definida pelo objetivo.** Uma claim
graduada `direct` para "TB-I + TAG" **não é** `direct` para outro objetivo. E `directness` é um dos
três multiplicandos da invariante central do sistema.

Isso força uma escolha de três, e ela é a decisão de projeto do item:

1. **Um banco por objetivo.** Trocar de objetivo é trocar de `data_dir`. Trivial de implementar,
   zero risco de contaminação, e o corpus não é reaproveitado — o que é caro, porque colher 400
   papers custa horas.
2. **`claims.objective_id` + directness por objetivo.** Tabela `claim_directness(claim_id,
   objective_id, directness)`. Reusa o corpus entre objetivos relacionados. Mas exige re-graduar
   toda claim relevante ao criar um objetivo novo — uma chamada de LLM por claim, ou seja o custo da
   extração outra vez.
3. **Trocar de objetivo invalida as graduações.** Mais simples que (2) e mais honesto que (1):
   claims e chunks sobrevivem (são fatos com PMID), `directness` e `population_tag` viram `NULL`, e
   a re-graduação acontece sob demanda quando a claim é recuperada. Corpus reaproveitado, custo
   amortizado, e nada fica com graduação errada em silêncio.

~~**Recomendação: (3)**~~ — **nenhuma das três foi escolhida.** Existe uma quarta, e ela só
aparece quando se corrige um erro de raciocínio na opção (2).

**O erro:** eu escrevi acima que (2) "exige re-graduar toda claim relevante — uma chamada de LLM
por claim, ou seja o custo da extração outra vez". Isso é falso, e a razão está no próprio
`extract_claims.md:46-50`: os quatro valores de `Directness` são **quatro descrições de
população**. Então `directness = f(população, alvo)` — e a claim não entra na função. Populações
se repetem maciçamente entre papers ("adults with generalized anxiety disorder"), então a unidade
de rejulgamento não é a claim, é a **população distinta**. Estimei ~5,8 s por população pelo
modelo de custo do repo, contra ~1,9 chamadas de LLM por claim na extração.

Isso derruba o argumento de custo que eliminava (2), e com ele cai a recomendação de (3) — que
era o remendo para um problema que não existia, e que tinha o defeito de perder a graduação
antiga ao voltar para um objetivo anterior.

`grade` **não** é invalidado por troca de alvo — desenho de estudo é propriedade do paper. Mas
ele **é** relativo à *escala*, que é outra coisa, e o usuário pediu escala trocável: uma claim
`rct` não tem imagem numa escala não-clínica sem reler a fonte.

### Desenho: *domain pack*

Um objetivo passa a ser um pacote declarativo, não código:

```
objectives/
  bipolar1_gad.toml        # o objetivo atual, extraído do código
  <novo>.toml
```

Contendo: `statement` (a frase do objetivo), a **escada de directness** (rótulo + descrição +
peso por degrau), as frentes de busca, a taxonomia de mecanismo e classes, as palavras-chave de
classe, as regras de segurança, e a exceção permanente do chat.

Consequências que precisam ser resolvidas junto:

- **`Directness` deixa de ser `StrEnum`.** Os rótulos passam a vir do pacote, então os `CHECK` do
  SQL e os schemas Pydantic precisam ser gerados a partir dele. O teste `test_types.py` que compara
  enum × `CHECK` × pesos passa a comparar contra o pacote carregado — e continua sendo a trava.
  Cuidado: a decodificação restrita precisa de um enum fechado em tempo de render, então o pacote é
  lido uma vez na subida e não pode mudar com o daemon no ar.
- **A invariante multiplicativa tem que sobreviver a qualquer pacote.** Um pacote com escada mal
  calibrada poderia fazer `rct × indirect` superar `cohort × direct`. Validação na carga: para todo
  par de degraus adjacentes, o produto com o `grade` imediatamente superior não pode inverter a
  ordem. É um teste de propriedade sobre o pacote, não sobre o código.
- **Memórias e lições são por objetivo.** Uma `search_lesson` sobre vocabulário de psiquiatria é
  ruído noutro domínio. `memories.objective_id`, e o `--forget` em cascata.
- **Os prompts recebem `$objective`**, que é o único lugar onde a composição de fragmentos do item
  13 deixa de ser Δ=0: ali ela passa a ser *necessária*, não estética.
- **O escopo clínico não é configurável.** "Ferramenta de síntese para revisão por especialista, não
  emite conduta" e a disciplina de citação valem para qualquer objetivo. Um pacote pode trocar o
  domínio; não pode desligar os portões.

### ~~Por que é o último item~~ — por que deixou de ser

O argumento original era: "todo item pendente (7, 7.5, 9–13) toca código que o pacote vai
parametrizar; fazer isto antes significa parametrizar duas vezes". Ele **valia e foi cumprido** —
7, 7.5 e 9 estão entregues, e o item 9 recusou as quatro fontes por medição em vez de construí-las
sobre uma base que ia mudar. O que resta pendente (10, 11, 12, 8) ou não existe ainda ou é o
último da fila, então a segunda parametrização não acontece.

E a segunda metade do argumento — "o valor só aparece quando você tiver um segundo objetivo real"
— foi respondida pelo usuário: ele quer poder trocar o foco quando quiser, e quer que o modelo
busque as próprias fontes, o que é inviável enquanto o alvo for constante de módulo.

### A decisão

| Questão | Resolução |
|---|---|
| Corpus entre focos | **Um banco, um corpus, uma memória.** Foco é lente, não partição |
| `directness` | Sai de `claims`; vira ~~`population_directness`~~ **`claim_directness(claim_id, focus_id, …)`** |
| Unidade de rejulgamento | ~~A população distinta~~ → **a claim**. Ver "Fase A — o que a medição mudou" |
| `grade` | Ganha `scale_id`; a escala pertence ao foco |
| Claim fora da escala do foco | **Legível e citável, sem peso** — a exclusão acontece por JOIN |
| Pacote de objetivo | Diretório `focuses/<slug>/`, não um `.toml` único |
| Fontes | Registro com `yields_evidence`; `EVIDENCE_KINDS` deixa de ser allowlist de API |
| Web | Canal de **reconhecimento** separado: propõe → te conta → você autoriza. Nunca vira claim |

As três consequências que o esboço acima já tinha antecipado corretamente e que seguem valendo:
`Directness` deixa de ser `StrEnum` fechado; a invariante multiplicativa precisa sobreviver a
qualquer escala (validação de não-inversão na carga); e **o escopo clínico não é configurável** —
um foco troca o domínio, não desliga os portões.

Uma que o esboço errou por omissão: ele propunha `memories.objective_id` para tudo. O plano
aprovado separa — memória sobre **você** (`chat`/`answer`/`manual`) atravessa focos, porque é
sobre a pessoa; lição de **processo** é do foco; e `dead_end` é o único caso em que o
compartilhamento é ativamente perigoso, porque ele é injetado sob "do not propose these again" e
removeria uma classe de mecanismo do espaço de busca de outro foco sem deixar assinatura
observável.

---

## Item 13 — Enxugar a camada de prompts → **absorvido pela Fase B**

> **Este item deixou de ser autônomo.** A medição posterior mediu a composição `_domain.md` em
> **Δ = 0 tokens** e ~34 linhas economizadas em disco — ou seja, ela nunca se pagou pelo motivo
> pelo qual foi proposta. Ela se paga por outro: é o **mecanismo da troca de foco**. Os
> fragmentos passam a receber `$target`, `$directness_definitions`, `$grade_scale`,
> `$standing_risks` e `$taxonomy` do perfil do foco ativo, e aí a composição deixa de ser
> estética e vira necessária. Executado dentro da Fase B, não separadamente.
>
> O que continua valendo por conta própria é o fragmento `_question_kinds.md`: `classify_question`
> e `generate_questions` escrevem a **mesma** coluna `questions.kind`, `AUTO_ANSWERABLE` roteia
> nela, e as duas definições **já divergiram uma vez**. Isso é consistência de roteamento, não
> economia de token, e é agnóstico ao foco.

**O problema, medido.** 626 linhas em 10 arquivos de prompt, e **8 deles reescrevem o
mesmo enquadramento clínico** (TB-I + TAG, o impasse do antidepressivo, o risco de
virada). Mudar como o domínio é apresentado hoje exige editar oito arquivos e torcer
para não divergirem. E a extração gasta **uma chamada de verificação por claim**, o que
domina o custo do pipeline.

O plano tem cinco frentes, em ordem de retorno:

**1. Composição em vez de duplicação.** Extrair `_domain.md`, `_grading.md` e
`_style.md` como fragmentos e compor via `prompts.render`. Uma edição passa a valer para
todos. Elimina ~150 linhas de repetição e, mais importante, remove a chance de os oito
enquadramentos divergirem em silêncio.

**2. Verificação em lote.** Hoje o portão 2 roda uma chamada por claim. Julgar todas as
claims de um mesmo chunk numa chamada só corta as chamadas de extração quase pela
metade, sem enfraquecer o portão — a separação que importa é entre *gerar* e *verificar*,
não entre verificar uma e verificar três.

**3. A LoRA é o mecanismo principal de encolhimento.** Boa parte do texto atual é
andaime didático: a explicação da taxonomia de grade e directness, os parágrafos de
calibração, os exemplos de o que conta como falha fatal. Depois da destilação, o modelo
internalizou isso e o andaime pode cair. É o propósito declarado da fase 2 — "formato,
vocabulário e estilo de raciocínio" — visto pelo outro lado.

**4. Instrumentação antes de otimizar.** Não há contagem de tokens por prompt hoje.
Registrar tokens de entrada e saída por tipo de tarefa é o que diz qual prompt custa de
verdade, em vez de otimizar o mais longo.

**5. Testes de contrato dos prompts.** Com fragmentos compartilhados, editar
`_domain.md` afeta oito prompts de uma vez. Testes de renderização golden protegem
contra a regressão que a consolidação introduz.

**Métrica de sucesso:** tokens de prompt por chamada caindo, com o `eval/goldset.yaml`
segurando a qualidade constante. Enxugar sem medir qualidade é só piorar mais rápido.

**MVP = itens 1–8:** um sistema que pesquisa sozinho e te pergunta no celular quando
trava.

---

## Verificação

**Por etapa** — `pytest` com fixtures HTTP gravadas (sem rede); round-trip do banco;
concorrência da fila; CI em `macos-latest` + `windows-latest` desde o commit 1;
`eval/goldset.yaml` (perguntas de resposta conhecida, ex: *"quetiapina tem RCT em
TAG?"* → sim, com PMIDs); `eval/safety_probes.yaml` (cenários de flag obrigatória).

**Ponta a ponta (MVP)**
```bash
lithium init
lithium serve &
lithium ask "Quetiapina em monoterapia reduz sintomas de TAG em TB-I?"
lithium status
lithium publish
# abrir o Space: auto-respondida OU escalada com trabalho parcial
lithium sync && lithium report
```

**Critérios de aceite**

| # | Critério | Situação |
|---|---|---|
| 1 | Pergunta factual se auto-responde em ≤3 rodadas, ≥2 fontes citadas | ⏳ item 7 |
| 2 | Pergunta `PREFERENCE` é escalada sem gastar rodada de pesquisa | ✅ |
| 3 | Toda alegação resolve para um PMID/NCT/DOI real | ✅ |
| 4 | Nenhuma menção a antidepressivo sem o alerta de virada anexado | ⏳ item 11 |
| 5 | `kill -9` e restart: nenhuma task perdida nem duplicada | ✅ |
| 6 | Resposta dada no Space chega ao banco em ≤10 min | ⏳ item 8 |
| 7 | Pergunta escalada chega até você sem que você consulte nada | ⏳ item 7.5 |

---

## Desvios e achados da implementação

Registrados aqui porque cada um custou tempo para descobrir e voltaria a custar.

**`--reasoning-budget 0` é obrigatório.** Gemma-4 é modelo de raciocínio. O llama-server
separa o thinking em `reasoning_content`, e com ele ligado o modelo consome o orçamento
inteiro de tokens **antes de emitir uma linha de JSON** — `content` volta vazio. O
parâmetro por requisição é ignorado; só a flag de servidor funciona. Custou 14 minutos em
três retries idênticos antes do diagnóstico; o cliente agora levanta `LLMTruncated` na
primeira ocorrência, porque é erro determinístico. Vale religar (`-1`) para síntese e
crítica, onde o raciocínio é o produto.

**Peso de evidência engolia a relevância no retrieval.** Scores RRF variam ~2x entre o
primeiro e o último candidato; o peso de evidência varia ~160x. Multiplicando os dois
crus, o peso decide sozinho — perguntar sobre TCC devolvia a claim de maior grade do
corpus, sobre o assunto que fosse. Corrigido normalizando a relevância min-max e
aplicando o peso como `(1 + w)`: **relevância decide quem está no páreo, peso decide a
ordem dentro dele.** Teste de regressão em `test_pipeline_retrieval.py`.

**`sqlite-vec` usa distância L2 por padrão, não cosseno.** O fallback em numpy calcula
cosseno. Os dois caminhos devolviam métricas diferentes conforme a extensão tivesse
carregado — e qualquer limiar calibrado em cima ficava errado em um dos ambientes. Pior:
eu medi "cosseno" num corpus real usando `1 − distância` com a tabela em L2 e calibrei um
piso com números que não eram cosseno. A tabela `vec0` agora declara
`distance_metric=cosine`, e `_migrate_vector_table` recria índices criados antes do fix
(`CREATE VIRTUAL TABLE IF NOT EXISTS` não conserta tabela existente — ficaria em L2 para
sempre, calado). `test_store_vector_metric.py` trava a equivalência entre os dois
backends.

**A normalização min-max zerava o último colocado.** Mapear para `[0, 1]` cru dava
exatamente `0.0` ao pior candidato, e o guarda `if relevance <= 0.0: continue` em
`search_claims` o descartava. "Não foi recuperado" e "foi recuperado em último lugar"
viraram a mesma coisa. Agora o teste é de pertinência, não de valor, e a faixa é
`[0.05, 1]`.

**Boilerplate de abstract estruturado estava indexado.** Num piloto real, um chunk cujo
texto era literalmente `"None."` (seção `funding`) apareceu em 2º lugar numa busca —
texto curto tem embedding próximo de qualquer coisa, então emerge justamente quando não
há resposta boa. `SKIP_SECTIONS` + `MIN_CHUNK_CHARS` resolvem. `limitations` fica de
fora da lista: é conteúdo, e conteúdo que importa para graduar evidência.

**Piso de similaridade: peneira grossa, não classificador.** Medido em 348 chunks reais
com bge-m3 e métrica corrigida — consulta fora do domínio chega a **0.704**, consulta
relevante fica em **0.845–0.865**. A faixa do bge-m3 é comprimida e alta (tudo acima de
0.6). Um limiar em 0.75 separaria esta amostra, mas com 0.04 de margem: evidência fraca
porém legítima sumiria calada. Ficou em **0.70** — pega o lixo evidente e deixa o resto
para o juiz de suficiência, que é o filtro de relevância de verdade.

**Conexão SQLite atravessando thread.** Passar `store.conn.execute` já vinculado para
`asyncio.to_thread` resolve a conexão na thread chamadora e executa em outra, o que o
sqlite3 proíbe. Toda operação de banco precisa de um método que acesse `conn` **dentro**
da thread trabalhadora.

**Lacunas no mapeamento do PubMed**, achadas testando contra XML real: `Network
Meta-Analysis` caía para `systematic_review` (é o desenho que melhor responde "qual
alternativa"), e `Practice Guideline` ficava sem grade — sendo diretriz uma das âncoras
de alto valor do plano.

**`string.Template` no lugar de jinja2.** Os prompts contêm exemplos de JSON cheios de
chaves; `str.format` engasgaria em todos. `$var` evita o problema e remove uma
dependência.

**`lithium/types.py` como fonte única.** Os mesmos enums aparecem nos `CHECK` do SQL, nos
schemas Pydantic que restringem a gramática e na lógica de pontuação. `test_types.py`
compara os três e falha no lugar certo quando saem de sincronia — sem isso o sintoma
seria um `IntegrityError` em produção, longe da causa.

**`DEFAULT_JOBS` só agenda handlers que existem.** Agendar um `task_kind` sem handler
manda a tarefa ao dead-letter a cada ciclo: ruído constante mascarando falha real.
`plan_tick`, `weekly_report` e `sync_answers` entram junto com os itens 6–8.

**Armadilha do `uv` com `VIRTUAL_ENV`.** Com um venv ativo, `uv pip install` instala
*nele*, ignorando o `.venv` local — o `lithium` foi parar no venv do `qyra-labs` na
primeira tentativa. Documentado no README.

**Dedup por similaridade de texto não funciona neste domínio.** Calibrei o limiar em
7 pares escritos à mão (paráfrases 0.788–0.952, distintas 0.661–0.711) e cheguei a 0.78
— já uma correção grande sobre o chute inicial de 0.90, que teria deixado passar 2 das
3 paráfrases. Mas num lote real gerado pelo 12B, cinco perguntas sobre cinco
intervenções **diferentes** (buspirona, pregabalina, antidepressivo, TCC, quetiapina)
ficaram em **0.767–0.874** entre si: compartilham o vocabulário do domínio, que domina
o embedding. Essa faixa sobrepõe inteiramente a das paráfrases genuínas, então nenhum
limiar separa as duas. Um plan tick de 5 propostas rendia 1 pergunta.

A correção é escopo, não limiar: duas perguntas sobre intervenções diferentes não são
duplicatas, por mais parecidas que soem. Dentro de um mesmo alvo o texto volta a
discriminar ("quetiapina tem RCT?" vs "quetiapina causa ganho de peso?" = 0.711). Exigiu
adicionar `questions.targets` e um migrador de coluna — `CREATE TABLE IF NOT EXISTS`
não altera tabela existente.

**O piso absoluto de similaridade não funciona — e eu calibrei errado duas vezes.**
Primeira medição, só com consultas em estilo de pesquisa, sugeriu 0.70: fora do domínio
dava 0.704, relevante dava 0.845. Parecia sólido. Ao testar o chat, a conclusão caiu:

    "O que o corpus já tem sobre quetiapina?" (assunto CERTO)  → 0.705
    "me conta o que você sabe de quetiapina"  (assunto CERTO)  → 0.706
    "anticoagulação em fibrilação atrial"     (assunto ERRADO) → 0.710

Pergunta conversacional sobre o tema certo pontua **abaixo** de uma de cardiologia: o
enquadramento ("o que você sabe sobre…") domina o embedding e afoga o conteúdo médico.
As faixas se sobrepõem por completo — nenhum limiar separa. E o piso em 0.70 apagava o
corpus inteiro numa conversa normal, fazendo o assistente afirmar que não tinha dados
que tinha.

Desceu para 0.45, onde nunca dispara exceto em caso degenerado. A discriminação de
relevância fica onde há informação para fazê-la: corte por `k`, juiz de suficiência, e
o prompt. O problema original — boilerplate emergindo quando não há resposta boa — já
tinha sido resolvido na origem por `SKIP_SECTIONS`, que é onde deveria ter sido
resolvido desde o começo.

**A crítica adversarial rejeitava 100% por confundir omissão com defeito.** Na primeira
rodada real, 0 de 3 hipóteses sobreviveram, as três com o mesmo veredito: *"the
mechanism would plausibly destabilise mood"*. Lendo as justificativas, o que o crítico
apontava era *"fails to address the switch-risk"* — omissão, não mecanismo
desestabilizador demonstrado. A causa estava escrita no meu prompt: *"A hypothesis that
ignores switch risk has a fatal flaw, not a gap."* Com essa instrução só sobrevive
tratamento estabelecido, que é a gaiola que a trilha existe para remover.

Separado agora: `fatal_flaw` só quando o **mecanismo descrito** produz a
desestabilização (ou é circular, contraindicado, já falhou, ou reduz a antidepressivo em
monoterapia). Omissão e dado ausente vão para `missing_risk`. O prompt inclui calibração
explícita — *"expect most hypotheses to survive; you are filtering the broken ones, not
the unproven ones"*. Depois da correção: 3/3 sobrevivem, cada uma com o elo fraco
identificado.

**A view SQL e a função Python discordavam sobre plausibilidade.** A view contava
`$.status = 'supported'`; o schema Pydantic gera `supported: bool` + `evidence`. A
plausibilidade em SQL era sempre 0, e o quadro ordenava por ineditismo puro sem que nada
acusasse. Mesma classe do bug L2-vs-cosseno: duas implementações da mesma métrica que
ninguém comparava. Agora `test_pipeline_explore` compara as duas em seis casos. Views
passaram a ser `DROP` + `CREATE` — `IF NOT EXISTS` deixaria a versão errada viva para
sempre em bancos existentes.

**Otimização para M1 saiu de escopo.** Medido: 45,6s por extração no M1 Pro/16 GB, com
~20 GB de swap. Produção roda em Windows, então isso é inconveniência de desenvolvimento,
não vetor de arquitetura. A config expõe o que precisa ser ajustado por máquina
(`n_parallel`, `n_ctx`, `kv_cache_type`, `reasoning_budget`).

---

## Pontos em aberto

1. **Nome** — `lithium` é provisório (o estabilizador canônico do TB-I).
2. **Elo fraco em síntese.** Gemma-4-12B é competente em extração estruturada, mas é o
   ponto frágil em síntese e crítica adversarial. Como a LLM já fica atrás de base URL
   configurável, dá para rotear **só** `synthesize`/`critique` a um modelo mais forte e
   manter harvest/extract 100% local.
3. **Cadência.** Harvest/plan diário, relatório semanal, sync 10 min é chute inicial.
4. **Idioma.** Prompts internos estão em inglês (a literatura é inglesa e a terminologia
   médica de um 12B degrada em português). Relatório final para o usuário em português,
   traduzido na camada de report. Reversível — prompts são arquivos de dados.
5. **Escopo de "tratamento".** O não-farmacológico já está nas estratégias de busca desde
   o início, por ser onde provavelmente está a resposta menos óbvia.
