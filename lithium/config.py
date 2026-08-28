"""Configuração — carregada de TOML, com defaults portáveis Mac/Windows.

Regra de portabilidade: nada aqui pode assumir separador de caminho, shell ou
serviço específico de SO. A raiz de dados sai de `platformdirs`, e a camada de
LLM é sempre uma base URL — nunca uma biblioteca in-process.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from platformdirs import user_data_dir
import logging

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)

APP_NAME = "lithium"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _resolve(value: Path | str | None) -> Path | None:
    """Caminho relativo no TOML é relativo à raiz do projeto, não ao CWD.

    Sem isto, `lithium serve` rodando como serviço supervisionado (que inicia em um
    CWD arbitrário) não acharia o modelo.
    """
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class LLMConfig(BaseModel):
    """O modelo de geração, atrás de uma API OpenAI-compatible.

    `llama-server` roda como subprocesso separado — build Metal no Mac, CUDA no
    Windows. Trocar de backend é trocar `base_url`, não código.
    """

    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "gemma-4-12b-it"
    # `PROJECT_ROOT / base_models`, não `PROJECT_ROOT.parent`: os pesos moram DENTRO do
    # repositório (ignorados pelo git) desde a migração para o repositório próprio. Antes
    # o projeto era `qyra-labs/lithium` e os GGUF eram um diretório irmão, um nível acima.
    model_path: Path = PROJECT_ROOT / "base_models" / "gemma-4-12b-it-Q5_K_M.gguf"
    lora_path: Path | None = None
    n_ctx: int = 8192
    n_parallel: int = 1
    n_gpu_layers: int = 99

    reasoning_budget: int = 0
    """0 desliga o bloco de raciocínio do Gemma-4; -1 deixa ilimitado.

    Medido em M1 Pro/16 GB com Q5_K_M: com raciocínio ligado, a extração estourava
    2048 tokens sem emitir uma linha de JSON — o thinking consome o orçamento
    inteiro e `content` volta vazio. Com 0, a mesma extração fecha em ~60s.

    Vale a pena religar (-1) para síntese e crítica, onde o raciocínio é o produto.
    Só a flag de servidor funciona; o parâmetro por requisição é ignorado.
    """

    kv_cache_type: str = "q8_0"
    """Quantizar o KV cache é o que faz o modelo caber em 16 GB junto com o SO."""
    timeout_s: float = 300.0
    max_retries: int = 3
    temperature: float = 0.2

    _abs_paths = field_validator("model_path", "lora_path", mode="after")(_resolve)


class EmbeddingConfig(BaseModel):
    """Segunda instância de llama-server, em modo --embedding."""

    base_url: str = "http://127.0.0.1:8081/v1"
    model: str = "bge-m3"
    model_path: Path | None = None
    dim: int = 1024
    batch_size: int = 16

    _abs_paths = field_validator("model_path", mode="after")(_resolve)


class WorkerConfig(BaseModel):
    concurrency: int = 3
    max_attempts: int = 3
    poll_interval_s: float = 2.0
    backoff_base_s: float = 5.0


class QuestionConfig(BaseModel):
    max_rounds: int = 3
    human_queue_limit: int = 5
    """Teto de perguntas escaladas simultâneas. Sistema que enche a fila não é usado."""

    dedup_threshold: float = 0.78
    """Similaridade acima da qual uma pergunta nova conta como duplicata.

    **Calibrado para bge-m3.** Medido em 7 pares reais:

        mesma lacuna, redação diferente  → 0.788 – 0.952
        perguntas genuinamente distintas → 0.661 – 0.711

    O default anterior de 0.90 era chute e teria deixado passar 2 das 3 paráfrases —
    o dedup mal funcionaria e o gerador acumularia reformulações da mesma lacuna.

    O valor fica junto ao piso das duplicatas, não no meio do intervalo, de propósito:
    dos dois erros possíveis, suprimir uma pergunta genuinamente nova é perda
    silenciosa de informação, enquanto deixar passar uma paráfrase é ruído visível e
    corrigível. A assimetria favorece não suprimir.
    """


class SourceConfig(BaseModel):
    enabled: bool = True
    rate_per_s: float = 3.0
    api_key: str | None = None


class SourcesConfig(BaseModel):
    # NCBI: 3 req/s sem key, 10 com key.
    pubmed: SourceConfig = SourceConfig(rate_per_s=3.0)
    epmc: SourceConfig = SourceConfig(rate_per_s=5.0)
    ctgov: SourceConfig = SourceConfig(rate_per_s=5.0)
    fda: SourceConfig = SourceConfig(rate_per_s=4.0)


class NotifyConfig(BaseModel):
    """Aviso no SO. `ntfy_url` é opcional e vira o canal preferido quando presente —
    o toast local não tem recibo de entrega, então o canal externo é redundância."""

    enabled: bool = True
    ntfy_url: str | None = None


class ReconConfig(BaseModel):
    """O batedor: busca na web aberta. **Desabilitado por padrão, de verdade.**

    `extra="forbid"` só neste modelo (risco zero — ele é novo) e a razão é medida:
    nem `Config` nem `SourceConfig` o declaram, então `api_kei = "..."` vira
    `api_key=None` em SILÊNCIO. Para uma feature cujo estado normal é "desabilitado,
    falta a chave", isso torna *errei o nome* indistinguível de *não configurei*. O
    remédio já estava escrito em `focus/profile.py`.

    E `enabled` aqui TEM leitor: `BraveSearch.__init__` levanta `ReconDisabled`.
    MEDIDO que `SourceConfig.enabled` e `NotifyConfig.enabled` não têm nenhum —
    `grep -rn '\\.enabled' lithium` não devolvia uma linha antes desta fase. Copiar
    esse padrão entregaria um "desabilitado por padrão" que não desabilita nada.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: str = "brave"
    api_key: str | None = None
    """Em `config.local.toml` (gitignored), NUNCA no banco, e enviada por HEADER.

    Header e não query param é mitigação ESTRUTURAL, não estilo: `str(HTTPStatusError)`
    carrega a URL completa, e `Runner._execute` a grava em `tasks.error`. Com a chave
    da NCBI em query param isso já acontece hoje (medido). Ver `redact_secrets`.
    """

    contact: str | None = None
    """E-mail ou URL que vai no User-Agent. Obrigatório quando `enabled`: acesso
    automatizado a terceiros sem identificação é o que o PLAN.md §7 recusa — uso
    pessoal não relaxa ToS."""

    max_calls_per_sweep: int = 6
    """Teto ESTRUTURAL: `recon_sweep` enfileira no máximo N tarefas `recon_query`, e
    cada uma faz exatamente uma busca faturada."""

    max_calls_per_day: int = 20
    """Teto DURÁVEL, em `recon_budget`. Brave: US$ 5/1.000 com US$ 5 de crédito mensal
    ≈ 1.000 grátis/mês ≈ 33/dia. 20 × 31 = 620 fica dentro com folga, então um teto
    mensal separado seria mais uma coisa para manter correta sem proteção adicional."""

    max_reads_per_query: int = 1
    """Quantas páginas UMA triagem pode mandar ler. Por QUERY e não por varredura
    porque o teto por varredura não tem onde ser aplicado: são N triagens
    independentes, e `max_pages_per_sweep` aplicado dentro de cada uma daria
    N × teto — 6 × 5 = 30 leituras numa varredura anunciada como 5."""

    max_pages_per_day: int = 8
    """Leituras de página por dia. 6 triagens × 42 s + 6 leituras × 57 s ≈ 9,9 min de
    GPU/dia (0,69% do dia), contra os ~38 min/dia (2,6%) do `answer_tick`."""

    max_robots_per_day: int = 12
    """robots.txt não é faturado, mas é acesso automatizado a terceiros e precisa de
    teto próprio. Contá-lo junto com as páginas faria o teto de leitura ser consumido
    por requisições que não leram nada."""

    pending_limit: int = 8
    """CONTRAPRESSÃO na fila humana, no idioma de `QuestionConfig.human_queue_limit`
    ("sistema que enche a fila não é usado"). Sem ele: 6 queries × 10 hits = 60
    triados por noite, e 30% de não-`skip` já são 18 descobertas no primeiro dia. O
    usuário pediu "comente comigo as suas descobertas", não uma fila de 200 que se
    aprova em lote sem ler. O que não couber é reencontrado na varredura seguinte —
    `freshness='pm'` cobre 31 dias."""

    expire_after_days: int = 14
    respect_robots: bool = True


class SyncConfig(BaseModel):
    dataset_repo: str | None = None
    model_repo: str | None = None
    interval_s: float = 600.0


class Config(BaseModel):
    data_dir: Path = Field(default_factory=lambda: Path(user_data_dir(APP_NAME)))
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    worker: WorkerConfig = WorkerConfig()
    question: QuestionConfig = QuestionConfig()
    sources: SourcesConfig = SourcesConfig()
    notify: NotifyConfig = NotifyConfig()
    recon: ReconConfig = ReconConfig()
    sync: SyncConfig = SyncConfig()

    focuses_dir: Path = PROJECT_ROOT / "focuses"
    """Onde vivem os perfis de foco, um diretório por slug.

    Na RAIZ DO PROJETO, não em `data_dir`: um perfil é configuração versionável —
    quatro TOML que o usuário edita à mão e diffa — enquanto `data_dir` é estado
    derivado (banco, cache, adapters) que ninguém commita. Misturar os dois faria a
    única parte do foco que se lê como código viver junto do que se apaga para
    recomeçar do zero.
    """

    _abs_paths = field_validator("data_dir", "focuses_dir", mode="after")(_resolve)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lithium.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def adapters_dir(self) -> Path:
        return self.data_dir / "adapters"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.cache_dir, self.adapters_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> Config:
    """Carrega config.toml, caindo para defaults quando ausente.

    Ordem de precedência: `path` explícito > config.local.toml > config.toml > defaults.
    `config.local.toml` é gitignored — é onde vão API keys e caminhos da máquina.
    """
    candidates = [path] if path else [
        PROJECT_ROOT / "config.local.toml",
        PROJECT_ROOT / "config.toml",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            with candidate.open("rb") as fh:
                raw = tomllib.load(fh)
            _warn_unknown_sections(candidate, raw)
            return Config.model_validate(raw)
    return Config()


def _warn_unknown_sections(path: Path, raw: dict) -> None:
    """Grita quando o TOML declara uma seção de topo que `Config` não conhece.

    AVISO e não exceção: `Config` não declara `extra="forbid"` e alargá-lo agora
    quebraria configs antigos por uma chave inócua. Mas o silêncio total é pior para a
    Fase C, cujo estado normal é "desabilitado, falta a chave" — um `[reccon]` com a
    chave certa dentro é indistinguível de nunca ter configurado nada.

    A mensagem NOMEIA o arquivo lido, e isso importa: `load_config` devolve o PRIMEIRO
    que existe, sem merge. Nesta máquina `config.local.toml` existe, então
    `config.toml` NUNCA é lido — visível hoje, ele declara `n_ctx = 16384` e o efetivo
    é 8192.
    """
    unknown = sorted(set(raw) - set(Config.model_fields))
    if unknown:
        log.warning(
            "%s declara seção/chave de topo que a configuração não conhece: %s. "
            "Ela foi IGNORADA. Conhecidas: %s",
            path, ", ".join(unknown), ", ".join(sorted(Config.model_fields)),
        )
