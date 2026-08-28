"""Ciclo de vida do llama-server como subprocesso.

Portabilidade é o ponto: `create_subprocess_exec` com lista de argumentos (sem shell,
sem string a ser parseada), `shutil.which` para achar o binário (resolve `.exe`
sozinho no Windows), e `terminate()` da API do asyncio, que vira `TerminateProcess`
lá e `SIGTERM` aqui. Nenhum `signal.SIGKILL`, nenhum `os.fork`.

Trocar Metal por CUDA é trocar a build do llama.cpp instalada. Nada aqui muda.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_BINARY = "llama-server"


class ServerError(RuntimeError):
    pass


@dataclass
class LlamaServer:
    model_path: Path
    port: int
    host: str = "127.0.0.1"
    n_ctx: int = 8192
    n_parallel: int = 1
    n_gpu_layers: int = 99
    kv_cache_type: str | None = "q8_0"
    reasoning_budget: int = 0
    lora_path: Path | None = None
    embedding: bool = False
    binary: str = DEFAULT_BINARY
    log_path: Path | None = None

    _process: asyncio.subprocess.Process | None = None
    _log_handle: object | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command(self) -> list[str]:
        exe = shutil.which(self.binary)
        if exe is None:
            raise ServerError(
                f"'{self.binary}' não está no PATH. "
                "macOS: `brew install llama.cpp`. "
                "Windows: baixe a build CUDA em github.com/ggml-org/llama.cpp/releases."
            )
        if not self.model_path.is_file():
            raise ServerError(f"modelo não encontrado: {self.model_path}")

        args = [
            exe,
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "--parallel", str(self.n_parallel),
            "-ngl", str(self.n_gpu_layers),
        ]
        if self.embedding:
            args += ["--embedding", "--pooling", "mean"]
        else:
            # --jinja usa o template de chat embutido no GGUF.
            args += ["--jinja", "--reasoning-budget", str(self.reasoning_budget)]
            if self.kv_cache_type:
                args += ["--cache-type-k", self.kv_cache_type,
                         "--cache-type-v", self.kv_cache_type]
        if self.lora_path:
            args += ["--lora", str(self.lora_path)]
        return args

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return

        command = self.command()
        log.info("subindo llama-server na porta %d: %s", self.port, self.model_path.name)

        stdout: object = asyncio.subprocess.DEVNULL
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("ab")
            stdout = self._log_handle

        self._process = await asyncio.create_subprocess_exec(
            *command, stdout=stdout, stderr=asyncio.subprocess.STDOUT
        )

    async def stop(self, *, timeout_s: float = 20.0) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_s)
            except TimeoutError:
                # Carregar/descarregar 8 GB pode demorar; matar é o último recurso.
                log.warning("llama-server na porta %d ignorou o terminate, matando", self.port)
                process.kill()
                await process.wait()

        handle = self._log_handle
        self._log_handle = None
        if handle is not None:
            handle.close()  # type: ignore[attr-defined]

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def __aenter__(self) -> LlamaServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
