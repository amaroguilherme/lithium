"""Testes de config — em especial as regras de portabilidade Mac↔Windows."""

from __future__ import annotations

from pathlib import Path

from lithium.config import PROJECT_ROOT, Config, load_config


def test_defaults_are_absolute():
    cfg = Config()
    assert cfg.data_dir.is_absolute()
    assert cfg.db_path.is_absolute()
    assert cfg.llm.model_path.is_absolute()


def test_relative_paths_resolve_against_project_root(tmp_path, monkeypatch):
    """Serviço supervisionado inicia em CWD arbitrário — o caminho não pode depender dele."""
    toml = tmp_path / "c.toml"
    toml.write_text(
        '[llm]\nmodel_path = "../base_models/x.gguf"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)  # CWD diferente da raiz do projeto
    cfg = load_config(toml)
    assert cfg.llm.model_path == (PROJECT_ROOT / ".." / "base_models" / "x.gguf").resolve()


def test_absolute_paths_pass_through(tmp_path):
    toml = tmp_path / "c.toml"
    absolute = tmp_path / "m.gguf"
    toml.write_text(f'[llm]\nmodel_path = "{absolute.as_posix()}"\n', encoding="utf-8")
    assert load_config(toml).llm.model_path == absolute


def test_missing_config_falls_back_to_defaults(tmp_path):
    assert load_config(tmp_path / "inexistente.toml") == Config()


def test_partial_toml_keeps_other_defaults(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text("[worker]\nconcurrency = 8\n", encoding="utf-8")
    cfg = load_config(toml)
    assert cfg.worker.concurrency == 8
    assert cfg.question.human_queue_limit == 5   # default preservado
    assert cfg.embedding.dim == 1024


def test_shipped_config_toml_is_valid():
    """O config.toml versionado precisa carregar — é o que o `lithium init` usa."""
    cfg = load_config(PROJECT_ROOT / "config.toml")
    assert cfg.llm.model_path.name == "gemma-4-12b-it-Q5_K_M.gguf"
    assert cfg.question.human_queue_limit == 5


def test_derived_dirs_live_under_data_dir(tmp_path):
    cfg = Config(data_dir=tmp_path)
    assert cfg.db_path.parent == tmp_path
    assert cfg.cache_dir.parent == tmp_path
    cfg.ensure_dirs()
    assert cfg.cache_dir.is_dir() and cfg.adapters_dir.is_dir()


def test_no_posix_only_apis_in_package():
    """Contrato de portabilidade: nada de fcntl/os.fork/waitpid no código do pacote.

    Analisa a AST em vez de fazer grep de texto. Grep acusaria a própria docstring
    que declara *não* usar essas APIs — e um lint com falso positivo é um lint que
    logo vira `# noqa`.
    """
    import ast

    banned_modules = {"fcntl", "pwd", "grp", "termios"}
    banned_attrs = {"fork", "waitpid", "kill", "setsid", "getuid", "SIGKILL"}
    offenders: list[str] = []

    for path in (PROJECT_ROOT / "lithium").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in banned_modules:
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in banned_modules:
                    offenders.append(f"{path.name}:{node.lineno} from {node.module}")
            elif isinstance(node, ast.Attribute) and node.attr in banned_attrs:
                # `proc.kill()` do asyncio é portável (vira TerminateProcess no
                # Windows); `os.kill`/`signal.SIGKILL` não são.
                base = node.value
                if isinstance(base, ast.Name) and base.id in {"os", "signal"}:
                    offenders.append(f"{path.name}:{node.lineno} {base.id}.{node.attr}")

    assert not offenders, offenders


def test_no_hardcoded_path_separators():
    """Caminho de arquivo vem de pathlib; separador literal quebra no Windows.

    URLs são exceção legítima — barra em URL é barra em todo SO. A heurística
    abaixo isenta linhas que claramente falam de URL/endpoint.
    """
    import re

    url_context = ("http", "://", "url", "endpoint", "base_url")
    # `/sair`, `/memorias` — comandos de barra do chat. Um único token minúsculo
    # depois da barra nunca é caminho de arquivo.
    slash_command = re.compile(r"""["']/[a-z]+["']""")

    offenders = []
    for path in (PROJECT_ROOT / "lithium").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if '"/' not in code and "'/" not in code:
                continue
            if any(token in code.lower() for token in url_context):
                continue
            if not re.sub(slash_command, "", code).count('"/') and not re.sub(
                slash_command, "", code
            ).count("'/"):
                continue
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}  {code.strip()}")
    assert not offenders, offenders


def test_project_is_hermetic_from_qyra():
    """Nenhum import de `qyra`. O único acoplamento permitido é o caminho do GGUF."""
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "lithium").rglob("*.py")
        if "import qyra" in path.read_text(encoding="utf-8")
        or "from qyra" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_package_data_includes_schema():
    assert (Path(__file__).parent.parent / "lithium" / "db" / "schema.sql").is_file()
