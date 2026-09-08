"""O log sobrevive ao processo — metade do item H.

`_setup_logging` instalava só `RichHandler(console)`. O `daemon.log` da primeira operação
(1-8/9) só existiu porque foi redirigido à mão com `nohup`; quem roda `lithium serve`
normalmente perde tudo ao fechar o terminal. E a taxa dos portões das cinco primeiras
fontes se perdeu num reboot que limpou o `/tmp`, ficando como "buraco declarado" em
`extraction_runs`.

É a mesma raiz que motivou aquela tabela: diagnóstico indo para um logger efêmero.
"""

from __future__ import annotations


def test_serve_writes_a_log_file_not_only_the_console(tmp_path):
    """O handler de arquivo faltava, e isso já custou dado real.

    O `daemon.log` da primeira operação só existiu porque foi redirigido à mão com
    `nohup`. Quem roda `lithium serve` normalmente perde tudo ao fechar o terminal — e a
    taxa dos portões das cinco primeiras fontes se perdeu num reboot que limpou o `/tmp`,
    ficando como "buraco declarado" em `extraction_runs`.

    MUTAÇÃO: voltar `handlers=[RichHandler(...)]` fixo em `_setup_logging`, ou chamar
    `_setup_logging(verbose)` sem o diretório.
    """
    import logging

    from typer.testing import CliRunner

    from lithium import cli
    from lithium.config import load_config
    from lithium.llm import LLMClient

    gen = tmp_path / "g.gguf"; gen.touch()
    emb = tmp_path / "e.gguf"; emb.touch()
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text(
        f'data_dir = "{tmp_path / "d"}"\n'
        f'[llm]\nmodel_path = "{gen}"\n[embedding]\nmodel_path = "{emb}"\n',
        encoding="utf-8")

    import lithium.daemon as D

    async def _pronto_embed(embedder, timeout_s=180.0):
        return None

    async def _pronto_gen(self, *a, **k):
        return None

    original_embed = D._wait_embedder
    original_gen = LLMClient.wait_healthy
    raiz = logging.getLogger()
    antes = list(raiz.handlers)
    D._wait_embedder = _pronto_embed
    LLMClient.wait_healthy = _pronto_gen
    try:
        r = CliRunner().invoke(cli.app, ["run", "-c", str(cfgfile), "--max-tasks", "0"])
        assert r.exit_code == 0, r.output
    finally:
        D._wait_embedder = original_embed
        LLMClient.wait_healthy = original_gen
        for h in list(raiz.handlers):
            if h not in antes:
                h.close()
                raiz.removeHandler(h)

    arquivo = load_config(cfgfile).data_dir / "lithium.log"
    assert arquivo.is_file(), "nenhum log em arquivo: o diagnóstico morre com o terminal"
    assert arquivo.stat().st_size > 0, "o arquivo existe e está vazio"


def test_a_bad_log_dir_does_not_kill_the_command(tmp_path, capsys):
    """Sem disco ou sem permissão, o console basta — o comando não pode morrer por causa
    do log.

    MUTAÇÃO: remover o `except OSError` de `_setup_logging`.
    """
    import logging

    from lithium.cli import _setup_logging

    raiz = logging.getLogger()
    antes = list(raiz.handlers)
    # um ARQUIVO onde se espera diretório: `mkdir` levanta OSError
    obstaculo = tmp_path / "obstaculo"
    obstaculo.write_text("nao sou diretorio", encoding="utf-8")
    try:
        _setup_logging(False, obstaculo / "dentro")
    finally:
        for h in list(raiz.handlers):
            if h not in antes:
                h.close()
                raiz.removeHandler(h)
    assert "desabilitado" in capsys.readouterr().out
