"""As superfícies onde VOCÊ decide: `lithium discoveries`, `recon`, `status`, `memories`.

Fiação de CLI é a classe que este repo já contou seis vezes: um comando que importa e
não faz nada passa em qualquer teste de unidade. Aqui tudo roda pelo `CliRunner` e as
asserções são sobre o BANCO depois.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lithium import cli

from reconkit import make_ctx, seed_discovery

runner = CliRunner()


def flat(result) -> str:
    """O Rich quebra linha em 80 colunas; a asserção é sobre o CONTEÚDO."""
    return " ".join(result.output.split())


@pytest.fixture
def wired(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, recon_enabled=False)
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: ctx.config)
    monkeypatch.setattr(cli, "_store", lambda cfg: ctx.store)
    yield ctx
    ctx.store.close()


def test_the_commands_exist_and_are_registered():
    out = runner.invoke(cli.app, ["--help"]).output
    assert "discoveries" in out and "recon" in out


def test_discoveries_lists_the_pending_ones_with_the_cost_footer(wired):
    seed_discovery(wired.store, kind="lead", lead_kind="pubmed",
                   lead_external_id="30712879", title="uma meta-análise")
    result = runner.invoke(cli.app, ["discoveries"])
    assert result.exit_code == 0, result.output
    assert "uma meta-análise" in result.output
    # O rodapé de consumo: é a resposta para "quanto isto me custou", e sem número
    # visível ela não existe em uso pessoal.
    assert "busca" in result.output and "leitura" in result.output


def test_approving_a_source_prints_the_explicit_refusal(wired):
    """No idioma do `RECUSADO:` de `_focus_new`. Marcar como aprovada e não fazer nada
    seria prometer o que a Fase D ainda não construiu."""
    did = seed_discovery(wired.store, kind="source")
    result = runner.invoke(cli.app, ["discoveries", "--approve", str(did)])
    assert result.exit_code == 0, result.output
    assert "PENDENTE" in flat(result) and "Fase D" in flat(result)
    assert wired.store.conn.execute(
        "SELECT status FROM discoveries WHERE id = ?", (did,)
    ).fetchone()["status"] == "deferred"


def test_rejecting_from_the_cli_writes_the_lesson(wired):
    did = seed_discovery(wired.store, kind="lead", lead_kind="pubmed",
                         lead_external_id="1", url="https://blog.invalid/a")
    result = runner.invoke(cli.app, ["discoveries", "--reject", str(did)])
    assert result.exit_code == 0, result.output
    assert wired.store.conn.execute(
        "SELECT COUNT(*) AS n FROM research_lessons").fetchone()["n"] == 1


def test_deciding_a_discovery_of_another_focus_is_refused_by_name(wired):
    """MUTAÇÃO: usar `active_focus()` no lugar de `discoveries.focus_id`.

    As duas alternativas silenciosas são piores: com `active_focus()`, uma observação
    lida sob um foco vira memória de OUTRO e é injetada em todo prompt dele; com
    `discoveries.focus_id` sem avisar, o CLI imprime "✓" sobre uma memória que não
    aparece em lugar nenhum enquanto este foco estiver ativo.
    """
    scale_id = int(wired.store.conn.execute(
        "SELECT id FROM evidence_scales LIMIT 1").fetchone()["id"])
    wired.store.conn.execute(
        "INSERT INTO focuses(id, slug, target, scale_id) VALUES(2, 'onco-x', 'y', ?)",
        (scale_id,))
    did = seed_discovery(wired.store, focus_id=2, kind="observation")

    result = runner.invoke(cli.app, ["discoveries", "--approve", str(did)])
    assert result.exit_code == 1
    assert "RECUSADO" in result.output
    assert "onco-x" in result.output and "bipolar-tag" in result.output
    assert wired.store.conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE source = 'recon'"
    ).fetchone()["n"] == 0


def test_deciding_twice_says_so_instead_of_doing_it_twice(wired):
    did = seed_discovery(wired.store, kind="source")
    runner.invoke(cli.app, ["discoveries", "--approve", str(did)])
    result = runner.invoke(cli.app, ["discoveries", "--approve", str(did)])
    assert result.exit_code == 1
    assert "já foi decidida" in result.output and "deferred" in result.output


def test_recon_without_the_key_says_exactly_how_to_turn_it_on(wired):
    result = runner.invoke(cli.app, ["recon"])
    assert result.exit_code == 0
    # `[recon]` LITERAL: o Rich trata colchete como tag de markup e o comeria, tirando
    # do terminal exatamente a linha que diz o que acrescentar.
    for token in ("config.local.toml", "[recon]", "enabled = true", "api_key",
                  "contact"):
        assert token in flat(result), token


def test_recon_now_without_the_key_refuses(wired):
    result = runner.invoke(cli.app, ["recon", "--now"])
    assert result.exit_code == 1
    assert wired.store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 0


def test_status_shows_the_recon_line_and_the_pending_count(wired):
    seed_discovery(wired.store)
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "recon" in result.output
    assert "buscas hoje" in result.output
    assert "esperando sua decisão" in result.output


def test_the_memories_screen_calls_a_web_note_web_and_not_you(wired):
    """MUTAÇÃO: deixar o mapa de origem com dois casos.

    A coluna "origem" existe precisamente para dizer se aquela memória passou por você.
    Com dois casos, uma memória lida na WEB aparece como **'você'** — falsa proveniência
    na única tela de auditoria que existe.
    """
    wired.store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active, focus_id) "
        "VALUES('a pagina diz X', 'fact', 'recon', 1, 1, 1)")
    wired.store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active) "
        "VALUES('voce prefere manha', 'preference', 'chat', 1, 1)")
    wired.store.conn.execute(
        "INSERT INTO memories(text, kind, source, confirmed, active) "
        "VALUES('queries com novel voltam vazias', 'search_lesson', 'research', 1, 1)")

    out = runner.invoke(cli.app, ["memories"]).output
    origens = {line.split()[1] for line in out.splitlines()
               if line.strip() and line.split()[0].isdigit()
               and len(line.split()) > 2 and line.split()[2] in
               ("fact", "preference", "search_lesson")}
    assert origens == {"web", "você", "pesquisa"}, out


def test_mode_off_says_the_scout_stops_costing_money(wired):
    out = runner.invoke(cli.app, ["mode", "off"]).output
    assert "batedor da web para de vez" in out
    assert "gasta dinheiro" in out.lower()
