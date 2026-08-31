"""CLI.

`lithium serve` é um processo comum de longa duração. A supervisão fica deliberadamente
fora do código — launchd no Mac, Task Scheduler no Windows — para não amarrar o daemon
a um SO. Ver README.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from lithium.config import load_config
from lithium.sources.http import resolve_credential
from lithium.db import Store
from lithium.db.store import DIRECTNESS_WEIGHTS, GRADE_WEIGHTS
from lithium.types import Directness, Grade
from lithium.worker.queue import TaskQueue

app = typer.Typer(add_completion=False, help="Assistente autônomo de síntese de evidência.")
console = Console()

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Caminho do config TOML")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # O httpx loga a URL COMPLETA em INFO, e `PubMedSource._params` injeta `api_key`
    # como query param: a credencial da NCBI iria para o console e para qualquer handler
    # de arquivo, uma vez por query, 19 por harvest_sweep. Vazamento de credencial não
    # tem teste que falhe sozinho — por isso a correção vem com
    # `test_the_api_key_never_reaches_the_log`.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _active_profile(store, cfg):
    """O perfil do foco ativo, com erro LEGÍVEL em vez de traceback.

    Erro de perfil não pode derrubar os comandos de diagnóstico — `focus.toml` é
    editado à mão, e se `lithium focus` morresse com o TOML quebrado o usuário ficaria
    sem o único caminho de saída. Por isso `focus` (a listagem) NÃO chama isto.
    """
    from lithium.focus import NoActiveFocus, ProfileError, active_profile
    try:
        return active_profile(store, cfg.focuses_dir)[1]
    except (NoActiveFocus, ProfileError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _store(cfg) -> Store:
    """Abre o banco e aplica migrações pendentes.

    `init_schema` é idempotente e custa milissegundos. Rodá-lo em toda abertura, e não
    só no `lithium init`, é o que impede uma coluna nova de quebrar bancos existentes
    até alguém lembrar de reinicializar — o que já aconteceu uma vez com
    `tasks.origin`.
    """
    if not cfg.db_path.is_file():
        console.print("[red]banco não existe.[/red] rode `lithium init`")
        raise typer.Exit(1)
    store = Store(cfg.db_path, embedding_dim=cfg.embedding.dim)
    store.init_schema()
    return store


@app.command()
def init(config: ConfigOpt = None) -> None:
    """Cria o banco e semeia as tabelas de peso de evidência."""
    cfg = load_config(config)
    cfg.ensure_dirs()
    store = Store(cfg.db_path, embedding_dim=cfg.embedding.dim)
    store.init_schema()

    backend = "sqlite-vec (nativo)" if store.vec_available else "numpy (fallback)"
    console.print(f"[green]✓[/green] banco em [cyan]{cfg.db_path}[/cyan]")
    console.print(f"  busca vetorial: {backend}, {cfg.embedding.dim} dims")

    for label, path in (("geração", cfg.llm.model_path), ("embedding", cfg.embedding.model_path)):
        if path and path.is_file():
            console.print(f"  modelo de {label}: [cyan]{path.name}[/cyan] "
                          f"({path.stat().st_size / 1e9:.1f} GB)")
        else:
            console.print(f"  [yellow]![/yellow] modelo de {label} ausente: {path or '(não configurado)'}")


@app.command()
def serve(
    config: ConfigOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    external_servers: Annotated[
        bool,
        typer.Option("--external-servers",
                     help="Não gerenciar os llama-server; assumir que já estão no ar."),
    ] = False,
) -> None:
    """Sobe os llama-server e o pool de workers. Ctrl-C encerra."""
    from lithium.daemon import Daemon, MissingModel

    _setup_logging(verbose)
    cfg = load_config(config)
    daemon = Daemon(cfg, manage_servers=not external_servers)

    try:
        asyncio.run(daemon.run())
    except MissingModel as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]encerrando…[/yellow]")


@app.command()
def run(
    config: ConfigOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    max_tasks: Annotated[int | None, typer.Option(help="Parar após N tarefas")] = None,
) -> None:
    """Drena a fila uma vez e sai. Para desenvolvimento e execução manual."""
    from lithium.daemon import Daemon

    _setup_logging(verbose)
    cfg = load_config(config)

    async def _drain() -> None:
        await Daemon(cfg, manage_servers=False).run(drain=True, drain_max=max_tasks)

    asyncio.run(_drain())


@app.command()
def harvest(
    config: ConfigOpt = None,
    strategy: Annotated[list[str] | None, typer.Option("--strategy", "-s")] = None,
) -> None:
    """Enfileira uma varredura completa (ou só das estratégias indicadas)."""
    from lithium.pipeline.strategy import all_strategies, strategy_by_name

    cfg = load_config(config)
    store = _store(cfg)
    profile = _active_profile(store, cfg)

    unknown = [s for s in (strategy or []) if s not in strategy_by_name(profile)]
    if unknown:
        console.print(f"[red]estratégia desconhecida:[/red] {', '.join(unknown)}")
        console.print("disponíveis: "
                      f"{', '.join(s.name for s in all_strategies(profile))}")
        raise typer.Exit(1)

    task_id = TaskQueue(store).enqueue(
        "harvest_sweep", {"strategies": strategy or []}, priority=0.6
    )
    scope = ", ".join(strategy) if strategy else "todas as estratégias"
    console.print(f"[green]✓[/green] varredura enfileirada ({scope}), tarefa {task_id}")


def _discovery_line(row) -> tuple[str, str, str]:
    """(#, tipo·domínio, título) — sem o resumo, que vai numa linha própria."""
    import json as _json

    payload = _json.loads(row["payload_json"] or "{}")
    mark = " [parcial]" if payload.get("truncated") else ""
    return (str(row["id"]),
            f"{row['kind']} · {payload.get('domain') or '?'}",
            (row["title"] or row["url"]) + mark)


def _print_discoveries(store, focus_id: int, cfg, *, all_: bool = False) -> None:
    from lithium.recon import budget as recon_budget

    where = "" if all_ else " AND status = 'pending'"
    rows = store.conn.execute(
        "SELECT id, kind, status, url, title, summary, payload_json FROM discoveries "
        f" WHERE focus_id = ?{where} ORDER BY id", (focus_id,),
    ).fetchall()
    if not rows:
        console.print("[dim]nenhuma descoberta pendente neste foco.[/dim] "
                      "[dim]`lithium recon --now` faz uma varredura[/dim]")
    else:
        table = Table(box=None, pad_edge=False)
        table.add_column("#", justify="right")
        table.add_column("tipo · domínio")
        if all_:
            table.add_column("estado")
        table.add_column("descoberta", overflow="fold")
        for row in rows:
            ident, kind, title = _discovery_line(row)
            cells = [ident, kind] + ([row["status"]] if all_ else []) + [title]
            table.add_row(*cells)
        console.print(table)
        for row in rows:
            if row["summary"]:
                console.print(f"  [dim]#{row['id']} {row['summary']}[/dim]")

    spent = recon_budget.spent(store)
    console.print(
        f"\n[dim]hoje: {spent['search_calls']}/{cfg.recon.max_calls_per_day} busca(s) "
        f"faturada(s) · {spent['page_fetches']}/{cfg.recon.max_pages_per_day} "
        f"leitura(s) · mês: {spent['month_search_calls']} busca(s)[/dim]"
    )
    console.print("[dim]`--approve N` autoriza · `--reject N` recusa e vira lição · "
                  "sem decisão, expira[/dim]")


@app.command()
def discoveries(
    config: ConfigOpt = None,
    approve: Annotated[int | None, typer.Option(
        "--approve", help="Autorizar: lead vira artigo, observation vira memória")] = None,
    reject: Annotated[int | None, typer.Option(
        "--reject", help="Recusar. lead/source viram lição de pesquisa")] = None,
    all_: Annotated[bool, typer.Option("--all", "-a", help="Incluir as já decididas")] = False,
) -> None:
    """O que o batedor achou na web e está esperando a sua decisão.

    Nada aqui é evidência e nada aqui vira claim. Um `lead` aprovado faz o ARTIGO ser
    recolhido do PubMed e passar pelos dois portões normais; uma `observation` aprovada
    vira uma memória escopada a este foco; um `source` fica pendente até a Fase D.
    """
    import asyncio

    from lithium.recon.verbs import (
        AlreadyDecided,
        ReconMemoryUnavailable,
        approve as approve_verb,
        fetch as fetch_discovery,
        reject as reject_verb,
    )

    cfg = load_config(config)
    store = _store(cfg)
    focus = store.active_focus()
    if focus is None:
        console.print("[red]nenhum foco ativo.[/red] "
                      "escolha um com `lithium focus --use <slug>`")
        raise typer.Exit(1)
    focus_id = int(focus["id"])

    target = approve if approve is not None else reject
    if target is not None:
        row = fetch_discovery(store, target)
        if row is None:
            console.print(f"[yellow]descoberta #{target} não existe[/yellow]")
            raise typer.Exit(1)
        if int(row["focus_id"]) != focus_id:
            # RECUSA explícita, no idioma do `RECUSADO:` de `_focus_new`. As duas
            # alternativas silenciosas são piores: usar `active_focus()` gravaria uma
            # observação lida sob um foco como memória de OUTRO, e usar
            # `discoveries.focus_id` sem avisar imprimiria "✓" sobre uma memória que
            # não aparece em lugar nenhum enquanto este foco estiver ativo.
            owner = store.conn.execute(
                "SELECT slug FROM focuses WHERE id = ?", (int(row["focus_id"]),)
            ).fetchone()
            console.print(
                f"[red]RECUSADO:[/red] a descoberta #{target} é do foco "
                f"[cyan]{owner['slug'] if owner else row['focus_id']}[/cyan] e o foco "
                f"ativo é [cyan]{focus['slug']}[/cyan]. Volte para ele com "
                f"`lithium focus --use {owner['slug'] if owner else ''}`."
            )
            raise typer.Exit(1)

    if approve is not None:
        async def _approve() -> None:
            from lithium.llm import EmbeddingClient

            async with EmbeddingClient(cfg.embedding.base_url, cfg.embedding.model,
                                       dim=cfg.embedding.dim) as emb:
                decision = await approve_verb(store, TaskQueue(store), approve,
                                              embedder=emb)
            console.print(f"[green]✓[/green] #{approve} → {decision.status} "
                          f"[dim]{decision.detail}[/dim]")
            if decision.discovery_kind == "source":
                # Os próximos passos são impressos AQUI, não na camada de domínio: os
                # nomes dos comandos são verdade no CLI, e `verbs.py` embutir sintaxe de
                # CLI seria acoplamento na direção errada.
                console.print(
                    "  [dim]descreva a busca e ative:[/dim] "
                    "`lithium sources --spec <slug>` → `lithium sources --approve <slug>`"
                )
        try:
            asyncio.run(_approve())
        except AlreadyDecided as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            raise typer.Exit(1) from exc
        except ReconMemoryUnavailable as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        return

    if reject is not None:
        try:
            decision = reject_verb(store, reject)
        except AlreadyDecided as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            raise typer.Exit(1) from exc
        console.print(f"[green]✓[/green] #{reject} recusada "
                      f"[dim]{decision.detail}[/dim]")
        return

    _print_discoveries(store, focus_id, cfg, all_=all_)


@app.command()
def recon(
    config: ConfigOpt = None,
    now: Annotated[bool, typer.Option("--now", help="Fazer uma varredura agora")] = False,
    record_fixture: Annotated[bool, typer.Option(
        "--record-fixture",
        help="Grava UMA resposta real da API em tests/fixtures/recon/ e sai")] = False,
) -> None:
    """O batedor da web: estado, consumo e varredura sob demanda."""
    from lithium.mode import ResearchMode, get_mode
    from lithium.recon import budget as recon_budget
    from lithium.recon.search import enable_hint

    cfg = load_config(config)
    store = _store(cfg)

    if record_fixture:
        _record_brave_fixture(cfg, store)
        return

    if not now:
        state = "[green]ligado[/green]" if cfg.recon.enabled else "[yellow]desligado[/yellow]"
        console.print(f"batedor da web: {state} ({cfg.recon.provider})")
        spent = recon_budget.spent(store)
        console.print(
            f"  [dim]hoje {spent['search_calls']}/{cfg.recon.max_calls_per_day} "
            f"busca(s) · {spent['page_fetches']}/{cfg.recon.max_pages_per_day} "
            f"leitura(s) · mês {spent['month_search_calls']} busca(s)[/dim]"
        )
        if not cfg.recon.enabled:
            # `markup=False`: a mensagem contém `[recon]`, e o Rich o comeria como
            # tag — a linha que diz ao usuário EXATAMENTE o que acrescentar sumiria do
            # terminal. É um defeito de saída, não de estilo.
            console.print()
            console.print(enable_hint(cfg.recon.provider), markup=False, style="dim")
        return

    if not cfg.recon.enabled:
        console.print(enable_hint(cfg.recon.provider), markup=False, style="red")
        raise typer.Exit(1)
    if get_mode(store) is not ResearchMode.ON:
        # RECUSA em vez de forçar `on_demand`, e é o que mantém `mode off` sendo uma
        # torneira real sobre a única coisa deste sistema que tem FATURA.
        console.print(
            "[yellow]o modo pesquisa está desligado.[/yellow] o batedor da web é a "
            "única parte deste sistema que GASTA DINHEIRO, então `mode off` também o "
            "para. `lithium mode on` religa."
        )
        raise typer.Exit(1)

    task_id = TaskQueue(store).enqueue("recon_sweep", {}, priority=0.6)
    console.print(f"[green]✓[/green] varredura enfileirada (tarefa {task_id})")
    console.print("[dim]o daemon busca, tria e lê; veja com `lithium discoveries`[/dim]")


@app.command()
def ask(
    text: Annotated[str, typer.Argument(help="A pergunta")],
    config: ConfigOpt = None,
) -> None:
    """Injeta uma pergunta manual, com prioridade máxima.

    Enfileirada, não respondida na hora: classificar exige o LLM, que vive no daemon.
    """
    cfg = load_config(config)
    store = _store(cfg)
    task_id = TaskQueue(store).enqueue("ask_question", {"text": text}, priority=1.0)
    console.print(f"[green]✓[/green] pergunta enfileirada (tarefa {task_id})")
    console.print("[dim]o daemon classifica e roteia; veja com `lithium questions`[/dim]")


@app.command()
def questions(
    config: ConfigOpt = None,
    all_: Annotated[bool, typer.Option("--all", "-a", help="Incluir as já fechadas")] = False,
) -> None:
    """Perguntas abertas: o que está na fila humana e o que a máquina persegue."""
    cfg = load_config(config)
    store = _store(cfg)

    escalated = store.conn.execute("SELECT * FROM escalated_queue").fetchall()
    if escalated:
        console.print(f"[bold yellow]precisam de você ({len(escalated)})[/bold yellow]")
        for r in escalated:
            console.print(f"\n  [bold]#{r['id']}[/bold] [dim]{r['kind']} · "
                          f"prio {r['priority']:.2f} · {r['stuck_reason']}[/dim]")
            console.print(f"  {r['text']}")
            if r["partial_work"]:
                console.print(f"  [dim]por que travei: {r['partial_work']}[/dim]")
        console.print(f"\n  [dim]responda com `lithium answer <id> \"...\"`[/dim]")

    held = store.conn.execute(
        "SELECT COUNT(*) AS n FROM questions "
        " WHERE status = 'OPEN' AND stuck_reason IS NOT NULL "
        "   AND focus_id = (SELECT id FROM active_focus)"
    ).fetchone()["n"]
    if held:
        console.print(f"\n[dim]{held} represadas — sobem quando abrir vaga na fila[/dim]")

    researching = store.conn.execute(
        "SELECT id, kind, priority, status, text FROM questions "
        " WHERE status IN ('OPEN', 'RESEARCHING') AND stuck_reason IS NULL "
        "   AND focus_id = (SELECT id FROM active_focus) "
        " ORDER BY priority DESC LIMIT 15"
    ).fetchall()
    if researching:
        console.print(f"\n[bold]na fila automática ({len(researching)})[/bold]")
        for r in researching:
            console.print(f"  [{r['priority']:.2f}] [dim]{r['kind']:10}[/dim] {r['text']}")

    if all_:
        closed = store.conn.execute(
            "SELECT id, kind, status, text, answer FROM questions "
            " WHERE status LIKE 'ANSWERED%' ORDER BY answered_at DESC LIMIT 20"
        ).fetchall()
        if closed:
            console.print(f"\n[bold green]respondidas ({len(closed)})[/bold green]")
            for r in closed:
                console.print(f"  [dim]{r['status']}[/dim] {r['text']}")
                if r["answer"]:
                    console.print(f"    [dim]→ {r['answer'][:220]}[/dim]")
                _print_finding(store, r["id"])

    if not (escalated or researching or held):
        console.print("[dim]nenhuma pergunta ainda. rode `lithium serve` "
                      "ou `lithium ask \"...\"`[/dim]")


@app.command()
def answer(
    question_id: Annotated[int, typer.Argument(help="ID da pergunta")],
    text: Annotated[str, typer.Argument(help="Sua resposta")],
    config: ConfigOpt = None,
) -> None:
    """Responde uma pergunta escalada e libera a vaga na fila."""
    cfg = load_config(config)
    store = _store(cfg)

    row = store.conn.execute(
        "SELECT text, status FROM questions WHERE id = ?", (question_id,)
    ).fetchone()
    if row is None:
        console.print(f"[red]pergunta #{question_id} não existe[/red]")
        raise typer.Exit(1)

    store.conn.execute(
        "UPDATE questions SET status = 'ANSWERED_HUMAN', answer = ?, "
        "  answer_origin = 'human', "
        "  answered_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (text, question_id),
    )
    # Libera a vaga imediatamente: a próxima represada sobe.
    free = 5 - store.conn.execute(
        "SELECT COUNT(*) AS n FROM questions WHERE status = 'ESCALATED' "
        "   AND focus_id = (SELECT id FROM active_focus)"
    ).fetchone()["n"]
    promoted = store.conn.execute(
        "UPDATE questions SET status = 'ESCALATED', "
        "  escalated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
        " WHERE id IN (SELECT id FROM questions "
        "               WHERE status = 'OPEN' AND stuck_reason IS NOT NULL "
        "                 AND focus_id = (SELECT id FROM active_focus) "
        "               ORDER BY priority DESC LIMIT ?)",
        (max(0, free),),
    ).rowcount

    console.print(f"[green]✓[/green] #{question_id} respondida")
    if promoted:
        console.print(f"  [dim]{promoted} pergunta(s) represada(s) subiram para a fila[/dim]")


@app.command()
def chat(
    config: ConfigOpt = None,
    new: Annotated[bool, typer.Option("--new", help="Começar do zero, apagando o histórico")] = False,
) -> None:
    """Conversa com o assistente. Ele enxerga o corpus e pergunta antes de memorizar.

    Exige os llama-server no ar (`lithium serve`, ou suba-os à mão).
    """
    import asyncio

    from lithium.chat import ChatEngine
    from lithium.llm import EmbeddingClient, LLMClient, LLMError
    from lithium.llm.prompts import PromptTooLarge
    from lithium.llm.usage import UsageSink

    cfg = load_config(config)
    store = _store(cfg)

    async def loop() -> None:
        async with LLMClient(cfg.llm.base_url, cfg.llm.model,
                             timeout_s=cfg.llm.timeout_s) as llm, \
                   EmbeddingClient(cfg.embedding.base_url, cfg.embedding.model,
                                   dim=cfg.embedding.dim) as emb:
            if not await llm.healthy():
                console.print(f"[red]llama-server fora do ar em {cfg.llm.base_url}[/red]")
                console.print("[dim]rode `lithium serve` em outro terminal[/dim]")
                raise typer.Exit(1)

            # O SINK É NOMEADO e recebe flush no fim, e isso fecha um buraco real:
            # MEDIDO que 19 `CallRecord` sem flush produzem ZERO linhas em `llm_calls`.
            # Só o daemon fechava o ciclo, então toda sessão de chat com menos de 20
            # POSTs registrava NADA — e esse é o precedente exato de um relens síncrono.
            sink = UsageSink(store)
            llm.on_usage = sink.record
            engine = ChatEngine(store, llm, emb, profile=_active_profile(store, cfg),
                                n_ctx=cfg.llm.n_ctx)
            if new:
                engine.clear_history()

            n = len(engine.memories())
            focus = store.active_focus()
            pend = int(store.conn.execute(
                "SELECT COUNT(*) AS n FROM discoveries "
                " WHERE status = 'pending' AND focus_id = ?",
                (int(focus["id"]) if focus else 0,),
            ).fetchone()["n"])
            console.print(
                "[dim]comandos: /sair · /novo · /memorias · /esquecer <id>"
                " · /descobertas · /aprovar <id> · /rejeitar <id>"
                f"  ·  {n} memória(s) ativa(s)"
                + (f" · {pend} descoberta(s) pendente(s)" if pend else "")
                + "[/dim]\n"
            )

            while True:
                try:
                    text = console.input("[bold cyan]você ›[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    await sink.flush()
                    return
                if not text:
                    continue
                if text in ("/sair", "/quit", "/exit"):
                    await sink.flush()
                    return
                if text == "/memorias":
                    _print_memories(engine)
                    continue
                if text == "/novo":
                    # IMPLEMENTADO. O `except PromptTooLarge` abaixo já oferecia `/novo`
                    # desde a Fase A, e ele não existia — a mensagem prescrevia um
                    # comando inexistente para um erro que, até esta fase, era
                    # inalcançável (o chat nunca chamava `budget_guard`).
                    console.print(f"[dim]{engine.clear_history()} mensagem(ns) "
                                  "apagada(s)[/dim]\n")
                    continue
                if text.startswith("/esquecer"):
                    parts = text.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        ok = engine.forget(int(parts[1]))
                        console.print("[green]esquecida[/green]" if ok
                                      else "[yellow]id não encontrado[/yellow]")
                    else:
                        console.print("[dim]uso: /esquecer <id>[/dim]")
                    continue
                if text.split()[0] in ("/descobertas", "/aprovar", "/rejeitar"):
                    # NO TERMINAL, fora do contexto do modelo — exatamente como
                    # `/memorias`. O texto de uma descoberta PENDENTE não entra em
                    # prompt nenhum; ver `ChatEngine._recon_notes`.
                    await _chat_discovery_verb(store, cfg, emb, text)
                    continue

                try:
                    with console.status("[dim]pensando…[/dim]"):
                        turn = await engine.send(text)
                except PromptTooLarge as exc:
                    console.print(f"[red]prompt grande demais:[/red] {exc}")
                    if engine.system_alone_overflows():
                        console.print(
                            "[yellow]nem o prompt de sistema cabe — limpar o histórico "
                            "não resolve.[/yellow] [dim]algum bloco injetado cresceu "
                            "sem teto: veja `lithium memories` e `lithium "
                            "discoveries`[/dim]\n"
                        )
                    else:
                        console.print("[dim]/novo limpa o histórico[/dim]\n")
                    continue
                except LLMError as exc:
                    # Sem isto, um turno longo derrubava a sessão inteira com traceback.
                    console.print(f"[red]falha na chamada:[/red] {exc}\n")
                    continue

                console.print(f"\n[bold]{turn.reply}[/bold]\n"
                              if False else f"\n{turn.reply}\n")

                if turn.proposal is not None:
                    console.print(f"[yellow]memorizar isto?[/yellow] "
                                  f"[dim]({turn.proposal.kind})[/dim] {turn.proposal.text}")
                    console.print(f"[dim]{turn.proposal.rationale}[/dim]")
                    answer = console.input("[dim]s/n › [/dim]").strip().lower()
                    if answer in ("s", "sim", "y", "yes"):
                        mid = await engine.remember(turn.proposal)
                        console.print(f"[green]✓ memória #{mid}[/green]\n")
                    else:
                        engine.decline(turn.proposal)
                        console.print("[dim]ok, não guardei[/dim]\n")

    asyncio.run(loop())


FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "recon"


def _record_brave_fixture(cfg, store) -> None:
    """Grava UMA resposta real da API de busca, para os testes pararem de ser sintéticos.

    O repo exige fixture REAL ("testar contra XML inventado esconde exatamente os casos
    que quebram na prática") e a fixture da Brave não pôde ser gravada durante a
    implementação — a conta exige cartão. Em vez de esconder isso, o buraco é declarado:
    a fixture atual se chama `.SYNTHETIC.json`, carrega um campo `_PROVENANCE` que diz
    exatamente o que ela é, e este comando é o caminho de UM passo para substituí-la.

    Debita a cota como qualquer outra busca — gravar fixture custa dinheiro igual.
    """
    import asyncio
    import json as _json

    from lithium.recon import budget as recon_budget
    from lithium.recon.search import BraveSearch, ReconDisabled

    try:
        searcher = BraveSearch(cfg.recon)
    except ReconDisabled as exc:
        console.print(str(exc), markup=False, style="red")
        raise typer.Exit(1) from exc
    if not recon_budget.debit_search(store, cfg.recon.max_calls_per_day):
        console.print("[yellow]teto diário de buscas atingido[/yellow]")
        raise typer.Exit(1)

    query = "bipolar maintenance lithium guideline"

    async def _go() -> dict:
        try:
            response = await searcher._client.get(  # noqa: SLF001
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 10, "freshness": "pm"},
                headers={"Accept": "application/json",
                         "X-Subscription-Token": searcher._api_key},  # noqa: SLF001
            )
            response.raise_for_status()
            return response.json()
        finally:
            await searcher.aclose()

    payload = asyncio.run(_go())
    # O header NUNCA vai para o arquivo: só o corpo, e o corpo não carrega a chave.
    destination = FIXTURE_PATH / "brave_web_search.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    console.print(f"[green]✓[/green] resposta real gravada em [cyan]{destination}[/cyan]")
    console.print("[dim]apague o .SYNTHETIC.json e tire as marcas @pytest.mark."
                  "provisional dos testes de parsing[/dim]")


async def _chat_discovery_verb(store, cfg, embedder, text: str) -> None:
    """`/descobertas`, `/aprovar <id>`, `/rejeitar <id>` — os mesmos verbos do CLI.

    Uma implementação só, em `lithium/recon/verbs.py`, para as duas superfícies. Duas
    cópias divergiriam na primeira correção, e a que importa (a ordem
    memória-antes-de-status) é justamente a que se perde numa reescrita apressada.
    """
    from lithium.recon.verbs import (
        AlreadyDecided,
        ReconMemoryUnavailable,
        approve as approve_verb,
        fetch as fetch_discovery,
        reject as reject_verb,
    )

    focus = store.active_focus()
    if focus is None:
        console.print("[red]nenhum foco ativo[/red]")
        return
    parts = text.split()
    verb = parts[0]
    if verb == "/descobertas":
        _print_discoveries(store, int(focus["id"]), cfg)
        return
    if len(parts) != 2 or not parts[1].isdigit():
        console.print(f"[dim]uso: {verb} <id>[/dim]")
        return

    discovery_id = int(parts[1])
    row = fetch_discovery(store, discovery_id)
    if row is None or int(row["focus_id"]) != int(focus["id"]):
        console.print(f"[yellow]#{discovery_id} não é uma descoberta deste foco[/yellow]")
        return
    try:
        if verb == "/aprovar":
            decision = await approve_verb(store, TaskQueue(store), discovery_id,
                                          embedder=embedder)
        else:
            decision = reject_verb(store, discovery_id)
    except AlreadyDecided as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return
    except ReconMemoryUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(f"[green]✓[/green] #{discovery_id} → {decision.status} "
                  f"[dim]{decision.detail}[/dim]\n")


def _print_memories(engine) -> None:
    rows = engine.memories()
    if not rows:
        console.print("[dim]nenhuma memória ainda[/dim]")
        return
    table = Table(box=None, pad_edge=False)
    table.add_column("#", justify="right")
    table.add_column("tipo")
    table.add_column("memória", overflow="fold")
    for r in rows:
        table.add_row(str(r["id"]), r["kind"], r["text"])
    console.print(table)


@app.command()
def memories(
    config: ConfigOpt = None,
    add: Annotated[str | None, typer.Option("--add", help="Adicionar uma memória")] = None,
    kind: Annotated[str, typer.Option("--kind", help="preference|context|constraint|fact")] = "fact",
    forget: Annotated[int | None, typer.Option("--forget", help="Desativar por id")] = None,
    all_: Annotated[bool, typer.Option("--all", "-a", help="Incluir recusadas e inativas")] = False,
    declined: Annotated[bool, typer.Option(
        "--declined", help="Listar só as recusadas (que não voltam a ser propostas)")] = False,
    allow: Annotated[int | None, typer.Option(
        "--allow", help="Desfazer uma recusa: volta a poder ser proposta")] = None,
    forget_duplicates: Annotated[bool, typer.Option(
        "--forget-duplicates", help="Desativar lições de pesquisa repetidas")] = False,
) -> None:
    """Lista, adiciona ou remove memórias."""
    import asyncio

    from lithium.chat import ChatEngine
    from lithium.llm import EmbeddingClient, LLMClient

    cfg = load_config(config)
    store = _store(cfg)

    if add is not None:
        async def _add() -> None:
            async with LLMClient(cfg.llm.base_url, cfg.llm.model) as llm, \
                       EmbeddingClient(cfg.embedding.base_url, cfg.embedding.model,
                                       dim=cfg.embedding.dim) as emb:
                mid = await ChatEngine(
                    store, llm, emb, profile=_active_profile(store, cfg)
                ).add_manual(add, kind)
                console.print(f"[green]✓[/green] memória #{mid}")
        asyncio.run(_add())
        return

    if forget is not None:
        cur = store.conn.execute(
            "UPDATE memories SET active = 0 WHERE id = ? AND active = 1", (forget,)
        )
        console.print("[green]✓ esquecida[/green]" if cur.rowcount
                      else "[yellow]id não encontrado ou já inativa[/yellow]")
        return

    if allow is not None:
        cur = store.conn.execute(
            "DELETE FROM memories WHERE id = ? AND confirmed = 0 AND active = 0",
            (allow,),
        )
        console.print("[green]✓ recusa desfeita — pode ser proposta de novo[/green]"
                      if cur.rowcount
                      else "[yellow]id não encontrado entre as recusadas[/yellow]")
        return

    if forget_duplicates:
        ids = store.forget_duplicate_lessons()
        if ids:
            console.print(f"[green]✓ {len(ids)} lição(ões) repetida(s) desativada(s)"
                          f"[/green] [dim]({', '.join(f'#{i}' for i in ids)})[/dim]")
        else:
            console.print("[dim]nenhuma lição repetida[/dim]")
        return

    where = "WHERE confirmed = 1 AND active = 1 "
    if all_:
        where = ""
    elif declined:
        where = "WHERE confirmed = 0 AND active = 0 "
    sql = ("SELECT id, text, kind, source, confirmed, active FROM memories "
           + where + "ORDER BY id")
    rows = store.conn.execute(sql).fetchall()
    if not rows:
        console.print("[dim]nenhuma memória recusada[/dim]" if declined
                      else "[dim]nenhuma memória. converse com `lithium chat`[/dim]")
        return

    table = Table(box=None, pad_edge=False)
    table.add_column("#", justify="right")
    table.add_column("origem")
    table.add_column("tipo")
    if all_ or declined:
        table.add_column("estado")
    table.add_column("memória", overflow="fold")
    for r in rows:
        estado = ("ativa" if r["active"] and r["confirmed"]
                  else "recusada" if not r["confirmed"] else "esquecida")
        # A origem é o que diz se aquela memória passou por você ou foi gravada
        # sozinha — auto-gravada não significa inauditável. TRÊS casos desde a Fase C:
        # sem o terceiro, uma memória lida na WEB apareceria como "você" na única tela
        # de auditoria de proveniência que existe.
        origem = {"research": "pesquisa", "recon": "web"}.get(r["source"], "você")
        cells = ([str(r["id"]), origem, r["kind"]]
                 + ([estado] if all_ or declined else [])
                 + [r["text"]])
        table.add_row(*cells)
    console.print(table)

    auto = sum(1 for r in rows if r["source"] == "research")
    if auto:
        console.print(f"\n[dim]{auto} gravada(s) automaticamente pela pesquisa · "
                      f"`--forget <id>` remove qualquer uma[/dim]")


def _print_finding(store, question_id: int) -> None:
    """As citações e os alertas do achado, ao lado dele.

    Aqui é onde o revisor lê. Um alerta que só existe em `findings.safety_json` é um
    alerta que ninguém vê — e a lista vazia NÃO é renderizada como "nada casou": um
    atestado afirmativo de limpeza é pior que silêncio, porque o screen é um casamento de
    termo pequeno, não uma checagem de interação medicamentosa.
    """
    import json as _json

    row = store.conn.execute(
        "SELECT citations_json, safety_json FROM findings "
        " WHERE question_id = ? ORDER BY id DESC LIMIT 1",
        (question_id,),
    ).fetchone()
    if row is None:
        return
    pmids = [c.get("external_id") for c in _json.loads(row["citations_json"] or "[]")]
    if pmids:
        console.print(f"    [dim]fontes: {', '.join(p for p in pmids if p)}[/dim]")
    for alert in _json.loads(row["safety_json"] or "[]"):
        colour = "red" if alert.get("severity") == "high" else "yellow"
        console.print(f"    [{colour}]! {alert['text']}[/{colour}]")


@app.command()
def mode(
    state: Annotated[str | None, typer.Argument(help="on | off (vazio = mostrar)")] = None,
    config: ConfigOpt = None,
) -> None:
    """Liga ou desliga o modo pesquisa 24/7.

    Desligado, o daemon continua no ar e atende o que você pedir — chat, `ask`,
    `harvest`, `explore` — mas não pesquisa por conta própria. A memória segue ativa
    nos dois modos: aprender com o que você diz é conversa, não pesquisa.
    """
    from lithium.mode import ResearchMode, backlog, get_mode, set_mode

    cfg = load_config(config)
    store = _store(cfg)

    if state is None:
        current = get_mode(store)
        colour = "green" if current is ResearchMode.ON else "yellow"
        console.print(f"modo pesquisa: [{colour}]{current.label}[/{colour}]")
        held = backlog(store)
        if held:
            console.print(f"  [dim]{held} tarefa(s) agendada(s) represada(s)[/dim]")
        return

    normalized = state.strip().lower()
    aliases = {"on": ResearchMode.ON, "ligado": ResearchMode.ON, "1": ResearchMode.ON,
               "off": ResearchMode.OFF, "desligado": ResearchMode.OFF, "0": ResearchMode.OFF}
    if normalized not in aliases:
        console.print(f"[red]valor inválido: {state}[/red]  [dim]use on | off[/dim]")
        raise typer.Exit(1)

    chosen = set_mode(store, aliases[normalized])
    if chosen is ResearchMode.ON:
        held = backlog(store)
        console.print("[green]✓[/green] pesquisa 24/7 ligada")
        if held:
            console.print(f"  [dim]{held} tarefa(s) represada(s) voltam a rodar[/dim]")
    else:
        console.print("[yellow]✓[/yellow] pesquisa 24/7 desligada — só sob demanda")
        console.print("  [dim]chat, ask, harvest e explore continuam funcionando[/dim]")
        console.print("  [dim]a memória segue ativa: ele continua aprendendo com você[/dim]")
        # E o recon NÃO continua, nem sob demanda. É a única coisa aqui que tem fatura,
        # então "sob demanda" para ele significaria "ainda gasta dinheiro".
        console.print("  [yellow]o batedor da web para de vez[/yellow] "
                      "[dim]— é a única parte que gasta dinheiro, e `lithium recon "
                      "--now` recusa enquanto o modo estiver desligado[/dim]")


@app.command()
def pursue(
    config: ConfigOpt = None,
    n: Annotated[int, typer.Option("--n", help="Quantas hipóteses perseguir")] = 2,
) -> None:
    """Transforma hipóteses da trilha exploratória em buscas dirigidas.

    É o que faz o sistema pesquisar um composto que ele mesmo propôs — as frentes
    fixas do harvest nunca mencionam sigma-1, orexina ou via transdérmica.
    """
    cfg = load_config(config)
    store = _store(cfg)
    task_id = TaskQueue(store).enqueue("pursue_speculation", {"limit": n}, priority=0.7)
    console.print(f"[green]✓[/green] perseguição enfileirada (tarefa {task_id})")


@app.command()
def reground(
    config: ConfigOpt = None,
    n: Annotated[int, typer.Option("--n")] = 3,
) -> None:
    """Reavalia cadeias mecanísticas contra o corpus atual.

    Elos `assumed` que a literatura recém-coletada passou a sustentar viram
    `supported` — é assim que a plausibilidade de uma hipótese sobe com o tempo.
    """
    cfg = load_config(config)
    store = _store(cfg)
    task_id = TaskQueue(store).enqueue("reground_speculations", {"limit": n}, priority=0.6)
    console.print(f"[green]✓[/green] reancoragem enfileirada (tarefa {task_id})")


@app.command()
def explore(
    config: ConfigOpt = None,
    show: Annotated[bool, typer.Option("--show", help="Só mostrar o quadro atual")] = False,
    n: Annotated[int, typer.Option("--n", help="Quantas hipóteses gerar")] = 3,
    refuted: Annotated[bool, typer.Option("--refuted", help="Incluir as reprovadas")] = False,
) -> None:
    """Trilha exploratória: hipóteses mecanísticas, incluindo o que ninguém testou."""
    cfg = load_config(config)
    store = _store(cfg)

    if not show:
        task_id = TaskQueue(store).enqueue("explore_tick", {"max_items": n}, priority=0.6)
        console.print(f"[green]✓[/green] tick exploratório enfileirado (tarefa {task_id})")
        console.print("[dim]o daemon gera e critica; veja com `lithium explore --show`[/dim]")
        return

    rows = store.conn.execute(
        "SELECT * FROM speculation_board "
        + ("" if refuted else "WHERE survives_critique = 1 ")
        + "ORDER BY plausibility * COALESCE(novelty, 0) DESC LIMIT 20"
    ).fetchall()
    if not rows:
        console.print("[dim]quadro especulativo vazio. rode `lithium explore`[/dim]")
        return

    for r in rows:
        score = r["plausibility"] * (r["novelty"] or 0)
        mark = "" if r["survives_critique"] else " [red](reprovada)[/red]"
        console.print(f"\n[bold]{r['statement']}[/bold]{mark}")
        console.print(
            f"  [dim]{r['mechanism_target']} · {r['intervention_class']}[/dim]"
        )
        console.print(
            f"  [dim]plausibilidade {r['plausibility']:.0%} "
            f"({r['supported_links']}/{r['chain_length']} elos citados) · "
            f"ineditismo {r['novelty']:.0%} · score {score:.2f}[/dim]"
        )
        console.print(f"  [cyan]refuta se:[/cyan] {r['falsifier']}")
        console.print(f"  [cyan]próximo teste:[/cyan] {r['test_proposal']}")
        if r["known_risks"]:
            console.print(f"  [yellow]riscos:[/yellow] {r['known_risks']}")


@app.command()
def tokens(
    config: ConfigOpt = None,
    days: Annotated[int | None, typer.Option("--days", help="Só os últimos N dias")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Custo de inferência por etapa. Uma linha por POST HTTP, não por chamada lógica."""
    import json as _json

    from lithium.llm.usage import summarize

    cfg = load_config(config)
    store = _store(cfg)
    rows = summarize(store, since_days=days)

    if as_json:
        console.print_json(_json.dumps(rows))
        return
    if not rows:
        console.print("[dim]nenhuma chamada registrada. rode `lithium serve`[/dim]")
        return

    table = Table(box=None, pad_edge=False)
    for column, justify in (("etapa", "left"), ("n", "right"), ("reparos", "right"),
                            ("falhas", "right"), ("prompt p50", "right"),
                            ("prompt p95", "right"), ("máx", "right"),
                            ("ms p50", "right"), ("horas", "right"),
                            ("tokens", "right")):
        table.add_column(column, justify=justify)
    for r in rows:
        table.add_row(
            r["label"], str(r["n"]), str(r["repairs"] or 0), str(r["failures"] or 0),
            str(r["p50_prompt"] or "—"), str(r["p95_prompt"] or "—"),
            str(r["max_prompt"] or "—"), str(r["p50_ms"] or "—"),
            f"{(r['total_ms'] or 0) / 3600000:.2f}",
            f"{r['total_tokens'] or 0:,}".replace(",", "."),
        )
    console.print(table)
    console.print("\n[dim]reparos = POSTs com attempt > 0; cada um carrega o prompt "
                  "inteiro mais os rounds falhos anteriores[/dim]")


@app.command()
def strategies(config: ConfigOpt = None) -> None:
    """Lista as frentes de busca do foco ATIVO e o directness esperado de cada uma."""
    from lithium.pipeline.strategy import all_strategies

    cfg = load_config(config)
    store = _store(cfg)
    for s in all_strategies(_active_profile(store, cfg)):
        console.print(f"\n[bold cyan]{s.name}[/bold cyan]  "
                      f"[dim]directness esperado: {s.expected_directness.value} · "
                      f"prioridade {s.priority}[/dim]")
        console.print(f"  {s.rationale}")
        for q in s.queries:
            console.print(f"    [dim]·[/dim] {q}")


@app.command()
def status(config: ConfigOpt = None) -> None:
    """Contagens do banco e estado da fila."""
    cfg = load_config(config)
    store = _store(cfg)

    counts = store.counts()
    focus = store.active_focus()
    table = Table(show_header=False, box=None)
    table.add_row("foco ativo", f"[bold]{focus['slug'] if focus else '— NENHUM —'}[/bold]")
    for key, value in counts.items():
        table.add_row(key, f"[bold]{value}[/bold]")
    console.print(table)
    # Três causas DISTINTAS para o mesmo placar zerado, e só a primeira tem remédio
    # de usuário. MEDIDO no contador único: uma claim COM aresta mas em escala
    # divergente contava como "sem julgamento de directness neste foco" — falso, e é a
    # mensagem que manda o usuário rodar 7,7 h de GPU que não podem mudar o número.
    if counts["claims_unjudged"]:
        console.print(
            f"[yellow]{counts['claims_unjudged']} claim(s) verificada(s) SEM "
            "julgamento de directness neste foco: elas não entram em nenhum peso, "
            "nenhum placar e nenhuma resposta.[/yellow] "
            "rode `lithium focus --relens` para julgá-las"
        )
    if counts["claims_off_scale"]:
        console.print(
            f"[yellow]{counts['claims_off_scale']} claim(s) graduada(s) em OUTRA "
            "escala de evidência: elas não podem ser pesadas neste foco.[/yellow] "
            "[dim]o relens NÃO resolve isto — re-julgar a população não muda a régua "
            "de grade. Enquanto `grade` for coluna da claim e não aresta, a única "
            "saída é um foco na mesma escala.[/dim]"
        )
    if counts["claims_out_of_scope"]:
        console.print(
            f"[dim]{counts['claims_out_of_scope']} claim(s) julgada(s) FORA DE ESCOPO "
            "neste foco — julgadas e irrelevantes, não esquecidas.[/dim]"
        )
    resto = (counts["claims_unweighted"] - counts["claims_unjudged"]
             - counts["claims_off_scale"] - counts["claims_out_of_scope"])
    if resto > 0:
        # O quarto modo: escala casa, aresta existe, e mesmo assim não há peso —
        # `scale_levels` incompleta num dos eixos. Não tem remédio de usuário, mas
        # sumir dos contadores seria pior: o placar ficaria zerado sem explicação.
        console.print(
            f"[red]{resto} claim(s) sem peso por motivo NÃO CLASSIFICADO[/red] — "
            "provavelmente a escala do foco não declara todos os níveis de grade ou "
            "de directness. Confira `scale_levels`."
        )
    if focus is None:
        console.print(
            "[red]nenhum foco ativo — todo placar do sistema vale zero.[/red] "
            "escolha um com `lithium focus --use <slug>`"
        )

    # O batedor. Duas linhas, e a de consumo existe porque ele é a ÚNICA parte deste
    # sistema com fatura — sem um número visível, "quanto isto me custou" não tem
    # resposta em uso pessoal, onde ninguém olha um dashboard.
    from lithium.recon import budget as recon_budget

    pending_discoveries = int(store.conn.execute(
        "SELECT COUNT(*) AS n FROM discoveries "
        " WHERE status = 'pending' AND focus_id = (SELECT id FROM active_focus)"
    ).fetchone()["n"])
    spent = recon_budget.spent(store)
    state = "ligado" if cfg.recon.enabled else "desligado"
    console.print(
        f"\n[bold]recon[/bold] ({state}) · buscas hoje "
        f"{spent['search_calls']}/{cfg.recon.max_calls_per_day} · leituras hoje "
        f"{spent['page_fetches']}/{cfg.recon.max_pages_per_day} · mês "
        f"{spent['month_search_calls']} busca(s)"
    )
    if pending_discoveries:
        console.print(f"[yellow]{pending_discoveries} descoberta(s) esperando sua "
                      "decisão[/yellow] — `lithium discoveries`")

    queue = TaskQueue(store)
    stats = queue.stats()
    if stats:
        console.print("\n[bold]fila[/bold]")
        line = Table(show_header=False, box=None)
        for state, n in sorted(stats.items()):
            colour = {"dead": "red", "running": "yellow", "done": "green"}.get(state, "white")
            line.add_row(state, f"[{colour}]{n}[/{colour}]")
        console.print(line)

    dead = queue.dead_letters(limit=5)
    if dead:
        console.print("\n[bold red]dead-letter[/bold red]")
        for letter in dead:
            console.print(f"  [{letter['id']}] {letter['kind']}: "
                          f"{(letter['error'] or '')[:110]}")


@app.command()
def claims(
    config: ConfigOpt = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Lista as claims verificadas, mais pesadas primeiro."""
    cfg = load_config(config)
    store = _store(cfg)

    focus = store.active_focus()
    rows = store.conn.execute(
        "SELECT c.statement, c.grade, cd.directness, cw.weight, s.external_id, s.year "
        "  FROM claims c "
        "  JOIN claim_weight cw ON cw.claim_id = c.id "
        "  JOIN claim_directness cd ON cd.claim_id = c.id "
        "                           AND cd.focus_id = (SELECT id FROM active_focus) "
        "  JOIN sources s ON s.id = c.source_id "
        " ORDER BY cw.weight DESC LIMIT ?",
        (limit,),
    ).fetchall()

    console.print(f"[dim]foco: {focus['slug'] if focus else '— NENHUM —'}"
                  f"{' · ' + focus['target'] if focus else ''}[/dim]")
    if not rows:
        console.print("[dim]nenhuma claim com peso neste foco.[/dim]")
        return

    table = Table(box=None, pad_edge=False)
    table.add_column("peso", justify="right")
    table.add_column("grade")
    table.add_column("direct.")
    table.add_column("PMID")
    table.add_column("alegação", overflow="fold")
    for r in rows:
        table.add_row(f"{r['weight']:.2f}", r["grade"], r["directness"],
                      r["external_id"], r["statement"])
    console.print(table)


@app.command()
def focus(
    config: ConfigOpt = None,
    use: Annotated[str | None, typer.Option("--use", help="Slug do foco a ativar")] = None,
    retire: Annotated[str | None, typer.Option("--retire")] = None,
    new: Annotated[str | None, typer.Option("--new", help="Cria perfil + linha")] = None,
    show: Annotated[bool, typer.Option("--show", help="Perfil do foco ativo")] = False,
    relens: Annotated[bool, typer.Option("--relens", help="Julga o que falta")] = False,
) -> None:
    """Lista os focos, ou troca o ativo.

    Existe porque sem ele `meta['active_focus']` só é escrita pelo seed, e uma chave que
    ninguém pode trocar não é um objeto de primeira classe: é uma constante com passos
    extras. É também onde a validação acontece — errar o slug aqui deixaria o placar
    zerado até a próxima abertura de CLI descobrir.
    """
    cfg = load_config(config)
    store = _store(cfg)

    def _resolve(slug: str):
        row = store.conn.execute(
            # `*`, não a lista curta: `_report_switch_cost` precisa de `scale_id`
            # para saber quantas claims o destino sequer PODE julgar.
            "SELECT * FROM focuses WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            console.print(f"[red]foco desconhecido:[/red] {slug}")
            raise typer.Exit(1)
        return row

    if retire:
        row = _resolve(retire)
        active = store.active_focus()
        if active is not None and int(active["id"]) == int(row["id"]):
            console.print(
                "[red]este é o foco ATIVO.[/red] aposentá-lo zeraria todo o placar em "
                "silêncio; troque antes com `lithium focus --use <outro>`"
            )
            raise typer.Exit(1)
        store.conn.execute(
            "UPDATE focuses SET retired_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') "
            " WHERE id = ?", (row["id"],),
        )
        console.print(f"[green]✓[/green] foco {retire} aposentado")
        return

    if new:
        _focus_new(store, cfg, new)
        return

    if show:
        _focus_show(store, cfg)
        return

    if relens:
        active = store.active_focus()
        if active is None:
            console.print("[red]nenhum foco ativo[/red]")
            raise typer.Exit(1)
        task_id = TaskQueue(store).enqueue(
            "relens_sweep", {}, priority=0.35, origin="scheduled",
            dedup_key=f"relens_sweep:{int(active['id'])}",
        )
        if task_id is None:
            console.print("[yellow]já existe uma varredura de relens na fila[/yellow]")
            return
        pending = _relens_cost(store, cfg, int(active["id"]), active["scale_id"])
        console.print(
            f"[green]✓[/green] relens enfileirado (tarefa {task_id}): "
            f"{pending[0]} claim(s) a julgar, ~{pending[1]:.1f} h de GPU.\n"
            "[dim]as tarefas entram como `scheduled`, então `lithium mode off` as "
            "para — é o único cancelamento que existe.[/dim]"
        )
        return

    if use:
        row = _resolve(use)
        if row["retired_at"]:
            console.print(
                f"[red]o foco {use} está aposentado[/red] — ativá-lo deixaria "
                "`claim_weight` vazia sem nenhum erro"
            )
            raise typer.Exit(1)
        _report_switch_cost(store, cfg, row)
        store.conn.execute(
            "INSERT INTO meta(key, value) VALUES('active_focus', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(int(row["id"])),),
        )
        console.print(f"[green]✓[/green] foco ativo: {use}")
        console.print(
            "[yellow]reinicie o daemon (`lithium serve` / `lithium work`).[/yellow]\n"
            "[dim]A razão NÃO é o cache de prompts: MEDIDO que `@cache _load(name)` "
            "guarda o Template PARSEADO e `render()` substitui a cada chamada, então o "
            "prompt acompanha a troca AO VIVO — e uma conexão já aberta enxerga "
            "imediatamente a troca de meta['active_focus'] feita por outra. O que "
            "exige reinício é a fila DURÁVEL: tarefas de colheita enfileiradas sob o "
            "foco anterior recusam com FocusDrift ao executar, e precisam ser drenadas "
            "ou descartadas.[/dim]"
        )
        return

    active = store.active_focus()
    table = Table(box=None, pad_edge=False)
    table.add_column("")
    table.add_column("slug")
    table.add_column("alvo", overflow="fold")
    table.add_column("estado")
    for r in store.conn.execute("SELECT * FROM focuses ORDER BY id"):
        mark = "→" if active is not None and int(r["id"]) == int(active["id"]) else " "
        table.add_row(mark, r["slug"], r["target"],
                      "aposentado" if r["retired_at"] else "ativo")
    console.print(table)
    if Store.hypotheses_are_globally_unique(store.conn):
        console.print(
            "[yellow]este banco carrega o `UNIQUE(statement)` legado em `hypotheses`: "
            "dois focos não podem sustentar a mesma hipótese aqui. O rebuild é da "
            "Fase B.[/yellow]"
        )


# ─────────────────────────────────────────────────────────────── focos: helpers


def _relens_cost(store, cfg, focus_id: int, scale_id) -> tuple[int, float]:
    """Quantas claims o foco destino não julgou, e quantas HORAS isso custa."""
    from lithium.pipeline.relens import SECONDS_PER_CLAIM, claims_to_judge

    n = len(claims_to_judge(store, focus_id, scale_id))
    return n, n * SECONDS_PER_CLAIM / 3600


def _report_switch_cost(store, cfg, row) -> None:
    """O custo da troca, ANTES de trocar.

    É o único lugar onde ele fica visível — antes desta fase `--use` imprimia
    `✓ foco ativo: {slug}` e mais nada. O idioma já existe no repo: `--retire` recusa
    com motivo escrito e `mode off` reporta o backlog.

    Os dois números são separados porque os remédios são OPOSTOS e só um existe: o
    relens julga as não-julgadas e NÃO pode fazer nada pelas de escala divergente.
    """
    focus_id, scale_id = int(row["id"]), row["scale_id"]
    n_unjudged, hours = _relens_cost(store, cfg, focus_id, scale_id)
    off_scale = int(store.conn.execute(
        "SELECT COUNT(*) AS n FROM claims WHERE verified = 1 AND scale_id IS NOT ?",
        (scale_id,),
    ).fetchone()["n"])

    console.print(f"[bold]custo da troca para {row['slug']}[/bold]")
    if n_unjudged:
        console.print(
            f"  {n_unjudged} claim(s) sem julgamento neste foco — "
            f"~{hours:.1f} h de GPU em `lithium focus --relens`"
        )
    if off_scale:
        console.print(
            f"  [yellow]{off_scale} claim(s) em escala divergente — o relens NÃO "
            f"resolve[/yellow]"
        )
    pendentes = int(store.conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE status IN ('pending', 'running') "
        "  AND kind IN ('harvest_query', 'fetch_source', 'extract_source')"
    ).fetchone()["n"])
    if pendentes:
        console.print(
            f"  [yellow]{pendentes} tarefa(s) de colheita pendentes do foco "
            f"anterior[/yellow] — elas vão RECUSAR ao executar (FocusDrift) em vez de "
            f"colher com a query de um foco e julgar contra o alvo do outro"
        )
    if not (n_unjudged or off_scale or pendentes):
        console.print("  [dim]nenhum — o corpus está em dia neste foco[/dim]")


def _focus_show(store, cfg) -> None:
    """O perfil do foco ativo, e a divergência de hash quando existe."""
    from lithium.focus import ProfileError, profile_for_slug

    active = store.active_focus()
    if active is None:
        console.print("[red]nenhum foco ativo[/red]")
        raise typer.Exit(1)
    try:
        profile = profile_for_slug(cfg.focuses_dir, active["slug"])
    except ProfileError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table(show_header=False, box=None)
    table.add_row("slug", f"[bold]{profile.slug}[/bold]")
    table.add_row("alvo (BANCO)", active["target"])
    table.add_row("alvo (TOML)", profile.focus.target)
    table.add_row("leitor", profile.focus.reader)
    table.add_row("pergunta", profile.focus.mechanistic_question)
    table.add_row("perfil em", str(profile.dir))
    table.add_row("estratégias", str(len(profile.strategies.strategies)))
    table.add_row("classes", str(len(profile.taxonomy.intervention_classes)))
    table.add_row("riscos permanentes",
                  ", ".join(r.name for r in profile.focus.standing_risks) or "— nenhum —")
    table.add_row("segurança",
                  "ausência DECLARADA" if profile.safety is None
                  else f"{len(profile.safety.rules)} regra(s)")
    console.print(table)

    console.print("\n[bold]directness neste foco[/bold]")
    console.print(profile.directness_definitions())

    # As DUAS metades, recomputadas separadamente: elas derivam por motivos diferentes
    # e pedem remédios OPOSTOS, então uma comparação só daria o conselho errado em
    # dois dos três casos.
    if active["profile_hash"] and active["profile_hash"] != Store.calibration_hash():
        console.print(
            "\n[yellow]a CALIBRAÇÃO divergiu[/yellow] — os pesos em types.py mudaram "
            "desde que este foco nasceu. O peso das claims NÃO foi reescrito, de "
            "propósito. O caminho é escala NOVA + foco NOVO."
        )
    esperado = Store.judgment_hash(active["target"], profile.directness_definitions())
    if active["judgment_hash"] is None:
        console.print(
            "\n[dim]este foco nasceu antes do `judgment_hash` — nada a comparar. "
            "Ele passa a ser gravado no próximo `focus --new`.[/dim]"
        )
    elif active["judgment_hash"] != esperado:
        console.print(
            "\n[yellow]o JULGAMENTO divergiu[/yellow] — `target` ou as definições de "
            "directness mudaram desde que as arestas deste foco foram gravadas. Toda "
            "linha em `claim_directness` foi julgada contra a régua ANTIGA.\n"
            "  o remédio é `lithium focus --relens`, não escala nova."
        )
    if profile.focus.target != active["target"]:
        console.print(
            f"\n[yellow]o alvo do TOML difere do alvo do BANCO[/yellow] — o BANCO "
            f"vence, porque toda aresta de `claim_directness` foi julgada contra ele. "
            f"Editar o TOML depois não reescreve julgamento nenhum."
        )


def _focus_new(store, cfg, slug: str) -> None:
    """Cria o diretório de perfil E a linha no banco, numa transação só.

    A recusa de ESCALA NOVA está aqui e não no relens, e ela é o item 19 desta fase.
    MEDIDO o que acontece sem ela: `UPDATE claims SET scale_id = 2` leva o peso da
    MESMA claim de 0,68 no foco 1 para NADA, `claim_weight` volta vazia ao reativar o
    foco antigo, sem erro e sem log — e é IRREVERSÍVEL, porque reconstruir exige
    re-derivar um `grade` que veio de julgamento de LLM e não é reproduzível bit a bit.
    O conserto correto é `grade` virar ARESTA, espelhando o que a Fase A fez com
    `directness`, e isso é fase própria — não um flag.
    """
    import shutil

    from lithium.focus import ProfileError, load_profile

    dest = cfg.focuses_dir / slug
    exists = store.conn.execute(
        "SELECT id FROM focuses WHERE slug = ?", (slug,)
    ).fetchone()
    if exists is not None:
        console.print(f"[red]o foco {slug} já existe no banco[/red] (id {exists['id']})")
        raise typer.Exit(1)

    if not dest.is_dir():
        template = Path(__file__).resolve().parent.parent / "focuses" / "bipolar-tag"
        if not template.is_dir():
            console.print(f"[red]nenhum perfil de referência em {template}[/red]")
            raise typer.Exit(1)
        shutil.copytree(template, dest)
        # `target` sai do template AUSENTE, nunca vazio: a carga tem de falhar ALTO na
        # primeira execução. Um `target = ""` passa o NOT NULL do SQLite, renderiza
        # `matches ""` no prompt, e — como o BANCO vence para `target` — corrigir o
        # TOML depois já não conserta as arestas que o relens gravou contra a string
        # vazia.
        focus_toml = dest / "focus.toml"
        focus_toml.write_text(
            focus_toml.read_text(encoding="utf-8")
            .replace(
                'slug   = "bipolar-tag"',
                f'slug   = "{slug}"\n\n'
                "# Apague esta linha DEPOIS de revisar os quatro arquivos. Enquanto ela\n"
                "# estiver aqui o perfil não carrega, e é de propósito: os outros três\n"
                "# TOML ainda são a cópia do foco de referência, com o domínio DELE.\n"
                "scaffold_pending = true",
            )
            .replace('target = "bipolar I + comorbid GAD"',
                     "# PREENCHA: sem esta chave a carga do perfil falha, de propósito.\n"
                     "# target = \"...\""),
            encoding="utf-8",
        )
        console.print(f"[green]✓[/green] perfil criado em {dest}")
        console.print(
            "[yellow]o perfil é uma CÓPIA do foco de referência[/yellow] — estratégias, "
            "taxonomia e segurança ainda são do domínio dele. Preencha `target`, revise "
            "os quatro arquivos, apague `scaffold_pending` e rode `lithium focus --new` "
            "de novo."
        )
        return

    try:
        profile = load_profile(dest)
    except ProfileError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    scale = store.conn.execute(
        "SELECT id FROM evidence_scales WHERE slug = ?", (profile.focus.scale,)
    ).fetchone()
    if scale is None:
        vivo = store.conn.execute(
            "SELECT f.slug FROM focuses f JOIN claims c ON c.scale_id = f.scale_id "
            " WHERE f.retired_at IS NULL AND c.verified = 1 LIMIT 1"
        ).fetchone()
        if vivo is not None:
            console.print(
                f"[red]RECUSADO: o perfil pede a escala nova {profile.focus.scale!r}, "
                f"mas o foco {vivo['slug']!r} está VIVO na escala das claims "
                f"atuais.[/red]\n"
                "Criar a escala e migrar as claims para ela levaria o peso das mesmas "
                "claims a ZERO no foco antigo — sem erro, sem log e sem reconstrução "
                "possível, porque re-derivar `grade` exige um julgamento de LLM que "
                "não é reproduzível bit a bit.\n"
                "[dim]Enquanto `grade` for coluna da claim e não aresta, escala nova "
                "só é segura com todos os outros focos aposentados. Aposente-os, ou "
                f"aponte este perfil para uma escala existente.[/dim]"
            )
            raise typer.Exit(1)

    with store.tx() as conn:
        if scale is None:
            # Escala NOVA é semeada dos DOIS eixos a partir de types.py. Semear só
            # directness (que é o que o perfil declara) deixaria `scale_levels` sem
            # nenhum nível de `grade`, e `claim_weight` — que dá JOIN nos dois —
            # ficaria PERMANENTEMENTE vazia, sem erro e sem log.
            cur = conn.execute(
                "INSERT INTO evidence_scales(slug, description) VALUES(?, ?) RETURNING id",
                (profile.focus.scale, f"escala do foco {slug}"),
            )
            scale_id = int(cur.fetchone()["id"])
            conn.executemany(
                "INSERT INTO scale_levels(scale_id, axis, value, weight, rank) "
                "VALUES(?, ?, ?, ?, ?)",
                [(scale_id, "grade", v, w, r) for v, w, r in GRADE_WEIGHTS]
                + [(scale_id, "directness", v, w, r) for v, w, r in DIRECTNESS_WEIGHTS],
            )
            n_grade = conn.execute(
                "SELECT COUNT(*) AS n FROM scale_levels WHERE scale_id = ? "
                "  AND axis = 'grade'", (scale_id,)).fetchone()["n"]
            n_dir = conn.execute(
                "SELECT COUNT(*) AS n FROM scale_levels WHERE scale_id = ? "
                "  AND axis = 'directness'", (scale_id,)).fetchone()["n"]
            if (n_grade, n_dir) != (len(Grade), len(Directness)):
                raise RuntimeError(
                    f"escala {profile.focus.scale!r} nasceria com {n_grade} grades e "
                    f"{n_dir} directness; `claim_weight` dá JOIN nos DOIS eixos e "
                    f"ficaria vazia para sempre, sem erro."
                )
        else:
            scale_id = int(scale["id"])
            console.print(
                f"[dim]escala {profile.focus.scale!r} já existe — reusada como está. "
                f"Reescrever peso de escala viva reescreveria julgamento "
                f"histórico.[/dim]"
            )
        conn.execute(
            "INSERT INTO focuses(slug, target, scale_id, profile_hash, judgment_hash) "
            "VALUES(?, ?, ?, ?, ?)",
            (slug, profile.focus.target, scale_id, Store.calibration_hash(),
             Store.judgment_hash(profile.focus.target,
                                 profile.directness_definitions())),
        )
    console.print(f"[green]✓[/green] foco {slug} criado. "
                  f"ative com `lithium focus --use {slug}`")





@app.command()
def sources(
    config: ConfigOpt = None,
    approve: Annotated[str | None, typer.Option(
        "--approve", help="Ativa a fonte: passa a ser consultada e a poder virar claim",
    )] = None,
    revoke: Annotated[str | None, typer.Option(
        "--revoke", help="Desativa sem apagar; o corpus já colhido continua válido",
    )] = None,
    spec: Annotated[str | None, typer.Option(
        "--spec", help="Slug cuja spec de busca/fetch será lida do stdin (JSON)",
    )] = None,
    evidence: Annotated[bool, typer.Option(
        "--evidence/--no-evidence",
        help="Junto de --approve: esta fonte pode produzir claims?",
    )] = False,
) -> None:
    """Lista o registro de fontes, ou aprova, revoga e descreve uma.

    Existe porque sem ele o registro é uma tabela que ninguém pode mudar — e uma tabela
    que ninguém pode mudar é uma constante com passos extras, que é exatamente o que
    `EVIDENCE_KINDS` era.

    `--approve` e `--evidence` são separados de propósito. "Consulte esta fonte" e "o que
    ela devolve pode virar evidência graduável" são duas afirmações diferentes: um registro
    de ensaios é útil para descobrir o que existe e não é desenho de estudo. Juntá-las faria
    a segunda pegar carona na primeira, que é como uma bula virou claim com peso 0,408 na
    medição do item 9.
    """
    import json as _json
    import sys as _sys

    cfg = load_config(config)
    store = _store(cfg)

    def _row(slug: str):
        row = store.conn.execute(
            "SELECT * FROM sources_registry WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            console.print(f"[red]fonte desconhecida:[/red] {slug}")
            raise typer.Exit(1)
        return row

    if spec:
        _row(spec)
        raw = _sys.stdin.read().strip()
        if not raw:
            console.print(
                "[red]nada no stdin.[/red] Passe o JSON da spec, por exemplo:\n"
                '  echo \'{"search": {"query_param": "search", "id_path": "results", '
                '"id_field": "id"}, "fetch": {"path": "works/{id}", '
                '"fields": {"title": "title", "passages": ["abstract"]}}}\' '
                "| lithium sources --spec openalex-org"
            )
            raise typer.Exit(1)
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError as exc:
            console.print(f"[red]JSON inválido:[/red] {exc}")
            raise typer.Exit(1) from exc
        store.conn.execute(
            "UPDATE sources_registry SET search_spec_json = ?, fetch_spec_json = ? "
            " WHERE slug = ?",
            (_json.dumps(parsed.get("search") or {}),
             _json.dumps(parsed.get("fetch") or {}), spec),
        )
        console.print(f"[green]✓[/green] spec de {spec} gravada")
        return

    if revoke:
        _row(revoke)
        store.conn.execute(
            "UPDATE sources_registry SET approved_at = NULL WHERE slug = ?", (revoke,))
        console.print(
            f"[green]✓[/green] {revoke} desativada. [dim]O que ela já colheu continua "
            f"no corpus e continua pesando: revogar não reescreve julgamento passado."
            f"[/dim]"
        )
        return

    if approve:
        row = _row(approve)
        if evidence and not (row["search_spec_json"] and row["fetch_spec_json"]) \
                and row["adapter"] == "http":
            console.print(
                "[red]RECUSADO:[/red] esta fonte não tem spec de busca, então ativá-la "
                "como fonte de EVIDÊNCIA enfileiraria tarefas que morrem no primeiro "
                "`search`. Descreva com `lithium sources --spec` primeiro."
            )
            raise typer.Exit(1)
        store.conn.execute(
            "UPDATE sources_registry "
            "   SET approved_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'), "
            "       yields_evidence = ? "
            " WHERE slug = ?",
            (1 if evidence else 0, approve),
        )
        console.print(
            f"[green]✓[/green] {approve} ativada"
            + (" [bold]como fonte de evidência[/bold]" if evidence
               else " [dim](não produz claims — só descoberta)[/dim]")
        )
        console.print("  [dim]reinicie o daemon: as fontes são montadas na subida.[/dim]")
        return

    rows = list(store.conn.execute(
        "SELECT r.slug, r.description, r.yields_evidence, r.approved_at, r.adapter,"
        "       r.credential_ref, r.search_spec_json,"
        "       (SELECT COUNT(*) FROM sources s WHERE s.kind = r.slug) AS colhidas "
        "  FROM sources_registry r ORDER BY r.approved_at IS NULL, r.slug"
    ))
    if not rows:
        console.print("[yellow]registro vazio.[/yellow]")
        return

    table = Table(box=None, pad_edge=False)
    for col in ("", "slug", "estado", "evidência", "adapter", "colhidas", "descrição"):
        table.add_column(col)
    for r in rows:
        ativa = r["approved_at"] is not None
        falta_spec = (r["adapter"] == "http" and not r["search_spec_json"])
        table.add_row(
            "→" if ativa else " ",
            f"[bold]{r['slug']}[/bold]" if ativa else r["slug"],
            "ativa" if ativa else "[yellow]proposta[/yellow]",
            "[bold]sim[/bold]" if r["yields_evidence"] else "não",
            r["adapter"] + ("[yellow] sem spec[/yellow]" if falta_spec else ""),
            str(r["colhidas"]),
            (r["description"] or "")[:44],
        )
    console.print(table)

    sem_cred = [r["slug"] for r in rows
                if r["approved_at"] and r["credential_ref"]
                and resolve_credential(r["credential_ref"], cfg) is None
                and r["adapter"] != "pubmed"]
    if sem_cred:
        console.print(
            f"[yellow]credencial não resolve para: {', '.join(sem_cred)}[/yellow] — "
            f"estas fontes ficam FORA do daemon em vez de tentar sem auth."
        )
    if not any(r["approved_at"] and r["yields_evidence"] for r in rows):
        console.print(
            "[red]nenhuma fonte de evidência ativa:[/red] a colheita não produz claims."
        )

if __name__ == "__main__":
    app()
