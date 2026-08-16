"""Rich interactive console GUI.

A task-oriented front end over the whole pipeline. Instead of remembering
subcommands and flags, the user picks a numbered task and answers a couple of
prompts; the task then runs end-to-end (e.g. "Collect data" gathers bars for
every timeframe plus ticks in one step).
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from slytrade import tasks

console = Console()


TASKS = [
    ("collect", "Collect market data (all timeframes + ticks)"),
    ("align", "Align bars & ticks into a research dataset"),
    ("backtest", "Run strategy backtests"),
    ("train", "Train an RL policy (PPO/SAC/TD3)"),
    ("walk-forward", "Walk-forward validation"),
    ("promote", "Promote a trained model through a stage"),
    ("paper", "Run the paper-trading loop"),
    ("demo", "Run the live demo-account loop"),
    ("reconcile", "Broker reconciliation / preflight"),
    ("doctor", "Environment health check"),
    ("quit", "Exit"),
]


def _print_header() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]SlyTrade RL Bot[/bold cyan] — production console\n"
            "Full pipeline: collect → align → backtest → train → walk-forward → paper → demo",
            title="SlyTrade",
            border_style="cyan",
        )
    )


def _print_menu() -> None:
    table = Table(title="Tasks", show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Task")
    table.add_column("Description")
    for index, (key, description) in enumerate(TASKS, start=1):
        table.add_row(str(index), key, description)
    table.add_row("0", "quit", "Exit")
    console.print(table)


def _prompt(prompt: str, default: str = "") -> str:
    return Prompt.ask(prompt, default=default).strip()


def _prompt_int(prompt: str, default: int) -> int:
    return IntPrompt.ask(prompt, default=default)


def task_collect() -> None:
    symbol = _prompt("Symbol", "XAUUSD").upper()
    lookback = _prompt("Lookback (1d / 1w / 1m / 1y / 2y)", "1y")
    source = _prompt("Source (hybrid / auto / mt5 / exness / samples)", "hybrid").lower()
    console.print(f"[cyan]Collecting {symbol} ({lookback}, source={source})…[/cyan]")
    result = tasks.collect_all(symbol, lookback=lookback, source=source)
    _render_result(result)


def task_align() -> None:
    symbol = _prompt("Symbol", "XAUUSD").upper()
    timeframe = _prompt("Timeframe", "M1")
    console.print(f"[cyan]Aligning {symbol} {timeframe}…[/cyan]")
    result = tasks.align(symbol, timeframe=timeframe)
    _render_result(result)


def task_backtest() -> None:
    symbol = _prompt("Symbol", "XAUUSD").upper()
    aligned = Path("data/processed/aligned") / symbol
    bars_file = _prompt("Aligned bars file", str(aligned / "bars.parquet"))
    if not Path(bars_file).exists():
        console.print("[yellow]Bars file not found; run align first (or check the path).[/yellow]")
        return
    strategy = _prompt("Strategy (persona-adaptive / ict-confluence / ma-cross)", "persona-adaptive")
    result = tasks.backtest(bars_file, strategy=strategy, symbol=symbol)
    _render_result(result)


def task_train() -> None:
    symbol = _prompt("Symbol", "XAUUSD").upper()
    aligned = Path("data/processed/aligned") / symbol
    bars_file = _prompt("Aligned bars file", str(aligned / "bars.parquet"))
    if not Path(bars_file).exists():
        console.print("[yellow]Bars file not found; run align first.[/yellow]")
        return
    algorithm = _prompt("Algorithm (ppo / sac / td3)", "ppo")
    policy = _prompt("Policy (mlp / lstm)", "mlp")
    reward = _prompt("Reward (r_multiple / trade_pnl / risk_adjusted / raw)", "r_multiple")
    timesteps = _prompt_int("Total timesteps", 50_000)
    console.print(f"[cyan]Training {algorithm.upper()} ({policy}) for {timesteps} steps…[/cyan]")
    result = tasks.train(bars_file, symbol=symbol, algorithm=algorithm, total_timesteps=timesteps, policy=policy, reward=reward)
    _render_result(result)


def task_walk_forward() -> None:
    symbol = _prompt("Symbol", "XAUUSD").upper()
    aligned = Path("data/processed/aligned") / symbol
    bars_file = _prompt("Aligned bars file", str(aligned / "bars.parquet"))
    if not Path(bars_file).exists():
        console.print("[yellow]Bars file not found; run align first.[/yellow]")
        return
    reward = _prompt("Reward (r_multiple / trade_pnl / risk_adjusted / raw)", "r_multiple")
    policy = _prompt("Policy (mlp / lstm)", "mlp")
    result = tasks.walk_forward(bars_file, symbol=symbol, reward=reward, policy=policy)
    _render_result(result)


def task_promote() -> None:
    model_id = _prompt("Model id (e.g. ppo-XAUUSD-42)", "ppo-XAUUSD-42")
    stage = _prompt("Stage (paper / shadow / demo)", "paper")
    result = tasks.promote(model_id, stage=stage)
    _render_result(result)


def task_paper() -> None:
    from slytrade.runtime.metrics_server import MetricsServer
    from slytrade.runtime.paper_loop import MT5QuoteProvider, PaperTradingLoop, ReplayQuoteProvider
    from slytrade.runtime.settings import RuntimeSettings

    symbol = _prompt("Symbol", "XAUUSD").upper()
    replay = _prompt("Replay ticks file (leave empty for live MT5)", "")
    settings = RuntimeSettings()
    settings.symbol = symbol

    provider: object
    if replay:
        from slytrade.backtest.reporting import load_ticks_file

        provider = ReplayQuoteProvider(load_ticks_file(Path(replay)), symbol=symbol)
    else:
        from slytrade.cli import load_mt5

        provider = MT5QuoteProvider(symbol, load_mt5(), poll_seconds=settings.poll_seconds)

    loop = PaperTradingLoop(settings, provider)  # type: ignore[arg-type]
    server = MetricsServer(port=settings.metrics_port, bind=settings.metrics_bind, metrics=loop.metrics) if settings.metrics_enabled else None
    if server:
        server.start()
        console.print(f"[green]Metrics on :{settings.metrics_port}[/green] (Ctrl+C to stop)")
    try:
        loop.run()
    except KeyboardInterrupt:
        console.print("[yellow]Stopping paper loop…[/yellow]")
    finally:
        if server:
            server.stop()


def task_demo() -> None:
    from slytrade.runtime.demo_loop import DemoTradingLoop
    from slytrade.runtime.settings import RuntimeSettings, TradingStage

    settings = RuntimeSettings()
    settings.symbol = _prompt("Symbol", "XAUUSD").upper()
    if not settings.allow_live or settings.stage != TradingStage.DEMO:
        console.print("[red]Demo trading requires SLYTRADE_ALLOW_LIVE=1 and SLYTRADE_STAGE=demo.[/red]")
        console.print("Set them in your .env, then re-run.")
        return
    from slytrade.cli import load_mt5

    loop = DemoTradingLoop(settings, load_mt5())
    console.print("[bold red]LIVE DEMO TRADING — real orders on the demo account.[/bold red]")
    if not Confirm.ask("Proceed?"):
        return
    try:
        loop.run()
    except KeyboardInterrupt:
        console.print("[yellow]Stopping demo loop…[/yellow]")


def task_reconcile() -> None:
    from slytrade.brokers.mt5_adapter import MT5BrokerAdapter
    from slytrade.cli import load_mt5
    from slytrade.execution.oms import OrderManagementSystem
    from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails

    symbol = _prompt("Symbol", "XAUUSD").upper()
    mt5 = load_mt5()
    adapter = MT5BrokerAdapter(
        mt5,
        oms=OrderManagementSystem(),
        guardrails=TradingGuardrails(GuardrailConfig(), initial_equity=1.0),
        allow_trading=False,
        expected_positions={},
    )
    try:
        adapter.connect()
        resolved = adapter.resolve_symbol(symbol)
        quote = adapter.quote(resolved)
        reconciliation = adapter.reconcile()
        console.print(f"[green]Connected. {symbol} -> {resolved}[/green]")
        console.print(f"  quote: bid={quote.bid} ask={quote.ask} spread={quote.spread}")
        console.print(f"  reconciliation: {'[green]OK[/green]' if reconciliation.reconciled else '[red]BLOCKED[/red]'} ({reconciliation.detail})")
    except Exception as exc:
        console.print(f"[red]Broker check failed: {exc}[/red]")
    finally:
        adapter.disconnect()


def task_doctor() -> None:
    from slytrade.cli import doctor

    doctor()


def _render_result(result: tasks.TaskResult) -> None:
    style = "green" if result.ok else "red"
    console.print(Panel.fit(result.message, border_style=style, title="Result"))
    if result.data:
        table = Table(show_header=True)
        table.add_column("Key")
        table.add_column("Value")
        for key, value in result.data.items():
            if isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            table.add_row(str(key), str(value))
        console.print(table)


def run_ui() -> None:
    """Interactive task loop (the `slytrade ui` entry point)."""
    _print_header()
    handlers = {
        "collect": task_collect,
        "align": task_align,
        "backtest": task_backtest,
        "train": task_train,
        "walk-forward": task_walk_forward,
        "promote": task_promote,
        "paper": task_paper,
        "demo": task_demo,
        "reconcile": task_reconcile,
        "doctor": task_doctor,
    }
    while True:
        _print_menu()
        try:
            raw = _prompt("Task (number or name)", "0").lower()
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye.")
            return
        if raw in ("0", "quit", "q", "exit"):
            console.print("Goodbye.")
            return

        key = raw
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(TASKS):
                key = TASKS[index][0]
            else:
                console.print("[red]Invalid selection[/red]")
                continue

        handler = handlers.get(key)
        if handler is None:
            console.print(f"[red]Unknown task: {raw}[/red]")
            continue
        try:
            handler()
        except KeyboardInterrupt:
            console.print("[yellow]Cancelled.[/yellow]")
        except Exception as exc:  # keep the UI alive on task failure
            console.print(f"[red]Task failed: {exc}[/red]")
