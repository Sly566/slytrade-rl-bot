from __future__ import annotations

import importlib.util
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="SlyTrade RL Bot CLI")
console = Console()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@app.command()
def doctor() -> None:
    """Check project health and dependencies."""
    table = Table(title="SlyTrade RL Bot Doctor")
    table.add_column("Check")
    table.add_column("Status")

    required = ["numpy", "pandas", "yaml", "pydantic", "typer", "rich"]
    optional = [
        "pyarrow",
        "polars",
        "torch",
        "gymnasium",
        "stable_baselines3",
        "optuna",
        "mlflow",
        "mt5linux",
    ]

    for mod in required:
        table.add_row(f"required:{mod}", "OK" if module_available(mod) else "MISSING")

    for mod in optional:
        table.add_row(f"optional:{mod}", "OK" if module_available(mod) else "MISSING")

    for path in [
        "configs/assets.yaml",
        "configs/broker.yaml",
        "configs/data.yaml",
        "configs/risk.yaml",
        "configs/training.yaml",
    ]:
        table.add_row(f"file:{path}", "OK" if Path(path).exists() else "MISSING")

    console.print(table)


@app.command()
def info() -> None:
    """Print project information."""
    console.print("[bold green]SlyTrade RL Bot[/bold green]")
    console.print("Production-grade MT5 tick-and-bar based ICT/SMC RL trading system.")
    console.print("Live trading is disabled by default.")


@app.command()
def live() -> None:
    """Live trading placeholder."""
    console.print("[bold red]Live trading is disabled at bootstrap stage.[/bold red]")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
