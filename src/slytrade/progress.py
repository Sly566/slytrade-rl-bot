"""Progress reporting for long-running pipeline tasks.

Every stage of the pipeline prints a bold banner and, where a loop is involved,
per-item progress, so the console never sits silent during a long run. Training
uses stable-baselines3's built-in progress bar (enabled via ``progress_bar``).
"""

from __future__ import annotations

from rich.console import Console

console = Console()


def stage(title: str) -> None:
    """Print a bold stage banner."""
    console.print(f"\n[bold cyan]━━━ {title} ━━━[/bold cyan]")


def info(message: str) -> None:
    console.print(f"  {message}")


def progress(index: int, total: int, label: str) -> None:
    """Print one-line loop progress (cheap, no flicker)."""
    console.print(f"  [{index}/{total}] {label}")
