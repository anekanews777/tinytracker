"""Terminal display helpers using Rich."""

from typing import Any, List

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tinytracker.models import Run

console = Console()

# Color scheme
COLOR_ID = "cyan"
COLOR_PROJECT = "dim"
COLOR_TIMESTAMP = "dim"
COLOR_PARAMS = "yellow"
COLOR_METRICS = "green"
COLOR_TAGS = "magenta"
COLOR_TAG_BG = "grey23"
COLOR_NOTES = "blue"
COLOR_SUCCESS = "bold green"
COLOR_ERROR = "bold red"
COLOR_WARNING = "bold yellow"
COLOR_INFO = "bold blue"
COLOR_BEST = "bold green"
COLOR_POSITIVE = "green"
COLOR_NEGATIVE = "red"


def _fmt(value: Any) -> str:
    """Format a value for display."""
    if isinstance(value, float):
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _is_lower_better(metric_name: str) -> bool:
    """Check if lower values are better for this metric (e.g., loss, error)."""
    lower_better_keywords = ["loss", "error", "mse", "mae", "rmse"]
    return any(keyword in metric_name.lower() for keyword in lower_better_keywords)


def print_run(run: Run) -> None:
    """Print detailed run info."""
    console.print()
    console.print(
        Panel(
            f"[bold cyan]Run #{run.id}[/bold cyan] • [dim]{run.project}[/dim]",
            subtitle=f"[dim]{run.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            box=box.ROUNDED,
        )
    )

    if run.params:
        console.print("\n[bold yellow]Parameters[/bold yellow]")
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="dim")
        t.add_column(style="white")
        for k, v in sorted(run.params.items()):
            t.add_row(k, _fmt(v))
        console.print(t)

    if run.metrics:
        console.print("\n[bold green]Metrics[/bold green]")
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="dim")
        t.add_column(style="bold white")
        for k, v in sorted(run.metrics.items()):
            t.add_row(k, _fmt(v))
        console.print(t)

    if run.tags:
        console.print(f"\n[bold {COLOR_TAGS}]Tags[/bold {COLOR_TAGS}]")
        console.print(
            "  "
            + " ".join(
                f"[on {COLOR_TAG_BG}] {tag} [/on {COLOR_TAG_BG}]" for tag in run.tags
            )
        )

    if run.notes:
        console.print("\n[bold blue]Notes[/bold blue]")
        console.print(f"  [italic]{run.notes}[/italic]")

    console.print()


def print_runs_table(runs: List[Run], show_project: bool = False) -> None:
    """Print runs as a table."""
    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    metric_keys = sorted(set(k for r in runs for k in r.metrics))

    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("ID", style="cyan", justify="right")
    if show_project:
        table.add_column("Project", style="dim")
    table.add_column("Timestamp", style="dim")
    for m in metric_keys:
        table.add_column(m, justify="right", style="green")
    table.add_column("Tags", style="magenta")

    for run in runs:
        row = [str(run.id)]
        if show_project:
            row.append(run.project)
        row.append(run.timestamp.strftime("%Y-%m-%d %H:%M"))
        for m in metric_keys:
            v = run.metrics.get(m)
            row.append(_fmt(v) if v is not None else "-")
        row.append(" ".join(run.tags) if run.tags else "")
        table.add_row(*row)

    console.print(table)


def print_comparison(runs: List[Run]) -> None:
    """Print side-by-side comparison."""
    if not runs:
        console.print("[dim]No runs to compare.[/dim]")
        return
    if len(runs) == 1:
        print_run(runs[0])
        return

    param_keys = sorted(set(k for r in runs for k in r.params))
    metric_keys = sorted(set(k for r in runs for k in r.metrics))

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, title="Run Comparison")
    table.add_column("", style="bold")
    for run in runs:
        table.add_column(f"Run #{run.id}", justify="right")

    table.add_row(
        "[dim]timestamp[/dim]",
        *[f"[dim]{r.timestamp.strftime('%m-%d %H:%M')}[/dim]" for r in runs],
    )
    table.add_row("[bold cyan]─ Parameters ─[/bold cyan]", *[""] * len(runs))

    for key in param_keys:
        vals = [
            _fmt(r.params.get(key)) if r.params.get(key) is not None else "[dim]-[/dim]"
            for r in runs
        ]
        table.add_row(f"[yellow]{key}[/yellow]", *vals)

    if metric_keys:
        table.add_row("[bold cyan]─ Metrics ─[/bold cyan]", *[""] * len(runs))

    for key in metric_keys:
        values = [r.metrics.get(key) for r in runs]
        numeric = [v for v in values if v is not None]

        lower_better = _is_lower_better(key)
        best = min(numeric) if lower_better else max(numeric) if numeric else None

        formatted = []
        for v in values:
            if v is None:
                formatted.append("[dim]-[/dim]")
            elif best is not None and v == best:
                formatted.append(f"[bold green]{_fmt(v)}[/bold green] ★")
            else:
                formatted.append(_fmt(v))
        table.add_row(f"[green]{key}[/green]", *formatted)

    table.add_row("[bold cyan]─ Tags ─[/bold cyan]", *[""] * len(runs))
    table.add_row(
        "[magenta]tags[/magenta]", *[" ".join(r.tags) or "[dim]-[/dim]" for r in runs]
    )

    console.print()
    console.print(table)
    console.print()


def print_diff(run_a: Run, run_b: Run) -> None:
    """Print only what changed between two runs."""
    console.print()
    console.print(f"[bold]Diff: Run #{run_a.id} → Run #{run_b.id}[/bold]")
    console.print()

    has_diff = False

    # Params diff
    all_params = set(run_a.params.keys()) | set(run_b.params.keys())
    param_diffs = []
    for key in sorted(all_params):
        val_a = run_a.params.get(key)
        val_b = run_b.params.get(key)
        if val_a != val_b:
            param_diffs.append((key, val_a, val_b))

    if param_diffs:
        has_diff = True
        console.print("[bold yellow]Parameters[/bold yellow]")
        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("", style="dim")
        t.add_column(f"#{run_a.id}", justify="right")
        t.add_column("→", justify="center", style="dim")
        t.add_column(f"#{run_b.id}", justify="right")
        for key, val_a, val_b in param_diffs:
            a_str = _fmt(val_a) if val_a is not None else "[dim]-[/dim]"
            b_str = _fmt(val_b) if val_b is not None else "[dim]-[/dim]"
            t.add_row(key, a_str, "→", f"[bold]{b_str}[/bold]")
        console.print(t)
        console.print()

    # Metrics diff
    all_metrics = set(run_a.metrics.keys()) | set(run_b.metrics.keys())
    metric_diffs = []
    for key in sorted(all_metrics):
        val_a = run_a.metrics.get(key)
        val_b = run_b.metrics.get(key)
        if val_a != val_b:
            metric_diffs.append((key, val_a, val_b))

    if metric_diffs:
        has_diff = True
        console.print("[bold green]Metrics[/bold green]")
        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("", style="dim")
        t.add_column(f"#{run_a.id}", justify="right")
        t.add_column("→", justify="center", style="dim")
        t.add_column(f"#{run_b.id}", justify="right")
        t.add_column("Δ", justify="right", style="dim")
        for key, val_a, val_b in metric_diffs:
            a_str = _fmt(val_a) if val_a is not None else "[dim]-[/dim]"
            b_str = _fmt(val_b) if val_b is not None else "[dim]-[/dim]"
            # Calculate delta
            delta_str = ""
            if val_a is not None and val_b is not None:
                delta = val_b - val_a
                lower_better = _is_lower_better(key)
                is_better = (delta < 0) if lower_better else (delta > 0)
                color = "green" if is_better else "red"
                sign = "+" if delta > 0 else ""
                delta_str = f"[{color}]{sign}{_fmt(delta)}[/{color}]"
            t.add_row(key, a_str, "→", f"[bold]{b_str}[/bold]", delta_str)
        console.print(t)
        console.print()

    # Tags diff
    tags_a = set(run_a.tags)
    tags_b = set(run_b.tags)
    if tags_a != tags_b:
        has_diff = True
        console.print("[bold magenta]Tags[/bold magenta]")
        added = tags_b - tags_a
        removed = tags_a - tags_b
        if added:
            console.print(f"  [green]+[/green] {' '.join(added)}")
        if removed:
            console.print(f"  [red]-[/red] {' '.join(removed)}")
        console.print()

    # Notes diff
    if run_a.notes != run_b.notes:
        has_diff = True
        console.print("[bold blue]Notes[/bold blue]")
        if run_a.notes:
            console.print(f"  [red]-[/red] [dim]{run_a.notes}[/dim]")
        if run_b.notes:
            console.print(f"  [green]+[/green] {run_b.notes}")
        console.print()

    if not has_diff:
        console.print("[dim]No differences found.[/dim]")
        console.print()


def print_best_run(run: Run, metric: str) -> None:
    """Print best run summary."""
    console.print()
    value = run.metrics.get(metric)
    console.print(
        f"[bold green]★[/bold green] Best [bold]{metric}[/bold]: [bold cyan]{_fmt(value)}[/bold cyan]"
    )
    console.print(
        f"  Run [bold]#{run.id}[/bold] from {run.timestamp.strftime('%Y-%m-%d %H:%M')}"
    )
    if run.params:
        params_str = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(run.params.items()))
        console.print(f"  [dim]{params_str}[/dim]")
    if run.tags:
        console.print(f"  [magenta]{' '.join(run.tags)}[/magenta]")
    console.print()


def print_projects(projects: List[dict]) -> None:
    """Print project list with stats."""
    if not projects:
        console.print("[dim]No projects found.[/dim]")
        return

    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("Project", style="cyan bold")
    table.add_column("Runs", justify="right")
    table.add_column("First Run", style="dim")
    table.add_column("Last Run", style="dim")

    for p in projects:
        table.add_row(
            p["name"],
            str(p["run_count"]),
            p["first_run"][:10] if p["first_run"] else "-",
            p["last_run"][:10] if p["last_run"] else "-",
        )
    console.print(table)


def print_success(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green] {msg}")


def print_error(msg: str) -> None:
    console.print(f"[bold red]✗[/bold red] {msg}")


def print_warning(msg: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {msg}")


def print_info(msg: str) -> None:
    console.print(f"[bold blue]ℹ[/bold blue] {msg}")
