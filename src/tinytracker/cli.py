"""Command-line interface."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from tinytracker.config import get_db_path_override, get_default_project
from tinytracker.display import (
    console,
    print_best_run,
    print_comparison,
    print_diff,
    print_error,
    print_info,
    print_projects,
    print_run,
    print_runs_table,
    print_success,
)
from tinytracker.storage import TINYTRACKER_DIR, Storage, get_db_path

app = typer.Typer(
    name="tinytracker",
    help="🔬 Minimal experiment tracker for ML projects.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _parse_kv(items: List[str]) -> dict:
    """Parse key=value pairs. Auto-converts to int/float when possible."""
    result = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid format: '{item}'. Use key=value.")
        key, val = item.split("=", 1)
        try:
            result[key] = int(val)
        except ValueError:
            try:
                result[key] = float(val)
            except ValueError:
                result[key] = val
    return result


def _get_storage() -> Storage:
    """Get storage, exit if not initialized."""
    db_path = get_db_path_override() or get_db_path()
    if not db_path.exists():
        print_error("TinyTracker not initialized in this directory.")
        print_info("Run [bold]tinytracker init <project>[/bold] to get started.")
        raise typer.Exit(1)
    return Storage(db_path)


def _resolve_project(project: Optional[str]) -> Optional[str]:
    """Get project from arg or config default."""
    return project or get_default_project()


@app.command()
def init(
    project: Annotated[str, typer.Argument(help="Project name")],
):
    """Initialize TinyTracker in current directory."""
    db_path = get_db_path()

    if db_path.exists():
        print_info(f"TinyTracker already initialized at [dim]{TINYTRACKER_DIR}/[/dim]")
    else:
        Storage(db_path)
        print_success(f"Initialized TinyTracker at [dim]{TINYTRACKER_DIR}/[/dim]")

    print_success(f"Project [bold cyan]{project}[/bold cyan] is ready!")
    console.print()
    console.print("[dim]Quick start:[/dim]")
    console.print(f"  tinytracker log -p {project} --metric acc=0.95 --param lr=0.001")
    console.print(f"  tinytracker list -p {project}")


@app.command()
def log(
    project: Annotated[
        Optional[str], typer.Option("--project", "-p", help="Project name")
    ] = None,
    metrics: Annotated[
        Optional[List[str]], typer.Option("--metric", "-m", help="key=value")
    ] = None,
    params: Annotated[
        Optional[List[str]], typer.Option("--param", "-P", help="key=value")
    ] = None,
    tags: Annotated[
        Optional[List[str]], typer.Option("--tag", "-t", help="Tag")
    ] = None,
    notes: Annotated[Optional[str], typer.Option("--notes", "-n", help="Notes")] = None,
):
    """Log a new experiment run."""
    project = _resolve_project(project)
    if not project:
        print_error(
            "Project required. Use -p or set default_project in .tinytracker.toml"
        )
        raise typer.Exit(1)
    storage = _get_storage()

    try:
        parsed_metrics = _parse_kv(metrics or [])
        parsed_params = _parse_kv(params or [])
    except typer.BadParameter as e:
        print_error(str(e))
        raise typer.Exit(1)

    run_id = storage.insert_run(
        project=project,
        params=parsed_params,
        metrics=parsed_metrics,
        tags=tags or [],
        notes=notes,
    )

    print_success(
        f"Logged run [bold cyan]#{run_id}[/bold cyan] to project [bold]{project}[/bold]"
    )
    if parsed_metrics:
        console.print(
            f"  [dim]Metrics:[/dim] {' | '.join(f'{k}={v}' for k, v in parsed_metrics.items())}"
        )


@app.command("list")
def list_runs(
    project: Annotated[
        Optional[str], typer.Option("--project", "-p", help="Project name")
    ] = None,
    tags: Annotated[
        Optional[List[str]], typer.Option("--tag", "-t", help="Filter by tag")
    ] = None,
    before: Annotated[
        Optional[str], typer.Option(help="Before date (YYYY-MM-DD)")
    ] = None,
    after: Annotated[
        Optional[str], typer.Option(help="After date (YYYY-MM-DD)")
    ] = None,
    order_by: Annotated[
        Optional[str], typer.Option("--order-by", "-o", help="Sort by metric")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max runs")] = 20,
    all_projects: Annotated[
        bool, typer.Option("--all", "-a", help="All projects")
    ] = False,
):
    """List experiment runs."""
    storage = _get_storage()

    before_dt, after_dt = None, None
    try:
        if before:
            before_dt = datetime.strptime(before, "%Y-%m-%d")
        if after:
            after_dt = datetime.strptime(after, "%Y-%m-%d")
    except ValueError:
        print_error("Invalid date format. Use YYYY-MM-DD.")
        raise typer.Exit(1)

    order_metric, order_desc = None, True
    if order_by:
        if ":" in order_by:
            order_metric, direction = order_by.split(":", 1)
            order_desc = direction.lower() != "asc"
        else:
            order_metric = order_by

    filter_project = None if all_projects else project

    if not all_projects and not project:
        projects = storage.get_projects()
        if projects:
            print_info(
                "Specify a project with [bold]-p PROJECT[/bold] or use [bold]--all[/bold]"
            )
            console.print()
            console.print("[dim]Available projects:[/dim]")
            for p in projects:
                stats = storage.get_project_stats(p)
                console.print(f"  • [cyan]{p}[/cyan] ({stats['run_count']} runs)")
            raise typer.Exit(0)
        print_info("No runs found. Log your first run with:")
        console.print("  tinytracker log -p my_project --metric acc=0.95")
        raise typer.Exit(0)

    runs = storage.list_runs(
        project=filter_project,
        tags=tags,
        before=before_dt,
        after=after_dt,
        order_by=order_metric,
        order_desc=order_desc,
        limit=limit,
    )

    if not runs:
        print_info("No runs found matching criteria.")
        raise typer.Exit(0)

    console.print()
    print_runs_table(runs, show_project=all_projects or filter_project is None)
    console.print()
    console.print(f"[dim]Showing {len(runs)} run(s)[/dim]")


@app.command()
def show(
    run_id: Annotated[int, typer.Argument(help="Run ID")],
):
    """Show run details."""
    storage = _get_storage()
    run = storage.get_run(run_id)

    if not run:
        print_error(f"Run #{run_id} not found.")
        raise typer.Exit(1)

    print_run(run)


@app.command()
def compare(
    run_ids: Annotated[List[int], typer.Argument(help="Run IDs to compare")],
):
    """Compare runs side-by-side."""
    storage = _get_storage()
    runs = storage.get_runs_by_ids(run_ids)

    if not runs:
        print_error("No runs found with specified IDs.")
        raise typer.Exit(1)

    found = {r.id for r in runs}
    missing = set(run_ids) - found
    if missing:
        print_error(f"Run(s) not found: {', '.join(map(str, sorted(missing)))}")

    print_comparison(runs)


@app.command()
def delete(
    run_id: Annotated[int, typer.Argument(help="Run ID")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation")
    ] = False,
):
    """Delete a run."""
    storage = _get_storage()
    run = storage.get_run(run_id)

    if not run:
        print_error(f"Run #{run_id} not found.")
        raise typer.Exit(1)

    if not force:
        console.print(
            f"About to delete run [bold cyan]#{run_id}[/bold cyan] from [bold]{run.project}[/bold]"
        )
        if not typer.confirm("Are you sure?"):
            print_info("Cancelled.")
            raise typer.Exit(0)

    storage.delete_run(run_id)
    print_success(f"Deleted run [bold cyan]#{run_id}[/bold cyan]")


@app.command()
def export(
    project: Annotated[
        Optional[str], typer.Option("--project", "-p", help="Project name")
    ] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="json or csv")] = "json",
    output: Annotated[
        Optional[str], typer.Option("--output", "-o", help="Output file")
    ] = None,
):
    """Export runs to JSON or CSV."""
    project = _resolve_project(project)
    if not project:
        print_error(
            "Project required. Use -p or set default_project in .tinytracker.toml"
        )
        raise typer.Exit(1)

    storage = _get_storage()

    if format not in ("json", "csv"):
        print_error(f"Unknown format: {format}. Use 'json' or 'csv'.")
        raise typer.Exit(1)

    data = storage.export_runs(project=project, format=format)

    if output:
        Path(output).write_text(data)
        print_success(f"Exported to [bold]{output}[/bold]")
    else:
        console.print(data)


@app.command()
def projects():
    """List all projects."""
    storage = _get_storage()
    names = storage.get_projects()

    if not names:
        print_info("No projects found.")
        raise typer.Exit(0)

    data = []
    for name in names:
        stats = storage.get_project_stats(name)
        data.append(
            {
                "name": name,
                "run_count": stats["run_count"],
                "first_run": stats["first_run"],
                "last_run": stats["last_run"],
            }
        )

    console.print()
    print_projects(data)
    console.print()


@app.command()
def best(
    project: Annotated[
        Optional[str], typer.Option("--project", "-p", help="Project name")
    ] = None,
    metric: Annotated[
        str, typer.Option("--metric", "-m", help="Metric to optimize")
    ] = "accuracy",
    minimize: Annotated[
        bool, typer.Option("--min", help="Minimize instead of maximize")
    ] = False,
):
    """Find best run by metric."""
    project = _resolve_project(project)
    if not project:
        print_error(
            "Project required. Use -p or set default_project in .tinytracker.toml"
        )
        raise typer.Exit(1)

    storage = _get_storage()
    run = storage.get_best_run(project, metric, minimize=minimize)

    if not run:
        print_info(f"No runs found with metric '{metric}' in project '{project}'")
        raise typer.Exit(1)

    print_best_run(run, metric)


@app.command()
def diff(
    run_a: Annotated[int, typer.Argument(help="First run ID")],
    run_b: Annotated[int, typer.Argument(help="Second run ID")],
):
    """Show what changed between two runs."""
    storage = _get_storage()

    first = storage.get_run(run_a)
    second = storage.get_run(run_b)

    if not first:
        print_error(f"Run #{run_a} not found.")
        raise typer.Exit(1)
    if not second:
        print_error(f"Run #{run_b} not found.")
        raise typer.Exit(1)

    print_diff(first, second)


@app.command()
def update(
    run_id: Annotated[int, typer.Argument(help="Run ID to update")],
    notes: Annotated[
        Optional[str], typer.Option("--notes", "-n", help="Set notes")
    ] = None,
    add_tag: Annotated[
        Optional[List[str]], typer.Option("--add-tag", "-t", help="Add tag")
    ] = None,
    remove_tag: Annotated[
        Optional[List[str]], typer.Option("--remove-tag", "-r", help="Remove tag")
    ] = None,
    set_tags: Annotated[
        Optional[str],
        typer.Option("--set-tags", help="Replace all tags (comma-separated)"),
    ] = None,
):
    """Update run notes or tags."""
    storage = _get_storage()

    run = storage.get_run(run_id)
    if not run:
        print_error(f"Run #{run_id} not found.")
        raise typer.Exit(1)

    tags = [t.strip() for t in set_tags.split(",")] if set_tags else None

    updated = storage.update_run(
        run_id,
        tags=tags,
        notes=notes,
        append_tags=add_tag,
        remove_tags=remove_tag,
    )

    if updated:
        print_success(f"Updated run [bold cyan]#{run_id}[/bold cyan]")
        # Show current state
        run = storage.get_run(run_id)
        if run.tags:
            console.print(f"  [dim]Tags:[/dim] {' '.join(run.tags)}")
        if run.notes:
            console.print(f"  [dim]Notes:[/dim] {run.notes}")


@app.command()
def config():
    """Show current configuration."""
    from tinytracker.config import load_config, _find_config_file

    config_path = _find_config_file()
    cfg = load_config()

    console.print()
    if config_path:
        console.print(f"[bold]Config file:[/bold] {config_path}")
    else:
        console.print("[dim]No config file found (.tinytracker.toml)[/dim]")

    console.print()
    if cfg:
        console.print("[bold]Settings:[/bold]")
        for key, value in cfg.items():
            console.print(f"  {key} = [cyan]{value}[/cyan]")
    else:
        console.print("[dim]No configuration set.[/dim]")
        console.print()
        console.print("[dim]Create .tinytracker.toml with:[/dim]")
        console.print('  default_project = "my_model"')

    console.print()


@app.command()
def status():
    """Show TinyTracker status."""
    db_path = get_db_path()

    if not db_path.exists():
        print_info("TinyTracker not initialized in this directory.")
        console.print()
        console.print("[dim]To get started:[/dim]")
        console.print("  tinytracker init my_project")
        raise typer.Exit(0)

    storage = Storage(db_path)
    names = storage.get_projects()

    console.print()
    console.print(
        f"[bold green]✓[/bold green] TinyTracker initialized at [dim]{TINYTRACKER_DIR}/[/dim]"
    )
    console.print()

    if names:
        total = 0
        console.print(f"[bold]Projects:[/bold] {len(names)}")
        for name in names:
            stats = storage.get_project_stats(name)
            total += stats["run_count"]
            console.print(f"  • [cyan]{name}[/cyan] ({stats['run_count']} runs)")
        console.print()
        console.print(f"[bold]Total runs:[/bold] {total}")
    else:
        console.print("[dim]No runs logged yet.[/dim]")
        console.print()
        console.print("[dim]Log your first run:[/dim]")
        console.print("  tinytracker log -p my_project --metric acc=0.95")

    console.print()


if __name__ == "__main__":
    app()
