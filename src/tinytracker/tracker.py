"""Python API for TinyTracker."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tinytracker.models import Epoch, Run
from tinytracker.storage import Storage, get_db_path


class Tracker:
    """
    Track ML experiments for a project.

    Example:
        tracker = Tracker("my_project")
        run_id = tracker.log(params={"lr": 0.001}, metrics={"acc": 0.95})
    """

    def __init__(self, project: str, root: Optional[Union[str, Path]] = None):
        """
        Create a tracker for your project.

        Args:
            project: Name of your project (e.g., "mnist_classifier")
            root: Directory to store data. Defaults to current directory.
        """
        self.project = project
        self.root = Path(root) if root else Path.cwd()
        self._storage = Storage(get_db_path(self.root))

    def log(
        self,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Log a new run with your hyperparameters and results.

        Args:
            params: Hyperparameters like learning rate, batch size, etc.
            metrics: Results like accuracy, loss, F1 score, etc.
            tags: Labels to organize runs (e.g., "baseline", "production")
            notes: Any extra info about this run

        Returns:
            The ID of the logged run
        """
        return self._storage.insert_run(
            project=self.project,
            params=params or {},
            metrics=metrics or {},
            tags=tags or [],
            notes=notes,
        )

    def get(self, run_id: int) -> Optional[Run]:
        """Fetch a specific run by its ID. Returns None if not found."""
        return self._storage.get_run(run_id)

    def list(
        self,
        tags: Optional[List[str]] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        order_by: Optional[str] = None,
        order_desc: bool = True,
        limit: Optional[int] = None,
    ) -> List[Run]:
        """
        List runs matching your filters.

        Args:
            tags: Only show runs with these tags
            before: Only runs before this date
            after: Only runs after this date
            order_by: Sort by this metric name
            order_desc: Sort highest first (False for lowest first)
            limit: Max number of runs to return

        Returns:
            List of matching runs, newest first by default
        """
        return self._storage.list_runs(
            project=self.project,
            tags=tags,
            before=before,
            after=after,
            order_by=order_by,
            order_desc=order_desc,
            limit=limit,
        )

    def compare(self, run_ids: List[int]) -> List[Run]:
        """Get multiple runs to compare side by side."""
        return self._storage.get_runs_by_ids(run_ids)

    def delete(self, run_id: int) -> bool:
        """Delete a run permanently. Returns True if successful."""
        return self._storage.delete_run(run_id)

    def update(
        self,
        run_id: int,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Update tags or notes for a run.

        Args:
            run_id: Which run to update
            tags: Replace all tags with these
            notes: Replace notes with this text
            add_tags: Add these tags (keeps existing ones)
            remove_tags: Remove these tags

        Returns:
            True if updated, False if run not found
        """
        return self._storage.update_run(
            run_id, tags=tags, notes=notes,
            append_tags=add_tags, remove_tags=remove_tags,
        )

    def best(self, metric: str, minimize: bool = False) -> Optional[Run]:
        """
        Find the run with the best value for a metric.

        Args:
            metric: Which metric to optimize (e.g., "accuracy", "loss")
            minimize: True for metrics like loss where lower is better

        Returns:
            The best run, or None if no runs have this metric
        """
        return self._storage.get_best_run(self.project, metric, minimize=minimize)

    def export(self, format: str = "json") -> str:
        """Export all runs to JSON or CSV format."""
        return self._storage.export_runs(project=self.project, format=format)

    @property
    def stats(self) -> dict:
        """Get basic stats about this project (run count, date range)."""
        return self._storage.get_project_stats(self.project)

    def log_epoch(
        self,
        run_id: int,
        epoch_num: int,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Log an epoch for a specific run.

        Args:
            run_id: The run this epoch belongs to
            epoch_num: Which epoch this is (1, 2, 3, etc.)
            metrics: Performance metrics for this epoch (loss, accuracy, etc.)
            notes: Any notes about this epoch

        Returns:
            The ID of the logged epoch
        """
        return self._storage.insert_epoch(
            run_id=run_id,
            epoch_num=epoch_num,
            metrics=metrics or {},
            notes=notes,
        )

    def get_epoch(self, epoch_id: int) -> Optional[Epoch]:
        """Fetch a specific epoch by its ID. Returns None if not found."""
        return self._storage.get_epoch(epoch_id)

    def list_epochs(
        self,
        run_id: int,
        order_by: Optional[str] = None,
        order_desc: bool = True,
        limit: Optional[int] = None,
    ) -> List[Epoch]:
        """
        List all epochs for a specific run.

        Args:
            run_id: The run to get epochs for
            order_by: Sort by this metric name
            order_desc: Sort highest first (False for lowest first)
            limit: Max number of epochs to return

        Returns:
            List of epochs, ordered by epoch_num by default
        """
        return self._storage.list_epochs(
            run_id=run_id,
            order_by=order_by,
            order_desc=order_desc,
            limit=limit,
        )

    def delete_epoch(self, epoch_id: int) -> bool:
        """Delete an epoch permanently. Returns True if successful."""
        return self._storage.delete_epoch(epoch_id)

    def update_epoch(
        self,
        epoch_id: int,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update notes for an epoch.

        Args:
            epoch_id: Which epoch to update
            notes: Replace notes with this text

        Returns:
            True if updated, False if epoch not found
        """
        return self._storage.update_epoch(epoch_id, notes=notes)

    def best_epoch(
        self,
        run_id: int,
        metric: str,
        minimize: bool = False,
    ) -> Optional[Epoch]:
        """
        Find the epoch with the best value for a metric within a run.

        Args:
            run_id: The run to search within
            metric: Which metric to optimize (e.g., "val_loss", "val_accuracy")
            minimize: True for metrics like loss where lower is better

        Returns:
            The best epoch, or None if no epochs have this metric
        """
        return self._storage.get_best_epoch(run_id, metric, minimize=minimize)

    def __repr__(self) -> str:
        return f"Tracker(project='{self.project}')"


def log(
    project: str,
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    root: Optional[Union[str, Path]] = None,
) -> int:
    """
    Quick one-liner to log a run without creating a Tracker object.

    Example:
        from tinytracker import log
        log(project="mnist", params={"lr": 0.01}, metrics={"acc": 0.95})

    Returns:
        The ID of the logged run
    """
    return Tracker(project, root=root).log(params=params, metrics=metrics, tags=tags, notes=notes)


def log_epoch(
    project: str,
    run_id: int,
    epoch_num: int,
    metrics: Optional[Dict[str, float]] = None,
    notes: Optional[str] = None,
    root: Optional[Union[str, Path]] = None,
) -> int:
    """
    Quick one-liner to log an epoch without creating a Tracker object.

    Example:
        from tinytracker import log_epoch
        log_epoch(project="mnist", run_id=1, epoch_num=5, metrics={"loss": 0.23, "acc": 0.92})

    Returns:
        The ID of the logged epoch
    """
    return Tracker(project, root=root).log_epoch(run_id=run_id, epoch_num=epoch_num, metrics=metrics, notes=notes)
