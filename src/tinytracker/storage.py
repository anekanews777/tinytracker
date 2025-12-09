"""SQLite storage backend."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from tinytracker.models import Epoch, Run

TINYTRACKER_DIR = ".tinytracker"
DB_NAME = "tracker.db"

# SQL column order for SELECT queries
_COLUMNS = "id, project, timestamp, params, metrics, tags, notes"
_EPOCH_COLUMNS = "id, run_id, epoch_num, timestamp, metrics, notes"


def _validate_identifier(name: str) -> None:
    """Validate that a string is safe to use as a SQL identifier (metric/param names)."""
    if not name:
        raise ValueError("Identifier cannot be empty")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"Invalid identifier: {name}. Only alphanumeric characters, underscores, and hyphens allowed."
        )


def get_db_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path.cwd()
    return root / TINYTRACKER_DIR / DB_NAME


class Storage:
    """
    Handles saving and loading experiment runs from SQLite.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_db_path()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    params TEXT,
                    metrics TEXT,
                    tags TEXT,
                    notes TEXT
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_project ON runs(project)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS epochs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    epoch_num INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    metrics TEXT,
                    notes TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON epochs(run_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_epoch_num ON epochs(epoch_num)"
            )
        finally:
            conn.close()

    def insert_run(
        self,
        project: str,
        params: dict,
        metrics: dict,
        tags: List[str],
        notes: Optional[str] = None,
    ) -> int:
        """Save a new run and return its ID."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO runs (project, timestamp, params, metrics, tags, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    project,
                    datetime.now().isoformat(),
                    json.dumps(params),
                    json.dumps(metrics),
                    json.dumps(tags),
                    notes,
                ),
            )
            return cursor.lastrowid
        finally:
            conn.close()

    def get_run(self, run_id: int) -> Optional[Run]:
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            return Run.from_row(tuple(row)) if row else None
        finally:
            conn.close()

    def list_runs(
        self,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        order_by: Optional[str] = None,
        order_desc: bool = True,
        limit: Optional[int] = None,
    ) -> List[Run]:
        """Get runs matching the specified filters."""
        query = f"SELECT {_COLUMNS} FROM runs WHERE 1=1"
        params: List[Any] = []

        if project:
            query += " AND project = ?"
            params.append(project)
        if before:
            query += " AND timestamp < ?"
            params.append(before.isoformat())
        if after:
            query += " AND timestamp > ?"
            params.append(after.isoformat())
        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')

        if order_by:
            _validate_identifier(order_by)
            direction = "DESC" if order_desc else "ASC"
            query += f" ORDER BY json_extract(metrics, '$.{order_by}') {direction}"
        else:
            query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        conn = self._connect()
        try:
            return [Run.from_row(tuple(row)) for row in conn.execute(query, params)]
        finally:
            conn.close()

    def get_runs_by_ids(self, run_ids: List[int]) -> List[Run]:
        """Get runs by IDs, preserving order."""
        if not run_ids:
            return []

        placeholders = ",".join("?" * len(run_ids))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM runs WHERE id IN ({placeholders})", run_ids
            )
            run_map = {row[0]: Run.from_row(tuple(row)) for row in rows}
            return [run_map[rid] for rid in run_ids if rid in run_map]
        finally:
            conn.close()

    def delete_run(self, run_id: int) -> bool:
        """Delete a run, returns True if deleted."""
        conn = self._connect()
        try:
            cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            return cursor.rowcount > 0
        finally:
            conn.close()

    def update_run(
        self,
        run_id: int,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        append_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
    ) -> bool:
        """Modify tags or notes for an existing run."""
        run = self.get_run(run_id)
        if not run:
            return False

        # Determine new tags
        if tags is not None:
            new_tags = tags
        else:
            new_tags = run.tags.copy()
            if append_tags:
                new_tags = list(set(new_tags) | set(append_tags))
            if remove_tags:
                new_tags = list(set(new_tags) - set(remove_tags))

        new_notes = notes if notes is not None else run.notes

        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET tags = ?, notes = ? WHERE id = ?",
                (json.dumps(new_tags), new_notes, run_id),
            )
            return True
        finally:
            conn.close()

    def get_best_run(
        self,
        project: str,
        metric: str,
        minimize: bool = False,
    ) -> Optional[Run]:
        """Find the run with the highest (or lowest) value for a specific metric."""
        _validate_identifier(metric)
        direction = "ASC" if minimize else "DESC"
        conn = self._connect()
        try:
            row = conn.execute(
                f"""SELECT {_COLUMNS} FROM runs
                    WHERE project = ? AND json_extract(metrics, '$.{metric}') IS NOT NULL
                    ORDER BY json_extract(metrics, '$.{metric}') {direction}
                    LIMIT 1""",
                (project,),
            ).fetchone()
            return Run.from_row(tuple(row)) if row else None
        finally:
            conn.close()

    def get_projects(self) -> List[str]:
        conn = self._connect()
        try:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT project FROM runs ORDER BY project"
                )
            ]
        finally:
            conn.close()

    def get_project_stats(self, project: str) -> dict:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM runs WHERE project = ?",
                (project,),
            ).fetchone()
            return {"run_count": row[0], "first_run": row[1], "last_run": row[2]}
        finally:
            conn.close()

    def export_runs(self, project: Optional[str] = None, format: str = "json") -> str:
        """Export runs as JSON or CSV formatted text."""
        runs = self.list_runs(project=project)

        if format == "json":
            return json.dumps([r.to_dict() for r in runs], indent=2)

        if format == "csv":
            if not runs:
                return ""

            all_params = sorted(set(k for r in runs for k in r.params))
            all_metrics = sorted(set(k for r in runs for k in r.metrics))

            headers = ["id", "project", "timestamp", "tags", "notes"]
            headers += [f"param:{k}" for k in all_params]
            headers += [f"metric:{k}" for k in all_metrics]

            lines = [",".join(headers)]
            for run in runs:
                row = [
                    str(run.id),
                    run.project,
                    run.timestamp.isoformat(),
                    "|".join(run.tags),
                    (run.notes or "").replace(",", ";").replace("\n", " "),
                ]
                row += [str(run.params.get(k, "")) for k in all_params]
                row += [str(run.metrics.get(k, "")) for k in all_metrics]
                lines.append(",".join(row))
            return "\n".join(lines)

        raise ValueError(f"Unknown format: {format}")

    def insert_epoch(
        self,
        run_id: int,
        epoch_num: int,
        metrics: dict,
        notes: Optional[str] = None,
    ) -> int:
        """Save a new epoch for a run and return its ID."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO epochs (run_id, epoch_num, timestamp, metrics, notes) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    epoch_num,
                    datetime.now().isoformat(),
                    json.dumps(metrics),
                    notes,
                ),
            )
            return cursor.lastrowid
        finally:
            conn.close()

    def get_epoch(self, epoch_id: int) -> Optional[Epoch]:
        """Fetch a specific epoch by its ID."""
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {_EPOCH_COLUMNS} FROM epochs WHERE id = ?", (epoch_id,)
            ).fetchone()
            return Epoch.from_row(tuple(row)) if row else None
        finally:
            conn.close()

    def list_epochs(
        self,
        run_id: int,
        order_by: Optional[str] = None,
        order_desc: bool = True,
        limit: Optional[int] = None,
    ) -> List[Epoch]:
        """Get all epochs for a specific run."""
        query = f"SELECT {_EPOCH_COLUMNS} FROM epochs WHERE run_id = ?"
        params: List[Any] = [run_id]

        if order_by:
            _validate_identifier(order_by)
            direction = "DESC" if order_desc else "ASC"
            query += f" ORDER BY json_extract(metrics, '$.{order_by}') {direction}"
        else:
            query += " ORDER BY epoch_num ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        conn = self._connect()
        try:
            return [Epoch.from_row(tuple(row)) for row in conn.execute(query, params)]
        finally:
            conn.close()

    def delete_epoch(self, epoch_id: int) -> bool:
        """Delete an epoch, returns True if deleted."""
        conn = self._connect()
        try:
            cursor = conn.execute("DELETE FROM epochs WHERE id = ?", (epoch_id,))
            return cursor.rowcount > 0
        finally:
            conn.close()

    def update_epoch(
        self,
        epoch_id: int,
        notes: Optional[str] = None,
    ) -> bool:
        """Update notes for an existing epoch."""
        epoch = self.get_epoch(epoch_id)
        if not epoch:
            return False

        conn = self._connect()
        try:
            conn.execute(
                "UPDATE epochs SET notes = ? WHERE id = ?",
                (notes, epoch_id),
            )
            return True
        finally:
            conn.close()

    def get_best_epoch(
        self,
        run_id: int,
        metric: str,
        minimize: bool = False,
    ) -> Optional[Epoch]:
        """Find the epoch with the best value for a specific metric within a run."""
        _validate_identifier(metric)
        direction = "ASC" if minimize else "DESC"
        conn = self._connect()
        try:
            row = conn.execute(
                f"""SELECT {_EPOCH_COLUMNS} FROM epochs
                    WHERE run_id = ? AND json_extract(metrics, '$.{metric}') IS NOT NULL
                    ORDER BY json_extract(metrics, '$.{metric}') {direction}
                    LIMIT 1""",
                (run_id,),
            ).fetchone()
            return Epoch.from_row(tuple(row)) if row else None
        finally:
            conn.close()
