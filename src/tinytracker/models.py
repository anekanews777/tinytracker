"""Data models."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Run:
    """
    Represents a single experiment run with its parameters, metrics, and metadata.

    Attributes:
        id: Unique identifier for this run
        project: Name of the project this run belongs to
        timestamp: When this run was logged
        params: Hyperparameters used (lr, batch_size, etc.)
        metrics: Results achieved (accuracy, loss, etc.)
        tags: Labels for organizing runs
        notes: Any additional notes about this run
    """

    id: int
    project: str
    timestamp: datetime
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project": self.project,
            "timestamp": self.timestamp.isoformat(),
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "Run":
        """Build a Run object from a database row."""
        return cls(
            id=row[0],
            project=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            params=json.loads(row[3]) if row[3] else {},
            metrics=json.loads(row[4]) if row[4] else {},
            tags=json.loads(row[5]) if row[5] else [],
            notes=row[6],
        )

    def __str__(self) -> str:
        return f"Run #{self.id} ({self.project}) - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


@dataclass
class Epoch:
    """
    Represents a single training epoch within a run.

    Attributes:
        id: Unique identifier for this epoch
        run_id: The run this epoch belongs to
        epoch_num: Which epoch this is (1, 2, 3, etc.)
        timestamp: When this epoch was logged
        metrics: Metrics for this epoch (loss, accuracy, etc.)
        notes: Any notes about this particular epoch
    """

    id: int
    run_id: int
    epoch_num: int
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "epoch_num": self.epoch_num,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "Epoch":
        """Build an Epoch object from a database row."""
        return cls(
            id=row[0],
            run_id=row[1],
            epoch_num=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            metrics=json.loads(row[4]) if row[4] else {},
            notes=row[5],
        )

    def __str__(self) -> str:
        return f"Epoch {self.epoch_num} (Run #{self.run_id}) - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
