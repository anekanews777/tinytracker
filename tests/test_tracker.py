"""Tests for TinyTracker."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tinytracker import Epoch, Tracker, Run, log, log_epoch


@pytest.fixture
def tmp_project():
    """Create a tracker in a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Tracker("test_project", root=tmpdir)


class TestTracker:
    def test_log_creates_run(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(
            params={"lr": 0.001, "epochs": 10},
            metrics={"accuracy": 0.95},
            tags=["test"],
            notes="Test run",
        )
        assert run_id == 1

    def test_get_returns_run(self, tmp_project):
        tracker = tmp_project
        tracker.log(params={"x": 1}, metrics={"y": 2.0})

        run = tracker.get(1)
        assert run is not None
        assert run.id == 1
        assert run.project == "test_project"
        assert run.params == {"x": 1}
        assert run.metrics == {"y": 2.0}

    def test_get_nonexistent_returns_none(self, tmp_project):
        assert tmp_project.get(999) is None

    def test_list_returns_all_runs(self, tmp_project):
        tracker = tmp_project
        tracker.log(metrics={"a": 1.0})
        tracker.log(metrics={"a": 2.0})
        tracker.log(metrics={"a": 3.0})

        runs = tracker.list()
        assert len(runs) == 3

    def test_list_filters_by_tags(self, tmp_project):
        tracker = tmp_project
        tracker.log(tags=["baseline"])
        tracker.log(tags=["improved"])
        tracker.log(tags=["baseline", "v2"])

        runs = tracker.list(tags=["baseline"])
        assert len(runs) == 2

    def test_list_orders_by_metric(self, tmp_project):
        tracker = tmp_project
        tracker.log(metrics={"acc": 0.8})
        tracker.log(metrics={"acc": 0.95})
        tracker.log(metrics={"acc": 0.7})

        runs = tracker.list(order_by="acc", order_desc=True)
        accs = [r.metrics["acc"] for r in runs]
        assert accs == [0.95, 0.8, 0.7]

    def test_list_limits_results(self, tmp_project):
        tracker = tmp_project
        for i in range(10):
            tracker.log(metrics={"i": float(i)})

        runs = tracker.list(limit=3)
        assert len(runs) == 3

    def test_compare_returns_runs_in_order(self, tmp_project):
        tracker = tmp_project
        tracker.log(metrics={"a": 1.0})
        tracker.log(metrics={"a": 2.0})
        tracker.log(metrics={"a": 3.0})

        runs = tracker.compare([3, 1, 2])
        assert [r.id for r in runs] == [3, 1, 2]

    def test_delete_removes_run(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(metrics={"x": 1.0})

        assert tracker.delete(run_id) is True
        assert tracker.get(run_id) is None

    def test_delete_nonexistent_returns_false(self, tmp_project):
        assert tmp_project.delete(999) is False

    def test_stats(self, tmp_project):
        tracker = tmp_project
        tracker.log(metrics={"a": 1.0})
        tracker.log(metrics={"a": 2.0})

        stats = tracker.stats
        assert stats["run_count"] == 2
        assert stats["first_run"] is not None
        assert stats["last_run"] is not None

    def test_export_json(self, tmp_project):
        tracker = tmp_project
        tracker.log(params={"x": 1}, metrics={"y": 2.0})

        data = json.loads(tracker.export(format="json"))
        assert len(data) == 1
        assert data[0]["params"] == {"x": 1}
        assert data[0]["metrics"] == {"y": 2.0}

    def test_export_csv(self, tmp_project):
        tracker = tmp_project
        tracker.log(params={"lr": 0.01}, metrics={"acc": 0.9})

        csv = tracker.export(format="csv")
        lines = csv.strip().split("\n")
        assert len(lines) == 2
        assert "param:lr" in lines[0]
        assert "metric:acc" in lines[0]


class TestRun:
    def test_to_dict(self):
        run = Run(
            id=1,
            project="test",
            timestamp=datetime(2024, 1, 15, 10, 30),
            params={"lr": 0.001},
            metrics={"acc": 0.95},
            tags=["baseline"],
            notes="Test",
        )

        d = run.to_dict()
        assert d["id"] == 1
        assert d["project"] == "test"
        assert d["timestamp"] == "2024-01-15T10:30:00"
        assert d["params"] == {"lr": 0.001}
        assert d["metrics"] == {"acc": 0.95}
        assert d["tags"] == ["baseline"]
        assert d["notes"] == "Test"

    def test_from_row(self):
        row = (
            1,
            "test",
            "2024-01-15T10:30:00",
            '{"lr": 0.001}',
            '{"acc": 0.95}',
            '["baseline"]',
            "Test",
        )

        run = Run.from_row(row)
        assert run.id == 1
        assert run.project == "test"
        assert run.params == {"lr": 0.001}
        assert run.metrics == {"acc": 0.95}
        assert run.tags == ["baseline"]
        assert run.notes == "Test"

    def test_str(self):
        run = Run(
            id=42,
            project="my_proj",
            timestamp=datetime(2024, 1, 15, 10, 30),
        )
        s = str(run)
        assert "42" in s
        assert "my_proj" in s


class TestTrackerUpdate:
    def test_update_notes(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(notes="original")

        tracker.update(run_id, notes="updated")
        run = tracker.get(run_id)
        assert run.notes == "updated"

    def test_update_replace_tags(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(tags=["a", "b"])

        tracker.update(run_id, tags=["x", "y"])
        run = tracker.get(run_id)
        assert set(run.tags) == {"x", "y"}

    def test_update_add_tags(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(tags=["a"])

        tracker.update(run_id, add_tags=["b", "c"])
        run = tracker.get(run_id)
        assert set(run.tags) == {"a", "b", "c"}

    def test_update_remove_tags(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(tags=["a", "b", "c"])

        tracker.update(run_id, remove_tags=["b"])
        run = tracker.get(run_id)
        assert set(run.tags) == {"a", "c"}

    def test_update_nonexistent_returns_false(self, tmp_project):
        assert tmp_project.update(999, notes="x") is False


class TestTrackerBest:
    def test_best_finds_max(self, tmp_project):
        tracker = tmp_project
        tracker.log(metrics={"acc": 0.8})
        tracker.log(metrics={"acc": 0.95})
        tracker.log(metrics={"acc": 0.7})

        best = tracker.best("acc")
        assert best.metrics["acc"] == 0.95

    def test_best_finds_min(self, tmp_project):
        tracker = tmp_project
        tracker.log(metrics={"loss": 0.5})
        tracker.log(metrics={"loss": 0.1})
        tracker.log(metrics={"loss": 0.3})

        best = tracker.best("loss", minimize=True)
        assert best.metrics["loss"] == 0.1

    def test_best_returns_none_if_no_metric(self, tmp_project):
        tracker = tmp_project
        tracker.log(metrics={"other": 1.0})

        assert tracker.best("acc") is None


class TestLogFunction:
    def test_log_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = log(
                project="quick_test",
                params={"x": 1},
                metrics={"y": 2.0},
                root=tmpdir,
            )
            assert run_id == 1

            # Verify it was stored
            tracker = Tracker("quick_test", root=tmpdir)
            run = tracker.get(1)
            assert run is not None
            assert run.params == {"x": 1}


class TestEpochTracking:
    def test_log_epoch_creates_epoch(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})

        epoch_id = tracker.log_epoch(
            run_id=run_id,
            epoch_num=1,
            metrics={"loss": 0.5, "accuracy": 0.85},
            notes="First epoch",
        )
        assert epoch_id == 1

    def test_get_epoch_returns_epoch(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})
        tracker.log_epoch(run_id=run_id, epoch_num=1, metrics={"loss": 0.5})

        epoch = tracker.get_epoch(1)
        assert epoch is not None
        assert epoch.id == 1
        assert epoch.run_id == run_id
        assert epoch.epoch_num == 1
        assert epoch.metrics == {"loss": 0.5}

    def test_get_nonexistent_epoch_returns_none(self, tmp_project):
        assert tmp_project.get_epoch(999) is None

    def test_list_epochs_returns_all_epochs(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})

        tracker.log_epoch(run_id=run_id, epoch_num=1, metrics={"loss": 0.5})
        tracker.log_epoch(run_id=run_id, epoch_num=2, metrics={"loss": 0.3})
        tracker.log_epoch(run_id=run_id, epoch_num=3, metrics={"loss": 0.2})

        epochs = tracker.list_epochs(run_id)
        assert len(epochs) == 3
        assert [e.epoch_num for e in epochs] == [1, 2, 3]

    def test_list_epochs_orders_by_metric(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})

        tracker.log_epoch(run_id=run_id, epoch_num=1, metrics={"acc": 0.8})
        tracker.log_epoch(run_id=run_id, epoch_num=2, metrics={"acc": 0.95})
        tracker.log_epoch(run_id=run_id, epoch_num=3, metrics={"acc": 0.7})

        epochs = tracker.list_epochs(run_id, order_by="acc", order_desc=True)
        accs = [e.metrics["acc"] for e in epochs]
        assert accs == [0.95, 0.8, 0.7]

    def test_list_epochs_limits_results(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})

        for i in range(10):
            tracker.log_epoch(run_id=run_id, epoch_num=i+1, metrics={"loss": float(i)})

        epochs = tracker.list_epochs(run_id, limit=3)
        assert len(epochs) == 3

    def test_list_epochs_only_returns_epochs_for_run(self, tmp_project):
        tracker = tmp_project
        run1 = tracker.log(params={"lr": 0.001})
        run2 = tracker.log(params={"lr": 0.01})

        tracker.log_epoch(run_id=run1, epoch_num=1, metrics={"loss": 0.5})
        tracker.log_epoch(run_id=run1, epoch_num=2, metrics={"loss": 0.3})
        tracker.log_epoch(run_id=run2, epoch_num=1, metrics={"loss": 0.6})

        epochs = tracker.list_epochs(run1)
        assert len(epochs) == 2
        assert all(e.run_id == run1 for e in epochs)

    def test_delete_epoch_removes_epoch(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})
        epoch_id = tracker.log_epoch(run_id=run_id, epoch_num=1, metrics={"loss": 0.5})

        assert tracker.delete_epoch(epoch_id) is True
        assert tracker.get_epoch(epoch_id) is None

    def test_delete_nonexistent_epoch_returns_false(self, tmp_project):
        assert tmp_project.delete_epoch(999) is False

    def test_update_epoch_notes(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})
        epoch_id = tracker.log_epoch(run_id=run_id, epoch_num=1, notes="original")

        tracker.update_epoch(epoch_id, notes="updated")
        epoch = tracker.get_epoch(epoch_id)
        assert epoch.notes == "updated"

    def test_update_nonexistent_epoch_returns_false(self, tmp_project):
        assert tmp_project.update_epoch(999, notes="x") is False

    def test_best_epoch_finds_max(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})

        tracker.log_epoch(run_id=run_id, epoch_num=1, metrics={"acc": 0.8})
        tracker.log_epoch(run_id=run_id, epoch_num=2, metrics={"acc": 0.95})
        tracker.log_epoch(run_id=run_id, epoch_num=3, metrics={"acc": 0.7})

        best = tracker.best_epoch(run_id, "acc")
        assert best.metrics["acc"] == 0.95
        assert best.epoch_num == 2

    def test_best_epoch_finds_min(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})

        tracker.log_epoch(run_id=run_id, epoch_num=1, metrics={"loss": 0.5})
        tracker.log_epoch(run_id=run_id, epoch_num=2, metrics={"loss": 0.1})
        tracker.log_epoch(run_id=run_id, epoch_num=3, metrics={"loss": 0.3})

        best = tracker.best_epoch(run_id, "loss", minimize=True)
        assert best.metrics["loss"] == 0.1
        assert best.epoch_num == 2

    def test_best_epoch_returns_none_if_no_metric(self, tmp_project):
        tracker = tmp_project
        run_id = tracker.log(params={"lr": 0.001})
        tracker.log_epoch(run_id=run_id, epoch_num=1, metrics={"other": 1.0})

        assert tracker.best_epoch(run_id, "acc") is None


class TestEpochModel:
    def test_to_dict(self):
        epoch = Epoch(
            id=1,
            run_id=5,
            epoch_num=3,
            timestamp=datetime(2024, 1, 15, 10, 30),
            metrics={"loss": 0.25, "acc": 0.92},
            notes="Good epoch",
        )

        d = epoch.to_dict()
        assert d["id"] == 1
        assert d["run_id"] == 5
        assert d["epoch_num"] == 3
        assert d["timestamp"] == "2024-01-15T10:30:00"
        assert d["metrics"] == {"loss": 0.25, "acc": 0.92}
        assert d["notes"] == "Good epoch"

    def test_from_row(self):
        row = (
            1,
            5,
            3,
            "2024-01-15T10:30:00",
            '{"loss": 0.25, "acc": 0.92}',
            "Good epoch",
        )

        epoch = Epoch.from_row(row)
        assert epoch.id == 1
        assert epoch.run_id == 5
        assert epoch.epoch_num == 3
        assert epoch.metrics == {"loss": 0.25, "acc": 0.92}
        assert epoch.notes == "Good epoch"

    def test_str(self):
        epoch = Epoch(
            id=1,
            run_id=5,
            epoch_num=10,
            timestamp=datetime(2024, 1, 15, 10, 30),
        )
        s = str(epoch)
        assert "10" in s
        assert "5" in s


class TestLogEpochFunction:
    def test_log_epoch_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = log(project="quick_test", params={"x": 1}, root=tmpdir)

            epoch_id = log_epoch(
                project="quick_test",
                run_id=run_id,
                epoch_num=1,
                metrics={"loss": 0.5},
                root=tmpdir,
            )
            assert epoch_id == 1

            # Verify it was stored
            tracker = Tracker("quick_test", root=tmpdir)
            epoch = tracker.get_epoch(1)
            assert epoch is not None
            assert epoch.run_id == run_id
            assert epoch.epoch_num == 1
            assert epoch.metrics == {"loss": 0.5}

