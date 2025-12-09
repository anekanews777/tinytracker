"""Tests for storage layer."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tinytracker.storage import Storage, get_db_path


@pytest.fixture
def storage():
    """Create storage in a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / ".tinytracker" / "tracker.db"
        yield Storage(db_path)


class TestStorage:
    def test_insert_and_get(self, storage):
        run_id = storage.insert_run(
            project="test",
            params={"lr": 0.01},
            metrics={"acc": 0.9},
            tags=["v1"],
            notes="Note",
        )

        run = storage.get_run(run_id)
        assert run is not None
        assert run.project == "test"
        assert run.params == {"lr": 0.01}
        assert run.metrics == {"acc": 0.9}
        assert run.tags == ["v1"]
        assert run.notes == "Note"

    def test_auto_increment_ids(self, storage):
        id1 = storage.insert_run("p1", {}, {}, [])
        id2 = storage.insert_run("p1", {}, {}, [])
        id3 = storage.insert_run("p1", {}, {}, [])

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_list_filters_by_project(self, storage):
        storage.insert_run("proj_a", {}, {}, [])
        storage.insert_run("proj_b", {}, {}, [])
        storage.insert_run("proj_a", {}, {}, [])

        runs = storage.list_runs(project="proj_a")
        assert len(runs) == 2
        assert all(r.project == "proj_a" for r in runs)

    def test_list_filters_by_date_range(self, storage):
        storage.insert_run("test", {}, {}, [])

        now = datetime.now()
        future = now + timedelta(days=1)
        past = now - timedelta(days=1)

        # Should find run (after yesterday)
        runs = storage.list_runs(after=past)
        assert len(runs) == 1

        # Should not find run (after tomorrow)
        runs = storage.list_runs(after=future)
        assert len(runs) == 0

    def test_get_runs_by_ids(self, storage):
        storage.insert_run("test", {}, {"a": 1.0}, [])
        storage.insert_run("test", {}, {"a": 2.0}, [])
        storage.insert_run("test", {}, {"a": 3.0}, [])

        runs = storage.get_runs_by_ids([3, 1])
        assert len(runs) == 2
        assert [r.id for r in runs] == [3, 1]

    def test_get_runs_by_ids_empty(self, storage):
        runs = storage.get_runs_by_ids([])
        assert runs == []

    def test_delete(self, storage):
        run_id = storage.insert_run("test", {}, {}, [])
        assert storage.delete_run(run_id) is True
        assert storage.get_run(run_id) is None
        assert storage.delete_run(run_id) is False

    def test_get_projects(self, storage):
        storage.insert_run("alpha", {}, {}, [])
        storage.insert_run("beta", {}, {}, [])
        storage.insert_run("alpha", {}, {}, [])

        projects = storage.get_projects()
        assert projects == ["alpha", "beta"]

    def test_get_project_stats(self, storage):
        storage.insert_run("test", {}, {}, [])
        storage.insert_run("test", {}, {}, [])

        stats = storage.get_project_stats("test")
        assert stats["run_count"] == 2
        assert stats["first_run"] is not None
        assert stats["last_run"] is not None

    def test_export_json(self, storage):
        storage.insert_run("test", {"x": 1}, {"y": 2.0}, [])
        data = storage.export_runs(format="json")
        assert '"x": 1' in data
        assert '"y": 2.0' in data

    def test_export_csv(self, storage):
        storage.insert_run("test", {"lr": 0.01}, {"acc": 0.9}, ["tag1"])
        csv = storage.export_runs(format="csv")

        lines = csv.split("\n")
        assert "param:lr" in lines[0]
        assert "metric:acc" in lines[0]
        assert "0.01" in lines[1]
        assert "0.9" in lines[1]

    def test_export_invalid_format(self, storage):
        with pytest.raises(ValueError, match="Unknown format"):
            storage.export_runs(format="xml")

    def test_update_run_notes(self, storage):
        run_id = storage.insert_run("test", {}, {}, [], notes="old")
        storage.update_run(run_id, notes="new")
        run = storage.get_run(run_id)
        assert run.notes == "new"

    def test_update_run_tags(self, storage):
        run_id = storage.insert_run("test", {}, {}, ["a", "b"])
        storage.update_run(run_id, tags=["x"])
        run = storage.get_run(run_id)
        assert run.tags == ["x"]

    def test_update_run_append_tags(self, storage):
        run_id = storage.insert_run("test", {}, {}, ["a"])
        storage.update_run(run_id, append_tags=["b"])
        run = storage.get_run(run_id)
        assert set(run.tags) == {"a", "b"}

    def test_update_run_remove_tags(self, storage):
        run_id = storage.insert_run("test", {}, {}, ["a", "b", "c"])
        storage.update_run(run_id, remove_tags=["b"])
        run = storage.get_run(run_id)
        assert set(run.tags) == {"a", "c"}

    def test_get_best_run_max(self, storage):
        storage.insert_run("test", {}, {"acc": 0.8}, [])
        storage.insert_run("test", {}, {"acc": 0.95}, [])
        storage.insert_run("test", {}, {"acc": 0.7}, [])

        best = storage.get_best_run("test", "acc", minimize=False)
        assert best.metrics["acc"] == 0.95

    def test_get_best_run_min(self, storage):
        storage.insert_run("test", {}, {"loss": 0.5}, [])
        storage.insert_run("test", {}, {"loss": 0.1}, [])
        storage.insert_run("test", {}, {"loss": 0.3}, [])

        best = storage.get_best_run("test", "loss", minimize=True)
        assert best.metrics["loss"] == 0.1

    def test_get_best_run_no_metric(self, storage):
        storage.insert_run("test", {}, {"other": 1.0}, [])
        assert storage.get_best_run("test", "acc") is None


class TestGetDbPath:
    def test_default_path(self):
        path = get_db_path()
        assert path.name == "tracker.db"
        assert ".tinytracker" in str(path)

    def test_custom_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = get_db_path(Path(tmpdir))
            assert str(tmpdir) in str(path)

