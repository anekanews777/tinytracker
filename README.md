# 🔬 TinyTracker

**A minimal, local-only experiment tracker for ML projects.**

Simple, fast, and local tracking with zero configuration.

```console
pip install tinytracker-ml
```

---

## Why TinyTracker?

| Feature | MLflow | W&B | TinyTracker |
|---------|--------|-----|-------------|
| Local-only | ⚠️ | ❌ | ✅ |
| Zero config | ❌ | ❌ | ✅ |
| No server | ⚠️ | ❌ | ✅ |
| Tiny deps | ❌ | ❌ | ✅ |
| CLI-first | ⚠️ | ❌ | ✅ |

TinyTracker is for people who want experiment tracking without the overhead.

---

## Quick Start

Initialize in your project:

```console
$ cd my-ml-project
$ tinytracker init my_model
✓ Initialized tracker for project 'my_model'
```

Log a run from CLI:

```console
$ tinytracker log -p my_model \
    --metric acc=0.92 \
    --metric loss=0.08 \
    --param lr=0.001 \
    --param epochs=100 \
    --tag baseline

✓ Logged run #1 to project my_model
```

Log from Python:

```python
from tinytracker import Tracker

tracker = Tracker("my_model")

run_id = tracker.log(
    params={"lr": 0.001, "epochs": 100},
    metrics={"accuracy": 0.92, "loss": 0.08},
    tags=["baseline"],
    notes="First experiment"
)
```

---

## Core Features

### List and View Runs

```console
$ tinytracker list -p my_model

 ID   Timestamp         accuracy   loss    Tags
 3    2024-12-08 14:30     0.95    0.05   improved
 2    2024-12-08 12:15     0.93    0.07   baseline v2
 1    2024-12-08 10:00     0.92    0.08   baseline

$ tinytracker show 3
Run #3
Project: my_model
Time: 2024-12-08 14:30
...
```

### Find Best Run

```console
$ tinytracker best -p my_model --metric accuracy

★ Best accuracy: 0.95
  Run #3 from 2024-12-08 14:30
  lr=0.001, epochs=100
```

### Filter and Sort

```console
$ tinytracker list -p my_model --tag baseline --order-by accuracy:desc -n 5
```

### Compare Runs

```console
$ tinytracker compare 1 2

Comparing 2 runs:

Parameters
         #1      #2
  lr     0.01    0.001

Metrics
         #1      #2      Δ
  acc    0.92    0.93    +0.01
  loss   0.08    0.07    -0.01
```

### Update and Export

```console
$ tinytracker update 3 --notes "Best model" --add-tag production
$ tinytracker export -p my_model -f json -o runs.json
```

---

## Epoch Tracking

Track individual epochs within training runs to monitor progress over time.

```python
from tinytracker import Tracker

tracker = Tracker("my_model")

# Start a new run
run_id = tracker.log(params={"lr": 0.001, "batch_size": 32})

# Log each epoch during training
for epoch in range(1, 11):
    train_loss = train_one_epoch()
    val_loss, val_acc = validate()

    tracker.log_epoch(
        run_id=run_id,
        epoch_num=epoch,
        metrics={"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
    )

# Find the best epoch
best_epoch = tracker.best_epoch(run_id, "val_acc")
print(f"Best: epoch {best_epoch.epoch_num} with acc={best_epoch.metrics['val_acc']}")

# List all epochs
epochs = tracker.list_epochs(run_id)
```

---

## PyTorch Integration

```python
from tinytracker import Tracker

tracker = Tracker("mnist_classifier")

# Log the run with hyperparameters
run_id = tracker.log(
    params={"lr": 0.001, "batch_size": 64, "epochs": 10},
    tags=["pytorch"]
)

# Training loop
for epoch in range(1, 11):
    # Training
    train_loss = 0.0
    for batch in train_loader:
        loss = train_step(model, batch)
        train_loss += loss

    # Validation
    val_loss, val_acc = evaluate(model, val_loader)

    # Log this epoch
    tracker.log_epoch(
        run_id=run_id,
        epoch_num=epoch,
        metrics={
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss,
            "val_acc": val_acc
        }
    )

# Find and save best epoch
best = tracker.best_epoch(run_id, "val_acc")
tracker.update(run_id, notes=f"Best: epoch {best.epoch_num}, acc={best.metrics['val_acc']:.4f}")
```

---

## CLI Reference

```console
tinytracker init <project>                # Initialize tracker
tinytracker log -p <project>              # Log a new run
tinytracker list -p <project>             # List runs
tinytracker show <run_id>                 # Show run details
tinytracker compare <id1> <id2> ...       # Compare runs side-by-side
tinytracker diff <id1> <id2>              # Show what changed between runs
tinytracker best -p <project> -m <metric> # Find best run by metric
tinytracker update <run_id>               # Update run notes/tags
tinytracker delete <run_id>               # Delete a run
tinytracker export -p <project>           # Export to JSON/CSV
tinytracker projects                      # List all projects
tinytracker status                        # Show tracker status
tinytracker config                        # Show configuration
```

Use `tt` as a short alias: `tt log -p my_model --metric acc=0.95`

---

## Python API

### Tracker

```python
from tinytracker import Tracker

tracker = Tracker("project_name")

# Runs
run_id = tracker.log(params={...}, metrics={...}, tags=[...], notes="...")
run = tracker.get(run_id)
runs = tracker.list(tags=["baseline"], order_by="acc", limit=10)
best = tracker.best("accuracy", minimize=False)
tracker.update(run_id, notes="...", add_tags=[...])
tracker.delete(run_id)

# Epochs
epoch_id = tracker.log_epoch(run_id, epoch_num=1, metrics={...})
epoch = tracker.get_epoch(epoch_id)
epochs = tracker.list_epochs(run_id, order_by="loss", limit=10)
best_epoch = tracker.best_epoch(run_id, "val_acc")

# Export
data = tracker.export(format="json")  # or "csv"
```

### Objects

```python
# Run
run.id, run.project, run.timestamp
run.params, run.metrics, run.tags, run.notes

# Epoch
epoch.id, epoch.run_id, epoch.epoch_num
epoch.timestamp, epoch.metrics, epoch.notes
```

---

## Configuration

Create `.tinytracker.toml` to set a default project:

```toml
default_project = "my_model"
```

Now you can skip the `-p` flag:

```console
$ tinytracker log --metric acc=0.95
$ tinytracker list
$ tinytracker best -m accuracy
```

Or use environment variables: `export TINYTRACKER_PROJECT=my_model`

---

## Storage

Data is stored in `.tinytracker/tracker.db` (SQLite):

```
my-ml-project/
├── .tinytracker/
│   └── tracker.db
├── train.py
└── ...
```

Add to `.gitignore`: `.tinytracker/`

### Direct SQL Access

```console
$ sqlite3 .tinytracker/tracker.db

sqlite> SELECT id, json_extract(metrics, '$.accuracy') as acc
        FROM runs WHERE acc > 0.9 ORDER BY acc DESC;
```

---

## License

Apache 2.0

---

