import os

import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own working directory / WANDB_DIR and a clean global state."""
    for key in list(os.environ):
        if key.startswith("WANDB_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_LOCAL_DISABLE_STATS", "true")
    yield tmp_path
    import wandb

    for run in list(wandb._run._active_runs):
        run.finish()
    wandb._run._sweep_context = None
