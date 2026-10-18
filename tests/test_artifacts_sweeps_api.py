from pathlib import Path

import pytest

import wandb


def test_artifact_versions_aliases_and_dedup(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("A")
    run = wandb.init(project="arts")
    art = wandb.Artifact("ds", type="dataset", metadata={"n": 1})
    art.add_dir(str(tmp_path / "data"))
    with art.new_file("extra/b.txt", mode="w") as fh:
        fh.write("B")
    run.log_artifact(art, aliases=["first"])
    assert art.version == "v0" and set(art.aliases) == {"first", "latest"}

    same = wandb.Artifact("ds", type="dataset")
    same.add_dir(str(tmp_path / "data"))
    with same.new_file("extra/b.txt", mode="w") as fh:
        fh.write("B")
    run.log_artifact(same, aliases=["again"])
    assert same.version == "v0"  # identical content -> same version

    (tmp_path / "data" / "a.txt").write_text("A2")
    changed = wandb.Artifact("ds", type="dataset")
    changed.add_dir(str(tmp_path / "data"))
    run.log_artifact(changed)
    assert changed.version == "v1"

    used = run.use_artifact("ds:first")
    assert used.version == "v0"
    path = Path(used.download())
    assert (path / "a.txt").read_text() == "A" and (path / "extra" / "b.txt").read_text() == "B"
    assert run.use_artifact("arts/ds:latest").version == "v1"
    out = used.download(root=str(tmp_path / "out"))
    assert (Path(out) / "extra" / "b.txt").exists()
    wandb.finish()


def test_artifact_add_table_and_get():
    run = wandb.init()
    art = wandb.Artifact("tab", type="dataset")
    art.add(wandb.Table(columns=["x"], data=[[1], [2]]), "mytable")
    run.log_artifact(art)
    t = run.use_artifact("tab:latest").get("mytable")
    assert t.columns == ["x"] and t.data == [[1], [2]]
    wandb.finish()


def test_log_model_and_module_level_log_artifact(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "weights.bin").write_bytes(b"123")
    run = wandb.init()
    wandb.log_artifact(str(tmp_path / "ckpt"), name="model-x", type="model", aliases=["epoch_1"])
    assert Path(run.use_model("model-x:epoch_1"), "weights.bin").read_bytes() == b"123"
    wandb.finish()


def test_sweep_agent_with_function():
    seen = []

    def train():
        run = wandb.init()
        seen.append(dict(run.config))
        wandb.log({"objective": run.config["x"] * 2})
        # user code trying to override a sweep parameter is ignored (like the real client)
        wandb.config.update({"x": -1}, allow_val_change=True)
        assert wandb.config.x != -1

    sweep_id = wandb.sweep({"method": "grid", "metric": {"name": "objective", "goal": "minimize"}, "parameters": {"x": {"values": [1, 2, 3]}, "y": {"value": "c"}}}, project="sw")
    wandb.agent(sweep_id, function=train)
    assert [s["x"] for s in seen] == [1, 2, 3] and all(s["y"] == "c" for s in seen)
    state = wandb.Api().sweep(sweep_id)
    assert [t["metric"] for t in state["trials"]] == [2, 4, 6]


def test_sweep_random_distributions():
    from wandb._sweeps import _sample
    import random

    rng = random.Random(0)
    v = _sample({"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2}, rng)
    assert 1e-5 <= v <= 1e-2
    assert _sample({"min": 1, "max": 3}, rng) in (1, 2, 3)
    assert _sample({"parameters": {"a": {"value": 1}}}, rng) == {"a": 1}


def test_public_api():
    pd = pytest.importorskip("pandas")
    run = wandb.init(project="apiproj", name="r1", config={"lr": 0.1})
    for i in range(10):
        wandb.log({"loss": 10 - i})
    wandb.finish()
    api = wandb.Api()
    runs = api.runs("apiproj")
    assert len(runs) == 1 and runs[0].name == "r1" and runs[0].config["lr"] == 0.1
    r = api.run(f"apiproj/{run.id}")
    df = r.history()
    assert isinstance(df, pd.DataFrame) and len(df) == 10 and df["loss"].iloc[-1] == 1
    assert len(list(r.scan_history(keys=["loss"]))) == 10
    assert r.summary["loss"] == 1 and r.state == "finished"
    assert api.runs("apiproj", filters={"config.lr": 0.1})
    assert not api.runs("apiproj", filters={"config.lr": {"$gt": 1}})
