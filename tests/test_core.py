import json
import math
import sys
from pathlib import Path

import pytest

import wandb
from wandb.sdk.lib.config_util import ConfigError


def history(run_dir):
    with open(Path(run_dir) / "files" / "wandb-history.jsonl") as fh:
        return [json.loads(line) for line in fh]


def test_init_creates_layout_and_globals(tmp_path):
    run = wandb.init(project="p", name="n", config={"a": 1}, tags=["x"], notes="hello")
    assert wandb.run is run and wandb.config is run.config and wandb.summary is run.summary
    assert run.project == "p" and run.name == "n" and run.tags == ("x",) and run.url is None
    rd = Path(run.run_dir)
    assert rd.parent == tmp_path / "wandb" and rd.name.endswith(run.id)
    assert (tmp_path / "wandb" / "latest-run").resolve() == rd.resolve()
    for name in ("config.yaml", "wandb-summary.json", "wandb-metadata.json", "requirements.txt", "wandb-history.jsonl"):
        assert (rd / "files" / name).exists(), name
    wandb.finish()
    assert wandb.run is None
    info = json.loads((rd / "wandb-run.json").read_text())
    assert info["state"] == "finished" and info["exit_code"] == 0 and info["notes"] == "hello"
    with pytest.raises(wandb.Error):
        wandb.log({"x": 1})


def test_log_step_semantics():
    run = wandb.init()
    wandb.log({"a": 1})
    wandb.log({"b": 2}, commit=False)
    wandb.log({"c": 3})
    wandb.log({"d": 4}, step=5)          # accumulates until a larger step arrives
    wandb.log({"e": 5}, step=5)
    wandb.log({"f": 6}, step=3)          # lower step: ignored
    wandb.log({"g": 7}, step=7)
    wandb.finish()
    rows = history(run.run_dir)
    assert [r["_step"] for r in rows] == [0, 1, 5, 7]
    assert rows[1]["b"] == 2 and rows[1]["c"] == 3
    assert rows[2]["d"] == 4 and rows[2]["e"] == 5
    assert all("f" not in r for r in rows)
    for r in rows:
        assert "_runtime" in r and "_timestamp" in r


def test_non_finite_values_are_strict_json():
    run = wandb.init()
    wandb.log({"nan": float("nan"), "inf": float("inf"), "ninf": -math.inf})
    wandb.finish()
    raw = (Path(run.run_dir) / "files" / "wandb-history.jsonl").read_text()
    row = json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert row["nan"] == "NaN" and row["inf"] == "Infinity" and row["ninf"] == "-Infinity"


def test_define_metric_step_sync_and_summary():
    run = wandb.init()
    wandb.define_metric("train/global_step")
    wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
    wandb.define_metric("eval/loss", summary="min,max,mean")
    wandb.log({"train/loss": 1.0, "train/global_step": 10})
    wandb.log({"profiling/t": 0.5})
    wandb.log({"eval/loss": 3.0})
    wandb.log({"eval/loss": 1.0})
    rows = history(run.run_dir)
    assert rows[1]["train/global_step"] == 10  # synced
    assert rows[2]["train/global_step"] == 10  # explicit metric inherits the glob's x-axis
    assert run.summary["eval/loss"] == {"min": 1.0, "max": 3.0, "mean": 2.0}
    assert run.summary["train/loss"] == 1.0
    wandb.finish()


def test_config_semantics():
    import argparse

    run = wandb.init(config=argparse.Namespace(lr=0.1, layers=2))
    assert wandb.config.lr == 0.1 and wandb.config["layers"] == 2
    with pytest.raises(ConfigError):
        wandb.config.update({"lr": 0.2})
    wandb.config.update({"lr": 0.2}, allow_val_change=True)
    wandb.config["model/num_parameters"] = 10
    wandb.config.update({"nested": {"a": [1, 2]}, "path": Path("/tmp/x")})
    assert vars(wandb.config)["_items"]["lr"] == 0.2
    assert dict(wandb.config)["path"] == "/tmp/x"
    with pytest.raises(AttributeError):
        _ = wandb.config.missing
    with pytest.raises(TypeError):
        wandb.config.update({"img": wandb.Histogram([1, 2, 3])})
    wandb.finish()
    text = (Path(run.run_dir) / "files" / "config.yaml").read_text()
    import yaml

    data = yaml.safe_load(text)
    assert data["lr"]["value"] == 0.2 and data["nested"]["value"] == {"a": [1, 2]} and "_wandb" in data


def test_media_types(tmp_path):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    run = wandb.init()
    img = wandb.Image(np.random.rand(8, 10, 3), caption="c")
    df = pd.DataFrame({"prompt": ["a", "b"], "score": [1.0, 2.0], "imgs": [[wandb.Image(np.zeros((4, 4)))], []]})
    wandb.log({
        "img": img,
        "imgs": [wandb.Image(np.ones((4, 4))), wandb.Image(np.zeros((4, 4)))],
        "table": wandb.Table(dataframe=df),
        "df": df,  # raw DataFrame (trl LogCompletionsCallback does this)
        "hist": wandb.Histogram(np.random.randn(100)),
        "arr": np.arange(100),  # arrays become histograms
        "scalar": np.float32(1.5),
        "html": wandb.Html("<b>x</b>"),
        "audio": wandb.Audio(np.zeros(800), sample_rate=8000),
        "plot": wandb.plot.line(wandb.Table(columns=["x", "y"], data=[[1, 2], [2, 3]]), "x", "y", title="t"),
    })
    wandb.finish()
    row = history(run.run_dir)[0]
    files = Path(run.run_dir) / "files"
    assert row["img"]["_type"] == "image-file" and (files / row["img"]["path"]).exists() and row["img"]["width"] == 10
    assert row["imgs"]["_type"] == "images/separated" and row["imgs"]["count"] == 2
    for key in ("table", "df"):
        ref = row[key]
        assert ref["_type"] == "table-file" and ref["nrows"] == 2
        payload = json.loads((files / ref["path"]).read_text())
        assert payload["columns"] == ["prompt", "score", "imgs"]
        assert payload["data"][0][2][0]["_type"] == "image-file"
    assert row["hist"]["_type"] == "histogram" and len(row["hist"]["bins"]) == len(row["hist"]["values"]) + 1
    assert row["arr"]["_type"] == "histogram" and row["scalar"] == 1.5
    assert row["html"]["_type"] == "html-file" and row["audio"]["_type"] == "audio-file"
    assert row["plot"]["_type"] == "plot" and row["plot"]["table"]["_type"] == "table-file"


def test_table_api():
    t = wandb.Table(columns=["a", "b"])
    t.add_data(1, "x")
    t.add_data(2, "y")
    t.add_column("c", [True, False])
    assert t.get_column("a") == [1, 2] and t.columns == ["a", "b", "c"] and len(t.data) == 2
    with pytest.raises(ValueError):
        t.add_data(1)


def test_console_capture():
    run = wandb.init()
    print("captured line")
    print("err line", file=sys.stderr)
    wandb.finish()
    out = (Path(run.run_dir) / "files" / "output.log").read_text()
    assert "captured line" in out and "err line" in out
    print("after finish")  # streams are restored
    assert "after finish" not in (Path(run.run_dir) / "files" / "output.log").read_text()


def test_disabled_mode(tmp_path):
    run = wandb.init(mode="disabled", config={"a": 1})
    wandb.log({"x": 1})
    assert run.disabled and wandb.config.a == 1
    wandb.finish()
    assert not (tmp_path / "wandb").exists()


def test_reinit_and_context_manager():
    r1 = wandb.init()
    assert wandb.init() is r1  # returns the active run by default
    r2 = wandb.init(reinit="finish_previous")
    assert r2 is not r1 and r1._finished
    r2.finish()
    with wandb.init() as r3:
        r3.log({"a": 1})
    assert r3._finished and r3._state == "finished"


def test_resume_and_rewind():
    r1 = wandb.init(id="resumeme", config={"a": 1})
    for i in range(5):
        wandb.log({"v": i})
    wandb.finish()
    r2 = wandb.init(id="resumeme", resume="allow")
    assert r2.resumed and r2.step == 5 and wandb.config.a == 1 and wandb.summary["v"] == 4
    wandb.log({"v": 5})
    wandb.finish()
    assert r2.run_dir == r1.run_dir
    assert [r["_step"] for r in history(r1.run_dir)] == [0, 1, 2, 3, 4, 5]
    r3 = wandb.init(resume_from="resumeme?_step=2")
    assert r3.step == 3
    wandb.log({"v": 30})
    wandb.finish()
    assert [r["_step"] for r in history(r1.run_dir)] == [0, 1, 2, 3]
    with pytest.raises(wandb.errors.UsageError):
        wandb.init(id="nope", resume="must")


def test_fork_from():
    r1 = wandb.init(id="parent", config={"a": 1})
    for i in range(5):
        wandb.log({"v": i})
    wandb.finish()
    r2 = wandb.init(fork_from="parent?_step=2")
    assert r2.id != "parent" and r2.step == 3 and wandb.config.a == 1
    wandb.log({"v": 99})
    wandb.finish()
    assert [r["v"] for r in history(r2.run_dir)] == [0, 1, 2, 99]


def test_save_and_restore(tmp_path):
    (tmp_path / "notes.txt").write_text("hi")
    run = wandb.init(id="saver")
    run.save("notes.txt", policy="now")
    assert (Path(run.dir) / "notes.txt").read_text() == "hi"
    wandb.finish()
    wandb.init()
    fh = wandb.restore("notes.txt", run_path="local/uncategorized/saver")
    assert fh.read() == "hi"
    wandb.finish()


def test_watch_logs_histograms():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
    run = wandb.init()
    wandb.watch(model, log="all", log_freq=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(4):
        loss = model(torch.randn(8, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        wandb.log({"loss": loss.item()})
    wandb.finish()
    keys = set().union(*(r.keys() for r in history(run.run_dir)))
    assert "gradients/0.weight" in keys and "parameters/2.bias" in keys


def test_alert_and_preempting():
    run = wandb.init()
    run.alert("title", "text", level=wandb.AlertLevel.WARN)
    run.mark_preempting()
    wandb.finish()
    alerts = (Path(run.run_dir) / "files" / "wandb-alerts.jsonl").read_text()
    assert '"level":"WARN"' in alerts
    assert json.loads((Path(run.run_dir) / "wandb-run.json").read_text())["preempting"] is True


def test_compat_import_paths():
    from wandb.sdk.lib import RunDisabled  # noqa: F401
    from wandb.sdk.wandb_run import Run  # noqa: F401
    from wandb.wandb_run import Run as Run2  # noqa: F401
    import wandb.env

    wandb.env.set_project("proj")
    assert wandb.env.get_project() == "proj"
    assert wandb.login() is True
    assert wandb.__version__.startswith("0.")
