# wandb-local

An offline, file-based drop-in replacement for the [Weights & Biases](https://wandb.ai) Python client.
It needs no account, server or network access.

It installs as the `wandb` package, so `report_to="wandb"` in transformers / TRL
(`SFTTrainer`, `GRPOTrainer`, `DPOTrainer`, distillation trainers…), accelerate trackers and plain
`wandb.init()` / `wandb.log()` scripts keep working unchanged. Everything is written to the
`./wandb/` folder, and a local dashboard shows it:

```bash
wandb ui            # live dashboard at http://127.0.0.1:8080
wandb report        # one self-contained HTML file you can share or archive
```

The dashboard has these tabs:

- **Charts:** run comparison, smoothing, log scale, zoom and hover.
- **System:** CPU, RAM, GPU, disk and network.
- **Tables:** for example the sample completions logged by TRL.
- **Media:** images, HTML, audio, video, histograms and plots.
- **Overview, Config and Summary:** run details and side-by-side diffs.
- **Logs:** console output.
- **Artifacts:** logged model checkpoints and other files.

## Install

The real client and this package share the import name `wandb`, so remove the real one first:

```bash
pip uninstall -y wandb
pip install /path/to/custom_wandb          # or: pip install -e /path/to/custom_wandb
```

Optional extras: `pip install "/path/to/custom_wandb[all]"` adds `psutil` (richer system metrics)
plus `numpy` and `pillow` (image and video encoding). The only hard dependency is PyYAML.

> The distribution is deliberately named `wandb` (version `0.28.1+local`). accelerate and TRL check
> `importlib.metadata.metadata("wandb")`, so any other name would make `report_to="wandb"` silently
> do nothing. Running `pip install -U wandb` later would replace this package with the real client.

## Use with TRL / transformers

You don't need to change anything:

```python
from trl import SFTConfig, SFTTrainer

args = SFTConfig(output_dir="out", report_to="wandb", run_name="my-sft", logging_steps=10)
trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
trainer.train()
```

The usual environment variables still apply:

| Variable | What it does |
|---|---|
| `WANDB_PROJECT` | Project name (default `huggingface` when set through transformers). |
| `WANDB_LOG_MODEL=end\|checkpoint` | Logs model artifacts. |
| `WANDB_WATCH=gradients\|parameters\|all` | Logs gradient and parameter histograms. |
| `WANDB_DIR` | Where the `wandb/` folder is created. |
| `WANDB_MODE=disabled` | Turns logging off. |
| `WANDB_RUN_ID` + `WANDB_RESUME=allow` | Continues an existing run. |

`trainer.hyperparameter_search(backend="wandb")` also works. It runs a local random or grid search.

## View your runs

```bash
wandb ui [PATH ...] [--port 8080] [--host 127.0.0.1] [-r] [--open]
```

`PATH` can be:

- a folder that contains `wandb/`,
- a `wandb/` folder itself,
- a single run directory,
- or a parent folder searched with `-r`.

While training runs, the page polls for new data every 5 s. **Export** in the dashboard, or the
command below, writes one HTML file with the data embedded. It opens from disk with no server, so
you can e-mail it or keep it with the checkpoint:

```bash
wandb report -o report.html [--run ID ...] [--project P] [--tables latest|all|none]
```

Other commands:

| Command | What it does |
|---|---|
| `wandb ls` | Lists runs. |
| `wandb show RUN` | Prints a run's info, config and summary as JSON. |
| `wandb export RUN -o history.csv` | Exports history (`--system` for system metrics, `--format jsonl`). |
| `wandb artifact ls` / `wandb artifact get NAME:ALIAS --root DIR` | Lists or fetches artifacts. |
| `wandb sweep cfg.yaml` / `wandb agent SWEEP_ID` | Runs a local sweep (`program` from the config). |
| `wandb sync [PATH]` | Imports runs recorded **offline by the real wandb client** (see below). |
| `wandb login`, `wandb offline`, `wandb online` | No-ops, kept so existing scripts don't break. |
| `wandb disabled`, `wandb enabled` | Turn logging off or on for this folder. |

### Runs recorded earlier with the real client

Offline runs from the real client keep their history in a binary `run-<id>.wandb` file.
`wandb ui` decodes these automatically, and `wandb sync` does it explicitly. Decoding uses a
built-in reader, so neither protobuf nor the real client is needed. It adds three small files next
to the original (`wandb-run.json`, `files/wandb-history.jsonl`, `files/wandb-events.jsonl`) and
never modifies or deletes anything.

## Python API

Logging (same signatures as the real client):

- `wandb.init` takes the usual arguments: project, name, config, tags, notes, group, job_type, dir,
  id, resume, `resume_from="id?_step=N"`, `fork_from=...`, mode, reinit, save_code and settings.
- `wandb.log` keeps the real step semantics: `step=`, `commit=False`, and monotonic steps.
- `wandb.config`, `run.summary`, `wandb.define_metric` (`step_metric`, `step_sync`, `summary="min,max,mean,best,..."`, `hidden`, `goal`).
- `wandb.finish`, `wandb.save`, `wandb.restore`, `wandb.watch`, `wandb.unwatch`, `run.alert`, `run.log_code`, `run.mark_preempting`.

Media:

- `wandb.Table` (including `dataframe=`, images inside cells, `add_data`, `add_column`, `get_dataframe`).
- `wandb.Image` (numpy, torch, PIL, matplotlib or a file path), `wandb.Html`, `wandb.Histogram`, `wandb.Audio`, `wandb.Video`, `wandb.Object3D`, `wandb.Molecule`.
- `wandb.plot.line / scatter / bar / histogram / line_series / confusion_matrix / pr_curve / roc_curve`.

Artifacts and analysis:

- `wandb.Artifact` with versioning, aliases and content-hash deduplication; `log_artifact`, `use_artifact`, `log_model`, `use_model`, `download()`, `get()`.
- `wandb.Api()` reads local runs: `api.runs("project")`, `api.run("id")`, `run.history()` (pandas), `run.scan_history()`, `api.artifact("name:alias")`.
- `wandb.sweep` and `wandb.agent` run grid or random search.

Compatibility import paths (`wandb.sdk.lib.config_util.ConfigError`, `wandb.sdk.lib.RunDisabled`,
`wandb.wandb_run.Run`, `wandb.env` …) exist so third-party integrations import cleanly.

## What gets written

```
wandb/
  latest-run -> run-20261018_120000-abcd1234
  run-20261018_120000-abcd1234/
    wandb-run.json             id, name, project, state, metric definitions, artifacts
    files/
      config.yaml              same format as the real client
      wandb-summary.json
      wandb-metadata.json      host, GPU, git commit, command line ...
      wandb-history.jsonl      one JSON object per step (strict JSON, NaN stored as "NaN")
      wandb-events.jsonl       system metrics, sampled every 15 s
      output.log               captured stdout/stderr
      requirements.txt
      media/table|images|...   tables and media, same layout as the real client
    logs/debug.log
  artifacts/<project>/<name>/v0/...   logged artifacts (index.json holds aliases)
  sweeps/<id>/sweep.json
```

Other knobs:

| Variable | What it does |
|---|---|
| `WANDB_LOCAL_STATS_INTERVAL` | Seconds between system-metric samples (default 15). |
| `WANDB_LOCAL_DISABLE_STATS=true` | Turns system metrics off. |
| `WANDB_CONSOLE=off` | Turns console capture off. |
| `WANDB_LOCAL_ARTIFACT_MODE=copy\|hardlink\|symlink` | How artifact files are stored. Use `hardlink` to avoid duplicating large checkpoints on the same filesystem. |
| `WANDB_LOCAL_ARTIFACTS_DIR` | A shared artifact store. |
| `WANDB_SILENT=true` | Silences terminal output. |

## Differences from the hosted service

- Nothing is uploaded, and `run.url` is `None` (the same as the real client in offline mode).
  Cloud-only features are not available: teams, reports, model registry, automations, Weave and
  launch.
- `method: bayes` sweeps fall back to random search, and early termination is ignored.
- Console capture covers Python-level `stdout`/`stderr`. Output written straight to the file
  descriptors by C extensions or subprocesses is not captured.
- `sync_tensorboard` is not supported. TensorBoard logs stay in their own directory.
- The dashboard has no authentication. It binds to `127.0.0.1` by default. To reach it from
  another machine, use an SSH tunnel: `ssh -L 8080:localhost:8080 host`.

## Development

```bash
pip install -e ".[all]" pytest
pytest tests        # the transformers test runs when torch and transformers are installed
```
