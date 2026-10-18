"""Local hyper-parameter sweeps: ``wandb.sweep`` + ``wandb.agent`` (grid / random search).

Supports the parameter syntax of the real sweep configs (``value``, ``values`` +
``probabilities``, ``min``/``max`` and the ``distribution`` family, nested
``parameters``).  ``method: bayes`` falls back to random search.
Sweep state lives in ``<wandb dir>/sweeps/<sweep id>/``.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from . import _run
from . import env as _env
from .errors import UsageError
from .util import atomic_write, generate_id, read_json, utc_iso


def _sweeps_dir() -> Path:
    return Path(_env.get_dir() or os.getcwd()) / "wandb" / "sweeps"


def _load_config(sweep: Any) -> dict:
    if callable(sweep):
        sweep = sweep()
    if isinstance(sweep, (str, os.PathLike)) and os.path.isfile(sweep):
        return _run._load_yaml(os.fspath(sweep))
    if isinstance(sweep, dict):
        return json.loads(json.dumps(sweep, default=str))
    raise UsageError("sweep config must be a dict or a path to a YAML file")


def sweep(sweep: Any, entity: str | None = None, project: str | None = None, prior_runs: list | None = None) -> str:
    config = _load_config(sweep)
    if "parameters" not in config:
        raise UsageError("sweep config needs a 'parameters' section")
    method = config.get("method", "random")
    if method not in ("grid", "random", "bayes"):
        raise UsageError(f"Unsupported sweep method: {method}")
    sweep_id = generate_id(8)
    d = _sweeps_dir() / sweep_id
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "id": sweep_id,
        "project": project or config.get("project") or _env.get_project() or "uncategorized",
        "entity": entity or config.get("entity") or _env.get_entity(),
        "config": config,
        "created_at": utc_iso(),
        "trials": [],
    }
    atomic_write(d / "sweep.json", json.dumps(state, indent=2))
    _run.termlog(f"Created sweep with ID: {sweep_id}\nRun sweep agent with: wandb agent {sweep_id}")
    return sweep_id


def _sample(spec: Any, rng: random.Random) -> Any:
    if not isinstance(spec, dict):
        return spec
    if "parameters" in spec:
        return {k: _sample(v, rng) for k, v in spec["parameters"].items()}
    if "value" in spec:
        return spec["value"]
    dist = spec.get("distribution")
    if "values" in spec and dist in (None, "categorical"):
        values = spec["values"]
        probs = spec.get("probabilities")
        return rng.choices(values, weights=probs, k=1)[0] if probs else rng.choice(values)
    lo, hi = spec.get("min"), spec.get("max")
    q = spec.get("q", 1)
    mu, sigma = spec.get("mu", 0.0), spec.get("sigma", 1.0)
    if dist is None:
        if lo is None or hi is None:
            raise UsageError(f"Cannot sample parameter spec {spec}")
        dist = "int_uniform" if isinstance(lo, int) and isinstance(hi, int) else "uniform"
    if dist == "constant":
        return spec.get("value")
    if dist == "uniform":
        return rng.uniform(lo, hi)
    if dist == "int_uniform":
        return rng.randint(int(lo), int(hi))
    if dist == "q_uniform":
        return round(rng.uniform(lo, hi) / q) * q
    if dist == "log_uniform":  # min/max are exponents (natural log)
        return math.exp(rng.uniform(lo, hi))
    if dist == "log_uniform_values":
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    if dist == "q_log_uniform_values":
        return round(math.exp(rng.uniform(math.log(lo), math.log(hi))) / q) * q
    if dist == "q_log_uniform":
        return round(math.exp(rng.uniform(lo, hi)) / q) * q
    if dist == "inv_log_uniform":
        return math.exp(-rng.uniform(lo, hi))
    if dist == "inv_log_uniform_values":
        return 1.0 / math.exp(rng.uniform(math.log(1.0 / hi), math.log(1.0 / lo)))
    if dist == "normal":
        return rng.gauss(mu, sigma)
    if dist == "q_normal":
        return round(rng.gauss(mu, sigma) / q) * q
    if dist == "log_normal":
        return math.exp(rng.gauss(mu, sigma))
    if dist == "q_log_normal":
        return round(math.exp(rng.gauss(mu, sigma)) / q) * q
    raise UsageError(f"Unsupported distribution: {dist}")


def _grid(params: dict) -> list[dict]:
    axes: list[tuple[list[str], list]] = []

    def walk(prefix: list[str], spec_map: dict) -> None:
        for name, spec in spec_map.items():
            if isinstance(spec, dict) and "parameters" in spec:
                walk(prefix + [name], spec["parameters"])
            elif isinstance(spec, dict) and "values" in spec:
                axes.append((prefix + [name], list(spec["values"])))
            elif isinstance(spec, dict) and "value" in spec:
                axes.append((prefix + [name], [spec["value"]]))
            elif isinstance(spec, dict) and spec.get("distribution") == "int_uniform" or (
                isinstance(spec, dict) and isinstance(spec.get("min"), int) and isinstance(spec.get("max"), int) and "distribution" not in spec
            ):
                axes.append((prefix + [name], list(range(int(spec["min"]), int(spec["max"]) + 1))))
            else:
                raise UsageError(f"Grid search requires discrete parameters; {'/'.join(prefix + [name])} is continuous")

    walk([], params)
    combos = []
    for values in itertools.product(*[a[1] for a in axes]):
        out: dict = {}
        for (path, _), v in zip(axes, values):
            node = out
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = v
        combos.append(out)
    return combos


def _metric_value(run_dir: str | None, metric: str) -> Any:
    if not run_dir:
        return None
    summary = read_json(Path(run_dir) / "files" / "wandb-summary.json") or {}
    value = summary.get(metric)
    if isinstance(value, dict):
        for k in ("best", "min", "max", "last"):
            if k in value:
                return value[k]
    return value


def agent(sweep_id: str, function: Callable | None = None, entity: str | None = None, project: str | None = None, count: int | None = None, forward_signals: bool | None = None) -> None:
    sweep_id = sweep_id.split("/")[-1]
    d = _sweeps_dir() / sweep_id
    state = read_json(d / "sweep.json")
    if state is None:
        raise UsageError(f"Sweep {sweep_id!r} not found in {d.parent}")
    config = state["config"]
    method = config.get("method", "random")
    if method == "bayes":
        _run.termwarn("method 'bayes' is not available locally; using random search.", repeat=False)
    rng = random.Random(config.get("seed"))
    grid = _grid(config["parameters"]) if method == "grid" else None
    done = len(state["trials"])
    run_cap = config.get("run_cap")
    metric = (config.get("metric") or {}).get("name")
    goal = (config.get("metric") or {}).get("goal", "minimize")
    executed = 0
    while True:
        if count is not None and executed >= count:
            break
        if run_cap is not None and done >= run_cap:
            break
        if grid is not None:
            if done >= len(grid):
                _run.termlog("Grid search complete.")
                break
            params = grid[done]
        else:
            params = {k: _sample(v, rng) for k, v in config["parameters"].items()}
        run_id = generate_id()
        trial = {"run_id": run_id, "params": params, "started_at": utc_iso(), "state": "running"}
        state["trials"].append(trial)
        atomic_write(d / "sweep.json", json.dumps(state, indent=2, default=str))
        _run.termlog(f"Agent starting run {run_id} with config: " + ", ".join(f"{k}={v}" for k, v in params.items()))
        exit_code = 0
        run_dir = None
        if function is not None:
            _run._sweep_context = {"sweep_id": sweep_id, "params": params, "run_id": run_id, "project": project or state.get("project")}
            try:
                function()
            except Exception as exc:  # the real agent logs and moves on
                exit_code = 1
                _run.termerror(f"Run {run_id} errored: {exc!r}")
                import traceback

                traceback.print_exc()
            finally:
                import wandb

                if wandb.run is not None:
                    run_dir = wandb.run.run_dir
                    wandb.run.finish(exit_code=exit_code)
                _run._sweep_context = None
        else:
            exit_code, run_dir = _run_program(config, sweep_id, run_id, params, project or state.get("project"))
        dirs = _run.find_run_dirs(Path(_env.get_dir() or os.getcwd()) / "wandb", run_id)
        run_dir = run_dir or (str(dirs[-1]) if dirs else None)
        trial.update({"state": "finished" if exit_code == 0 else "failed", "finished_at": utc_iso(), "run_dir": run_dir})
        if metric:
            trial["metric"] = _metric_value(run_dir, metric)
        atomic_write(d / "sweep.json", json.dumps(state, indent=2, default=str))
        done += 1
        executed += 1
    if metric:
        scored = [t for t in state["trials"] if isinstance(t.get("metric"), (int, float))]
        if scored:
            best = (max if goal == "maximize" else min)(scored, key=lambda t: t["metric"])
            _run.termlog(f"Best run so far: {best['run_id']} {metric}={best['metric']} params={best['params']}")


def _run_program(config: dict, sweep_id: str, run_id: str, params: dict, project: str | None) -> tuple[int, str | None]:
    program = config.get("program")
    if not program:
        raise UsageError("wandb.agent() without a function requires 'program' in the sweep config")
    fd, param_path = tempfile.mkstemp(prefix="wandb-sweep-", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump({"params": params, "project": project}, fh)
    args = []
    for k, v in params.items():
        args.append(f"--{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}")
    command = config.get("command")
    if command:
        cmd: list[str] = []
        for part in command:
            if part == "${env}":
                continue
            if part in ("${interpreter}", "${python}"):
                cmd.append(sys.executable)
            elif part == "${program}":
                cmd.append(program)
            elif part == "${args}":
                cmd.extend(args)
            elif part == "${args_no_hyphens}":
                cmd.extend(a[2:] for a in args)
            else:
                cmd.extend(shlex.split(part))
    else:
        cmd = [sys.executable, program, *args]
    env = dict(os.environ)
    env.update({_env.SWEEP_ID: sweep_id, _env.SWEEP_PARAM_PATH: param_path, _env.RUN_ID: run_id})
    if project:
        env[_env.PROJECT] = project
    try:
        code = subprocess.call(cmd, env=env)
    finally:
        os.unlink(param_path)
    return code, None


def controller(sweep_id_or_config: Any = None, entity: str | None = None, project: str | None = None):
    raise UsageError("wandb.controller() (custom sweep controllers) is not available in wandb-local; use wandb.agent().")
