"""The local Run object and ``wandb.init``.

On-disk layout of one run (compatible with the real client's ``files/`` dir)::

    <dir>/wandb/
        latest-run -> run-20261018_120000-abcd1234
        run-20261018_120000-abcd1234/
            wandb-run.json            # identity, state, metric definitions, artifacts
            files/
                config.yaml           # {key: {value: ...}} like the real client
                wandb-summary.json
                wandb-metadata.json
                wandb-history.jsonl   # one JSON object per committed step
                wandb-events.jsonl    # system metrics
                output.log            # captured stdout/stderr
                requirements.txt
                media/...             # tables, images, html, ...
            logs/debug.log
"""

from __future__ import annotations

import atexit
import datetime
import fnmatch
import glob as _glob
import json
import logging
import math
import os
import shutil
import socket
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from . import env as _env
from ._config import Config
from ._settings import Settings
from ._summary import Summary
from ._version import FORMAT, LOCAL_VERSION, __version__
from .errors import Error, UsageError
from .util import atomic_write, dumps, generate_id, generate_name, is_numeric, read_json, to_jsonable, utc_iso

logger = logging.getLogger("wandb")

_active_runs: list["Run"] = []
_atexit_registered = False
_exit_code: int | None = None
_sweep_context: dict | None = None
_printed: set = set()
_lock = threading.RLock()


# ----------------------------------------------------------------- terminal --
def _is_silent() -> bool:
    return _env.truthy(os.environ.get(_env.SILENT))


def termlog(string: str = "", newline: bool = True, repeat: bool = True, prefix: bool = True) -> None:
    if _is_silent():
        return
    if not repeat:
        if string in _printed:
            return
        _printed.add(string)
    stream = sys.stderr
    if stream is None:
        return
    tty = hasattr(stream, "isatty") and stream.isatty()
    pre = ("\033[34m\033[1mwandb\033[0m: " if tty else "wandb: ") if prefix else ""
    text = "\n".join(pre + line for line in str(string).split("\n"))
    try:
        stream.write(text + ("\n" if newline else ""))
        stream.flush()
    except Exception:
        pass


def termwarn(string: str, repeat: bool = True, **kwargs: Any) -> None:
    termlog("WARNING " + str(string), repeat=repeat, **{k: v for k, v in kwargs.items() if k in ("newline", "prefix")})


def termerror(string: str, repeat: bool = True, **kwargs: Any) -> None:
    termlog("ERROR " + str(string), repeat=repeat, **{k: v for k, v in kwargs.items() if k in ("newline", "prefix")})


def termsetup(settings: Any = None, logger: Any = None) -> None:
    return None


class _PreInitObject:
    """Placeholder for ``wandb.config`` / ``wandb.summary`` before ``wandb.init()``."""

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("__"):
            raise AttributeError(key)
        raise Error(f"You must call wandb.init() before {self._name}.{key}")

    def __setattr__(self, key: str, value: Any) -> None:
        raise Error(f"You must call wandb.init() before {self._name}.{key}")

    def __getitem__(self, key: str) -> Any:
        raise Error(f"You must call wandb.init() before {self._name}[{key!r}]")

    __setitem__ = __getitem__

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<PreInitObject {self._name}>"


def _set_globals(run: "Run | None") -> None:
    import wandb

    if run is None:
        wandb.run = None
        wandb.config = _PreInitObject("wandb.config")
        wandb.summary = _PreInitObject("wandb.summary")
    else:
        wandb.run = run
        wandb.config = run.config
        wandb.summary = run.summary


# ------------------------------------------------------------------ helpers --
class _Spark:
    """Keeps a bounded, evenly decimated sample of a series for the end-of-run sparkline."""

    def __init__(self, cap: int = 256) -> None:
        self.values: list[float] = []
        self.stride = 1
        self.count = 0
        self.cap = cap

    def add(self, v: float) -> None:
        if self.count % self.stride == 0:
            self.values.append(v)
            if len(self.values) > self.cap:
                self.values = self.values[::2]
                self.stride *= 2
        self.count += 1

    def render(self, width: int = 20) -> str:
        vals = [v for v in self.values if math.isfinite(v)]
        if not vals:
            return ""
        if len(vals) > width:
            step = len(vals) / width
            vals = [vals[int(i * step)] for i in range(width)]
        lo, hi = min(vals), max(vals)
        blocks = "▁▂▃▄▅▆▇█"
        if hi == lo:
            return blocks[0] * len(vals)
        return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.4e}"
        return f"{v:.5g}"
    if isinstance(v, dict) and "_type" in v:
        t = v["_type"]
        if t == "table-file":
            return f"table {v.get('nrows', '?')}x{v.get('ncols', '?')}"
        return t
    return str(v)


class Metric:
    def __init__(self, run: "Run", spec: dict) -> None:
        self._run = run
        self._spec = spec

    name = property(lambda self: self._spec["name"])
    step_metric = property(lambda self: self._spec.get("step_metric"))
    step_sync = property(lambda self: self._spec.get("step_sync"))
    hidden = property(lambda self: self._spec.get("hidden"))
    summary = property(lambda self: tuple(self._spec.get("summary") or ()))
    goal = property(lambda self: self._spec.get("goal"))

    def __repr__(self) -> str:
        return f"<Metric {self._spec}>"


_SUMMARY_OPTS = {"min", "max", "mean", "best", "last", "none", "copy", "first"}


# --------------------------------------------------------------------- Run --
class Run:
    """A single tracked run, written to the local filesystem."""

    def __init__(self, settings: Settings, disabled: bool = False) -> None:
        self._settings = settings
        self._disabled = disabled
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self.config = Config()
        self.summary = Summary(serializer=self._summary_serialize, callback=self._mark_summary_dirty)
        self._id = generate_id()
        self._name: str | None = None
        self._project = "uncategorized"
        self._entity = "local"
        self._group: str | None = None
        self._job_type: str | None = None
        self._tags: tuple = ()
        self._notes: str | None = None
        self._sweep_id: str | None = None
        self._wandb_dir: Path | None = None
        self._run_dir: Path | None = None
        self.files_dir: Path | None = None
        self._start_time = time.time()
        self._created_at = self._start_time
        self._runtime_offset = 0.0
        self._step = 0
        self._pending: dict = {}
        self._extra: dict = {}
        self._last_values: dict = {}
        self._metrics: dict[str, dict] = {}
        self._metric_cache: dict[str, dict | None] = {}
        self._summary_agg: dict[str, dict] = {}
        self._sparks: dict[str, _Spark] = {}
        self._history_fh = None
        self._events_fh = None
        self._config_dirty = False
        self._summary_dirty = False
        self._info_dirty = False
        self._last_info_write = 0.0
        self._finished = False
        self._exit_code: int | None = None
        self._state = "running"
        self._console = None
        self._bg: _Background | None = None
        self._watch_hooks: dict[int, list] = {}
        self._save_globs: list[tuple[str, str, str]] = []
        self._labels: dict = {}
        self._artifacts_logged: list[dict] = []
        self._artifacts_used: list[dict] = []
        self._resumed = False
        self._forked_from: str | None = None
        self._preempting = False
        self._log_handler: logging.Handler | None = None
        self._quiet = False
        self.config._set_callback(self._mark_config_dirty)

    # -- identity ------------------------------------------------------------
    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)
        self._info_dirty = True

    display_name = name

    @property
    def project(self) -> str:
        return self._project

    def project_name(self) -> str:
        return self._project

    @property
    def entity(self) -> str:
        return self._entity

    @property
    def group(self) -> str | None:
        return self._group

    @property
    def job_type(self) -> str | None:
        return self._job_type

    @property
    def tags(self) -> tuple:
        return self._tags

    @tags.setter
    def tags(self, value: Sequence[str]) -> None:
        self._tags = tuple(value)
        self._info_dirty = True

    @property
    def notes(self) -> str | None:
        return self._notes

    @notes.setter
    def notes(self, value: str) -> None:
        self._notes = value
        self._info_dirty = True

    @property
    def sweep_id(self) -> str | None:
        return self._sweep_id

    @property
    def dir(self) -> str:
        return str(self.files_dir) if self.files_dir is not None else ""

    @property
    def path(self) -> str:
        return f"{self._entity}/{self._project}/{self._id}"

    @property
    def url(self) -> None:
        """No web UI URL exists for local runs (same as the real client in offline mode)."""
        return None

    def get_url(self) -> None:
        return None

    project_url = property(lambda self: None)
    sweep_url = property(lambda self: None)

    def get_project_url(self) -> None:
        return None

    def get_sweep_url(self) -> None:
        return None

    @property
    def start_time(self) -> float:
        return self._created_at

    @property
    def starting_step(self) -> int:
        return self._step

    @property
    def step(self) -> int:
        return self._step

    @property
    def resumed(self) -> bool:
        return self._resumed

    @property
    def offline(self) -> bool:
        return True

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def run_dir(self) -> str:
        return str(self._run_dir) if self._run_dir else ""

    def __repr__(self) -> str:
        return f"<Run {self.path} ({self._state}) dir={self.run_dir!r}>"

    # -- dirty flags ---------------------------------------------------------
    def _mark_config_dirty(self) -> None:
        self._config_dirty = True

    def _mark_summary_dirty(self) -> None:
        self._summary_dirty = True

    def _runtime(self, now: float | None = None) -> float:
        return self._runtime_offset + ((now or time.time()) - self._start_time)

    # -- persistence ---------------------------------------------------------
    def _info(self) -> dict:
        now = time.time()
        return {
            "format": FORMAT,
            "wandb_local_version": LOCAL_VERSION,
            "id": self._id,
            "name": self._name,
            "project": self._project,
            "entity": self._entity,
            "group": self._group,
            "job_type": self._job_type,
            "tags": list(self._tags),
            "notes": self._notes,
            "sweep_id": self._sweep_id,
            "state": self._state,
            "exit_code": self._exit_code,
            "created_at": utc_iso(self._created_at),
            "started_at": utc_iso(self._start_time),
            "heartbeat_at": utc_iso(now),
            "finished_at": utc_iso(now) if self._finished else None,
            "runtime": self._runtime(now),
            "step": self._step,
            "pid": self._pid,
            "host": socket.gethostname(),
            "resumed": self._resumed,
            "forked_from": self._forked_from,
            "preempting": self._preempting,
            "metrics": list(self._metrics.values()),
            "labels": self._labels,
            "artifacts": self._artifacts_logged,
            "used_artifacts": self._artifacts_used,
        }

    def _write_info(self) -> None:
        if self._run_dir is None:
            return
        atomic_write(self._run_dir / "wandb-run.json", json.dumps(self._info(), indent=2, default=str))
        self._info_dirty = False
        self._last_info_write = time.time()

    def _config_payload(self) -> dict:
        data: dict = {k: {"value": v} for k, v in self.config._items.items() if k != "_wandb"}
        data["_wandb"] = {
            "value": {
                "cli_version": __version__,
                "python_version": sys.version.split()[0],
                "start_time": self._created_at,
                "local": True,
                "labels": self._labels,
            }
        }
        return data

    def _write_config(self) -> None:
        if self.files_dir is None:
            return
        payload = self._config_payload()
        try:
            import yaml

            text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
        except Exception:  # pyyaml missing or exotic value: JSON is valid YAML
            text = dumps(payload, indent=2)
        atomic_write(self.files_dir / "config.yaml", text)
        self._config_dirty = False

    def _write_summary(self) -> None:
        if self.files_dir is None:
            return
        data = dict(self.summary._d)
        data["_wandb"] = {"runtime": int(self._runtime())}
        atomic_write(self.files_dir / "wandb-summary.json", dumps(data))
        self._summary_dirty = False

    def _flush(self) -> None:
        if self._disabled or self.files_dir is None:
            return
        with self._lock:
            try:
                if self._config_dirty:
                    self._write_config()
                if self._summary_dirty:
                    self._write_summary()
                if self._info_dirty or time.time() - self._last_info_write > 15:
                    self._write_info()
                for fh in (self._history_fh, self._events_fh):
                    if fh is not None and not fh.closed:
                        fh.flush()
                if self._console is not None:
                    self._console.flush()
            except Exception:
                logger.debug("flush failed", exc_info=True)

    def _write_event(self, metrics: dict) -> None:
        if not metrics:
            return
        with self._lock:
            if self._events_fh is None or self._events_fh.closed:
                return
            now = time.time()
            row = dict(metrics)
            row["_timestamp"] = now
            row["_runtime"] = self._runtime(now)
            self._events_fh.write(dumps(row) + "\n")

    # -- serialisation ---------------------------------------------------------
    def _serialize(self, key: str, value: Any, step: Any) -> Any:
        from .data_types import Histogram, Image, Table, WBValue

        if isinstance(value, WBValue):
            return value.to_json(self, key, step)
        if isinstance(value, Mapping):
            return {str(k): self._serialize(f"{key}.{k}", v, step) for k, v in value.items()}
        if isinstance(value, (list, tuple)) and value:
            if all(isinstance(v, Image) for v in value):
                refs = [v.to_json(self, key, f"{step}_{i}") for i, v in enumerate(value)]
                out = {"_type": "images/separated", "count": len(refs), "filenames": [r["path"] for r in refs], "format": refs[0].get("format", "png")}
                captions = [r.get("caption") for r in refs]
                out["captions"] = captions if any(c is not None for c in captions) else False
                if refs[0].get("width") is not None:
                    out["width"], out["height"] = refs[0]["width"], refs[0]["height"]
                return out
            if any(isinstance(v, WBValue) for v in value):
                return [self._serialize(f"{key}_{i}", v, step) for i, v in enumerate(value)]
        mod = type(value).__module__
        if mod.startswith("pandas") and type(value).__name__ == "DataFrame":
            return Table(dataframe=value).to_json(self, key, step)
        if mod.startswith(("numpy", "torch", "jax", "tensorflow")):
            try:
                size = value.numel() if hasattr(value, "numel") else getattr(value, "size", 1)
            except Exception:
                size = 1
            if size == 1 or not hasattr(value, "shape") or getattr(value, "shape", ()) == ():
                return to_jsonable(value)
            return Histogram(value).to_json(self, key, step)
        if mod.startswith("matplotlib") and hasattr(value, "savefig"):
            return Image(value).to_json(self, key, step)
        return to_jsonable(value)

    def _summary_serialize(self, key: str, value: Any) -> Any:
        if self._disabled or self.files_dir is None:
            return to_jsonable(value)
        return self._serialize(key, value, None)

    # -- metrics ------------------------------------------------------------------
    def _metric_for(self, key: str) -> dict | None:
        if key in self._metric_cache:
            return self._metric_cache[key]
        found = self._metrics.get(key)
        if (found is None or not found.get("step_metric")) and not key.startswith("_"):
            glob_spec, best_len = None, -1
            for name, spec in self._metrics.items():
                if "*" in name and fnmatch.fnmatchcase(key, name):
                    plen = len(name.split("*", 1)[0])
                    if plen > best_len:
                        glob_spec, best_len = spec, plen
            if found is None:
                found = glob_spec
            elif glob_spec is not None and glob_spec.get("step_metric") and glob_spec["step_metric"] != key:
                # An explicit definition without an x-axis inherits the glob's step metric.
                found = {**found, "step_metric": glob_spec["step_metric"], "step_sync": glob_spec.get("step_sync")}
        self._metric_cache[key] = found
        return found

    def define_metric(
        self,
        name: str,
        step_metric: Any = None,
        step_sync: bool | None = None,
        hidden: bool | None = None,
        summary: str | None = None,
        goal: str | None = None,
        overwrite: bool | None = None,
        **kwargs: Any,
    ) -> Metric:
        if not isinstance(name, str) or not name:
            raise UsageError("define_metric() requires a non-empty string name")
        if isinstance(step_metric, Metric):
            step_metric = step_metric.name
        summary_list = None
        if summary:
            summary_list = [s.strip().lower() for s in summary.split(",") if s.strip()]
            bad = [s for s in summary_list if s not in _SUMMARY_OPTS]
            if bad:
                raise UsageError(f"Unhandled define_metric() arg: summary op: {bad[0]}")
        if goal is not None:
            goal = {"min": "minimize", "max": "maximize"}.get(goal.lower(), goal.lower())
            if goal not in ("minimize", "maximize"):
                raise UsageError("goal must be 'minimize' or 'maximize'")
        spec = {"name": name, "step_metric": step_metric, "step_sync": (True if step_sync is None else bool(step_sync)) if step_metric else None, "hidden": hidden, "summary": summary_list, "goal": goal}
        with self._lock:
            existing = self._metrics.get(name)
            if existing and not overwrite:
                existing.update({k: v for k, v in spec.items() if v is not None})
                spec = existing
            self._metrics[name] = spec
            self._metric_cache.clear()
            self._info_dirty = True
        return Metric(self, spec)

    # -- logging --------------------------------------------------------------------
    def log(self, data: Mapping, step: int | None = None, commit: bool | None = None, sync: bool | None = None) -> None:
        if not isinstance(data, Mapping):
            raise ValueError("wandb.log must be passed a dictionary")
        if any(not isinstance(k, str) for k in data):
            raise ValueError("Key values passed to `wandb.log` must be strings.")
        if self._disabled:
            return
        if os.getpid() != self._pid:
            return
        with self._lock:
            if self._finished:
                termwarn("Run is finished; wandb.log() call ignored.", repeat=False)
                return
            if step is not None:
                step = int(step)
                if step < self._step:
                    termwarn(
                        f"Tried to log to step {step} that is less than the current step {self._step}. "
                        "Steps must be monotonically increasing, so this data will be ignored.",
                        repeat=False,
                    )
                    return
                if step > self._step:
                    self._commit_locked()
                    self._step = step
            for k, v in data.items():
                self._pending[k] = self._serialize(k, v, self._step)
            if commit is None:
                commit = step is None
            if commit:
                self._commit_locked()

    def _add_pending(self, key: str, value: Any) -> None:
        with self._lock:
            self._extra[key] = value

    def _commit_locked(self) -> None:
        if not self._pending:
            return
        row: dict = {}
        if self._extra:
            row.update(self._extra)
            self._extra = {}
        row.update(self._pending)
        self._pending = {}
        for key in list(row):
            spec = self._metric_for(key)
            if spec and spec.get("step_metric") and spec.get("step_sync"):
                sm = spec["step_metric"]
                if sm not in row and sm in self._last_values:
                    row[sm] = self._last_values[sm]
        now = time.time()
        row["_step"] = self._step
        row["_runtime"] = self._runtime(now)
        row["_timestamp"] = now
        if self._history_fh is not None and not self._history_fh.closed:
            self._history_fh.write(dumps(row) + "\n")
            self._history_fh.flush()
        for key, value in row.items():
            self._last_values[key] = value
            self._update_summary(key, value)
            if is_numeric(value) and not key.startswith("_"):
                spark = self._sparks.get(key)
                if spark is None:
                    if len(self._sparks) >= 500:
                        continue
                    spark = self._sparks[key] = _Spark()
                spark.add(float(value))
        self._step += 1
        self._summary_dirty = True

    def _update_summary(self, key: str, value: Any) -> None:
        spec = None if key.startswith("_") else self._metric_for(key)
        opts = (spec or {}).get("summary")
        if not opts or opts in (["last"], ["copy"]):
            self.summary._d[key] = value
            return
        if "none" in opts:
            return
        if not is_numeric(value) or not math.isfinite(float(value)):
            return
        v = float(value)
        agg = self._summary_agg.setdefault(key, {"min": v, "max": v, "sum": 0.0, "count": 0, "first": v})
        agg["min"], agg["max"] = min(agg["min"], v), max(agg["max"], v)
        agg["sum"] += v
        agg["count"] += 1
        out: dict = {}
        for op in opts:
            if op in ("min", "max", "first"):
                out[op] = agg[op]
            elif op == "mean":
                out["mean"] = agg["sum"] / agg["count"]
            elif op in ("last", "copy"):
                out["last"] = value
            elif op == "best":
                out["best"] = agg["max"] if spec.get("goal") == "maximize" else agg["min"]
        self.summary._d[key] = out

    # -- files ----------------------------------------------------------------------
    def save(self, glob_str: str | os.PathLike | None = None, base_path: str | None = None, policy: str = "live") -> bool | list[str]:
        if glob_str is None:
            return False
        if policy not in ("live", "end", "now"):
            raise ValueError('Only "live", "end" and "now" policies are currently supported.')
        glob_str = os.fspath(glob_str)
        if "://" in glob_str:
            termwarn(f"wandb.save({glob_str!r}): remote paths are not supported locally; ignored.")
            return []
        glob_path = os.path.abspath(glob_str)
        if base_path is None:
            parts = glob_path.split(os.sep)
            fixed = []
            for p in parts:
                if any(c in p for c in "*?["):
                    break
                fixed.append(p)
            base_path = os.sep.join(fixed[:-1]) if len(fixed) == len(parts) else os.sep.join(fixed)
        base_path = os.path.abspath(base_path or os.sep)
        self._save_globs.append((glob_path, base_path, policy))
        if policy in ("now", "live"):
            return self._copy_glob(glob_path, base_path)
        return []

    def _copy_glob(self, glob_path: str, base_path: str) -> list[str]:
        if self.files_dir is None:
            return []
        copied = []
        files_dir = str(self.files_dir.resolve())
        for path in _glob.glob(glob_path, recursive=True):
            if not os.path.isfile(path) or os.path.abspath(path).startswith(files_dir + os.sep):
                continue
            rel = os.path.relpath(path, base_path)
            if rel.startswith(".."):
                rel = os.path.basename(path)
            dst = self.files_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, dst)
                copied.append(str(dst))
            except OSError as exc:
                logger.warning("could not save %s: %s", path, exc)
        return copied

    def restore(self, name: str, run_path: str | None = None, replace: bool = False, root: str | None = None):
        return restore(name, run_path=run_path, replace=replace, root=root or self.dir)

    # -- artifacts ------------------------------------------------------------------
    def _as_artifact(self, artifact_or_path: Any, name: str | None, type: str | None):
        from .artifacts import Artifact

        if isinstance(artifact_or_path, Artifact):
            return artifact_or_path
        path = os.fspath(artifact_or_path)
        if "://" in path:
            art = Artifact(name or f"run-{self._id}-{os.path.basename(path.rstrip('/'))}", type or "unspecified")
            art.add_reference(path)
            return art
        art = Artifact(name or f"run-{self._id}-{os.path.basename(path.rstrip(os.sep))}", type or "unspecified")
        if os.path.isdir(path):
            art.add_dir(path)
        elif os.path.isfile(path):
            art.add_file(path)
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")
        return art

    def _artifacts_root(self) -> Path:
        from .artifacts import artifacts_root

        return artifacts_root(self._wandb_dir)

    def log_artifact(self, artifact_or_path: Any, name: str | None = None, type: str | None = None, aliases: list[str] | None = None, tags: list[str] | None = None, **kwargs: Any):
        from .artifacts import commit_artifact

        art = self._as_artifact(artifact_or_path, name, type)
        if self._disabled:
            return art
        if not art.is_draft():
            self.use_artifact(art)
            return art
        commit_artifact(art, self._artifacts_root(), self._project, self._entity, {"id": self._id, "name": self._name, "run_dir": self.run_dir}, aliases, tags)
        with self._lock:
            self._artifacts_logged.append({"name": art._name, "version": art.version, "type": art.type, "aliases": art.aliases, "dir": str(art._dir)})
            self._info_dirty = True
        logger.info("logged artifact %s", art.qualified_name)
        return art

    upsert_artifact = log_artifact

    def finish_artifact(self, artifact_or_path: Any, name: str | None = None, type: str | None = None, aliases: list[str] | None = None, **kwargs: Any):
        return self.log_artifact(artifact_or_path, name, type, aliases)

    def use_artifact(self, artifact_or_name: Any, type: str | None = None, aliases: list[str] | None = None, use_as: str | None = None, **kwargs: Any):
        from .artifacts import Artifact, find_artifact

        if isinstance(artifact_or_name, Artifact):
            art = artifact_or_name
            if art.is_draft():
                art = self.log_artifact(art, aliases=aliases)
        else:
            art = find_artifact(str(artifact_or_name), [self._artifacts_root()], default_project=self._project)
        if type is not None and art.type != type:
            raise ValueError(f"Supplied type {type} does not match type {art.type} of artifact {art.name}")
        if aliases and art._dir is not None:
            from .artifacts import _add_aliases

            _add_aliases(art._dir.parent, art.version or "v0", list(aliases))
        with self._lock:
            self._artifacts_used.append({"name": art._name, "version": art.version, "type": art.type})
            self._info_dirty = True
        return art

    def log_model(self, path: str, name: str | None = None, aliases: list[str] | None = None) -> None:
        art = self._as_artifact(path, name or f"run-{self._id}-model", "model")
        self.log_artifact(art, aliases=aliases)

    def use_model(self, name: str) -> str:
        art = self.use_artifact(name)
        return art.download()

    def link_model(self, path: str, registered_model_name: str, name: str | None = None, aliases: list[str] | None = None) -> None:
        art = self._as_artifact(path, name or registered_model_name, "model")
        self.log_artifact(art, aliases=aliases)

    def link_artifact(self, artifact: Any, target_path: str, aliases: list[str] | None = None) -> Any:
        if artifact.is_draft():
            self.log_artifact(artifact)
        artifact.link(target_path, aliases)
        return artifact

    def log_code(self, root: str = ".", name: str | None = None, include_fn: Any = None, exclude_fn: Any = None):
        from .artifacts import Artifact

        root = os.path.abspath(root or ".")
        skip = {".git", "wandb", "venv", ".venv", "env", "__pycache__", "node_modules", "site-packages", ".mypy_cache", ".pytest_cache"}

        def call(fn, path):
            try:
                return fn(path, root)
            except TypeError:
                return fn(path)

        include = include_fn or (lambda p, r=None: p.endswith(".py"))
        art = Artifact(name or f"source-{self._project}-{self._id}", "code")
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                if not call(include, full) or (exclude_fn and call(exclude_fn, full)):
                    continue
                rel = os.path.relpath(full, root)
                art.add_file(full, rel)
                if self.files_dir is not None:
                    dst = self.files_dir / "code" / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full, dst)
                count += 1
        if count == 0:
            termwarn("log_code(): no files matched")
            return None
        return self.log_artifact(art)

    # -- misc ---------------------------------------------------------------------------
    def alert(self, title: str, text: str, level: Any = None, wait_duration: Any = None) -> None:
        level_name = getattr(level, "value", level) or "INFO"
        termwarn(f"ALERT [{level_name}] {title}: {text}")
        if self.files_dir is None or self._disabled:
            return
        with self._lock, open(self.files_dir / "wandb-alerts.jsonl", "a", encoding="utf-8") as fh:
            fh.write(dumps({"title": title, "text": text, "level": level_name, "_timestamp": time.time(), "_step": self._step}) + "\n")

    def mark_preempting(self) -> None:
        self._preempting = True
        self._info_dirty = True

    def _label(self, code: str | None = None, repo: str | None = None, code_version: str | None = None, **kwargs: Any) -> None:
        for k, v in (("code", code), ("repo", repo), ("code_version", code_version)):
            if v is not None:
                self._labels[k] = v
        self._info_dirty = True

    def _label_probe_notebook(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _set_config_wandb(self, key: str, val: Any) -> None:
        self._labels[key] = val

    def status(self) -> dict:
        return {"state": self._state, "step": self._step, "dir": self.run_dir, "synced": True}

    def to_html(self, height: int = 420, hidden: bool = False) -> str:
        return f"<div>wandb-local run <b>{self._name}</b> ({self._id}) stored in <code>{self.run_dir}</code></div>"

    def display(self, height: int = 420, hidden: bool = False) -> bool:
        return False

    def watch(self, models: Any, criterion: Any = None, log: str | None = "gradients", log_freq: int = 1000, idx: int | None = None, log_graph: bool = False) -> None:
        if self._disabled:
            return
        from ._watch import watch as _w

        _w(self, models, criterion, log, log_freq, idx, log_graph)

    def unwatch(self, models: Any = None) -> None:
        from ._watch import unwatch as _u

        _u(self, models)

    # -- lifecycle --------------------------------------------------------------------------
    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(exit_code=0 if exc_type is None else 1)
        return False

    def finish(self, exit_code: int | None = None, quiet: bool | None = None) -> None:
        if os.getpid() != self._pid:
            return
        with self._lock:
            if self._finished:
                return
            self._finished = True
        if self._disabled:
            self._state = "finished"
            _deactivate(self)
            return
        with self._lock:
            self._commit_locked()
        if self._bg is not None:
            self._bg.stop()
        for glob_path, base_path, policy in self._save_globs:
            if policy in ("live", "end"):
                self._copy_glob(glob_path, base_path)
        self._exit_code = 0 if exit_code is None else int(exit_code)
        self._state = "finished" if self._exit_code == 0 else ("killed" if self._exit_code == 255 else "failed")
        quiet = self._quiet if quiet is None else quiet
        if not quiet:
            self._print_footer()
        if self._console is not None:
            self._console.stop()
            self._console = None
        with self._lock:
            self._write_config()
            self._write_summary()
            self._write_info()
            for fh in (self._history_fh, self._events_fh):
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
        if self._state == "finished" and self._wandb_dir is not None:
            resume_file = self._wandb_dir / "wandb-resume.json"
            if (read_json(resume_file) or {}).get("run_id") == self._id:
                try:
                    resume_file.unlink()
                except OSError:
                    pass
        self.unwatch()
        if self._log_handler is not None:
            logging.getLogger("wandb").removeHandler(self._log_handler)
            self._log_handler.close()
        _deactivate(self)

    def _print_footer(self) -> None:
        if _is_silent():
            return
        lines = []
        keys = sorted(k for k in self._sparks if not k.startswith("_"))
        hidden = {k for k in keys if (self._metric_for(k) or {}).get("hidden")}
        keys = [k for k in keys if k not in hidden]
        if keys:
            width = min(max(len(k) for k in keys[:40]), 48)
            lines.append("")
            lines.append("Run history:")
            for k in keys[:40]:
                lines.append(f"  {k[:48]:>{width}} {self._sparks[k].render()}")
        summary_keys = sorted(k for k in self.summary._d if not k.startswith("_"))
        if summary_keys:
            width = min(max(len(k) for k in summary_keys[:40]), 48)
            lines.append("")
            lines.append("Run summary:")
            for k in summary_keys[:40]:
                lines.append(f"  {k[:48]:>{width}} {_fmt_value(self.summary._d[k])}")
        if len(keys) > 40 or len(summary_keys) > 40:
            lines.append("  ... (more metrics in the local dashboard)")
        lines.append("")
        lines.append(f"Run {self._name!r} ({self._id}) {self._state}. Data saved in {self.run_dir}")
        lines.append(f"View it with: wandb ui {self._wandb_dir.parent if self._wandb_dir else '.'}")
        termlog("\n".join(lines))


class _Background(threading.Thread):
    def __init__(self, run: Run, stats_interval: float | None, flush_interval: float = 2.0) -> None:
        super().__init__(name="wandb-local-background", daemon=True)
        self.run_ref = run
        self.stats_interval = stats_interval
        self.flush_interval = flush_interval
        self._stop_event = threading.Event()

    def run(self) -> None:  # noqa: D401 - thread entry point
        sampler = None
        if self.stats_interval:
            try:
                from ._system import SystemSampler

                sampler = SystemSampler()
            except Exception:
                logger.debug("system metrics disabled", exc_info=True)
        start = time.monotonic()
        next_stats = start + min(2.0, self.stats_interval or 0)
        last_flush = start
        while not self._stop_event.wait(0.5):
            now = time.monotonic()
            if sampler is not None and now >= next_stats:
                try:
                    self.run_ref._write_event(sampler.sample())
                except Exception:
                    logger.debug("system sample failed", exc_info=True)
                next_stats = now + float(self.stats_interval)
            if now - last_flush >= self.flush_interval:
                self.run_ref._flush()
                last_flush = now

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=15)


# -------------------------------------------------------------------- init --
def _deactivate(run: Run) -> None:
    import wandb

    with _lock:
        if run in _active_runs:
            _active_runs.remove(run)
        if wandb.run is run:
            _set_globals(_active_runs[-1] if _active_runs else None)


def _atexit_handler() -> None:
    for run in list(_active_runs):
        try:
            run.finish(exit_code=_exit_code if _exit_code is not None else 0)
        except Exception:
            logger.debug("finish at exit failed", exc_info=True)


def _install_hooks() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True
    atexit.register(_atexit_handler)
    original = sys.excepthook

    def hook(exc_type, exc, tb):
        global _exit_code
        if not issubclass(exc_type, SystemExit):
            _exit_code = 255 if issubclass(exc_type, KeyboardInterrupt) else 1
        original(exc_type, exc, tb)

    sys.excepthook = hook


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ImportError:
        data = json.loads(text or "{}")
    return data if isinstance(data, dict) else {}


def _config_file_values(path: str) -> dict:
    out = {}
    for k, v in _load_yaml(path).items():
        if k == "wandb_version" or k == "_wandb":
            continue
        out[k] = v["value"] if isinstance(v, dict) and "value" in v else v
    return out


def _parse_run_ref(ref: str) -> tuple[str, int | None]:
    run_id, _, query = ref.partition("?")
    step = None
    for part in query.split("&"):
        k, _, v = part.partition("=")
        if k == "_step" and v:
            step = int(float(v))
    return run_id.split("/")[-1], step


def find_run_dirs(wandb_dir: Path, run_id: str) -> list[Path]:
    if not wandb_dir.is_dir():
        return []
    found = [p for p in wandb_dir.iterdir() if p.is_dir() and (p.name.startswith(("run-", "offline-run-"))) and p.name.endswith("-" + run_id)]
    return sorted(found)


def _read_last_history_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        block = min(size, 1 << 20)
        while True:
            fh.seek(size - block)
            lines = [ln for ln in fh.read(block).splitlines() if ln.strip()]
            for line in reversed(lines):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
            if block >= size:
                return None
            block = min(size, block * 4)


def _truncate_history(path: Path, max_step: int) -> None:
    """Keep rows with _step <= max_step (the old file is kept as a backup)."""
    if not path.exists():
        return
    backup = path.with_name(f"wandb-history.{datetime.datetime.now():%Y%m%d_%H%M%S}.bak.jsonl")
    shutil.copy2(path, backup)
    kept = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                if json.loads(line).get("_step", 0) <= max_step:
                    kept.append(line if line.endswith("\n") else line + "\n")
            except ValueError:
                continue
    atomic_write(path, "".join(kept))


def _make_link(link: Path, target: str) -> None:
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    except OSError:
        pass


def init(
    entity: str | None = None,
    project: str | None = None,
    dir: str | os.PathLike | None = None,
    id: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    tags: Sequence[str] | None = None,
    config: Any = None,
    config_exclude_keys: list[str] | None = None,
    config_include_keys: list[str] | None = None,
    allow_val_change: bool | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: str | None = None,
    force: bool | None = None,
    anonymous: str | None = None,
    reinit: Any = None,
    resume: Any = None,
    resume_from: str | None = None,
    fork_from: str | None = None,
    save_code: bool | None = None,
    tensorboard: bool | None = None,
    sync_tensorboard: bool | None = None,
    monitor_gym: bool | None = None,
    settings: Any = None,
    **kwargs: Any,
) -> Run:
    """Start a new run that logs to the local filesystem. Mirrors ``wandb.init``."""
    import wandb

    s = Settings()
    if isinstance(settings, Settings):
        s.update(settings)
    elif isinstance(settings, dict):
        s.update(settings)
    reinit = reinit if reinit is not None else s.reinit

    # mode -----------------------------------------------------------------
    root = os.fspath(dir) if dir is not None else (s.root_dir or _env.get_dir() or os.getcwd())
    root = os.path.abspath(os.path.expanduser(root))
    wandb_dir = Path(root) / "wandb"
    mode = (mode or s.mode or _env.get(_env.MODE) or _read_mode_file(wandb_dir) or "offline").lower()
    disabled = mode == "disabled" or _env.truthy(os.environ.get(_env.DISABLED))

    # reinit ------------------------------------------------------------
    with _lock:
        active = [r for r in _active_runs if not r._finished]
    if active:
        if reinit in (True, "finish_previous"):
            for r in active:
                r.finish()
        elif reinit != "create_new":
            termwarn("wandb.init() called while a run is active and reinit is set to 'default', so returning the previous run.", repeat=False)
            return active[-1]

    sweep = _sweep_context or _sweep_context_from_env()
    run = Run(s, disabled=disabled)
    run._quiet = bool(s.quiet or _env.truthy(os.environ.get(_env.QUIET)))
    run._project = project or s.project or _env.get_project() or (sweep or {}).get("project") or "uncategorized"
    run._entity = entity or s.entity or _env.get_entity() or _default_entity()
    run._group = group or s.run_group or _env.get(_env.RUN_GROUP)
    run._job_type = job_type or s.run_job_type or _env.get(_env.JOB_TYPE)
    env_tags = _env.get(_env.TAGS)
    run._tags = tuple(tags if tags is not None else (env_tags.split(",") if env_tags else ()))
    run._notes = notes or s.run_notes or _env.get(_env.NOTES)
    run_id = id or s.run_id or _env.get(_env.RUN_ID) or (sweep or {}).get("run_id")
    resume = resume if resume is not None else (s.resume or _env.get(_env.RESUME))
    if sweep and resume is None:
        resume = "allow"

    if disabled:
        run._id = run_id or generate_id()
        run._name = name or s.run_name or _env.get(_env.NAME) or generate_name()
        run._state = "disabled"
        _apply_config(run, config, config_include_keys, config_exclude_keys, allow_val_change, sweep, load_defaults=False)
        with _lock:
            _active_runs.append(run)
        _set_globals(run)
        return run

    wandb_dir.mkdir(parents=True, exist_ok=True)
    existing_dir: Path | None = None
    if resume in (True, "auto") and not run_id:
        run_id = (read_json(wandb_dir / "wandb-resume.json") or {}).get("run_id")
    if resume_from:
        rid, rstep = _parse_run_ref(resume_from)
        run_id = rid
        resume = "must"
    if run_id:
        dirs = find_run_dirs(wandb_dir, run_id)
        if resume == "never" and dirs:
            raise UsageError(f"resume='never' but run {run_id!r} already exists in {wandb_dir}")
        if resume == "must" and not dirs:
            raise UsageError(f"resume='must' but run {run_id!r} does not exist in {wandb_dir}")
        if dirs and resume in (True, "auto", "allow", "must"):
            existing_dir = dirs[-1]
        elif dirs:
            termwarn(f"A local run with id {run_id!r} already exists; creating a new run directory (pass resume='allow' to continue it).", repeat=False)
    run._id = run_id or generate_id()

    if existing_dir is not None:
        run._run_dir = existing_dir
        run._resumed = True
        info = read_json(existing_dir / "wandb-run.json") or {}
        files_dir = existing_dir / "files"
        if resume_from and rstep is not None:
            _truncate_history(files_dir / "wandb-history.jsonl", rstep)
        last = _read_last_history_row(files_dir / "wandb-history.jsonl")
        run._step = int(last.get("_step", -1)) + 1 if last else 0
        run._runtime_offset = float((last or {}).get("_runtime", info.get("runtime") or 0.0))
        if info.get("created_at"):
            from .util import parse_iso

            run._created_at = parse_iso(info["created_at"]) or run._start_time
        run._name = name or info.get("name")
        run._project = project or info.get("project") or run._project
        run._group = run._group or info.get("group")
        run._job_type = run._job_type or info.get("job_type")
        run._tags = run._tags or tuple(info.get("tags") or ())
        run._notes = run._notes or info.get("notes")
        run._artifacts_logged = list(info.get("artifacts") or [])
        run._artifacts_used = list(info.get("used_artifacts") or [])
        run._labels = dict(info.get("labels") or {})
        for spec in info.get("metrics") or []:
            if spec.get("name"):
                run._metrics[spec["name"]] = spec
        if (files_dir / "config.yaml").exists():
            try:
                run.config._load(_config_file_values(str(files_dir / "config.yaml")))
            except Exception:
                logger.warning("could not reload config of resumed run", exc_info=True)
        summary = read_json(files_dir / "wandb-summary.json") or {}
        summary.pop("_wandb", None)
        run.summary._update_raw(summary)
        if last:
            run._last_values.update(last)
    else:
        stamp = datetime.datetime.fromtimestamp(run._start_time).strftime("%Y%m%d_%H%M%S")
        run._run_dir = wandb_dir / f"run-{stamp}-{run._id}"
        n_runs = sum(1 for p in wandb_dir.iterdir() if p.name.startswith(("run-", "offline-run-")))
        run._name = name or s.run_name or _env.get(_env.NAME) or generate_name(n_runs + 1)

    run._wandb_dir = wandb_dir
    run.files_dir = run._run_dir / "files"
    run.files_dir.mkdir(parents=True, exist_ok=True)
    (run._run_dir / "logs").mkdir(exist_ok=True)
    run._sweep_id = (sweep or {}).get("sweep_id")

    if fork_from and existing_dir is None:
        _fork(run, wandb_dir, fork_from)

    _apply_config(run, config, config_include_keys, config_exclude_keys, allow_val_change, sweep, load_defaults=existing_dir is None)

    # files ---------------------------------------------------------------------
    handler = logging.FileHandler(run._run_dir / "logs" / "debug.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)s:%(name)s %(message)s"))
    handler.setLevel(logging.DEBUG)
    wandb_logger = logging.getLogger("wandb")
    wandb_logger.addHandler(handler)
    if wandb_logger.level == logging.NOTSET or wandb_logger.level > logging.INFO:
        wandb_logger.setLevel(logging.INFO)
    run._log_handler = handler
    logger.info("wandb-local %s: run %s in %s (resumed=%s)", LOCAL_VERSION, run._id, run._run_dir, run._resumed)

    program = _program_path()
    try:
        from ._system import collect_metadata, requirements

        meta = collect_metadata(run._start_time, program, os.getcwd())
        atomic_write(run.files_dir / "wandb-metadata.json", json.dumps(meta, indent=2, default=str))
        if not (run.files_dir / "requirements.txt").exists() or run._resumed:
            atomic_write(run.files_dir / "requirements.txt", requirements())
    except Exception:
        logger.warning("could not collect run metadata", exc_info=True)
    save_code = save_code if save_code is not None else (s.save_code or _env.truthy(os.environ.get(_env.SAVE_CODE)))
    if save_code and program and os.path.isfile(program):
        dst = run.files_dir / "code" / os.path.basename(program)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(program, dst)

    run._history_fh = open(run.files_dir / "wandb-history.jsonl", "a", encoding="utf-8")
    run._events_fh = open(run.files_dir / "wandb-events.jsonl", "a", encoding="utf-8")
    run._write_config()
    run._write_summary()
    run._write_info()
    _make_link(wandb_dir / "latest-run", run._run_dir.name)
    _make_link(wandb_dir / "debug.log", os.path.join(run._run_dir.name, "logs", "debug.log"))
    atomic_write(wandb_dir / "wandb-resume.json", json.dumps({"run_id": run._id}))

    if tensorboard or sync_tensorboard:
        termwarn("sync_tensorboard is not supported by wandb-local; TensorBoard logs stay in their own log dir.", repeat=False)

    if not run._quiet:
        verb = "Resuming" if run._resumed else "Tracking"
        termlog(
            f"{verb} run with wandb-local {LOCAL_VERSION} (offline replacement for wandb {__version__.split('+')[0]})\n"
            f"Run data is saved locally in {run._run_dir}\n"
            f"Run {run._name!r} (id {run._id}) in project {run._project!r}. View with: wandb ui {wandb_dir.parent}"
        )

    console = (s.console or _env.get(_env.CONSOLE) or "auto").lower()
    if console not in ("off", "none", "false"):
        try:
            from ._console import ConsoleCapture

            run._console = ConsoleCapture(str(run.files_dir / "output.log"))
            run._console.start()
        except Exception:
            logger.warning("console capture disabled", exc_info=True)
            run._console = None

    stats_disabled = s.x_disable_stats or s._disable_stats or _env.truthy(os.environ.get(_env.LOCAL_DISABLE_STATS))
    interval = s.x_stats_sampling_interval or s._stats_sample_rate_seconds or _env.get(_env.LOCAL_STATS_INTERVAL) or 15
    run._bg = _Background(run, None if stats_disabled else float(interval))
    run._bg.start()

    with _lock:
        _active_runs.append(run)
    _install_hooks()
    _set_globals(run)
    return run


def _apply_config(run: Run, config: Any, include: list | None, exclude: list | None, allow_val_change: bool | None, sweep: dict | None, load_defaults: bool) -> None:
    if load_defaults and os.path.isfile("config-defaults.yaml"):
        try:
            run.config.setdefaults(_config_file_values("config-defaults.yaml"))
        except Exception:
            logger.warning("could not read config-defaults.yaml", exc_info=True)
    for path in filter(None, (_env.get(_env.CONFIG_PATHS) or "").split(",")):
        try:
            run.config.update(_config_file_values(path.strip()), allow_val_change=True)
        except Exception:
            logger.warning("could not read config file %s", path, exc_info=True)
    if config is not None:
        if isinstance(config, (str, os.PathLike)):
            values = _config_file_values(os.fspath(config))
        else:
            from ._config import _as_dict

            values = _as_dict(config)
        if include:
            values = {k: v for k, v in values.items() if k in include}
        if exclude:
            values = {k: v for k, v in values.items() if k not in exclude}
        run.config.update(values, allow_val_change=True if run._resumed else allow_val_change)
    if sweep and sweep.get("params"):
        run.config.update_locked(sweep["params"], user="sweep")


def _fork(run: Run, wandb_dir: Path, fork_from: str) -> None:
    parent_id, step = _parse_run_ref(fork_from)
    dirs = find_run_dirs(wandb_dir, parent_id)
    if not dirs:
        raise UsageError(f"fork_from: run {parent_id!r} not found in {wandb_dir}")
    parent_files = dirs[-1] / "files"
    hist = parent_files / "wandb-history.jsonl"
    rows = []
    if hist.exists():
        with open(hist, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if step is None or row.get("_step", 0) <= step:
                    rows.append(line if line.endswith("\n") else line + "\n")
    (run.files_dir / "wandb-history.jsonl").write_text("".join(rows), encoding="utf-8")
    if rows:
        last = json.loads(rows[-1])
        run._step = int(last.get("_step", -1)) + 1
        run._runtime_offset = float(last.get("_runtime", 0.0))
        run._last_values.update(last)
        for line in rows:
            for k, v in json.loads(line).items():
                run.summary._d[k] = v
    if (parent_files / "config.yaml").exists():
        run.config._load(_config_file_values(str(parent_files / "config.yaml")))
    parent_info = read_json(dirs[-1] / "wandb-run.json") or {}
    for spec in parent_info.get("metrics") or []:
        if spec.get("name"):
            run._metrics[spec["name"]] = spec
    run._forked_from = fork_from


def _read_mode_file(wandb_dir: Path) -> str | None:
    path = wandb_dir / "settings"
    try:
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "mode":
                return value.strip()
    except OSError:
        pass
    return None


def _default_entity() -> str:
    try:
        import getpass

        return getpass.getuser() or "local"
    except Exception:
        return "local"


def _program_path() -> str | None:
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None) or (sys.argv[0] if sys.argv and sys.argv[0] not in ("", "-c") else None)
    return os.path.abspath(path) if path and os.path.exists(path) else None


def _sweep_context_from_env() -> dict | None:
    sweep_id = _env.get(_env.SWEEP_ID)
    param_path = _env.get(_env.SWEEP_PARAM_PATH)
    if not sweep_id or not param_path or not os.path.isfile(param_path):
        return None
    data = read_json(param_path) or {}
    return {"sweep_id": sweep_id, "params": data.get("params", {}), "run_id": _env.get(_env.RUN_ID), "project": data.get("project")}


# ---------------------------------------------------------------- restore --
def restore(name: str, run_path: str | None = None, replace: bool = False, root: str | None = None):
    import wandb

    current = wandb.run
    wandb_dir = current._wandb_dir if current is not None and current._wandb_dir else Path(_env.get_dir() or os.getcwd()) / "wandb"
    run_id = run_path.split("/")[-1] if run_path else (current.id if current is not None else None)
    if run_id is None:
        raise UsageError("restore() needs run_path when no run is active")
    for run_dir in reversed(find_run_dirs(wandb_dir, run_id)):
        src = run_dir / "files" / name
        if src.exists():
            root = root or (current.dir if current is not None else os.getcwd())
            dst = Path(root) / name
            if os.path.abspath(dst) != os.path.abspath(src) and (replace or not dst.exists()):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return open(dst if dst.exists() else src)
    raise ValueError(f"File {name} not found in run {run_id}")
