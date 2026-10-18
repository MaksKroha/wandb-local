"""``wandb.Api`` over local run directories (read-only analysis API).

    api = wandb.Api()                       # runs below $WANDB_DIR or the cwd
    run = api.run("my-project/abc123")      # or just "abc123"
    df = run.history()                      # pandas DataFrame (if pandas is installed)
    for row in run.scan_history(keys=["train/loss"]): ...
"""

from __future__ import annotations

import re
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

from .. import env as _env
from ..artifacts import find_artifact, list_artifacts, load_artifact
from ..errors import CommError
from ..util import parse_iso
from ..viewer.store import RunRecord, RunStore


class File:
    def __init__(self, run: "Run", path: str, size: int) -> None:
        self._run, self.name, self.size = run, path, size

    @property
    def url(self) -> str:
        return "file://" + str(self._run._rec.files / self.name)

    def download(self, root: str = ".", replace: bool = False, exist_ok: bool = False):
        dst = Path(root) / self.name
        if dst.exists() and not (replace or exist_ok):
            raise ValueError(f"File already exists, pass replace=True to overwrite: {dst}")
        if not dst.exists() or replace:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._run._rec.files / self.name, dst)
        return open(dst)

    def __repr__(self) -> str:
        return f"<File {self.name} ({self.size} bytes)>"


class Run:
    def __init__(self, record: RunRecord) -> None:
        self._rec = record
        self._info = record.info(convert=True)
        self.id = self._info.get("id")
        self.name = self._info.get("name")
        self.display_name = self.name
        self.project = self._info.get("project")
        self.entity = self._info.get("entity")
        self.group = self._info.get("group")
        self.job_type = self._info.get("job_type")
        self.tags = list(self._info.get("tags") or [])
        self.notes = self._info.get("notes")
        self.state = self._info.get("state")
        self.sweep_name = self._info.get("sweep_id")
        self.created_at = self._info.get("created_at")
        self.heartbeat_at = self._info.get("heartbeat_at")
        self.dir = str(record.path)
        self.url = None
        self.path = [self.entity, self.project, self.id]
        self.config = record.config()
        self.summary = record.summary()
        self.metadata = record.metadata()
        self.system_metrics = {}

    @property
    def lastHistoryStep(self) -> int:  # noqa: N802 - real API name
        return int(self._info.get("step") or 0) - 1

    def _rows(self, stream: str) -> Iterator[dict]:
        return self._rec.iter_rows("history" if stream in ("default", "history") else "events")

    def scan_history(self, keys: list[str] | None = None, page_size: int = 1000, min_step: int | None = None, max_step: int | None = None) -> Iterator[dict]:
        for row in self._rows("default"):
            step = row.get("_step", 0)
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step >= max_step:
                continue
            if keys:
                if not all(k in row for k in keys):
                    continue
                row = {k: row[k] for k in ["_step", *keys] if k in row}
            yield row

    def history(self, samples: int | None = 500, keys: list[str] | None = None, x_axis: str = "_step", pandas: bool = True, stream: str = "default"):
        rows = []
        for row in self._rows(stream):
            if keys:
                if not any(k in row for k in keys):
                    continue
                row = {k: row.get(k) for k in [x_axis, *keys] if k in row}
            rows.append(row)
        if samples and len(rows) > samples:
            stride = len(rows) / samples
            picked = [rows[int(i * stride)] for i in range(samples - 1)] + [rows[-1]]
            rows = picked
        if pandas:
            try:
                import pandas as pd

                return pd.DataFrame(rows)
            except ImportError:
                pass
        return rows

    def files(self, names: list[str] | None = None, per_page: int = 50) -> list[File]:
        out = [File(self, f["path"], f["size"]) for f in self._rec.list_files()]
        if names:
            out = [f for f in out if f.name in names]
        return out

    def file(self, name: str) -> File:
        path = self._rec.file_path(name)
        if path is None:
            raise ValueError(f"File {name} not found in run {self.id}")
        return File(self, name, path.stat().st_size)

    def logged_artifacts(self) -> list:
        out = []
        for a in self._info.get("artifacts") or []:
            if a.get("dir"):
                vdir = Path(a["dir"])
                try:
                    out.append(load_artifact(vdir.parent, vdir.name))
                except Exception:
                    continue
        return out

    def used_artifacts(self) -> list:
        return list(self._info.get("used_artifacts") or [])

    def output(self) -> str:
        return self._rec.output()

    def __repr__(self) -> str:
        return f"<Run {self.entity}/{self.project}/{self.id} ({self.state})>"


class Runs(list):
    pass


class Api:
    def __init__(self, overrides: dict | None = None, timeout: int | None = None, api_key: str | None = None, **kwargs: Any) -> None:
        overrides = overrides or {}
        roots = overrides.get("dirs") or overrides.get("dir") or kwargs.get("dirs") or _env.get_dir() or os.getcwd()
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        self._store = RunStore(list(roots), recursive=bool(overrides.get("recursive", True)))
        self.settings = {"entity": overrides.get("entity") or _env.get_entity(), "project": overrides.get("project") or _env.get_project(), "base_url": None}

    @property
    def default_entity(self) -> str | None:
        return self.settings["entity"]

    def _split(self, path: str | None) -> tuple[str | None, str | None]:
        if not path:
            return self.settings["entity"], self.settings["project"]
        parts = [p for p in path.split("/") if p]
        if len(parts) == 1:
            return None, parts[0]
        return parts[-2], parts[-1]

    def runs(self, path: str | None = None, filters: dict | None = None, order: str = "-created_at", per_page: int = 50, include_sweeps: bool = True, **kwargs: Any) -> Runs:
        entity, project = self._split(path)
        out = Runs()
        for rec in self._store.runs():
            run = Run(rec)
            if project and run.project != project:
                continue
            if entity and run.entity not in (entity, None):
                continue
            if filters and not _match_filters(run, filters):
                continue
            out.append(run)
        key = order.lstrip("+-")
        reverse = order.startswith("-")
        out.sort(key=lambda r: (parse_iso(r.created_at) or 0) if key == "created_at" else str(getattr(r, key, "") or r.summary.get(key.replace("summary_metrics.", ""), "")), reverse=reverse)
        return out

    def run(self, path: str) -> Run:
        rec = self._store.find(path)
        if rec is None:
            raise CommError(f"Could not find run {path!r} in {', '.join(map(str, self._store.roots))}")
        return Run(rec)

    def projects(self, entity: str | None = None, per_page: int = 200) -> list[dict]:
        names = sorted({Run(r).project for r in self._store.runs()})
        return [{"name": n, "entity": entity or self.default_entity} for n in names]

    def artifact(self, name: str, type: str | None = None):
        art = find_artifact(name, self._store.artifact_roots())
        if type is not None and art.type != type:
            raise ValueError(f"Artifact {name} has type {art.type}, not {type}")
        return art

    def artifacts(self, type_name: str | None = None, name: str | None = None, per_page: int = 50) -> list:
        out = []
        for entry in list_artifacts(self._store.artifact_roots()):
            if type_name and entry["type"] != type_name:
                continue
            if name and entry["name"] != name.split("/")[-1].split(":")[0]:
                continue
            for v in entry["versions"]:
                out.append(load_artifact(Path(v["dir"]).parent, v["version"]))
        return out

    artifact_versions = artifacts

    def artifact_exists(self, name: str, type: str | None = None) -> bool:
        try:
            self.artifact(name, type)
            return True
        except Exception:
            return False

    def sweep(self, path: str) -> dict:
        from ..util import read_json

        sweep_id = path.split("/")[-1]
        for d in self._store.wandb_dirs():
            data = read_json(d / "sweeps" / sweep_id / "sweep.json")
            if data:
                return data
        raise CommError(f"Sweep {path!r} not found")

    def flush(self) -> None:
        return None


def _match_filters(run: Run, filters: dict) -> bool:
    """Supports simple equality filters plus ``$regex`` / ``$in`` / ``$gt`` / ``$lt`` operators."""
    for key, cond in filters.items():
        if key == "$and":
            if not all(_match_filters(run, f) for f in cond):
                return False
            continue
        if key == "$or":
            if not any(_match_filters(run, f) for f in cond):
                return False
            continue
        if key.startswith("config."):
            value = run.config.get(key[7:].replace(".value", ""))
        elif key.startswith("summary_metrics."):
            value = run.summary.get(key[16:])
        elif key == "display_name":
            value = run.name
        else:
            value = getattr(run, key, None)
        if isinstance(cond, dict):
            for op, arg in cond.items():
                if op == "$regex" and not (isinstance(value, str) and re.search(arg, value)):
                    return False
                if op == "$in" and value not in arg:
                    return False
                if op == "$nin" and value in arg:
                    return False
                if op == "$ne" and value == arg:
                    return False
                if op == "$gt" and not (value is not None and value > arg):
                    return False
                if op == "$gte" and not (value is not None and value >= arg):
                    return False
                if op == "$lt" and not (value is not None and value < arg):
                    return False
                if op == "$lte" and not (value is not None and value <= arg):
                    return False
        elif isinstance(value, list):
            if cond not in value:
                return False
        elif value != cond:
            return False
    return True


PublicApi = Api
