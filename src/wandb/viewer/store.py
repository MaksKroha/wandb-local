"""Discover and read runs on disk (used by the dashboard, reports, CLI and ``wandb.Api``)."""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import zlib
from pathlib import Path
from typing import Any, Iterator

from ..util import parse_iso, pid_alive, read_json

RUN_DIR_RE = re.compile(r"^(?:offline-)?run-(\d{8}_\d{6})-([a-z0-9]+)$")
_convert_lock = threading.Lock()


def _load_yaml_values(path: Path) -> tuple[dict, dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}, {}
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except Exception:
        try:
            data = json.loads(text)
        except ValueError:
            return {}, {}
    if not isinstance(data, dict):
        return {}, {}
    values, internal = {}, {}
    for k, v in data.items():
        if k in ("_wandb", "wandb_version"):
            internal = v.get("value", v) if isinstance(v, dict) else {}
            continue
        values[k] = v["value"] if isinstance(v, dict) and "value" in v else v
    return values, internal


class RunRecord:
    """A run directory on disk, written either by wandb-local or by the real client."""

    def __init__(self, path: Path, key: str) -> None:
        self.path = Path(path)
        self.key = key
        self.files = self.path / "files"
        m = RUN_DIR_RE.match(self.path.name)
        self.dir_id = m.group(2) if m else self.path.name

    # -- legacy ------------------------------------------------------------------
    @property
    def is_legacy(self) -> bool:
        return not (self.path / "wandb-run.json").exists() and any(self.path.glob("*.wandb"))

    def ensure_converted(self) -> None:
        if not self.is_legacy:
            return
        from .._legacy import convert_run_dir

        with _convert_lock:
            if self.is_legacy:
                convert_run_dir(self.path)

    # -- metadata ----------------------------------------------------------------
    def info(self, convert: bool = False) -> dict:
        if convert:
            try:
                self.ensure_converted()
            except Exception as exc:  # pragma: no cover - corrupt files
                return {**self._fallback_info(), "error": f"could not import .wandb file: {exc}"}
        info = read_json(self.path / "wandb-run.json")
        if info is None:
            info = self._fallback_info()
        info = dict(info)
        info.setdefault("id", self.dir_id)
        info["key"] = self.key
        info["dir"] = str(self.path)
        info["legacy"] = self.is_legacy or bool(info.get("imported_from"))
        if not info.get("name"):
            info["name"] = info["id"]
        if info.get("state") == "running" and not info.get("imported_from"):
            same_host = info.get("host") in (None, socket.gethostname())
            if same_host and not pid_alive(info.get("pid")):
                info["state"] = "crashed"
        if not info.get("created_at"):
            m = RUN_DIR_RE.match(self.path.name)
            if m:
                import datetime

                info["created_at"] = datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").astimezone().isoformat()
        return info

    def _fallback_info(self) -> dict:
        meta = self.metadata()
        state = "unknown"
        if self.is_legacy:
            state = "not imported"
        return {"id": self.dir_id, "name": None, "project": "uncategorized", "state": state, "created_at": meta.get("startedAt"), "host": meta.get("host")}

    def config(self) -> dict:
        return _load_yaml_values(self.files / "config.yaml")[0]

    def summary(self) -> dict:
        data = read_json(self.files / "wandb-summary.json") or {}
        data.pop("_wandb", None)
        return data

    def metadata(self) -> dict:
        return read_json(self.files / "wandb-metadata.json") or {}

    def created_ts(self) -> float:
        info = read_json(self.path / "wandb-run.json") or {}
        ts = parse_iso(info.get("created_at"))
        if ts:
            return ts
        m = RUN_DIR_RE.match(self.path.name)
        if m:
            import datetime

            return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
        return self.path.stat().st_mtime

    # -- streams -------------------------------------------------------------------
    def history_path(self) -> Path:
        return self.files / "wandb-history.jsonl"

    def events_path(self) -> Path:
        return self.files / "wandb-events.jsonl"

    @staticmethod
    def read_lines(path: Path, offset: int = 0, max_bytes: int | None = None) -> tuple[list[str], int]:
        """Return complete JSON lines starting at byte ``offset`` and the new offset."""
        if not path.exists():
            return [], 0
        size = path.stat().st_size
        if offset > size:  # file was rewritten (rewind/import): start over
            offset = 0
        with open(path, "rb") as fh:
            fh.seek(offset)
            data = fh.read(max_bytes) if max_bytes else fh.read()
        end = data.rfind(b"\n")
        if end < 0:
            return [], offset
        chunk = data[: end + 1]
        lines = [ln.decode("utf-8", "replace") for ln in chunk.split(b"\n") if ln.strip()]
        return lines, offset + end + 1

    def iter_rows(self, stream: str = "history") -> Iterator[dict]:
        path = self.history_path() if stream == "history" else self.events_path()
        if not path.exists():
            return
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue

    def output(self, tail_bytes: int = 2_000_000) -> str:
        path = self.files / "output.log"
        if not path.exists():
            return ""
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
                data = fh.read()
                data = data[data.find(b"\n") + 1:]
                return f"[... {size - tail_bytes} earlier bytes not shown ...]\n" + data.decode("utf-8", "replace")
            return fh.read().decode("utf-8", "replace")

    def file_path(self, rel: str) -> Path | None:
        """Resolve ``rel`` inside the run's files dir (rejects path traversal)."""
        base = self.files.resolve()
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return None
        return target if target.is_file() else None

    def list_files(self) -> list[dict]:
        out = []
        if not self.files.is_dir():
            return out
        for root, _dirs, names in os.walk(self.files):
            for name in names:
                if name.startswith(".tmp-"):
                    continue
                full = Path(root) / name
                out.append({"path": str(full.relative_to(self.files)).replace(os.sep, "/"), "size": full.stat().st_size})
        return sorted(out, key=lambda f: f["path"])


def _scan_dir(directory: Path) -> list[Path]:
    try:
        return [p for p in directory.iterdir() if p.is_dir() and RUN_DIR_RE.match(p.name) and not p.is_symlink()]
    except OSError:
        return []


def wandb_dirs_for(root: Path, recursive: bool = False, max_depth: int = 4) -> list[Path]:
    """Directories that directly contain run-* folders, found from ``root``."""
    root = Path(root).expanduser().resolve()
    if RUN_DIR_RE.match(root.name):
        return [root.parent]
    found = []
    if _scan_dir(root):
        found.append(root)
    if (root / "wandb").is_dir() and _scan_dir(root / "wandb"):
        found.append(root / "wandb")
    if recursive:
        for dirpath, dirnames, _files in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "__pycache__", "site-packages") and not RUN_DIR_RE.match(d)]
            if depth >= max_depth:
                dirnames[:] = []
            p = Path(dirpath)
            if p != root and p.name == "wandb" and _scan_dir(p) and p not in found:
                found.append(p)
    return found


class RunStore:
    def __init__(self, roots: list[str | Path], recursive: bool = False) -> None:
        self.roots = [Path(r).expanduser().resolve() for r in roots] or [Path.cwd()]
        self.recursive = recursive
        self._records: dict[str, RunRecord] = {}
        self._single: list[Path] = []
        for r in self.roots:
            if RUN_DIR_RE.match(r.name):
                self._single.append(r)

    def wandb_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        for root in self.roots:
            if RUN_DIR_RE.match(root.name):
                continue
            for d in wandb_dirs_for(root, self.recursive):
                if d not in dirs:
                    dirs.append(d)
        return dirs

    def artifact_roots(self) -> list[Path]:
        from ..artifacts import artifacts_root

        roots = []
        for d in self.wandb_dirs() + [p.parent for p in self._single]:
            r = artifacts_root(d)
            if r not in roots:
                roots.append(r)
        return roots

    def runs(self) -> list[RunRecord]:
        paths: list[Path] = list(self._single)
        for d in self.wandb_dirs():
            paths.extend(_scan_dir(d))
        records = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            key = stable_key(path)
            rec = self._records.get(key)
            if rec is None:
                rec = self._records[key] = RunRecord(path, key)
            records.append(rec)
        records.sort(key=lambda r: r.created_ts(), reverse=True)
        return records

    def get(self, key: str) -> RunRecord | None:
        if key not in self._records:
            self.runs()
        return self._records.get(key)

    def find(self, ref: str) -> RunRecord | None:
        """Find a run by key, directory name, run id, ``entity/project/id`` or name."""
        ref = ref.rstrip("/")
        last = ref.split("/")[-1]
        candidate = Path(ref).expanduser()
        if candidate.is_dir() and RUN_DIR_RE.match(candidate.resolve().name):
            return RunRecord(candidate.resolve(), "path-" + candidate.resolve().name)
        runs = self.runs()
        for rec in runs:
            if rec.key == ref or rec.path.name == last or rec.dir_id == last:
                return rec
        for rec in runs:
            if (read_json(rec.path / "wandb-run.json") or {}).get("name") == ref:
                return rec
        return None


def stable_key(path: Path) -> str:
    """Deterministic (across processes) key for a run directory."""
    return f"{zlib.crc32(str(path.parent).encode()) % 10000:04d}-{path.name}"


def summarize_value(value: Any) -> Any:
    """Short display form of a summary/config value (media refs become labels)."""
    if isinstance(value, dict) and "_type" in value:
        t = value["_type"]
        if t == "table-file":
            return f"[table {value.get('nrows', '?')}×{value.get('ncols', '?')}]"
        return f"[{t}]"
    return value
