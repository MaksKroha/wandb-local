"""Read ``run-<id>.wandb`` transaction logs written by the real wandb client.

The file is a LevelDB-style log (32 KiB blocks, 7-byte record headers) of
protobuf ``wandb_internal.Record`` messages.  A tiny protobuf wire-format
decoder is used so neither ``protobuf`` nor the real ``wandb`` package is needed.
The result is converted to the wandb-local layout (``wandb-history.jsonl``,
``wandb-events.jsonl``, ``wandb-run.json``) so the dashboard and ``wandb.Api``
can show runs that were recorded offline with the real client.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any, Iterator

from ._version import FORMAT
from .util import atomic_write, dumps, utc_iso

_BLOCK = 32768
_HEADER = 7
_FULL, _FIRST, _MIDDLE, _LAST = 1, 2, 3, 4


class LegacyFormatError(Exception):
    pass


def iter_raw_records(path: str | Path, verify_crc: bool = False) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < _HEADER or data[:4] != b":W&B":
        raise LegacyFormatError(f"{path} is not a wandb transaction log")
    ident, magic, version = struct.unpack("<4sHB", data[:_HEADER])
    if magic != 0xBEE1 or version != 0:
        raise LegacyFormatError(f"{path}: unsupported wandb log header")
    pos = _HEADER
    size = len(data)
    pending = b""
    while pos + _HEADER <= size:
        left = _BLOCK - (pos % _BLOCK)
        if left < _HEADER:
            pos += left
            continue
        checksum, length, rtype = struct.unpack_from("<IHB", data, pos)
        pos += _HEADER
        if rtype == 0 and length == 0:  # zero padding / preallocated tail
            break
        chunk = data[pos:pos + length]
        if len(chunk) < length:
            break  # truncated record (run still being written or crashed)
        pos += length
        if verify_crc and zlib.crc32(chunk, zlib.crc32(bytes([rtype]))) & 0xFFFFFFFF != checksum:
            raise LegacyFormatError("record checksum mismatch")
        if rtype == _FULL:
            yield chunk
        elif rtype == _FIRST:
            pending = chunk
        elif rtype == _MIDDLE:
            pending += chunk
        elif rtype == _LAST:
            yield pending + chunk
            pending = b""
        else:
            break


# ---------------------------------------------------------------- protobuf --
def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def fields(buf: bytes) -> Iterator[tuple[int, int, Any]]:
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            val = buf[pos:pos + length]
            pos += length
        elif wire == 1:
            val = buf[pos:pos + 8]
            pos += 8
        elif wire == 5:
            val = buf[pos:pos + 4]
            pos += 4
        else:
            raise LegacyFormatError(f"unsupported wire type {wire}")
        yield num, wire, val


def _s(b: bytes) -> str:
    return b.decode("utf-8", "replace")


def _signed(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def _item(item: bytes) -> tuple[list[str], Any]:
    """Decode one {key=1, nested_key=2, value_json=16} item."""
    key, nested, value = None, [], None
    for n, _w, v in fields(item):
        if n == 1:
            key = _s(v)
        elif n == 2:
            nested.append(_s(v))
        elif n == 16:
            try:
                value = json.loads(_s(v))
            except ValueError:
                value = _s(v)
    return (nested if nested else [key or ""]), value


def _set_path(target: dict, path: list[str], value: Any) -> None:
    node = target
    for p in path[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = node[p] = {}
        node = nxt
    node[path[-1]] = value


def _timestamp(buf: bytes) -> float:
    seconds = nanos = 0
    for n, _w, v in fields(buf):
        if n == 1:
            seconds = v
        elif n == 2:
            nanos = v
    return seconds + nanos / 1e9


def read_legacy_run(path: str | Path) -> dict:
    """Decode the parts of a .wandb file that the dashboard needs."""
    history: list[dict] = []
    events: list[dict] = []
    summary: dict = {}
    config: dict = {}
    metrics: dict[str, dict] = {}
    alerts: list[dict] = []
    run: dict = {}
    exit_info: dict = {}
    for raw in iter_raw_records(path):
        for num, _w, val in fields(raw):
            if num == 2:  # history
                row: dict = {}
                step = None
                for n, _w2, v in fields(val):
                    if n == 1:
                        _set_path(row, *_item(v))
                    elif n == 2:
                        for n3, _w3, v3 in fields(v):
                            if n3 == 1:
                                step = v3
                if step is not None and "_step" not in row:
                    row["_step"] = step
                history.append(row)
            elif num == 3:  # summary
                for n, _w2, v in fields(val):
                    if n == 1:
                        _set_path(summary, *_item(v))
            elif num == 5:  # config
                _apply_config(config, val)
            elif num == 7:  # stats
                row = {}
                for n, _w2, v in fields(val):
                    if n == 2:
                        row["_timestamp"] = _timestamp(v)
                    elif n == 3:
                        key = value = None
                        for n3, _w3, v3 in fields(v):
                            if n3 == 1:
                                key = _s(v3)
                            elif n3 == 16:
                                try:
                                    value = json.loads(_s(v3))
                                except ValueError:
                                    value = None
                        if key is not None:
                            row[key] = value
                events.append(row)
            elif num == 10:  # alert
                alert = {}
                for n, _w2, v in fields(val):
                    if n in (1, 2, 3):
                        alert[{1: "title", 2: "text", 3: "level"}[n]] = _s(v)
                alerts.append(alert)
            elif num == 12:  # metric
                spec: dict = {}
                for n, _w2, v in fields(val):
                    if n == 1:
                        spec["name"] = _s(v)
                    elif n == 2:
                        spec["name"] = _s(v)
                        spec["glob"] = True
                    elif n == 4:
                        spec["step_metric"] = _s(v)
                    elif n == 6:
                        for n3, _w3, v3 in fields(v):
                            if n3 == 1:
                                spec["step_sync"] = bool(v3)
                            elif n3 == 2:
                                spec["hidden"] = bool(v3)
                    elif n == 8:
                        spec["goal"] = {1: "minimize", 2: "maximize"}.get(v)
                if spec.get("name"):
                    metrics.setdefault(spec["name"], {}).update(spec)
            elif num == 17:  # run
                for n, _w2, v in fields(val):
                    if n == 1:
                        run["id"] = _s(v)
                    elif n == 2:
                        run["entity"] = _s(v)
                    elif n == 3:
                        run["project"] = _s(v)
                    elif n == 4:
                        _apply_config(config, v)
                    elif n == 6:
                        run["group"] = _s(v)
                    elif n == 7:
                        run["job_type"] = _s(v)
                    elif n == 8:
                        run["name"] = _s(v)
                    elif n == 9:
                        run["notes"] = _s(v)
                    elif n == 10:
                        run.setdefault("tags", []).append(_s(v))
                    elif n == 12:
                        run["sweep_id"] = _s(v)
                    elif n == 13:
                        run["host"] = _s(v)
                    elif n == 17:
                        run["start_time"] = _timestamp(v)
                    elif n == 18:
                        run["resumed"] = bool(v)
            elif num == 18:  # exit
                for n, _w2, v in fields(val):
                    if n == 1:
                        exit_info["exit_code"] = _signed(v)
                    elif n == 2:
                        exit_info["runtime"] = v
    return {"history": history, "events": events, "summary": summary, "config": config, "metrics": list(metrics.values()), "alerts": alerts, "run": run, "exit": exit_info}


def _apply_config(config: dict, buf: bytes) -> None:
    for n, _w, v in fields(buf):
        if n == 1:
            _set_path(config, *_item(v))
        elif n == 2:
            config.pop(_item(v)[0][0], None)


def find_legacy_file(run_dir: Path) -> Path | None:
    for p in run_dir.glob("*.wandb"):
        return p
    return None


def convert_run_dir(run_dir: str | Path, force: bool = False) -> dict:
    """Write wandb-local files for a run directory recorded by the real client."""
    run_dir = Path(run_dir)
    files = run_dir / "files"
    marker = run_dir / "wandb-run.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())
    src = find_legacy_file(run_dir)
    if src is None:
        raise LegacyFormatError(f"No .wandb file in {run_dir}")
    data = read_legacy_run(src)
    files.mkdir(exist_ok=True)
    start = data["run"].get("start_time")
    with open(files / "wandb-history.jsonl.tmp", "w", encoding="utf-8") as fh:
        for row in data["history"]:
            fh.write(dumps(row) + "\n")
    (files / "wandb-history.jsonl.tmp").replace(files / "wandb-history.jsonl")
    with open(files / "wandb-events.jsonl.tmp", "w", encoding="utf-8") as fh:
        for row in data["events"]:
            if start and "_timestamp" in row:
                row["_runtime"] = row["_timestamp"] - start
            fh.write(dumps(row) + "\n")
    (files / "wandb-events.jsonl.tmp").replace(files / "wandb-events.jsonl")
    if not (files / "wandb-summary.json").exists() and data["summary"]:
        atomic_write(files / "wandb-summary.json", dumps(data["summary"]))
    if not (files / "config.yaml").exists() and data["config"]:
        atomic_write(files / "config.yaml", dumps(data["config"], indent=2))
    exit_info = data["exit"]
    runtime = exit_info.get("runtime") or (data["history"][-1].get("_runtime") if data["history"] else None)
    if exit_info:
        state = "finished" if exit_info.get("exit_code", 0) == 0 else "failed"
    else:
        state = "crashed"
    info = {
        "format": FORMAT,
        "imported_from": src.name,
        "id": data["run"].get("id") or run_dir.name.rsplit("-", 1)[-1],
        "name": data["run"].get("name"),
        "project": data["run"].get("project") or "uncategorized",
        "entity": data["run"].get("entity"),
        "group": data["run"].get("group"),
        "job_type": data["run"].get("job_type"),
        "tags": data["run"].get("tags", []),
        "notes": data["run"].get("notes"),
        "sweep_id": data["run"].get("sweep_id"),
        "host": data["run"].get("host"),
        "state": state,
        "exit_code": exit_info.get("exit_code"),
        "created_at": utc_iso(start) if start else None,
        "finished_at": utc_iso(start + runtime) if start and runtime else None,
        "runtime": runtime,
        "step": (data["history"][-1].get("_step", -1) + 1) if data["history"] else 0,
        "metrics": data["metrics"],
        "artifacts": [],
        "used_artifacts": [],
    }
    if data["alerts"]:
        with open(files / "wandb-alerts.jsonl", "w", encoding="utf-8") as fh:
            for alert in data["alerts"]:
                fh.write(dumps(alert) + "\n")
    atomic_write(marker, json.dumps(info, indent=2, default=str))
    return info
