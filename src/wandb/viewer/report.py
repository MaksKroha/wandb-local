"""Self-contained HTML reports: the dashboard with run data embedded (no server needed)."""

from __future__ import annotations

import base64
import json
import mimetypes
from typing import Any, Iterator

from ..artifacts import list_artifacts
from ..util import finite, utc_iso
from .server import app_html
from .store import RunRecord, RunStore

_MEDIA_TYPES = {"image-file", "html-file", "audio-file", "video-file"}


def _refs(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        if value.get("_type") in _MEDIA_TYPES | {"table-file"} and "path" in value:
            yield value
        elif value.get("_type") == "images/separated":
            for p in value.get("filenames") or []:
                yield {"_type": "image-file", "path": p}
        elif value.get("_type") == "plot" and isinstance(value.get("table"), dict):
            yield value["table"]
        else:
            for v in value.values():
                yield from _refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from _refs(v)


def _run_payload(rec: RunRecord, tables: str, budget: list[int], max_rows: int) -> dict:
    info = rec.info(convert=True)
    history = list(rec.iter_rows("history"))
    events = list(rec.iter_rows("events"))
    files: dict[str, Any] = {}
    latest_table: dict[str, str] = {}
    media: list[str] = []
    for row in history:
        for key, value in row.items():
            for ref in _refs(value):
                if ref["_type"] == "table-file":
                    if tables == "all":
                        media.append(ref["path"])
                    latest_table[key] = ref["path"]
                else:
                    media.append(ref["path"])
    for ref in _refs(rec.summary()):
        if ref["_type"] == "table-file":
            latest_table.setdefault(ref["path"], ref["path"])
        else:
            media.append(ref["path"])
    table_paths = list(latest_table.values()) if tables != "none" else []
    if tables == "all":
        table_paths = list(dict.fromkeys([p for p in media if p.endswith(".table.json")] + table_paths))
    for rel in table_paths:
        path = rec.file_path(rel)
        if path is None:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and len(data.get("data") or []) > max_rows:
            data["truncated_from"] = len(data["data"])
            data["data"] = data["data"][:max_rows]
        files[rel] = data
        for ref in _refs(data.get("data") if isinstance(data, dict) else None):
            media.append(ref["path"])
    for rel in dict.fromkeys(m for m in media if not m.endswith(".table.json")):
        path = rec.file_path(rel)
        if path is None:
            continue
        size = path.stat().st_size
        if size > budget[0]:
            continue
        budget[0] -= size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        files[rel] = f"data:{ctype};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "info": {"info": info, "config": rec.config(), "summary": rec.summary(), "metadata": rec.metadata()},
        "history": history,
        "events": events,
        "output": rec.output(tail_bytes=300_000),
        "files": files,
    }


def build_report(store: RunStore, records: list[RunRecord], tables: str = "latest", max_media_mb: float = 50, max_table_rows: int = 2000) -> str:
    budget = [int(max_media_mb * 1024 * 1024)]
    data = {rec.key: _run_payload(rec, tables, budget, max_table_rows) for rec in records}
    embed = {
        "static": True,
        "generated_at": utc_iso(),
        "runs": [d["info"]["info"] for d in data.values()],
        "data": data,
        "artifacts": list_artifacts(store.artifact_roots()),
    }
    payload = json.dumps(finite(embed), ensure_ascii=False, default=str).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = app_html()
    tag = f'<script id="wandb-embed" type="application/json">{payload}</script>\n'
    marker = "<!--WANDB_EMBED-->"
    return html.replace(marker, tag, 1) if marker in html else html.replace("</head>", tag + "</head>", 1)
