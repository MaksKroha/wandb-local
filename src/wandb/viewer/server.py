"""``wandb ui`` — a small local web server for browsing runs (stdlib only)."""

from __future__ import annotations

import gzip
import json
import mimetypes
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..artifacts import list_artifacts
from ..util import finite
from .store import RunStore

STATIC = Path(__file__).parent / "static"


def app_html() -> str:
    return (STATIC / "app.html").read_text(encoding="utf-8")


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(finite(obj), ensure_ascii=False, default=str).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    server_version = "wandb-local"
    store: RunStore

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    # -- helpers -------------------------------------------------------------
    def _send(self, body: bytes, content_type: str, status: int = 200, headers: dict | None = None) -> None:
        gz = len(body) > 32768 and "gzip" in (self.headers.get("Accept-Encoding") or "") and not content_type.startswith(("image/", "video/", "audio/"))
        if gz:
            body = gzip.compress(body, 5)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(_json_bytes(obj), "application/json; charset=utf-8", status)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    # -- routing ---------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route()
        except BrokenPipeError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            try:
                self._error(500, f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

    def _route(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(url.query)
        parts = [urllib.parse.unquote(p) for p in url.path.split("/") if p]
        store = self.server.store  # type: ignore[attr-defined]
        if not parts or parts == ["index.html"]:
            self._send(app_html().encode("utf-8"), "text/html; charset=utf-8")
            return
        if parts[0] != "api":
            self._error(404, "not found")
            return
        if parts[1:] == ["runs"]:
            runs = [rec.info() for rec in store.runs()]
            self._json({"runs": runs, "live": True, "roots": [str(r) for r in store.roots]})
            return
        if parts[1:] == ["artifacts"]:
            self._json({"artifacts": list_artifacts(store.artifact_roots())})
            return
        if parts[1:] == ["report"]:
            from .report import build_report

            keys = [k for k in (query.get("keys", [""])[0]).split(",") if k]
            records = [store.get(k) for k in keys] or store.runs()[:8]
            html = build_report(store, [r for r in records if r is not None])
            self._send(html.encode("utf-8"), "text/html; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="wandb-report.html"'})
            return
        if len(parts) >= 4 and parts[1] == "run":
            rec = store.get(parts[2])
            if rec is None:
                self._error(404, f"unknown run {parts[2]}")
                return
            what = parts[3]
            if what == "info":
                info = rec.info(convert=True)
                self._json({"info": info, "config": rec.config(), "summary": rec.summary(), "metadata": rec.metadata()})
            elif what in ("history", "events"):
                rec.ensure_converted()
                offset = int(query.get("offset", ["0"])[0] or 0)
                path = rec.history_path() if what == "history" else rec.events_path()
                lines, new_offset = rec.read_lines(path, offset)
                body = b'{"offset":' + str(new_offset).encode() + b',"reset":' + (b"true" if new_offset < offset or (offset and new_offset == 0) else b"false") + b',"rows":[' + ",".join(lines).encode("utf-8") + b"]}"
                self._send(body, "application/json; charset=utf-8")
            elif what == "output":
                self._send(rec.output().encode("utf-8"), "text/plain; charset=utf-8")
            elif what == "files":
                self._json({"files": rec.list_files()})
            elif what == "file":
                rel = "/".join(parts[4:]) or query.get("path", [""])[0]
                path = rec.file_path(rel)
                if path is None:
                    self._error(404, f"no such file: {rel}")
                    return
                ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if path.name.endswith(".json"):
                    ctype = "application/json; charset=utf-8"
                headers = {}
                if ctype.startswith("text/html"):
                    # logged HTML is untrusted: render it sandboxed
                    headers["Content-Security-Policy"] = "sandbox allow-scripts"
                self._send(path.read_bytes(), ctype, headers=headers)
            else:
                self._error(404, "unknown endpoint")
            return
        self._error(404, "not found")


def _convert_legacy_runs(store: RunStore) -> None:
    for rec in store.runs():
        if rec.is_legacy:
            try:
                rec.ensure_converted()
            except Exception:
                pass


def serve(roots: list[str], host: str = "127.0.0.1", port: int = 8080, recursive: bool = False, open_browser: bool = False, verbose: bool = False) -> None:
    store = RunStore(roots, recursive=recursive)
    server = None
    last_error: OSError | None = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), _Handler)
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        raise SystemExit(f"Could not bind to {host}:{port}-{port + 19}: {last_error}")
    server.daemon_threads = True
    server.store = store  # type: ignore[attr-defined]
    server.verbose = verbose  # type: ignore[attr-defined]
    actual = server.server_address[1]
    runs = store.runs()
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{actual}"
    print(f"wandb-local dashboard: {url}")
    print(f"  {len(runs)} run(s) found in: " + ", ".join(str(d) for d in store.wandb_dirs() or store.roots))
    legacy = sum(1 for r in runs if r.is_legacy)
    if legacy:
        print(f"  importing {legacy} run(s) recorded by the real wandb client in the background...")
        threading.Thread(target=_convert_legacy_runs, args=(store,), daemon=True).start()
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: the dashboard has no authentication; anyone who can reach this address can read your runs.")
    print("  Press Ctrl+C to stop.")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
