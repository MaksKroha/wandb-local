"""``wandb`` command line (wandb-local edition)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from ._version import LOCAL_VERSION, __version__


def _default_roots(paths: list[str] | None) -> list[str]:
    if paths:
        return paths
    return [os.environ.get("WANDB_DIR") or os.getcwd()]


def _store(args):
    from .viewer.store import RunStore

    return RunStore(_default_roots(getattr(args, "paths", None)), recursive=getattr(args, "recursive", False))


def cmd_ui(args) -> int:
    from .viewer.server import serve

    serve(_default_roots(args.paths), host=args.host, port=args.port, recursive=args.recursive, open_browser=args.open, verbose=args.verbose)
    return 0


def cmd_report(args) -> int:
    from .viewer.report import build_report

    store = _store(args)
    if args.run:
        records = []
        for ref in args.run:
            rec = store.find(ref)
            if rec is None:
                print(f"run not found: {ref}", file=sys.stderr)
                return 1
            records.append(rec)
    else:
        records = store.runs()
        if args.project:
            records = [r for r in records if r.info().get("project") == args.project]
        records = records[: args.limit]
    if not records:
        print("no runs found", file=sys.stderr)
        return 1
    html = build_report(store, records, tables=args.tables, max_media_mb=args.max_media_mb, max_table_rows=args.max_table_rows)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(records)} run(s))")
    return 0


def cmd_ls(args) -> int:
    store = _store(args)
    runs = store.runs()
    if not runs:
        print("no runs found in " + ", ".join(map(str, store.roots)))
        return 0
    rows = []
    for rec in runs:
        info = rec.info()
        if args.project and info.get("project") != args.project:
            continue
        rows.append((info.get("id", ""), str(info.get("name") or "")[:40], str(info.get("project") or "")[:24], str(info.get("state") or ""), str(info.get("created_at") or "")[:19], str(rec.path)))
    widths = [max(len(r[i]) for r in rows + [("ID", "NAME", "PROJECT", "STATE", "CREATED", "DIR")]) for i in range(6)]
    header = ("ID", "NAME", "PROJECT", "STATE", "CREATED", "DIR")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return 0


def cmd_show(args) -> int:
    store = _store(args)
    rec = store.find(args.run)
    if rec is None:
        print(f"run not found: {args.run}", file=sys.stderr)
        return 1
    info = rec.info(convert=True)
    print(json.dumps({"info": info, "config": rec.config(), "summary": rec.summary()}, indent=2, default=str))
    return 0


def cmd_export(args) -> int:
    store = _store(args)
    rec = store.find(args.run)
    if rec is None:
        print(f"run not found: {args.run}", file=sys.stderr)
        return 1
    rec.ensure_converted()
    rows = list(rec.iter_rows("events" if args.system else "history"))
    out = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        if args.format == "jsonl":
            for row in rows:
                out.write(json.dumps(row) + "\n")
        else:
            keys: list[str] = []
            for row in rows:
                for k, v in row.items():
                    if k not in keys and not isinstance(v, (dict, list)):
                        keys.append(k)
            if args.keys:
                keys = [k for k in ["_step", *args.keys] if k in keys]
            writer = csv.DictWriter(out, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in keys})
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def cmd_sync(args) -> int:
    """Convert runs recorded by the real wandb client (.wandb files) for local viewing."""
    from ._legacy import convert_run_dir
    from .viewer.store import RUN_DIR_RE, wandb_dirs_for

    targets: list[Path] = []
    for p in args.paths or [os.getcwd()]:
        path = Path(p).expanduser().resolve()
        if RUN_DIR_RE.match(path.name):
            targets.append(path)
        else:
            for d in wandb_dirs_for(path, recursive=True):
                targets.extend(sorted(x for x in d.iterdir() if RUN_DIR_RE.match(x.name) and not x.is_symlink()))
    done = 0
    for run_dir in targets:
        if not any(run_dir.glob("*.wandb")):
            continue
        if (run_dir / "wandb-run.json").exists() and not args.force:
            continue
        try:
            info = convert_run_dir(run_dir, force=args.force)
            print(f"imported {run_dir.name}: {info.get('name')} ({info.get('state')})")
            done += 1
        except Exception as exc:
            print(f"failed {run_dir}: {exc}", file=sys.stderr)
    print(f"{done} run(s) imported. Nothing is uploaded: wandb-local keeps everything on disk.")
    return 0


def cmd_artifact(args) -> int:
    from .artifacts import find_artifact, list_artifacts

    store = _store(args)
    if args.artifact_cmd == "ls":
        for entry in list_artifacts(store.artifact_roots()):
            for v in entry["versions"]:
                aliases = ",".join(v["aliases"])
                print(f"{entry['project']}/{entry['name']}:{v['version']}  type={entry['type']}  aliases={aliases}  files={v['files']}  size={v['size']}  {v['dir']}")
        return 0
    art = find_artifact(args.name, store.artifact_roots())
    path = art.download(args.root)
    print(path)
    return 0


def cmd_sweep(args) -> int:
    from ._sweeps import sweep

    if args.project:
        os.environ["WANDB_PROJECT"] = args.project
    sweep(args.config, project=args.project)
    return 0


def cmd_agent(args) -> int:
    from ._sweeps import agent

    agent(args.sweep_id, count=args.count)
    return 0


def _write_mode(mode: str) -> int:
    root = Path(os.environ.get("WANDB_DIR") or os.getcwd()) / "wandb"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings").write_text(f"[default]\nmode = {mode}\n")
    print(f"wandb-local mode set to {mode!r} in {root / 'settings'}")
    return 0


def cmd_status(args) -> int:
    store = _store(args)
    root = Path(os.environ.get("WANDB_DIR") or os.getcwd())
    print(f"wandb-local {LOCAL_VERSION} (API compatible with wandb {__version__.split('+')[0]})")
    print(f"directory: {root / 'wandb'}")
    mode = None
    try:
        from ._run import _read_mode_file

        mode = _read_mode_file(root / "wandb")
    except Exception:
        pass
    print(f"mode: {os.environ.get('WANDB_MODE') or mode or 'offline (local)'}")
    print(f"runs: {len(store.runs())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wandb", description="wandb-local: offline experiment tracking with a local dashboard.")
    p.add_argument("--version", action="version", version=f"wandb-local {LOCAL_VERSION} (wandb API {__version__})")
    sub = p.add_subparsers(dest="cmd")

    def add_paths(sp, help_text="directories containing a wandb/ folder, wandb/ folders or run directories (default: $WANDB_DIR or cwd)"):
        sp.add_argument("paths", nargs="*", help=help_text)
        sp.add_argument("-r", "--recursive", action="store_true", help="search sub-directories for wandb/ folders")

    for name in ("ui", "local", "server", "serve"):
        sp = sub.add_parser(name, help="start the local dashboard" if name == "ui" else argparse.SUPPRESS)
        add_paths(sp)
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--port", type=int, default=8080)
        sp.add_argument("--open", action="store_true", help="open a browser")
        sp.add_argument("--verbose", action="store_true")
        sp.set_defaults(func=cmd_ui)

    sp = sub.add_parser("report", help="write a self-contained HTML report (no server needed)")
    add_paths(sp)
    sp.add_argument("-o", "--output", default="wandb-report.html")
    sp.add_argument("--run", action="append", help="run id / dir name / name (repeatable); default: latest runs")
    sp.add_argument("--project")
    sp.add_argument("--limit", type=int, default=8)
    sp.add_argument("--tables", choices=["latest", "all", "none"], default="latest")
    sp.add_argument("--max-media-mb", type=float, default=50)
    sp.add_argument("--max-table-rows", type=int, default=2000)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("ls", help="list runs")
    add_paths(sp)
    sp.add_argument("--project")
    sp.set_defaults(func=cmd_ls)

    sp = sub.add_parser("show", help="print a run's info, config and summary as JSON")
    sp.add_argument("run")
    add_paths(sp)
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("export", help="export a run's history (or system metrics) as CSV/JSONL")
    sp.add_argument("run")
    add_paths(sp)
    sp.add_argument("-o", "--output")
    sp.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    sp.add_argument("--keys", nargs="*")
    sp.add_argument("--system", action="store_true", help="export system metrics instead of history")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("sync", help="import runs recorded by the real wandb client (.wandb files) for local viewing")
    add_paths(sp, "run directories or folders to search (default: cwd)")
    sp.add_argument("--force", action="store_true", help="re-import already imported runs")
    sp.add_argument("--sync-all", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("artifact", help="list or fetch local artifacts")
    asub = sp.add_subparsers(dest="artifact_cmd", required=True)
    a = asub.add_parser("ls")
    add_paths(a)
    a = asub.add_parser("get")
    a.add_argument("name", help="name[:alias|:vN], optionally project/name:alias")
    a.add_argument("--root", help="copy files to this directory")
    add_paths(a)
    sp.set_defaults(func=cmd_artifact)

    sp = sub.add_parser("sweep", help="create a local sweep from a YAML config")
    sp.add_argument("config")
    sp.add_argument("--project")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("agent", help="run a local sweep agent (uses 'program' from the sweep config)")
    sp.add_argument("sweep_id")
    sp.add_argument("--count", type=int)
    sp.set_defaults(func=cmd_agent)

    sp = sub.add_parser("status", help="show local settings")
    add_paths(sp)
    sp.set_defaults(func=cmd_status)

    for mode in ("offline", "online", "disabled", "enabled"):
        sp = sub.add_parser(mode, help=f"set mode to {mode}" if mode in ("disabled", "enabled") else argparse.SUPPRESS)
        sp.set_defaults(func=lambda a, m=mode: _write_mode({"enabled": "offline", "online": "offline"}.get(m, m)))

    sp = sub.add_parser("login", help="no-op (wandb-local needs no account)")
    sp.add_argument("key", nargs="?")
    sp.add_argument("--relogin", action="store_true")
    sp.add_argument("--host")
    sp.set_defaults(func=lambda a: print("wandb-local runs fully offline; no login required.") or 0)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
