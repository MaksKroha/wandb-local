"""Legacy .wandb import, the dashboard server and static reports."""

import json
import struct
import threading
import urllib.request
import zlib
from pathlib import Path

import wandb
from wandb._legacy import convert_run_dir, read_legacy_run
from wandb.viewer.report import build_report
from wandb.viewer.store import RunStore


# -- a tiny writer for the real client's transaction-log format (test fixture) --
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _field(num, payload):
    if isinstance(payload, int):
        return _varint(num << 3) + _varint(payload)
    if isinstance(payload, str):
        payload = payload.encode()
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _item(key, value):
    return _field(1, key) + _field(16, json.dumps(value))


def write_legacy(path, records):
    """Write records as the real client does (32 KiB blocks, FULL/FIRST/MIDDLE/LAST)."""
    data = bytearray(struct.pack("<4sHB", b":W&B", 0xBEE1, 0))
    for rec in records:
        rest = rec
        first = True
        while True:
            left = 32768 - len(data) % 32768
            if left < 7:
                data += b"\x00" * left
                continue
            room = left - 7
            chunk, rest = rest[:room], rest[room:]
            if first and not rest:
                rtype = 1
            elif first:
                rtype = 2
            elif rest:
                rtype = 3
            else:
                rtype = 4
            crc = zlib.crc32(chunk, zlib.crc32(bytes([rtype]))) & 0xFFFFFFFF
            data += struct.pack("<IHB", crc, len(chunk), rtype) + chunk
            first = False
            if not rest:
                break
    Path(path).write_bytes(bytes(data))


def make_legacy_run(root: Path, n=500):
    rd = root / "wandb" / "run-20260101_000000-legacy01"
    (rd / "files").mkdir(parents=True)
    recs = [_field(17, _field(1, "legacy01") + _field(3, "oldproj") + _field(8, "old-run") + _field(17, _field(1, 1767225600)))]
    recs.append(_field(12, _field(2, "*") + _field(4, "train/global_step") + _field(6, _field(1, 1))))
    for i in range(n):
        hist = _field(1, _item("train/loss", 1 / (i + 1))) + _field(1, _item("train/global_step", i)) + _field(1, _item("_step", i)) + _field(1, _item("blob", "x" * 400)) + _field(2, _field(1, i))
        recs.append(_field(2, hist))
    recs.append(_field(7, _field(2, _field(1, 1767225610)) + _field(3, _field(1, "cpu") + _field(16, "12.5"))))
    recs.append(_field(18, _field(2, 42)))
    write_legacy(rd / "run-legacy01.wandb", recs)
    return rd


def test_legacy_decode_and_convert(tmp_path):
    rd = make_legacy_run(tmp_path)
    data = read_legacy_run(rd / "run-legacy01.wandb")
    assert len(data["history"]) == 500 and data["history"][-1]["train/loss"] == 1 / 500
    assert data["run"]["name"] == "old-run" and data["events"][0]["cpu"] == 12.5
    info = convert_run_dir(rd)
    assert info["state"] == "finished" and info["project"] == "oldproj"
    api_run = wandb.Api().run("legacy01")
    assert api_run.name == "old-run" and len(api_run.history(samples=None)) == 500


def _serve(store):
    from http.server import ThreadingHTTPServer

    from wandb.viewer.server import _Handler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.store = store
    srv.verbose = False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def test_server_endpoints(tmp_path):
    np = __import__("pytest").importorskip("numpy")
    make_legacy_run(tmp_path, n=20)
    run = wandb.init(project="srv", name="live")
    wandb.log({"loss": 1.0, "img": wandb.Image(np.zeros((4, 4)))})
    wandb.finish()
    srv, base = _serve(RunStore([str(tmp_path)]))
    try:
        assert b"wandb-local" in _get(base + "/")
        runs = json.loads(_get(base + "/api/runs"))["runs"]
        assert {r["name"] for r in runs} >= {"live"}
        key = next(r["key"] for r in runs if r["name"] == "live")
        hist = json.loads(_get(f"{base}/api/run/{key}/history?offset=0"))
        assert hist["rows"][0]["loss"] == 1.0 and hist["offset"] > 0
        again = json.loads(_get(f"{base}/api/run/{key}/history?offset={hist['offset']}"))
        assert again["rows"] == []
        info = json.loads(_get(f"{base}/api/run/{key}/info"))
        assert info["info"]["state"] == "finished"
        img_path = hist["rows"][0]["img"]["path"]
        assert _get(f"{base}/api/run/{key}/file/{img_path}")[:4] == b"\x89PNG"
        try:
            _get(f"{base}/api/run/{key}/file/../../../../etc/passwd")
            raise AssertionError("path traversal not blocked")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        legacy_key = next(r["key"] for r in runs if r["id"] == "legacy01")
        rows = json.loads(_get(f"{base}/api/run/{legacy_key}/history"))["rows"]
        assert len(rows) == 20
    finally:
        srv.shutdown()


def test_static_report(tmp_path):
    run = wandb.init(project="rep", name="report-run")
    wandb.log({"loss": 0.5, "t": wandb.Table(columns=["a"], data=[["</script><script>alert(1)</script>"]])})
    wandb.finish()
    store = RunStore([str(tmp_path)])
    html = build_report(store, store.runs())
    assert '<script id="wandb-embed"' in html
    # embedded data must not be able to close the script tag
    embed = html.split('<script id="wandb-embed" type="application/json">', 1)[1].split("</script>", 1)[0]
    payload = json.loads(embed)
    key = payload["runs"][0]["key"]
    assert payload["data"][key]["history"][0]["loss"] == 0.5
    assert any(p.endswith(".table.json") for p in payload["data"][key]["files"])


def test_cli_ls_export_report(tmp_path, capsys):
    from wandb.cli import main

    run = wandb.init(project="cli", name="cli-run")
    for i in range(3):
        wandb.log({"loss": i})
    wandb.finish()
    assert main(["ls"]) == 0
    assert "cli-run" in capsys.readouterr().out
    assert main(["export", run.id, "-o", str(tmp_path / "h.csv")]) == 0
    assert (tmp_path / "h.csv").read_text().splitlines()[0].startswith("loss")
    assert main(["report", "-o", str(tmp_path / "r.html")]) == 0
    assert (tmp_path / "r.html").stat().st_size > 10000
