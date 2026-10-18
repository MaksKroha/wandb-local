"""Small helpers shared by the SDK, the viewer and the CLI."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import hashlib
import json
import math
import os
import pathlib
import random
import string
import sys
import tempfile
from typing import Any

_ID_ALPHABET = string.ascii_lowercase + string.digits
_rng = random.SystemRandom()


def generate_id(length: int = 8) -> str:
    """Random run id in the same alphabet the real client uses."""
    return "".join(_rng.choice(_ID_ALPHABET) for _ in range(length))


_ADJECTIVES = (
    "amber", "ancient", "astral", "autumn", "blooming", "bold", "brisk", "celestial", "clear", "cosmic",
    "crimson", "dainty", "dashing", "dauntless", "deep", "dulcet", "eager", "electric", "elegant", "ethereal",
    "fanciful", "fearless", "fiery", "fluent", "fresh", "gentle", "giddy", "glorious", "golden", "graceful",
    "happy", "hardy", "honest", "icy", "jolly", "lively", "lucky", "lunar", "misty", "noble",
    "polar", "proud", "quiet", "radiant", "rosy", "royal", "rural", "serene", "silver", "solar",
    "spring", "stellar", "stilted", "sunny", "swift", "twilight", "vibrant", "vivid", "wandering", "wise",
)
_NOUNS = (
    "aurora", "bird", "breeze", "brook", "butterfly", "canyon", "cloud", "comet", "cosmos", "dawn",
    "deluge", "dew", "dream", "dust", "eon", "feather", "field", "firefly", "flower", "fog",
    "forest", "frost", "galaxy", "glade", "glitter", "grass", "haze", "hill", "lake", "leaf",
    "meadow", "moon", "morning", "mountain", "night", "oasis", "ocean", "paper", "pine", "planet",
    "pond", "rain", "resonance", "river", "sea", "shadow", "sky", "smoke", "snow", "sound",
    "star", "sun", "sunset", "terrain", "thunder", "tree", "valley", "violet", "water", "wave",
)


def generate_name(number: int | None = None) -> str:
    name = f"{_rng.choice(_ADJECTIVES)}-{_rng.choice(_NOUNS)}"
    return f"{name}-{number}" if number is not None else name


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert arbitrary Python objects into JSON-compatible values.

    Non-finite floats are kept as floats here; :func:`finite` turns them into
    strings right before anything is written to disk.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if _depth > 48:
        return str(obj)
    d = _depth + 1
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, d) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, d) for v in obj]
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, d)
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", "replace")
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist(), d)
    torch = sys.modules.get("torch")
    if torch is not None:
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu()
            return t.item() if t.numel() == 1 else to_jsonable(t.tolist(), d)
        if isinstance(obj, (torch.dtype, torch.device)):
            return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name, None), d) for f in dataclasses.fields(obj)}
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict) and not isinstance(obj, type):
        try:
            return to_jsonable(to_dict(), d)
        except Exception:
            pass
    if type(obj).__name__ == "Namespace" and hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj), d)
    return str(obj)


def finite(obj: Any) -> Any:
    """Replace NaN/Inf with strings so files stay strict JSON (readable by browsers)."""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if math.isnan(obj):
            return "NaN"
        return "Infinity" if obj > 0 else "-Infinity"
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    return obj


def dumps(obj: Any, indent: int | None = None) -> str:
    return json.dumps(finite(obj), ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))


def atomic_write(path: str | os.PathLike, text: str) -> None:
    path = os.fspath(path)
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def utc_iso(ts: float | None = None) -> str:
    dt = datetime.datetime.fromtimestamp(ts if ts is not None else datetime.datetime.now().timestamp(), datetime.timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def parse_iso(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
