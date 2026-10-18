"""Rich media types: Table, Image, Html, Histogram, Audio, Video, Object3D, Molecule, Plotly.

Every type serialises to a small JSON reference (``{"_type": "...-file", "path": ...}``)
stored in the history row, while the payload itself is written under
``<run>/files/media/<kind>/`` — the same layout the real client uses, so the
local viewer can also display tables/images from runs produced by real wandb.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Sequence

from .util import sha256_bytes, to_jsonable

_SAFE = re.compile(r"[^A-Za-z0-9_.\-/]+")


def _safe_key(key: str) -> str:
    key = _SAFE.sub("_", str(key)).strip("/")
    parts = [p for p in key.split("/") if p not in ("", ".", "..")]
    return "/".join(parts) or "media"


def _files_dir(target: Any) -> Path:
    files_dir = getattr(target, "files_dir", None)
    if files_dir is None:
        raise ValueError("Media objects can only be serialised inside a run or an artifact")
    return Path(files_dir)


def _write_media(target: Any, subdir: str, key: str, step: Any, data: bytes, ext: str) -> tuple[str, str]:
    sha = sha256_bytes(data)
    step_part = "summary" if step is None else str(step)
    rel = f"media/{subdir}/{_safe_key(key)}_{step_part}_{sha[:20]}.{ext}"
    path = _files_dir(target) / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return rel, sha


def _to_numpy(data: Any):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError("numpy is required to log array data (pip install numpy)") from exc
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(data, torch.Tensor):
        data = data.detach().cpu()
        if data.dtype in (torch.bfloat16, torch.float16):
            data = data.float()
        data = data.numpy()
    return np.asarray(data)


class WBValue:
    """Base class of every loggable media object."""

    _log_type: str = "unknown"

    def to_json(self, target: Any = None, key: str = "value", step: Any = None) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_obj: dict, source_artifact: Any = None):
        raise NotImplementedError


class Media(WBValue):
    """A WBValue backed by a file (bytes in memory or a source path on disk)."""

    _subdir = "files"

    def __init__(self, caption: str | None = None) -> None:
        self._caption = caption
        self._bytes: bytes | None = None
        self._src_path: str | None = None
        self._ext = "bin"

    @property
    def caption(self) -> str | None:
        return self._caption

    def _load_path(self, path: str | os.PathLike, default_ext: str | None = None) -> None:
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise ValueError(f"File not found: {path}")
        self._src_path = os.path.abspath(path)
        name = os.path.basename(path)
        ext = name.split(".", 1)[1] if "." in name else (default_ext or "bin")
        self._ext = ext.lower()

    def _content(self) -> bytes:
        if self._bytes is None and self._src_path is not None:
            with open(self._src_path, "rb") as fh:
                return fh.read()
        return self._bytes or b""

    def _base_json(self, target: Any, key: str, step: Any) -> dict:
        data = self._content()
        rel, sha = _write_media(target, self._subdir, key, step, data, self._ext)
        out = {"_type": self._log_type, "path": rel, "sha256": sha, "size": len(data), "format": self._ext}
        if self._caption is not None:
            out["caption"] = str(self._caption)
        return out

    def to_json(self, target: Any = None, key: str = "value", step: Any = None) -> dict:
        return self._base_json(target, key, step)


# ---------------------------------------------------------------- PNG ------
def _encode_png(arr) -> bytes:
    """Minimal PNG encoder for uint8 HxW / HxWx{1,3,4} arrays (used when Pillow is missing)."""
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    h, w = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def _png_size(data: bytes) -> tuple[int, int] | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    return None


def _normalize_image_array(arr, normalize: bool = True):
    import numpy as np

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {arr.shape}")
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr.astype(np.float64))
        if normalize:
            lo, hi = float(arr.min()) if arr.size else 0.0, float(arr.max()) if arr.size else 0.0
            if lo >= 0 and hi <= 1:
                arr = arr * 255
            elif lo >= -1 and hi <= 1:
                arr = (arr + 1) * 127.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class Image(Media):
    _log_type = "image-file"
    _subdir = "images"

    def __init__(
        self,
        data_or_path: Any,
        mode: str | None = None,
        caption: str | None = None,
        grouping: int | None = None,
        classes: Any = None,
        boxes: Any = None,
        masks: Any = None,
        file_type: str | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__(caption)
        self._width: int | None = None
        self._height: int | None = None
        self._grouping = grouping
        self._boxes = boxes
        self._masks = masks
        file_type = (file_type or "png").lower().lstrip(".")
        if isinstance(data_or_path, Image):
            self.__dict__.update({k: v for k, v in data_or_path.__dict__.items() if k != "_caption"})
            self._caption = caption if caption is not None else data_or_path._caption
            return
        if isinstance(data_or_path, (str, os.PathLike)):
            self._load_path(data_or_path, "png")
            self._probe_size_from_file()
            return
        if isinstance(data_or_path, (bytes, bytearray)):
            self._bytes = bytes(data_or_path)
            self._ext = "png" if _png_size(self._bytes) else file_type
            size = _png_size(self._bytes)
            if size:
                self._width, self._height = size
            return
        self._from_object(data_or_path, mode, normalize, file_type)

    def _probe_size_from_file(self) -> None:
        try:
            with open(self._src_path, "rb") as fh:  # type: ignore[arg-type]
                size = _png_size(fh.read(32))
            if size:
                self._width, self._height = size
                return
            from PIL import Image as PILImage

            with PILImage.open(self._src_path) as im:
                self._width, self._height = im.size
        except Exception:
            pass

    def _from_object(self, obj: Any, mode: str | None, normalize: bool, file_type: str) -> None:
        pil_module = sys.modules.get("PIL.Image")
        # matplotlib figure / pyplot module
        if hasattr(obj, "savefig") or getattr(obj, "__name__", "") == "matplotlib.pyplot":
            fig = obj.gcf() if getattr(obj, "__name__", "") == "matplotlib.pyplot" else obj
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            self._bytes = buf.getvalue()
            self._ext = "png"
            size = _png_size(self._bytes)
            if size:
                self._width, self._height = size
            return
        if pil_module is not None and isinstance(obj, pil_module.Image):
            self._from_pil(obj, file_type, mode)
            return
        arr = _normalize_image_array(_to_numpy(obj), normalize)
        self._height, self._width = int(arr.shape[0]), int(arr.shape[1])
        try:
            from PIL import Image as PILImage

            self._from_pil(PILImage.fromarray(arr, mode=mode) if mode else PILImage.fromarray(arr), file_type, None)
        except ImportError:
            self._bytes = _encode_png(arr)
            self._ext = "png"

    def _from_pil(self, im: Any, file_type: str, mode: str | None) -> None:
        if mode:
            im = im.convert(mode)
        fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "gif": "GIF", "bmp": "BMP", "webp": "WEBP"}.get(file_type, "PNG")
        if fmt == "JPEG" and im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format=fmt)
        self._bytes = buf.getvalue()
        self._ext = "jpg" if fmt == "JPEG" else fmt.lower()
        self._width, self._height = im.size

    @property
    def image(self):
        from PIL import Image as PILImage

        return PILImage.open(io.BytesIO(self._content()))

    def to_json(self, target: Any = None, key: str = "image", step: Any = None) -> dict:
        out = self._base_json(target, key, step)
        if self._width is not None:
            out["width"], out["height"] = self._width, self._height
        if self._grouping is not None:
            out["grouping"] = self._grouping
        return out


class Html(Media):
    _log_type = "html-file"
    _subdir = "html"

    def __init__(self, data: Any, inject: bool = True, data_is_not_path: bool = False) -> None:
        super().__init__()
        self._ext = "html"
        if hasattr(data, "read"):
            text = data.read()
        elif isinstance(data, (str, os.PathLike)) and not data_is_not_path and str(data).lower().endswith((".html", ".htm")) and os.path.isfile(data):
            with open(data, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        else:
            text = str(data)
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        self.html = text
        self._bytes = text.encode("utf-8")


class Histogram(WBValue):
    _log_type = "histogram"
    MAX_LENGTH = 512

    def __init__(self, sequence: Sequence | None = None, np_histogram: tuple | None = None, num_bins: int = 64) -> None:
        if np_histogram is not None:
            counts, bins = np_histogram
            self.histogram = [float(c) for c in to_jsonable(counts)]
            self.bins = [float(b) for b in to_jsonable(bins)]
        else:
            if sequence is None:
                raise ValueError("Histogram requires either a sequence or np_histogram")
            num_bins = max(1, min(int(num_bins), self.MAX_LENGTH))
            try:
                import numpy as np

                values = _to_numpy(sequence).astype(np.float64).ravel()
                values = values[np.isfinite(values)]
                counts, bins = np.histogram(values, bins=num_bins) if values.size else (np.zeros(num_bins), np.linspace(0, 1, num_bins + 1))
                self.histogram = counts.tolist()
                self.bins = bins.tolist()
            except ImportError:
                values = [float(v) for v in _flatten(sequence) if isinstance(v, (int, float)) and math.isfinite(v)]
                lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
                if lo == hi:
                    lo, hi = lo - 0.5, hi + 0.5
                width = (hi - lo) / num_bins
                counts = [0] * num_bins
                for v in values:
                    counts[min(int((v - lo) / width), num_bins - 1)] += 1
                self.histogram = counts
                self.bins = [lo + i * width for i in range(num_bins + 1)]
        if len(self.histogram) > self.MAX_LENGTH:
            raise ValueError(f"The maximum length of a histogram is {self.MAX_LENGTH}")
        if len(self.bins) != len(self.histogram) + 1:
            raise ValueError("len(bins) must be len(histogram) + 1")

    def to_json(self, target: Any = None, key: str = "histogram", step: Any = None) -> dict:
        return {"_type": "histogram", "values": self.histogram, "bins": self.bins}


def _flatten(seq: Any):
    for item in seq:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


class Audio(Media):
    _log_type = "audio-file"
    _subdir = "audio"

    def __init__(self, data_or_path: Any, sample_rate: int | None = None, caption: str | None = None) -> None:
        super().__init__(caption)
        self._sample_rate = sample_rate
        self._duration: float | None = None
        if isinstance(data_or_path, (str, os.PathLike)):
            self._load_path(data_or_path, "wav")
            return
        if sample_rate is None:
            raise ValueError("Argument 'sample_rate' is required when instantiating wandb.Audio with raw data.")
        import wave

        import numpy as np

        arr = _to_numpy(data_or_path)
        if arr.ndim == 1:
            arr = arr[:, None]
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, -1.0, 1.0) * 32767).astype("<i2")
        else:
            arr = arr.astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(arr.shape[1])
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(np.ascontiguousarray(arr).tobytes())
        self._bytes = buf.getvalue()
        self._ext = "wav"
        self._duration = arr.shape[0] / float(sample_rate)

    def to_json(self, target: Any = None, key: str = "audio", step: Any = None) -> dict:
        out = self._base_json(target, key, step)
        if self._sample_rate:
            out["sample_rate"] = self._sample_rate
        if self._duration is not None:
            out["duration"] = self._duration
        return out


class Video(Media):
    _log_type = "video-file"
    _subdir = "videos"

    def __init__(self, data_or_path: Any, caption: str | None = None, fps: int | None = None, format: str | None = None) -> None:
        super().__init__(caption)
        self._fps = fps or 4
        if isinstance(data_or_path, (str, os.PathLike)):
            self._load_path(data_or_path, "mp4")
            return
        import numpy as np

        arr = _to_numpy(data_or_path)
        if arr.ndim == 5:  # (N, T, C, H, W) -> tile batch horizontally
            arr = np.concatenate(list(arr), axis=-1)
        if arr.ndim != 4:
            raise ValueError("wandb.Video requires a (time, channel, height, width) array")
        frames = [_normalize_image_array(frame) for frame in arr]
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError("Pillow is required to encode videos from arrays (pip install pillow)") from exc
        images = [PILImage.fromarray(f) for f in frames]
        buf = io.BytesIO()
        images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=int(1000 / self._fps), loop=0)
        self._bytes = buf.getvalue()
        self._ext = "gif"

    def to_json(self, target: Any = None, key: str = "video", step: Any = None) -> dict:
        out = self._base_json(target, key, step)
        out["fps"] = self._fps
        return out


class Object3D(Media):
    _log_type = "object3D-file"
    _subdir = "object3D"

    def __init__(self, data_or_path: Any, caption: str | None = None, **kwargs: Any) -> None:
        super().__init__(caption)
        if isinstance(data_or_path, (str, os.PathLike)):
            self._load_path(data_or_path, "obj")
        elif isinstance(data_or_path, dict):
            self._bytes = json.dumps(to_jsonable(data_or_path)).encode()
            self._ext = "pts.json"
        else:
            self._bytes = json.dumps(to_jsonable(_to_numpy(data_or_path).tolist())).encode()
            self._ext = "pts.json"


class Molecule(Media):
    _log_type = "molecule-file"
    _subdir = "molecule"

    def __init__(self, data_or_path: Any, caption: str | None = None, **kwargs: Any) -> None:
        super().__init__(caption)
        if isinstance(data_or_path, (str, os.PathLike)):
            self._load_path(data_or_path, "pdb")
        else:
            self._bytes = data_or_path.read().encode() if hasattr(data_or_path, "read") else str(data_or_path).encode()
            self._ext = "pdb"


class Plotly(Media):
    _log_type = "plotly-file"
    _subdir = "plotly"

    def __init__(self, val: Any) -> None:
        super().__init__()
        if hasattr(val, "savefig"):  # matplotlib -> store as image instead
            raise TypeError("Use wandb.Image for matplotlib figures in wandb-local")
        payload = val.to_plotly_json() if hasattr(val, "to_plotly_json") else val
        self._bytes = json.dumps(to_jsonable(payload)).encode()
        self._ext = "plotly.json"


class Graph(WBValue):
    _log_type = "graph"

    def __init__(self, text: str = "") -> None:
        self.text = text

    def to_json(self, target: Any = None, key: str = "graph", step: Any = None) -> dict:
        return {"_type": "graph", "text": self.text}


# --------------------------------------------------------------- Table -----
def _is_dataframe(obj: Any) -> bool:
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(obj, pd.DataFrame)


def _cell_to_json(target: Any, key: str, step: Any, value: Any) -> Any:
    if isinstance(value, WBValue):
        return value.to_json(target, key, step)
    if isinstance(value, (list, tuple)) and any(isinstance(v, WBValue) for v in value):
        return [_cell_to_json(target, key, step, v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    pd = sys.modules.get("pandas")
    if pd is not None:
        try:
            if value is pd.NA or value is pd.NaT:
                return None
        except Exception:
            pass
    return to_jsonable(value)


class Table(Media):
    """A 2D table of values; may contain other media (e.g. lists of ``wandb.Image``)."""

    _log_type = "table-file"
    _subdir = "table"
    MAX_ROWS = 10000
    MAX_ARTIFACT_ROWS = 200000

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        data: Any = None,
        rows: Any = None,
        dataframe: Any = None,
        dtype: Any = None,
        optional: Any = True,
        allow_mixed_types: bool = False,
        log_mode: str | None = "IMMUTABLE",
    ) -> None:
        super().__init__()
        self._ext = "table.json"
        self.log_mode = (log_mode or "IMMUTABLE").upper()
        if self.log_mode not in ("IMMUTABLE", "MUTABLE", "INCREMENTAL"):
            raise ValueError("log_mode must be one of IMMUTABLE, MUTABLE, INCREMENTAL")
        if data is None and rows is not None:
            data = rows
        if dataframe is None and _is_dataframe(data):
            dataframe, data = data, None
        if dataframe is not None:
            self.columns = [str(c) for c in dataframe.columns]
            self.data = [list(r) for r in dataframe.to_numpy(dtype=object).tolist()]
        else:
            if data is not None and type(data).__module__.startswith(("numpy", "torch")):
                data = _to_numpy(data).tolist()
            self.data = [list(r) for r in (data or [])]
            if columns is None:
                width = len(self.data[0]) if self.data else 3
                columns = ["Input", "Output", "Expected"] if width == 3 and not self.data else [f"Column {i + 1}" for i in range(width)]
            self.columns = [str(c) for c in columns]
        for row in self.data:
            if len(row) != len(self.columns):
                raise ValueError(f"This table expects {len(self.columns)} columns: {self.columns}, found {len(row)}")
        self._last_logged_rows = 0

    # -- API ---------------------------------------------------------------
    def add_data(self, *data: Any) -> None:
        if len(data) != len(self.columns):
            raise ValueError(f"This table expects {len(self.columns)} columns: {self.columns}, found {len(data)}")
        self.data.append(list(data))

    add_row = add_data

    def add_column(self, name: str, data: Sequence, optional: bool = False) -> None:
        if name in self.columns:
            raise ValueError(f"Column {name} already exists")
        data = list(data)
        if self.data and len(data) != len(self.data):
            raise ValueError(f"Expected column length {len(self.data)}, found {len(data)}")
        if not self.data:
            self.data = [[] for _ in data]
        self.columns.append(str(name))
        for row, value in zip(self.data, data):
            row.append(value)

    def add_computed_columns(self, fn) -> None:
        for idx, row in enumerate(self.data):
            extra = fn(idx, dict(zip(self.columns, row)))
            for name, value in extra.items():
                if name not in self.columns:
                    self.columns.append(name)
                    for r in self.data:
                        r.append(None)
                row[self.columns.index(name)] = value

    def get_column(self, name: str, convert_to: str | None = None) -> list:
        idx = self.columns.index(name)
        values = [row[idx] for row in self.data]
        if convert_to == "numpy":
            import numpy as np

            return np.array(values)
        return values

    def get_index(self) -> list:
        return list(range(len(self.data)))

    def get_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.data, columns=self.columns)

    def cast(self, col_name: str, dtype: Any, optional: bool = False) -> None:
        """Type casting is a server-side concept; kept for API compatibility."""

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Table) and self.columns == other.columns and self.data == other.data

    @property
    def ncols(self) -> int:
        return len(self.columns)

    @property
    def nrows(self) -> int:
        return len(self.data)

    def to_payload(self, target: Any, key: str, step: Any, rows: list | None = None) -> dict:
        rows = self.data if rows is None else rows
        return {"columns": self.columns, "data": [[_cell_to_json(target, key, step, v) for v in row] for row in rows]}

    def to_json(self, target: Any = None, key: str = "table", step: Any = None) -> dict:
        from .util import dumps

        payload = self.to_payload(target, key, step)
        self._bytes = dumps(payload).encode("utf-8")
        out = self._base_json(target, key, step)
        out.update({"ncols": self.ncols, "nrows": self.nrows, "log_mode": self.log_mode})
        self._bytes = None
        return out

    @classmethod
    def from_json(cls, json_obj: dict, source_artifact: Any = None) -> "Table":
        return cls(columns=json_obj.get("columns"), data=json_obj.get("data"))


class JoinedTable(WBValue):
    _log_type = "joined-table"

    def __init__(self, table1: Table, table2: Table, join_key: Any) -> None:
        self.table1, self.table2, self.join_key = table1, table2, join_key

    def to_json(self, target: Any = None, key: str = "joined", step: Any = None) -> dict:
        return {
            "_type": "joined-table",
            "table1": self.table1.to_json(target, f"{key}_t1", step),
            "table2": self.table2.to_json(target, f"{key}_t2", step),
            "join_key": to_jsonable(self.join_key),
        }


class _LazyPath:
    """Helper used by Artifact.add to persist a WBValue into a staging directory."""

    def __init__(self, files_dir: str) -> None:
        self.files_dir = files_dir


def save_value_into(directory: str, name: str, value: WBValue) -> str:
    """Serialise ``value`` into ``directory`` as ``<name>.<type>.json``; returns that file name."""
    target = _LazyPath(directory)
    ref = value.to_json(target, name, None)
    if isinstance(value, Table):
        # Move the table payload next to its descriptor with a stable name.
        src = Path(directory) / ref["path"]
        dst_name = f"{name}.table.json"
        dst = Path(directory) / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return dst_name
    dst_name = f"{name}.{value._log_type}.json"
    (Path(directory) / dst_name).parent.mkdir(parents=True, exist_ok=True)
    (Path(directory) / dst_name).write_text(json.dumps(ref), encoding="utf-8")
    return dst_name


Classes = None  # placeholder for API compatibility (bounding-box class sets are not rendered locally)
