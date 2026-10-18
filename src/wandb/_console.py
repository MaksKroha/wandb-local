"""Tee stdout/stderr into ``files/output.log``.

Like the real client's "wrap" console mode, the ``write`` method of the
existing stream objects is patched, so loggers and progress bars that captured
``sys.stderr`` before ``wandb.init()`` are recorded too.
"""

from __future__ import annotations

import os
import sys
import threading


class _StreamProxy:
    """Fallback used when a stream does not allow patching its ``write`` attribute."""

    def __init__(self, stream, tee):
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_tee", tee)

    def write(self, s):
        n = self._stream.write(s)
        self._tee(s)
        return n

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __setattr__(self, name, value):
        setattr(self._stream, name, value)


class ConsoleCapture:
    def __init__(self, path: str) -> None:
        self._fh = open(path, "a", encoding="utf-8", errors="replace")
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._closed = False
        self._patched: list[tuple[object, object]] = []
        self._proxied: list[tuple[str, object]] = []

    def _tee(self, s) -> None:
        if self._closed or os.getpid() != self._pid:
            return
        if isinstance(s, (bytes, bytearray)):
            s = bytes(s).decode("utf-8", "replace")
        elif not isinstance(s, str):
            return
        try:
            with self._lock:
                self._fh.write(s)
        except Exception:
            pass

    def start(self) -> None:
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name, None)
            if stream is None:
                continue
            original = stream.write
            tee = self._tee

            def write(s, _orig=original, _tee=tee):
                n = _orig(s)
                _tee(s)
                return n

            write._wandb_local_tee = True  # type: ignore[attr-defined]
            try:
                stream.write = write
                self._patched.append((stream, original))
            except (AttributeError, TypeError):
                setattr(sys, name, _StreamProxy(stream, self._tee))
                self._proxied.append((name, stream))

    def flush(self) -> None:
        try:
            with self._lock:
                if not self._closed:
                    self._fh.flush()
        except Exception:
            pass

    def stop(self) -> None:
        for stream, original in self._patched:
            try:
                current = getattr(stream, "write", None)
                if getattr(current, "_wandb_local_tee", False):
                    if "write" in getattr(stream, "__dict__", {}):
                        del stream.__dict__["write"]
                    if getattr(stream, "write", None) is not original and getattr(stream.write, "_wandb_local_tee", False):
                        stream.write = original
            except Exception:
                pass
        self._patched.clear()
        for name, stream in self._proxied:
            if isinstance(getattr(sys, name, None), _StreamProxy):
                setattr(sys, name, stream)
        self._proxied.clear()
        with self._lock:
            self._closed = True
            try:
                self._fh.close()
            except Exception:
                pass
