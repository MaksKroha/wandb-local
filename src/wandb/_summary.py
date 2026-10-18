"""``run.summary`` — last/aggregated value of each metric, persisted to wandb-summary.json."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


class Summary:
    def __init__(self, serializer: Callable[[str, Any], Any] | None = None, callback: Callable[[], None] | None = None):
        object.__setattr__(self, "_d", {})
        object.__setattr__(self, "_serializer", serializer)
        object.__setattr__(self, "_callback", callback)

    def _encode(self, key: str, value: Any) -> Any:
        ser = self._serializer
        return ser(key, value) if ser is not None else value

    def _changed(self) -> None:
        if self._callback is not None:
            self._callback()

    def __setitem__(self, key: str, value: Any) -> None:
        self._d[str(key)] = self._encode(str(key), value)
        self._changed()

    def __setattr__(self, key: str, value: Any) -> None:
        self.__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __getattr__(self, key: str) -> Any:
        if key.startswith("__"):
            raise AttributeError(key)
        try:
            return self._d[key]
        except KeyError:
            raise AttributeError(key) from None

    def __delitem__(self, key: str) -> None:
        del self._d[key]
        self._changed()

    __delattr__ = __delitem__

    def __contains__(self, key: object) -> bool:
        return key in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        return f"Summary({self._d!r})"

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def update(self, d: Mapping | None = None, **kwargs: Any) -> None:
        for k, v in dict(d or {}, **kwargs).items():
            self._d[str(k)] = self._encode(str(k), v)
        self._changed()

    def _as_dict(self) -> dict:
        return dict(self._d)

    def _update_raw(self, d: Mapping) -> None:
        """Update with values that are already serialised (no media conversion)."""
        self._d.update(d)
