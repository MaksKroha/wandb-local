"""``wandb.config`` — a dict-like object with attribute access and change checks."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Callable

from .sdk.lib.config_util import ConfigError
from .util import to_jsonable

logger = logging.getLogger("wandb")


def _as_dict(d: Any) -> dict:
    if d is None:
        return {}
    if isinstance(d, Config):
        return d.as_dict()
    if isinstance(d, Mapping):
        return dict(d)
    if hasattr(d, "__dict__"):  # argparse.Namespace, absl flags, simple objects
        return {k: v for k, v in vars(d).items() if not k.startswith("_")}
    raise TypeError(f"Cannot convert {type(d).__name__} to a config dictionary")


class Config:
    """Mirror of the real ``wandb.sdk.wandb_config.Config`` semantics.

    * Values are normalised to JSON-friendly types.
    * Changing an existing key without ``allow_val_change=True`` raises ``ConfigError``.
    * Keys locked by a sweep cannot be overridden by user code (the update is ignored).
    * ``vars(config)["_items"]`` exposes the raw dict (transformers' sweep integration relies on it).
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_items", {})
        object.__setattr__(self, "_locked", {})
        object.__setattr__(self, "_callback", None)
        object.__setattr__(self, "_warned_locked", set())

    # -- internals ---------------------------------------------------------
    def _set_callback(self, callback: Callable[[], None] | None) -> None:
        object.__setattr__(self, "_callback", callback)

    def _changed(self) -> None:
        cb = self._callback
        if cb is not None:
            cb()

    def _sanitize(self, key: Any, val: Any, allow_val_change: bool | None = None):
        from .data_types import WBValue

        if isinstance(val, WBValue):
            raise TypeError("WBValue objects cannot be added to the run config")
        key = str(key).strip("-")
        from .artifacts import Artifact

        if isinstance(val, Artifact):
            val = {"_type": "artifactVersion", "name": val.qualified_name}
        else:
            val = to_jsonable(val)
        if not allow_val_change and key in self._items and val != self._items[key]:
            raise ConfigError(
                f'Attempted to change value of key "{key}" from {self._items[key]} to {val}\n'
                "If you really want to do this, pass allow_val_change=True to config.update()"
            )
        return key, val

    def _check_locked(self, key: str, ignore_locked: bool = False) -> bool:
        if key in self._locked and not ignore_locked:
            if key not in self._warned_locked:
                self._warned_locked.add(key)
                logger.warning("Config item '%s' was locked by '%s' (ignored update).", key, self._locked[key])
            return True
        return False

    # -- mapping API -------------------------------------------------------
    def __setitem__(self, key: str, val: Any) -> None:
        if self._check_locked(str(key)):
            return
        key, val = self._sanitize(key, val)
        self._items[key] = val
        self._changed()

    def __setattr__(self, key: str, val: Any) -> None:
        self.__setitem__(key, val)

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __getattr__(self, key: str) -> Any:
        if key.startswith("__"):
            raise AttributeError(key)
        try:
            return self._items[key]
        except KeyError:
            raise AttributeError(f"{type(self)!r} object has no attribute {key!r}") from None

    def __delitem__(self, key: str) -> None:
        del self._items[key]
        self._changed()

    __delattr__ = __delitem__

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Config({self._items!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._items == other._items
        if isinstance(other, Mapping):
            return self._items == dict(other)
        return NotImplemented

    def keys(self):
        return [k for k in self._items if not k.startswith("_wandb")]

    def values(self):
        return [self._items[k] for k in self.keys()]

    def items(self):
        return [(k, self._items[k]) for k in self.keys()]

    def get(self, *args: Any) -> Any:
        return self._items.get(*args)

    def as_dict(self) -> dict:
        return dict(self._items)

    _as_dict = as_dict

    def update(self, d: Any, allow_val_change: bool | None = None) -> None:
        changed = False
        for k, v in _as_dict(d).items():
            if self._check_locked(str(k)):
                continue
            k, v = self._sanitize(k, v, allow_val_change)
            self._items[k] = v
            changed = True
        if changed:
            self._changed()

    def setdefaults(self, d: Any) -> None:
        changed = False
        for k, v in _as_dict(d).items():
            k = str(k).strip("-")
            if k not in self._items:
                self._items[k] = to_jsonable(v)
                changed = True
        if changed:
            self._changed()

    def update_locked(self, d: Any, user: str = "sweep", _allow_val_change: bool | None = True) -> None:
        for k, v in _as_dict(d).items():
            k, v = self._sanitize(k, v, allow_val_change=True)
            self._items[k] = v
            self._locked[k] = user
        self._changed()

    merge_locked = update_locked

    def persist(self) -> None:
        self._changed()

    def _load(self, d: Mapping) -> None:
        self._items.update(d)
