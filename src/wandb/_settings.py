"""A permissive ``wandb.Settings`` replacement.

The real class validates hundreds of fields.  Here every keyword is accepted and
stored, unknown attributes read as ``None``, and only the handful of settings
that matter for local logging are interpreted by :mod:`wandb._run`.
"""

from __future__ import annotations

from typing import Any


class Settings:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def get(self, name: str, default: Any = None) -> Any:
        value = self.__dict__.get(name)
        return default if value is None else value

    def update(self, other: Any = None, **kwargs: Any) -> "Settings":
        if isinstance(other, Settings):
            self.__dict__.update({k: v for k, v in other.__dict__.items() if v is not None})
        elif isinstance(other, dict):
            self.__dict__.update({k: v for k, v in other.items() if v is not None})
        self.__dict__.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def model_copy(self) -> "Settings":
        return Settings(**self.__dict__)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"Settings({items})"
