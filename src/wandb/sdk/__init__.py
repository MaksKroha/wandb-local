"""Compatibility package mirroring import paths of the real ``wandb.sdk``.

Attributes are resolved lazily to avoid import cycles with ``wandb._config``.
"""


def __getattr__(name):
    if name == "Config":
        from .._config import Config

        return Config
    if name in ("Run", "init"):
        from .. import _run

        return getattr(_run, name)
    if name == "Settings":
        from .._settings import Settings

        return Settings
    if name == "Artifact":
        from ..artifacts import Artifact

        return Artifact
    import importlib

    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(name) from None
