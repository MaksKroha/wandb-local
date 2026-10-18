"""wandb-local — an offline, file-based drop-in replacement for the ``wandb`` client.

Everything is written below ``<dir>/wandb/`` (``WANDB_DIR`` or the current
directory). No account, network access or server is needed. Browse runs with::

    wandb ui            # local dashboard at http://127.0.0.1:8080
    wandb report -o report.html   # self-contained HTML report

The public API mirrors the parts of the real client used by transformers,
TRL, accelerate, PEFT and typical training scripts.
"""

from __future__ import annotations

import enum as _enum
from typing import Any

from . import env, errors, util
from ._run import (
    Run,
    _PreInitObject,
    init,
    restore,
    termerror,
    termlog,
    termsetup,
    termwarn,
)
from ._settings import Settings
from ._sweeps import agent, controller, sweep
from ._version import LOCAL_VERSION as __local_version__
from ._version import __version__
from .artifacts import Artifact
from .data_types import (
    Audio,
    Classes,
    Graph,
    Histogram,
    Html,
    Image,
    JoinedTable,
    Molecule,
    Object3D,
    Plotly,
    Table,
    Video,
    WBValue,
)
from .errors import Error
from .apis.public import Api

run: Run | None = None
config: Any = _PreInitObject("wandb.config")
summary: Any = _PreInitObject("wandb.summary")
setup_called = False


class AlertLevel(str, _enum.Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


def _require_run(fn: str) -> Run:
    if run is None:
        raise Error(f"You must call wandb.init() before wandb.{fn}()")
    return run


def log(data: dict, step: int | None = None, commit: bool | None = None, sync: bool | None = None) -> None:
    _require_run("log").log(data, step=step, commit=commit, sync=sync)


def finish(exit_code: int | None = None, quiet: bool | None = None) -> None:
    if run is not None:
        run.finish(exit_code=exit_code, quiet=quiet)


def define_metric(name: str, step_metric: Any = None, step_sync: bool | None = None, hidden: bool | None = None, summary: str | None = None, goal: str | None = None, overwrite: bool | None = None, **kwargs: Any):
    return _require_run("define_metric").define_metric(name, step_metric, step_sync, hidden, summary, goal, overwrite, **kwargs)


def save(glob_str: Any = None, base_path: str | None = None, policy: str = "live"):
    return _require_run("save").save(glob_str, base_path, policy)


def watch(models: Any, criterion: Any = None, log: str | None = "gradients", log_freq: int = 1000, idx: int | None = None, log_graph: bool = False):
    return _require_run("watch").watch(models, criterion, log, log_freq, idx, log_graph)


def unwatch(models: Any = None) -> None:
    if run is not None:
        run.unwatch(models)


def log_artifact(artifact_or_path: Any, name: str | None = None, type: str | None = None, aliases: list | None = None, tags: list | None = None):
    return _require_run("log_artifact").log_artifact(artifact_or_path, name, type, aliases, tags)


def use_artifact(artifact_or_name: Any, type: str | None = None, aliases: list | None = None, use_as: str | None = None):
    if run is not None:
        return run.use_artifact(artifact_or_name, type, aliases, use_as)
    return Api().artifact(str(artifact_or_name), type=type)


def log_model(path: str, name: str | None = None, aliases: list | None = None):
    return _require_run("log_model").log_model(path, name, aliases)


def use_model(name: str) -> str:
    return _require_run("use_model").use_model(name)


def link_model(path: str, registered_model_name: str, name: str | None = None, aliases: list | None = None):
    return _require_run("link_model").link_model(path, registered_model_name, name, aliases)


def alert(title: str, text: str, level: Any = None, wait_duration: Any = None) -> None:
    _require_run("alert").alert(title, text, level, wait_duration)


def mark_preempting() -> None:
    _require_run("mark_preempting").mark_preempting()


def login(anonymous: str | None = None, key: str | None = None, relogin: bool | None = None, host: str | None = None, force: bool | None = None, timeout: int | None = None, verify: bool = False, **kwargs: Any) -> bool:
    """No-op: wandb-local never talks to a server."""
    termlog("wandb-local: running fully offline; no login required.", repeat=False)
    return True


def setup(settings: Any = None) -> Any:
    global setup_called
    setup_called = True
    return None


def teardown(exit_code: int | None = None) -> None:
    for r in list(_run._active_runs):
        r.finish(exit_code=exit_code)


def require(requirement: Any = None, experiment: Any = None) -> None:
    return None


def attach(attach_id: str | None = None, run_id: str | None = None, **kwargs: Any) -> Run:
    if run is not None and (run_id in (None, run.id)):
        return run
    raise errors.UsageError("wandb.attach() is not supported by wandb-local")


def __getattr__(name: str) -> Any:
    # Lazily expose heavier submodules (``wandb.plot``, ``wandb.apis``, ``wandb.sdk``).
    import importlib

    if name in ("plot", "apis", "sdk", "viewer", "wandb_run"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'wandb' has no attribute {name!r}")


from . import _run  # noqa: E402  (needed by teardown)

__all__ = [
    "AlertLevel",
    "Api",
    "Artifact",
    "Audio",
    "Error",
    "Graph",
    "Histogram",
    "Html",
    "Image",
    "Molecule",
    "Object3D",
    "Plotly",
    "Run",
    "Settings",
    "Table",
    "Video",
    "agent",
    "alert",
    "config",
    "controller",
    "define_metric",
    "finish",
    "init",
    "log",
    "log_artifact",
    "log_model",
    "login",
    "restore",
    "run",
    "save",
    "setup",
    "summary",
    "sweep",
    "termerror",
    "termlog",
    "termwarn",
    "unwatch",
    "use_artifact",
    "use_model",
    "watch",
]
