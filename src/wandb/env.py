"""Environment variable names and helpers (subset of ``wandb.env``)."""

from __future__ import annotations

import os

PROJECT = "WANDB_PROJECT"
ENTITY = "WANDB_ENTITY"
DIR = "WANDB_DIR"
NAME = "WANDB_NAME"
NOTES = "WANDB_NOTES"
TAGS = "WANDB_TAGS"
RUN_ID = "WANDB_RUN_ID"
RUN_GROUP = "WANDB_RUN_GROUP"
JOB_TYPE = "WANDB_JOB_TYPE"
MODE = "WANDB_MODE"
DISABLED = "WANDB_DISABLED"
RESUME = "WANDB_RESUME"
SILENT = "WANDB_SILENT"
QUIET = "WANDB_QUIET"
CONSOLE = "WANDB_CONSOLE"
SAVE_CODE = "WANDB_SAVE_CODE"
SWEEP_ID = "WANDB_SWEEP_ID"
SWEEP_PARAM_PATH = "WANDB_SWEEP_PARAM_PATH"
CONFIG_PATHS = "WANDB_CONFIG_PATHS"
API_KEY = "WANDB_API_KEY"
BASE_URL = "WANDB_BASE_URL"
CACHE_DIR = "WANDB_CACHE_DIR"
ARTIFACT_DIR = "WANDB_ARTIFACT_DIR"
# wandb-local specific knobs
LOCAL_ARTIFACTS_DIR = "WANDB_LOCAL_ARTIFACTS_DIR"
LOCAL_ARTIFACT_MODE = "WANDB_LOCAL_ARTIFACT_MODE"  # copy | hardlink | symlink
LOCAL_STATS_INTERVAL = "WANDB_LOCAL_STATS_INTERVAL"
LOCAL_DISABLE_STATS = "WANDB_LOCAL_DISABLE_STATS"

_TRUE = {"1", "true", "yes", "on", "y", "t"}


def truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUE


def get(name: str, default: str | None = None, env=None) -> str | None:
    env = os.environ if env is None else env
    value = env.get(name)
    return default if value in (None, "") else value


def get_project(default=None, env=None):
    return get(PROJECT, default, env)


def get_entity(default=None, env=None):
    return get(ENTITY, default, env)


def get_dir(default=None, env=None):
    return get(DIR, default, env)


def get_run(default=None, env=None):
    return get(RUN_ID, default, env)


def set_project(value: str | None, env=None) -> None:
    env = os.environ if env is None else env
    if value:
        env[PROJECT] = value


def set_entity(value: str | None, env=None) -> None:
    env = os.environ if env is None else env
    if value:
        env[ENTITY] = value


def is_disabled(env=None) -> bool:
    return truthy(get(DISABLED, None, env)) or (get(MODE, "", env) or "").lower() == "disabled"
