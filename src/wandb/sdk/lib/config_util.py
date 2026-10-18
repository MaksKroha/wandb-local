from wandb.errors import Error


class ConfigError(Error):
    """Raised when a config value is changed without ``allow_val_change=True``."""
