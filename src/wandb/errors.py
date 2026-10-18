"""Exception types mirroring ``wandb.errors``."""


class Error(Exception):
    """Base class for wandb-local errors."""

    def __init__(self, message="", context=None):
        super().__init__(message)
        self.message = message
        self.context = context


class CommError(Error):
    """Raised where the real client would fail to talk to the server."""

    def __init__(self, msg="", exc=None):
        super().__init__(msg)
        self.exc = exc


class AuthenticationError(CommError):
    pass


class UsageError(Error):
    """Raised when the API is used incorrectly."""


class UnsupportedError(UsageError):
    """Raised when a feature has no local equivalent."""


class WandbCoreNotAvailableError(Error):
    pass
