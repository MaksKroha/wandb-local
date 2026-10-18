def __getattr__(name):
    if name in ("RunDisabled", "SummaryDisabled"):
        from . import disabled

        return getattr(disabled, name)
    import importlib

    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(name) from None
