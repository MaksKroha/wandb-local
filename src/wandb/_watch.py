"""``wandb.watch`` — gradient / parameter histograms for torch models."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("wandb")
_BINS = 64


def _histogram(tensor: Any) -> dict | None:
    import torch

    t = tensor.detach()
    if t.is_sparse:
        t = t.coalesce().values()
    t = t.float().flatten()
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return None
    lo, hi = t.min().item(), t.max().item()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    counts = torch.histc(t, bins=_BINS, min=lo, max=hi).cpu().tolist()
    step = (hi - lo) / _BINS
    return {"_type": "histogram", "values": counts, "bins": [lo + i * step for i in range(_BINS + 1)]}


def watch(run: Any, models: Any, criterion: Any = None, log: str | None = "gradients", log_freq: int = 1000, idx: int | None = None, log_graph: bool = False) -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError("wandb.watch requires torch") from exc
    if log not in ("gradients", "parameters", "all", None):
        raise ValueError("log must be one of 'gradients', 'parameters', 'all', or None")
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    log_freq = max(1, int(log_freq or 1))
    for i, model in enumerate(models):
        if not hasattr(model, "named_parameters"):
            raise ValueError("Expected a torch.nn.Module")
        if id(model) in run._watch_hooks:
            continue
        prefix = "" if len(models) == 1 and idx is None else f"graph_{idx if idx is not None else i}."
        state = {"count": 0}
        hooks: list[Any] = []

        def forward_hook(module, _inp, _out, _state=state, _model=model, _prefix=prefix):
            if not module.training:
                return
            _state["count"] += 1
            if log in ("parameters", "all") and _state["count"] % log_freq == 0:
                for name, p in _model.named_parameters():
                    hist = _histogram(p)
                    if hist is not None:
                        run._add_pending(f"parameters/{_prefix}{name}", hist)

        hooks.append(model.register_forward_hook(forward_hook))
        if log in ("gradients", "all"):
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue

                def grad_hook(grad, _name=name, _state=state, _prefix=prefix):
                    if _state["count"] and _state["count"] % log_freq == 0:
                        try:
                            hist = _histogram(grad)
                        except Exception:  # pragma: no cover - defensive
                            hist = None
                        if hist is not None:
                            run._add_pending(f"gradients/{_prefix}{_name}", hist)
                    return None

                hooks.append(p.register_hook(grad_hook))
        run._watch_hooks[id(model)] = hooks
        if log_graph:
            try:
                (run.files_dir / f"model_graph_{i}.txt").write_text(str(model), encoding="utf-8")
            except Exception:  # pragma: no cover - defensive
                logger.debug("could not save model graph", exc_info=True)


def unwatch(run: Any, models: Any = None) -> None:
    if models is None:
        keys = list(run._watch_hooks)
    else:
        models = models if isinstance(models, (list, tuple)) else [models]
        keys = [id(m) for m in models]
    for key in keys:
        for hook in run._watch_hooks.pop(key, []):
            try:
                hook.remove()
            except Exception:
                pass
