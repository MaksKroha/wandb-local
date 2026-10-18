"""``wandb.plot`` — chart presets backed by a Table.

Each helper returns a :class:`CustomChart` that logs its data table plus a small
chart description (``{"_type": "plot", "kind": ...}``) rendered by the local viewer.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..data_types import Table, WBValue, _to_numpy


class CustomChart(WBValue):
    _log_type = "plot"

    def __init__(self, kind: str, table: Table, fields: dict, title: str | None = None, extra: dict | None = None) -> None:
        self.kind = kind
        self.table = table
        self.fields = fields
        self.title = title or ""
        self.extra = extra or {}

    def to_json(self, target: Any = None, key: str = "chart", step: Any = None) -> dict:
        ref = self.table.to_json(target, f"{key}_table", step)
        return {"_type": "plot", "kind": self.kind, "title": self.title, "fields": self.fields, "table": ref, **self.extra}


Visualize = CustomChart


def line(table: Table, x: str, y: str, stroke: str | None = None, title: str = "", split_table: bool = False) -> CustomChart:
    return CustomChart("line", table, {"x": x, "y": y, "stroke": stroke}, title)


def scatter(table: Table, x: str, y: str, title: str = "", split_table: bool = False) -> CustomChart:
    return CustomChart("scatter", table, {"x": x, "y": y}, title)


def bar(table: Table, label: str, value: str, title: str = "", split_table: bool = False) -> CustomChart:
    return CustomChart("bar", table, {"label": label, "value": value}, title)


def histogram(table: Table, value: str, title: str = "", split_table: bool = False) -> CustomChart:
    return CustomChart("histogram", table, {"value": value}, title)


def line_series(xs: Sequence, ys: Sequence[Sequence], keys: Sequence[str] | None = None, title: str = "", xname: str = "x", split_table: bool = False) -> CustomChart:
    ys = [list(y) for y in ys]
    if xs and not isinstance(xs[0], (list, tuple)) and not hasattr(xs[0], "__len__"):
        xs = [list(xs)] * len(ys)
    keys = list(keys) if keys is not None else [f"key_{i}" for i in range(len(ys))]
    table = Table(columns=["step", "lineKey", "lineVal"])
    for key, x_list, y_list in zip(keys, xs, ys):
        for xv, yv in zip(x_list, y_list):
            table.add_data(xv, key, yv)
    return CustomChart("line_series", table, {"x": "step", "series": "lineKey", "y": "lineVal", "xname": xname}, title)


def confusion_matrix(probs: Sequence | None = None, y_true: Sequence | None = None, preds: Sequence | None = None, class_names: Sequence[str] | None = None, title: str = "Confusion Matrix", split_table: bool = False) -> CustomChart:
    import numpy as np

    if preds is None:
        if probs is None:
            raise ValueError("confusion_matrix requires preds or probs")
        preds = np.argmax(_to_numpy(probs), axis=1).tolist()
    y_true = list(_to_numpy(y_true).tolist())
    preds = list(_to_numpy(preds).tolist())
    n = len(class_names) if class_names else int(max(max(y_true), max(preds))) + 1
    names = list(class_names) if class_names else [f"Class_{i}" for i in range(n)]
    counts = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, preds):
        counts[int(t)][int(p)] += 1
    table = Table(columns=["Actual", "Predicted", "nPredictions"])
    for i in range(n):
        for j in range(n):
            table.add_data(names[i], names[j], counts[i][j])
    return CustomChart("confusion_matrix", table, {"actual": "Actual", "predicted": "Predicted", "value": "nPredictions"}, title)


def _curves(y_true: Sequence, y_probas: Sequence, labels: Sequence[str] | None, classes_to_plot: Sequence | None, kind: str):
    import numpy as np

    y_true = _to_numpy(y_true).ravel()
    probas = _to_numpy(y_probas)
    if probas.ndim == 1:
        probas = np.stack([1 - probas, probas], axis=1)
    classes = np.unique(y_true) if classes_to_plot is None else np.asarray(classes_to_plot)
    rows = []
    for c in classes:
        idx = int(c) if np.issubdtype(np.asarray(c).dtype, np.number) else list(labels or []).index(c)
        scores = probas[:, idx]
        positive = (y_true == c).astype(np.float64)
        order = np.argsort(-scores, kind="mergesort")
        tp = np.cumsum(positive[order])
        fp = np.cumsum(1 - positive[order])
        total_p = max(positive.sum(), 1)
        total_n = max((1 - positive).sum(), 1)
        name = labels[idx] if labels is not None and idx < len(labels) else str(c)
        if kind == "pr":
            xs, ys = tp / total_p, tp / np.maximum(tp + fp, 1)
        else:
            xs, ys = np.concatenate([[0], fp / total_n]), np.concatenate([[0], tp / total_p])
        stride = max(1, len(xs) // 200)
        for x, y in list(zip(xs, ys))[::stride]:
            rows.append([name, float(x), float(y)])
    return rows


def pr_curve(y_true: Sequence, y_probas: Sequence, labels: Sequence[str] | None = None, classes_to_plot: Sequence | None = None, interp_size: int = 21, title: str = "Precision-Recall Curve", split_table: bool = False) -> CustomChart:
    table = Table(columns=["class", "recall", "precision"], data=_curves(y_true, y_probas, labels, classes_to_plot, "pr"))
    return CustomChart("line_series", table, {"x": "recall", "series": "class", "y": "precision"}, title)


def roc_curve(y_true: Sequence, y_probas: Sequence, labels: Sequence[str] | None = None, classes_to_plot: Sequence | None = None, title: str = "ROC Curve", split_table: bool = False) -> CustomChart:
    table = Table(columns=["class", "fpr", "tpr"], data=_curves(y_true, y_probas, labels, classes_to_plot, "roc"))
    return CustomChart("line_series", table, {"x": "fpr", "series": "class", "y": "tpr"}, title)


def plot_table(vega_spec_name: str, data_table: Table, fields: dict, string_fields: dict | None = None, split_table: bool = False) -> CustomChart:
    return CustomChart("custom", data_table, dict(fields), (string_fields or {}).get("title", vega_spec_name), {"vega_spec_name": vega_spec_name})


__all__ = ["CustomChart", "Visualize", "bar", "confusion_matrix", "histogram", "line", "line_series", "plot_table", "pr_curve", "roc_curve", "scatter"]
