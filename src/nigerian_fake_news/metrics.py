from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: list[int] | np.ndarray, scores: list[float] | np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    y_pred = (score_arr >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    try:
        auc: float | None = float(roc_auc_score(y_true_arr, score_arr))
    except ValueError:
        auc = None
    return {
        "n": int(y_true_arr.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision_misleading": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall_misleading": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1_misleading": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred)) if y_true_arr.size else 0.0,
        "roc_auc": auc,
        "confusion_matrix_labels_0_credible_1_misleading": cm.tolist(),
        "classification_report": classification_report(
            y_true_arr,
            y_pred,
            labels=[0, 1],
            target_names=["credible", "misleading"],
            output_dict=True,
            zero_division=0,
        ),
    }
