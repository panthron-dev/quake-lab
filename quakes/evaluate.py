"""Part 2, Task C: metrics, threshold search, cross-validation."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score

_THRESH_METRICS = {"f1": f1_score, "precision": precision_score, "recall": recall_score}


def regression_report(y_true, y_pred) -> dict:
    """Return {'mae': ..., 'rmse': ..., 'r2': ...} as plain floats."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # RMSE punishes big misses more than MAE (errors are squared first).
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_report_dict(y_true, y_pred, y_proba) -> dict:
    """Return accuracy, precision, recall, f1, roc_auc as plain floats.
    Use zero_division=0 so a model that predicts no positives does not crash."""
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:  # only one class present in y_true
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # AUC uses the probabilities, not the 0/1 predictions: it measures
        # ranking quality across ALL thresholds at once.
        "roc_auc": auc,
    }


def best_threshold(y_true, y_proba, metric: str = "f1") -> float:
    """Try thresholds 0.05, 0.10, ... 0.95 (predict 1 when proba >= t) and
    return the one with the highest score for `metric` (f1, precision, recall)."""
    score_fn = _THRESH_METRICS[metric]
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    scores = [score_fn(y_true, (y_proba >= t).astype(int), zero_division=0)
              for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])  # first best on ties


def cross_validate_model(pipe, X, y, scoring: str, cv: int = 5) -> dict:
    """Return {'mean': ..., 'std': ..., 'scores': [...]} using cross_val_score."""
    scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv)
    return {"mean": float(scores.mean()), "std": float(scores.std()),
            "scores": [float(s) for s in scores]}
