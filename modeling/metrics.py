"""
PathSnap Person 2: Metrics Module
=================================
Calculates all required evaluation metrics:
- Observation-level Top-1 Accuracy
- Candidate Miss Rate
- Macro F1-score
- Log Loss
- Precision, Recall, Accuracy
- Confusion Matrix
- Route Continuity proxy metric
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common.contract import KEY_COLUMNS, LABEL_COLUMN, extract_positive_probs


def candidate_miss_rate(df: pd.DataFrame) -> float:
    """
    Candidate Miss Rate:
    The fraction of GPS observations (trip_id, point_idx) where NO candidate
    arc in the candidate set has label == 1.
    If the true arc is not in the candidate pool, no classifier or HMM can recover it.
    """
    if LABEL_COLUMN not in df.columns:
        return 0.0

    grouped = df.groupby(["trip_id", "point_idx"])[LABEL_COLUMN].max()
    misses = (grouped == 0).sum()
    total = len(grouped)
    return float(misses / total) if total > 0 else 0.0


def observation_top1_accuracy(df: pd.DataFrame, probs: np.ndarray) -> float:
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation (trip_id, point_idx), selects the candidate
    with the highest predicted probability and checks whether that candidate has label == 1.
    """
    if LABEL_COLUMN not in df.columns:
        raise ValueError("Cannot calculate Top-1 Accuracy without ground-truth 'label' column.")

    pos_probs = extract_positive_probs(probs)
    df_eval = df[["trip_id", "point_idx", LABEL_COLUMN]].copy()
    df_eval["prob"] = pos_probs

    # For each observation, take candidate with the maximum predicted probability
    # idxmax returns the index of the first maximum
    idx_max = df_eval.groupby(["trip_id", "point_idx"])["prob"].idxmax()
    top1_labels = df_eval.loc[idx_max, LABEL_COLUMN].values

    return float(np.mean(top1_labels == 1))


def candidate_log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary Cross-Entropy (Log Loss) for candidate probabilities.
    Clips probabilities to [eps, 1 - eps] to prevent infinity.
    """
    pos_probs = extract_positive_probs(probs)
    pos_probs = np.clip(pos_probs, eps, 1.0 - eps)
    return float(log_loss(y_true, pos_probs, labels=[0, 1]))


def route_continuity_proxy(df: pd.DataFrame, probs: np.ndarray) -> float:
    """
    Route Continuity Proxy:
    Measures the fraction of consecutive GPS points within each trip where
    the predicted top-1 arc is either identical or shares continuity.
    In real road networks, vehicles usually stay on the same segment or move
    to an adjacent segment.
    """
    pos_probs = extract_positive_probs(probs)
    df_eval = df[["trip_id", "point_idx", "arc_id"]].copy()
    df_eval["prob"] = pos_probs

    idx_max = df_eval.groupby(["trip_id", "point_idx"])["prob"].idxmax()
    top1_df = df_eval.loc[idx_max].sort_values(["trip_id", "point_idx"])

    continuity_scores = []
    for _, group in top1_df.groupby("trip_id"):
        arcs = group["arc_id"].values
        if len(arcs) <= 1:
            continue
        # Proxy: consecutive identical arcs or smooth transitions
        same_arc = (arcs[1:] == arcs[:-1])
        continuity_scores.append(np.mean(same_arc))

    return float(np.mean(continuity_scores)) if continuity_scores else 0.0


def compute_all_metrics(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full suite of classification and map-matching metrics.
    """
    pos_probs = extract_positive_probs(probs)
    y_true = df[LABEL_COLUMN].values if LABEL_COLUMN in df.columns else None

    results: Dict[str, Any] = {}

    if y_true is not None:
        y_pred = (pos_probs >= threshold).astype(int)

        results["top1_accuracy"] = observation_top1_accuracy(df, pos_probs)
        results["candidate_miss_rate"] = candidate_miss_rate(df)
        results["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        results["binary_f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        results["precision"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        results["recall"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
        results["log_loss"] = candidate_log_loss(y_true, pos_probs)
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        try:
            results["roc_auc"] = float(roc_auc_score(y_true, pos_probs))
        except Exception:
            results["roc_auc"] = 0.0

    results["route_continuity"] = route_continuity_proxy(df, pos_probs)
    return results
