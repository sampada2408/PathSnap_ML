"""
PathSnap Shared Contracts (FROZEN - Step 0)
===========================================
This module defines the shared contracts between all 4 sub-teams:
  P1: data_pipeline/
  P2: modeling/
  P3: decoder/
  P4: api/ ui/ viz/

RULE: Do not modify this file without consensus across all sub-teams.
"""

from __future__ import annotations

# Initialize OpenMP runtimes early on Windows
try:
    import lightgbm
except ImportError:
    pass

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

# ==============================================================================
# 0. Canonical Directory Layout
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure essential directories exist
for _p in [PROCESSED_DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


def df_to_markdown_safe(df: pd.DataFrame) -> str:
    """
    Format a pandas DataFrame as a Markdown table safely without requiring tabulate.
    """
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = list(df.columns)
        lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)


# ==============================================================================
# Contract 1: Candidate Table Schema (P1 -> P2, P3)
# ==============================================================================
# Key columns identifying the observation and candidate arc
KEY_COLUMNS = ["trip_id", "point_idx", "arc_id"]

# Ground-truth label (present in train & student_test; absent or ignored at test time)
LABEL_COLUMN = "label"

# Canonical feature list frozen for model consumption
# dist_m: perpendicular or nearest distance in meters from GPS observation to arc
# bearing_diff: absolute difference between GPS heading and candidate arc heading (degrees [0, 180])
# speed: estimated GPS speed (m/s)
# frac_along: normalized projection position along arc segment [0.0, 1.0]
# arc_len_m: length of candidate road segment in meters
# dist_rank: integer distance rank among candidates for this GPS point (0 = closest)
# delta_time_s: time gap from previous GPS point (seconds)
# perp_dist_m: perpendicular distance from GPS point to segment line (meters)
FEATURE_COLUMNS = [
    "dist_m",
    "bearing_diff",
    "speed",
    "frac_along",
    "arc_len_m",
    "dist_rank",
    "delta_time_s",
    "perp_dist_m",
]

# Minimal subset supported for v0 fast-checkpoint iterations
V0_FEATURE_COLUMNS = [
    "dist_m",
    "bearing_diff",
    "dist_rank",
]


def validate_candidate_table(
    df: pd.DataFrame,
    require_label: bool = False,
    feature_cols: Optional[List[str]] = None,
) -> bool:
    """
    Validates that a DataFrame strictly adheres to Contract 1.
    """
    expected_features = feature_cols if feature_cols is not None else FEATURE_COLUMNS
    missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing_keys:
        raise ValueError(f"Contract 1 Violation: Missing key columns {missing_keys}")

    if require_label and LABEL_COLUMN not in df.columns:
        raise ValueError(f"Contract 1 Violation: Missing required label column '{LABEL_COLUMN}'")

    missing_features = [c for c in expected_features if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Contract 1 Violation: Missing expected feature columns {missing_features}. "
            f"Present columns: {list(df.columns)}"
        )

    return True


# ==============================================================================
# Contract 2: The Model Interface (P2 -> P3, P4)
# ==============================================================================
# Saved Artifacts:
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURE_COLS_ARTIFACT_PATH = ARTIFACTS_DIR / "feature_columns.json"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


class ModelContract(Protocol):
    """
    Contract 2 Protocol:
    The trained model must provide predict_proba(X) -> np.ndarray.
    X is a DataFrame or 2D array with columns matching feature_columns.json.
    Output: 1D or 2D array of probability P(arc is correct).
    If 2D, column 1 is P(arc is correct).
    """

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        ...


def extract_positive_probs(probs: np.ndarray) -> np.ndarray:
    """
    Helper to normalize predict_proba output to 1D float array of P(positive class).
    Handles both 1D and 2D sklearn-style outputs.
    """
    probs = np.asarray(probs)
    if probs.ndim == 2:
        if probs.shape[1] >= 2:
            return probs[:, 1].astype(float)
        elif probs.shape[1] == 1:
            return probs[:, 0].astype(float)
    elif probs.ndim == 1:
        return probs.astype(float)
    raise ValueError(f"Unexpected probability array shape: {probs.shape}")


# ==============================================================================
# Contract 3: The Decoder Interface (P3 -> P4)
# ==============================================================================
# Signature:
#   decode(cands, probs, arcs, nodes, times) -> list[int]
# Input:
#   cands: DataFrame of candidate rows adhering to Contract 1 (with trip_id, point_idx, arc_id)
#   probs: 1D np.ndarray of emission probabilities matching len(cands)
#   arcs: 2D np.ndarray or DataFrame of road network arcs [from_node, to_node]
#   nodes: 2D np.ndarray or DataFrame of road network nodes [lon, lat]
#   times: 1D np.ndarray of GPS observation timestamps (seconds)
# Output:
#   list[int]: sequence of predicted arc_ids, exactly one arc_id per unique point_idx
DecoderFunction = Callable[
    [pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    List[int],
]


# ==============================================================================
# Contract 4: The Predictor Interface (P4 API Endpoint)
# ==============================================================================
# Signature:
#   predict_trip(trip_dir: Path | str) -> dict[str, Any]
# Returns:
#   {
#       "trip_id": str,
#       "points": [{"lon": float, "lat": float, "time": float}, ...],
#       "pred_arcs": [int, int, ...],
#       "point_count": int,
#       "pred_count": int
#   }
# STRICT RULE: Must NEVER read .route files at inference time.
PredictorFunction = Callable[[Union[Path, str]], Dict[str, Any]]
