"""
PathSnap Person 4 Predictor: Integrated Production Pipeline (Contracts 1-4)
===========================================================================
Executes the full end-to-end map-matching prediction pipeline:
1. P1: Candidate generation & spatial feature engineering (data_pipeline/)
2. P2: Positive arc emission probability scoring via trained ML model (artifacts/model.joblib)
3. P3: Sequence reconstruction via Viterbi trellis dynamic programming (decoder/viterbi.py)
4. Robust KDTree nearest-arc fallback ensuring zero failures on edge cases.

STRICT CONTRACT RULE: Must NEVER read .route files at inference time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common.contract import (
    ARTIFACTS_DIR,
    FEATURE_COLS_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    extract_positive_probs,
)
from data_pipeline.candidates import generate_candidates
from data_pipeline.features import add_features
from data_pipeline.io_utils import load_trip as p1_load_trip
from decoder.viterbi import decode as viterbi_decode_contract3

HMM_CONFIG_PATH = ARTIFACTS_DIR / "hmm_config.json"

log = logging.getLogger("pathsnap.predictor")

# Module-level cache to keep inference latency sub-second
_CACHE: Dict[str, Any] = {
    "model": None,
    "feature_columns": None,
    "hmm_config": None,
}


def _get_artifacts() -> tuple[Any, List[str], Dict[str, Any]]:
    """Loads and caches P2 model artifact and P3 HMM config."""
    if _CACHE["model"] is None and MODEL_ARTIFACT_PATH.exists():
        try:
            _CACHE["model"] = joblib.load(MODEL_ARTIFACT_PATH)
        except Exception as e:
            log.warning("Failed to load model artifact: %s", e)

    if _CACHE["feature_columns"] is None and FEATURE_COLS_ARTIFACT_PATH.exists():
        try:
            with open(FEATURE_COLS_ARTIFACT_PATH, "r", encoding="utf-8") as f:
                _CACHE["feature_columns"] = json.load(f)
        except Exception as e:
            log.warning("Failed to load feature columns: %s", e)

    if _CACHE["hmm_config"] is None and HMM_CONFIG_PATH.exists():
        try:
            with open(HMM_CONFIG_PATH, "r", encoding="utf-8") as f:
                _CACHE["hmm_config"] = json.load(f)
        except Exception as e:
            log.warning("Failed to load HMM config: %s", e)

    return _CACHE["model"], _CACHE["feature_columns"] or [], _CACHE["hmm_config"] or {}


def _fallback_nearest_kdtree(
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    k: int = 15,
    sigma: float = 20.0,
    break_pen: float = 12.0,
    step_pen: float = 1.0,
    uturn_pen: float = 6.0,
) -> List[int]:
    """Fast, self-contained fallback using KDTree nearest arcs and topological Viterbi."""
    lat0 = track[:, 1].mean()
    scale = np.array([111320.0 * np.cos(np.radians(lat0)), 110540.0])
    p_proj = track[:, :2] * scale
    n_proj = nodes * scale
    a_proj, b_proj = n_proj[arcs[:, 0]], n_proj[arcs[:, 1]]

    tree = cKDTree((a_proj + b_proj) / 2.0)
    k_eff = min(k, len(arcs))
    _, idx = tree.query(p_proj, k=k_eff)
    idx = idx.reshape(len(p_proj), k_eff)

    a_sub, b_sub = a_proj[idx], b_proj[idx]
    ab = b_sub - a_sub
    ap = p_proj[:, None, :] - a_sub
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9), 0, 1)
    d = np.linalg.norm(p_proj[:, None, :] - (a_sub + t[..., None] * ab), axis=-1)
    emit = -(d ** 2) / (2 * sigma ** 2)

    src, dst = arcs[:, 0], arcs[:, 1]
    n = len(p_proj)
    score = emit[0].copy()
    back = np.zeros((n, k_eff), dtype=np.int32)
    for i in range(1, n):
        p, c = idx[i - 1], idx[i]
        same = p[:, None] == c[None, :]
        succ = dst[p][:, None] == src[c][None, :]
        rev = (dst[c][:, None] == src[p][None, :]).T & (src[c][None, :] == dst[p][:, None]) & ~same
        trans = np.full((k_eff, k_eff), -break_pen)
        trans[rev] = -uturn_pen
        trans[succ] = -step_pen
        trans[same] = 0.0
        tot = score[:, None] + trans
        back[i] = tot.argmax(0)
        score = tot.max(0) + emit[i]

    path = np.empty(n, dtype=np.int64)
    j = int(score.argmax())
    for i in range(n - 1, -1, -1):
        path[i] = idx[i, j]
        j = back[i, j]
    return [int(x) for x in path]


def predict_trip(trip_dir: Path | str) -> Dict[str, Any]:
    """
    Contract 4 Predictor Function:
    predict_trip(trip_dir) -> {"pred_arcs": list[int], ...}
    
    Accepts path to a trip folder. Never reads .route files.
    """
    p = Path(trip_dir)
    try:
        # 1. Load raw trip observations & projection (P1)
        tr = p1_load_trip(p, with_route=False)
        track, nodes, arcs = tr["track"], tr["nodes"], tr["arcs"]
        n_points = len(track)

        # 2. Check artifacts availability
        model, feature_cols, hmm_cfg = _get_artifacts()

        if model is None or not feature_cols:
            log.warning("Artifacts missing, using KDTree topological fallback")
            pred_arcs = _fallback_nearest_kdtree(track, nodes, arcs)
            return {"pred_arcs": pred_arcs, "point_count": n_points, "pred_count": len(pred_arcs)}

        # 3. Generate candidate arcs within spatial radius (P1)
        cands_dict = generate_candidates(tr, radius_m=60.0, k=15)
        if len(cands_dict["point_idx"]) == 0:
            log.warning("No candidates found within 60m, using fallback")
            pred_arcs = _fallback_nearest_kdtree(track, nodes, arcs)
            return {"pred_arcs": pred_arcs, "point_count": n_points, "pred_count": len(pred_arcs)}

        # 4. Compute kinematic & spatial features (P1)
        df_cands = add_features(cands_dict, tr)

        # Ensure required feature column aliases exist
        if "delta_time_s" not in df_cands.columns and "dt_prev" in df_cands.columns:
            df_cands["delta_time_s"] = df_cands["dt_prev"]
        if "perp_dist_m" not in df_cands.columns and "dist_m" in df_cands.columns and "bearing_diff" in df_cands.columns:
            df_cands["perp_dist_m"] = df_cands["dist_m"] * np.sin(np.radians(df_cands["bearing_diff"]))

        # Fill any missing feature columns with default zeros
        for col in feature_cols:
            if col not in df_cands.columns:
                df_cands[col] = 0.0

        # 5. Score candidate emission probabilities with ML model (P2)
        X = df_cands[feature_cols]
        raw_probs = model.predict_proba(X)
        probs = extract_positive_probs(raw_probs)

        # 6. Reconstruct sequence with Viterbi Trellis Decoder (P3)
        # Note: if any GPS point had 0 candidates within 60m, fill missing points
        unique_points = set(df_cands["point_idx"].unique())
        if len(unique_points) < n_points:
            # Fallback for completely missing points: insert nearest arc for uncovered points
            missing_pts = [i for i in range(n_points) if i not in unique_points]
            fallback_arcs = _fallback_nearest_kdtree(track, nodes, arcs, k=1)
            new_rows = []
            for m_pt in missing_pts:
                new_rows.append({
                    "point_idx": m_pt,
                    "arc_id": fallback_arcs[m_pt],
                    "dist_m": 10.0,
                    "frac_along": 0.5,
                    "dist_rank": 0,
                    "prob": 0.5,
                })
            df_missing = pd.DataFrame(new_rows)
            df_cands = pd.concat([df_cands, df_missing], ignore_index=True)
            # Recompute probs slice
            probs = df_cands.get("prob", pd.Series(probs)).values

        pred_arcs = viterbi_decode_contract3(
            cands=df_cands,
            probs=probs,
            arcs=arcs,
            nodes=nodes,
            times=tr.get("t"),
            decoder_config=hmm_cfg,
        )

        # Ensure output length matches number of GPS points
        if len(pred_arcs) != n_points:
            log.warning("Viterbi output length mismatch (%d vs %d), using fallback", len(pred_arcs), n_points)
            pred_arcs = _fallback_nearest_kdtree(track, nodes, arcs)

        return {
            "pred_arcs": [int(a) for a in pred_arcs],
            "point_count": n_points,
            "pred_count": len(pred_arcs),
        }

    except Exception as e:
        log.error("Prediction pipeline error: %s, using safe KDTree fallback", e, exc_info=True)
        track, nodes, arcs = p1_load_trip(p, with_route=False)
        pred_arcs = _fallback_nearest_kdtree(track, nodes, arcs)
        return {
            "pred_arcs": pred_arcs,
            "point_count": len(track),
            "pred_count": len(pred_arcs),
        }
