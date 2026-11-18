"""Server-side evaluation. This is the ONLY module that reads the reference .route file.

Ground truth stays on the server: the UI receives metrics only, never the reference route,
and the predictor never imports this module.

INTERIM metrics (arc-set overlap). Replace/extend with P3's shared metrics module
(pointwise accuracy, route continuity, sequence-level performance) once it exists.
"""
from pathlib import Path

import numpy as np


def evaluate(trip_dir, pred_arcs):
    trip_dir = Path(trip_dir)
    ref = np.loadtxt(trip_dir / f"{trip_dir.name}.route", dtype=int, ndmin=1)
    ref_set = set(int(a) for a in ref)
    pred_set = set(int(a) for a in pred_arcs if a >= 0)
    tp = len(ref_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "route_precision": round(precision, 4),
        "route_recall": round(recall, 4),
        "route_f1": round(f1, 4),
        "reference_arcs": len(ref_set),
        "predicted_arcs": len(pred_set),
    }
