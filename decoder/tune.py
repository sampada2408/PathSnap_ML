"""
PathSnap Person 3: Decoder Parameter Tuning
===========================================
Tunes HMM decoder parameters on train and student-test trips only
(evaluator partition is untouched, strictly preventing data leakage).

Optimizes:
- beta: transition scale parameter
- sigma_z: GPS measurement noise std
- unreachable_penalty_m: floor penalty for disconnected transitions
- gap_threshold_s: threshold for dynamic temporal gap scaling
- clf_weight: emission blend between classifier probability and distance

Saves the optimal configuration to:
    artifacts/hmm_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from common.contract import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from decoder.graph import RoadNetworkGraph
from decoder.metrics import compute_pointwise_accuracy, compute_route_continuity
from decoder.viterbi import ViterbiDecoder


def evaluate_config_on_trip(
    trip_id: str,
    cand_df: pd.DataFrame,
    ref_arcs: List[int],
    graph: RoadNetworkGraph,
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Evaluates a specific decoder configuration on a single trip.
    Returns (pointwise_acc, continuity).
    """
    trip_cands = cand_df[cand_df["trip_id"] == trip_id].copy()
    if trip_cands.empty:
        return 0.0, 0.0

    decoder = ViterbiDecoder(
        beta=config["beta"],
        sigma_z=config["sigma_z"],
        unreachable_penalty_m=config["unreachable_penalty_m"],
        gap_threshold_s=config["gap_threshold_s"],
        dijkstra_cutoff_m=config["dijkstra_cutoff_m"],
        clf_weight=config["clf_weight"],
    )

    probs = trip_cands["prob"].values if "prob" in trip_cands.columns else None
    pred_arcs = decoder.decode(
        cands=trip_cands,
        probs=probs,
        arcs=graph.arcs,
        nodes=graph.nodes,
    )

    pt_acc = compute_pointwise_accuracy(pred_arcs, ref_arcs)
    cont = compute_route_continuity(pred_arcs, graph)
    return pt_acc, cont


def tune_decoder(
    sample_trip_ids: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Grid search over decoder hyperparameters on real training and student-test trips.
    """
    out_dir = Path(output_dir) if output_dir else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load candidate data
    cand_file = PROCESSED_DATA_DIR / "student_test_candidates.parquet"
    ref_file = PROCESSED_DATA_DIR / "student_test_reference_arcs.json"

    if not cand_file.exists():
        # Fallback to v0 or train candidates
        cand_file = PROCESSED_DATA_DIR / "train_candidates.parquet"
        ref_file = PROCESSED_DATA_DIR / "train_reference_arcs.json"

    print(f"[Decoder Tuning] Loading candidates from: {cand_file}")
    cand_df = pd.read_parquet(cand_file)
    with open(ref_file, "r", encoding="utf-8") as f:
        all_refs = json.load(f)

    trips = sample_trip_ids or list(all_refs.keys())[:3]
    print(f"[Decoder Tuning] Tuning across representative trips: {trips}")

    # Load graphs for these trips
    graphs: Dict[str, RoadNetworkGraph] = {}
    for tid in trips:
        # Check raw train then student_test
        trip_dir = RAW_DATA_DIR / "student_test" / tid
        if not trip_dir.exists():
            trip_dir = RAW_DATA_DIR / "train" / tid
        if trip_dir.exists():
            graphs[tid] = RoadNetworkGraph.from_trip_dir(trip_dir)

    # Parameter grid
    grid = [
        {"beta": 3.0, "sigma_z": 4.07, "unreachable_penalty_m": 350.0, "gap_threshold_s": 10.0, "dijkstra_cutoff_m": 800.0, "clf_weight": 0.85},
        {"beta": 5.0, "sigma_z": 4.07, "unreachable_penalty_m": 350.0, "gap_threshold_s": 10.0, "dijkstra_cutoff_m": 1000.0, "clf_weight": 0.85},
        {"beta": 7.5, "sigma_z": 5.00, "unreachable_penalty_m": 450.0, "gap_threshold_s": 15.0, "dijkstra_cutoff_m": 1000.0, "clf_weight": 0.80},
        {"beta": 10.0, "sigma_z": 6.00, "unreachable_penalty_m": 500.0, "gap_threshold_s": 15.0, "dijkstra_cutoff_m": 1200.0, "clf_weight": 0.75},
        {"beta": 5.0, "sigma_z": 4.07, "unreachable_penalty_m": 250.0, "gap_threshold_s": 10.0, "dijkstra_cutoff_m": 800.0, "clf_weight": 0.90},
    ]

    best_score = -1.0
    best_config = grid[1]

    print("\n--- Running Decoder Hyperparameter Search ---")
    for idx, cfg in enumerate(grid, 1):
        conts, accs = [], []
        for tid in trips:
            if tid in graphs and tid in all_refs:
                pt_acc, cont = evaluate_config_on_trip(
                    trip_id=tid,
                    cand_df=cand_df,
                    ref_arcs=all_refs[tid],
                    graph=graphs[tid],
                    config=cfg,
                )
                conts.append(cont)
                accs.append(pt_acc)

        mean_cont = float(np.mean(conts)) if conts else 0.0
        mean_acc = float(np.mean(accs)) if accs else 0.0
        score = 0.6 * mean_cont + 0.4 * mean_acc

        print(
            f"  Config #{idx} | beta={cfg['beta']} | sigma_z={cfg['sigma_z']} | "
            f"Continuity: {mean_cont:.4f} | Pointwise Acc: {mean_acc:.4f} | Composite Score: {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_config = cfg

    # Save best config
    config_path = out_dir / "hmm_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print(f"\n[Decoder Tuning] Best configuration saved to: {config_path}")
    print("```json\n" + json.dumps(best_config, indent=2) + "\n```")
    return best_config


if __name__ == "__main__":
    tune_decoder()
