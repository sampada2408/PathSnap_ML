"""
PathSnap Person 3: Route Reconstruction Evaluation
==================================================
Runs the 3-way comparison required by Hackathon Section 5:
1. Baseline: Distance-based Gaussian Emission (Newson & Krumm 2009)
2. Classifier-Only: Independent Top-1 predictions (Greedy)
3. Proposed Method: Classifier + HMM/Viterbi Sequential Decoding

Outputs:
- reports/hmm_comparison_table.csv & .md
- reports/figures/hmm_continuity_comparison.png
- reports/figures/hmm_accuracy_comparison.png
- reports/figures/hmm_sequence_f1_comparison.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.contract import (
    ARTIFACTS_DIR,
    FEATURE_COLS_ARTIFACT_PATH,
    FIGURES_DIR,
    MODEL_ARTIFACT_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    df_to_markdown_safe,
    extract_positive_probs,
)
from decoder.graph import RoadNetworkGraph
from decoder.metrics import (
    compute_pointwise_accuracy,
    compute_route_continuity,
    compute_sequence_precision_recall_f1,
)
from decoder.viterbi import ViterbiDecoder


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def run_route_reconstruction_experiments(
    sample_size: int = 5,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates Newson-Krumm, Classifier-Only, and Classifier+HMM across evaluation trips.
    """
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load candidate data
    cand_file = PROCESSED_DATA_DIR / "student_test_candidates.parquet"
    ref_file = PROCESSED_DATA_DIR / "student_test_reference_arcs.json"
    partition = "student_test"

    if not cand_file.exists():
        cand_file = PROCESSED_DATA_DIR / "train_candidates.parquet"
        ref_file = PROCESSED_DATA_DIR / "train_reference_arcs.json"
        partition = "train"

    print(f"\n[HMM Eval] Loading candidate data from {cand_file} ({partition})...")
    df_cands = pd.read_parquet(cand_file)
    with open(ref_file, "r", encoding="utf-8") as f:
        all_refs = json.load(f)

    trip_ids = list(all_refs.keys())[:sample_size]
    print(f"[HMM Eval] Evaluating on {len(trip_ids)} real trips: {trip_ids}")

    # 2. Load trained classifier and features if available
    classifier = None
    feature_cols = None
    if MODEL_ARTIFACT_PATH.exists() and FEATURE_COLS_ARTIFACT_PATH.exists():
        try:
            classifier = joblib.load(MODEL_ARTIFACT_PATH)
            with open(FEATURE_COLS_ARTIFACT_PATH, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
            print(f"[HMM Eval] Successfully loaded trained champion classifier from {MODEL_ARTIFACT_PATH}")
        except Exception as e:
            print(f"[HMM Eval] Warning: Could not load classifier ({e}). Using candidate dist/speed.")

    # 3. Load HMM config
    config_path = ARTIFACTS_DIR / "hmm_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            hmm_cfg = json.load(f)
    else:
        hmm_cfg = {
            "beta": 5.0,
            "sigma_z": 4.07,
            "unreachable_penalty_m": 350.0,
            "gap_threshold_s": 10.0,
            "dijkstra_cutoff_m": 1000.0,
            "clf_weight": 0.85,
        }

    # Initialize decoders
    decoder_gaussian = ViterbiDecoder(
        beta=hmm_cfg["beta"],
        sigma_z=hmm_cfg["sigma_z"],
        unreachable_penalty_m=hmm_cfg["unreachable_penalty_m"],
        gap_threshold_s=hmm_cfg["gap_threshold_s"],
        dijkstra_cutoff_m=hmm_cfg["dijkstra_cutoff_m"],
        use_hybrid_emission=False,
    )

    decoder_ml = ViterbiDecoder(
        beta=hmm_cfg["beta"],
        sigma_z=hmm_cfg["sigma_z"],
        unreachable_penalty_m=hmm_cfg["unreachable_penalty_m"],
        gap_threshold_s=hmm_cfg["gap_threshold_s"],
        dijkstra_cutoff_m=hmm_cfg["dijkstra_cutoff_m"],
        clf_weight=hmm_cfg.get("clf_weight", 0.85),
        use_hybrid_emission=True,
    )

    # Accumulators for metrics
    results_agg: Dict[str, Dict[str, List[float]]] = {
        "Newson-Krumm (Gaussian Baseline)": {"pt_acc": [], "continuity": [], "seq_p": [], "seq_r": [], "seq_f1": [], "lat_ms": []},
        "Classifier-Only (Greedy Top-1)": {"pt_acc": [], "continuity": [], "seq_p": [], "seq_r": [], "seq_f1": [], "lat_ms": []},
        "Classifier + HMM/Viterbi (Proposed)": {"pt_acc": [], "continuity": [], "seq_p": [], "seq_r": [], "seq_f1": [], "lat_ms": []},
    }

    for tid in trip_ids:
        trip_dir = RAW_DATA_DIR / partition / tid
        if not trip_dir.exists():
            trip_dir = RAW_DATA_DIR / "train" / tid
        if not trip_dir.exists():
            continue

        graph = RoadNetworkGraph.from_trip_dir(trip_dir)
        route_file = trip_dir / f"{tid}.route"
        gt_route = np.loadtxt(route_file, dtype=int).tolist() if route_file.exists() else []
        ref_arcs = all_refs[tid]

        c_df = df_cands[df_cands["trip_id"] == tid].copy()
        if c_df.empty:
            continue

        # Predict candidate probabilities
        if classifier is not None and feature_cols is not None:
            available_cols = [c for c in feature_cols if c in c_df.columns]
            if len(available_cols) == len(feature_cols):
                raw_p = classifier.predict_proba(c_df[feature_cols])
                c_df["prob"] = extract_positive_probs(raw_p)
            else:
                # Fallback: inverse distance rank
                c_df["prob"] = 1.0 / (c_df["dist_rank"] + 1.0)
        else:
            c_df["prob"] = 1.0 / (c_df["dist_rank"] + 1.0)

        # 1. Newson-Krumm (Gaussian HMM)
        t0 = time.perf_counter()
        nk_arcs = decoder_gaussian.decode(cands=c_df, probs=None, arcs=graph.arcs, nodes=graph.nodes)
        t_nk = (time.perf_counter() - t0) * 1000.0 / len(ref_arcs) * 1000.0

        # 2. Classifier-Only (Greedy Top-1)
        t0 = time.perf_counter()
        idx_max = c_df.groupby("point_idx")["prob"].idxmax()
        clf_arcs = c_df.loc[idx_max, "arc_id"].values.astype(int).tolist()
        t_clf = (time.perf_counter() - t0) * 1000.0 / len(ref_arcs) * 1000.0

        # 3. Classifier + HMM / Viterbi
        t0 = time.perf_counter()
        hmm_arcs = decoder_ml.decode(cands=c_df, probs=c_df["prob"].values, arcs=graph.arcs, nodes=graph.nodes)
        t_hmm = (time.perf_counter() - t0) * 1000.0 / len(ref_arcs) * 1000.0

        # Record metrics
        for name, pred_seq, latency in [
            ("Newson-Krumm (Gaussian Baseline)", nk_arcs, t_nk),
            ("Classifier-Only (Greedy Top-1)", clf_arcs, t_clf),
            ("Classifier + HMM/Viterbi (Proposed)", hmm_arcs, t_hmm),
        ]:
            pt = compute_pointwise_accuracy(pred_seq, ref_arcs)
            cont = compute_route_continuity(pred_seq, graph)
            seq = compute_sequence_precision_recall_f1(pred_seq, gt_route)

            results_agg[name]["pt_acc"].append(pt)
            results_agg[name]["continuity"].append(cont)
            results_agg[name]["seq_p"].append(seq["precision"])
            results_agg[name]["seq_r"].append(seq["recall"])
            results_agg[name]["seq_f1"].append(seq["f1"])
            results_agg[name]["lat_ms"].append(latency)

    # Build Comparison Summary Table
    rows = []
    for name, m in results_agg.items():
        rows.append(
            {
                "Method": name,
                "Pointwise Accuracy": f"{np.mean(m['pt_acc']):.4f}",
                "Route Continuity": f"{np.mean(m['continuity']):.4f}",
                "Sequence Precision": f"{np.mean(m['seq_p']):.4f}",
                "Sequence Recall": f"{np.mean(m['seq_r']):.4f}",
                "Sequence F1-Score": f"{np.mean(m['seq_f1']):.4f}",
                "Latency (ms / 1k pts)": f"{np.mean(m['lat_ms']):.2f}",
            }
        )

    comp_df = pd.DataFrame(rows)

    # Save outputs
    csv_path = out_dir / "hmm_comparison_table.csv"
    md_path = out_dir / "hmm_comparison_table.md"
    comp_df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 5. Route Reconstruction Comparison Table\n\n")
        f.write(df_to_markdown_safe(comp_df))
        f.write("\n")

    print("\n======================================================================")
    print("5. ROUTE RECONSTRUCTION COMPARISON TABLE")
    print("======================================================================")
    print(comp_df.to_string(index=False))

    # Generate Visualizations
    plot_hmm_comparison_charts(comp_df, FIGURES_DIR)

    return comp_df, results_agg


def plot_hmm_comparison_charts(comp_df: pd.DataFrame, out_dir: Path):
    """
    Generates comparison plots for Section 5:
    - Route Continuity comparison
    - Pointwise Accuracy comparison
    - Sequence F1 comparison
    """
    set_plot_style()
    methods = [
        "Newson–Krumm\n(Gaussian)",
        "Classifier-Only\n(Greedy)",
        "Classifier + HMM\n(Viterbi)",
    ]

    conts = [float(v) for v in comp_df["Route Continuity"].values]
    pts = [float(v) for v in comp_df["Pointwise Accuracy"].values]
    f1s = [float(v) for v in comp_df["Sequence F1-Score"].values]

    # 1. Route Continuity Plot
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = ["#457b9d", "#e63946", "#2a9d8f"]
    bars = ax.bar(methods, conts, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    ax.set_title("Route Continuity: Classifier-Only vs. HMM/Viterbi", fontsize=12, weight="bold", pad=12)
    ax.set_ylabel("Continuity Ratio", fontsize=10.5)
    ax.set_ylim(0.0, 1.08)

    for bar, val in zip(bars, conts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "hmm_continuity_comparison.png", dpi=300)
    plt.close(fig)

    # 2. Side-by-side Pointwise vs Sequence F1
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    r1 = ax.bar(x - width / 2, pts, width, label="Pointwise Accuracy", color="#3a86ff")
    r2 = ax.bar(x + width / 2, f1s, width, label="Sequence F1-Score", color="#ff006e")

    ax.set_title("Pointwise Accuracy vs. Sequence-Level F1-Score", fontsize=12, weight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.0, 1.08)
    ax.legend(frameon=True, facecolor="white")

    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9, weight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "hmm_accuracy_comparison.png", dpi=300)
    plt.close(fig)
    print(f"\n[HMM Visualizations] Saved Section 5 charts to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Route Reconstruction (Section 5)")
    parser.add_argument("--samples", type=int, default=5, help="Number of trips to evaluate")
    args = parser.parse_args()

    run_route_reconstruction_experiments(sample_size=args.samples)
