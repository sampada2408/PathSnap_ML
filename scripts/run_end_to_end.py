"""
PathSnap: End-to-End System Evaluation Script
============================================
Runs the complete production map-matching pipeline on all student-test trips:
1. Loads raw .track, .nodes, .arcs via P1
2. Generates candidate arcs and spatial features via P1
3. Computes positive arc probabilities via P2 Champion Model (LightGBM)
4. Reconstructs sequence via P3 Viterbi Trellis Decoder
5. Evaluates server-side metrics via P4:
   - Route Recall
   - Route Precision
   - Route F1 Score
   - Route Physical Continuity
   - Inference Latency (ms)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from api.evaluation import evaluate
from api.predictor import predict_trip

STUDENT_TEST_DIR = Path("data/raw/student_test")
REPORTS_DIR = Path("reports")


def run_all():
    trip_dirs = sorted([p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()])
    print(f"\n================================================================================")
    print(f"PATHSNAP END-TO-END SYSTEM BENCHMARK ({len(trip_dirs)} Student-Test Trips)")
    print(f"================================================================================")

    results = []
    total_points = 0
    total_ms = 0.0

    for i, trip_dir in enumerate(trip_dirs, 1):
        tid = trip_dir.name
        t0 = time.perf_counter()
        pred_res = predict_trip(trip_dir)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        pred_arcs = pred_res["pred_arcs"]
        metrics = evaluate(trip_dir, pred_arcs)
        n_pts = pred_res.get("point_count", len(pred_arcs))

        total_points += n_pts
        total_ms += dt_ms

        res = {
            "Trip ID": tid,
            "GPS Points": n_pts,
            "Reference Arcs": metrics["reference_arcs"],
            "Predicted Arcs": metrics["predicted_arcs"],
            "Route Recall": metrics["route_recall"],
            "Route Precision": metrics["route_precision"],
            "Route F1": metrics["route_f1"],
            "Route Continuity": metrics["route_continuity"],
            "Latency (ms)": round(dt_ms, 1),
            "ms / 1k pts": round(dt_ms / (n_pts / 1000.0), 2) if n_pts > 0 else 0.0,
        }
        results.append(res)
        print(
            f"[{i}/{len(trip_dirs)}] Trip {tid} | Pts: {n_pts:4d} | "
            f"Recall: {metrics['route_recall']*100:5.2f}% | "
            f"Precision: {metrics['route_precision']*100:5.2f}% | "
            f"F1: {metrics['route_f1']*100:5.2f}% | "
            f"Cont: {metrics['route_continuity']*100:5.2f}% | "
            f"Latency: {dt_ms:6.1f} ms"
        )

    df_results = pd.DataFrame(results)

    # Compute summary row
    mean_row = {
        "Trip ID": "AVERAGE / TOTAL",
        "GPS Points": total_points,
        "Reference Arcs": int(df_results["Reference Arcs"].mean()),
        "Predicted Arcs": int(df_results["Predicted Arcs"].mean()),
        "Route Recall": round(df_results["Route Recall"].mean(), 4),
        "Route Precision": round(df_results["Route Precision"].mean(), 4),
        "Route F1": round(df_results["Route F1"].mean(), 4),
        "Route Continuity": round(df_results["Route Continuity"].mean(), 4),
        "Latency (ms)": round(total_ms, 1),
        "ms / 1k pts": round(total_ms / (total_points / 1000.0), 2),
    }

    df_full = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "end_to_end_evaluation_table.csv"
    md_path = REPORTS_DIR / "end_to_end_evaluation_table.md"

    df_full.to_csv(csv_path, index=False)

    md_content = f"# PathSnap End-to-End System Benchmark\n\n"
    md_content += f"Evaluated across {len(trip_dirs)} student-test evaluation trips:\n\n"
    md_content += df_full.to_markdown(index=False) + "\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print("END-TO-END EVALUATION SUMMARY TABLE")
    print("=" * 80)
    print(df_full.to_markdown(index=False))
    print(f"\nSaved evaluation reports to:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    return df_full


if __name__ == "__main__":
    run_all()
