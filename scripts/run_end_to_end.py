
"""
PathSnap: End-to-End System Evaluation
======================================
P1: Candidate generation and feature engineering
P2: LightGBM candidate probability prediction
P3: HMM/Viterbi route reconstruction
P4: Route-level evaluation and report generation

Usage:
    python -m scripts.run_end_to_end
    python -m scripts.run_end_to_end --data "data/raw/Student folder/student_test"
    python -m scripts.run_end_to_end --data evaluator_test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from api.evaluation import evaluate
from api.predictor import predict_trip
from common.contract import df_to_markdown_safe


REPORTS_DIR = Path("reports")


def contains_valid_trips(directory: Path) -> bool:
    """Check whether a directory contains usable benchmark trip folders."""
    if not directory.is_dir():
        return False

    for trip_dir in directory.iterdir():
        if not trip_dir.is_dir():
            continue

        tid = trip_dir.name
        required = [
            trip_dir / f"{tid}.track",
            trip_dir / f"{tid}.nodes",
            trip_dir / f"{tid}.arcs",
            trip_dir / f"{tid}.route",
        ]

        if all(path.is_file() for path in required):
            return True

    return False


def resolve_data_dir(data_dir: str | None = None) -> Path:
    """Resolve an explicit dataset path or discover a supported default."""
    if data_dir:
        target = Path(data_dir)

        if not contains_valid_trips(target):
            raise FileNotFoundError(
                f"No valid benchmark trips found in: {target}\n"
                "Each trip must contain .track, .nodes, .arcs and .route files."
            )

        return target

    candidates = [
        Path("evaluator_test"),
        Path("data/raw/evaluator_test"),
        Path("data/raw/student_test"),
        Path("data/raw/Student folder/student_test"),
    ]

    for candidate in candidates:
        if contains_valid_trips(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate a valid evaluation dataset.\n"
        "Use --data to specify the folder containing the trip directories."
    )


def run_all(data_dir: str | None = None) -> pd.DataFrame:
    """Run the complete P1-P2-P3-P4 pipeline on every trip."""
    target_dir = resolve_data_dir(data_dir)

    trip_dirs = sorted(
        path for path in target_dir.iterdir()
        if path.is_dir()
        and (path / f"{path.name}.track").is_file()
        and (path / f"{path.name}.nodes").is_file()
        and (path / f"{path.name}.arcs").is_file()
        and (path / f"{path.name}.route").is_file()
    )

    if not trip_dirs:
        raise RuntimeError(f"No valid trips found in {target_dir}")

    print("\n" + "=" * 80)
    print(f"PATHSNAP END-TO-END BENCHMARK: {target_dir}")
    print(f"Number of trips: {len(trip_dirs)}")
    print("=" * 80)

    results = []
    total_points = 0
    total_ms = 0.0

    for i, trip_dir in enumerate(trip_dirs, start=1):
        tid = trip_dir.name

        print(f"\n[{i}/{len(trip_dirs)}] Processing trip {tid}...")

        try:
            start = time.perf_counter()

            # P1 + P2 + P3: production prediction pipeline.
            prediction = predict_trip(trip_dir)

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            predicted_arcs = prediction["pred_arcs"]

            # P4: ground-truth evaluation happens after inference.
            metrics = evaluate(trip_dir, predicted_arcs)

            point_count = prediction.get(
                "point_count", len(predicted_arcs)
            )

            total_points += point_count
            total_ms += elapsed_ms

            result = {
                "Trip ID": tid,
                "GPS Points": point_count,
                "Reference Arcs": metrics["reference_arcs"],
                "Predicted Arcs": metrics["predicted_arcs"],
                "Route Recall": metrics["route_recall"],
                "Route Precision": metrics["route_precision"],
                "Route F1": metrics["route_f1"],
                "Route Continuity": metrics["route_continuity"],
                "Latency (ms)": round(elapsed_ms, 1),
                "ms / 1k pts": (
                    round(elapsed_ms / (point_count / 1000), 2)
                    if point_count > 0 else 0.0
                ),
                "Status": "Success",
            }

            results.append(result)

            print(
                f"Points: {point_count} | "
                f"Recall: {metrics['route_recall'] * 100:.2f}% | "
                f"Precision: {metrics['route_precision'] * 100:.2f}% | "
                f"F1: {metrics['route_f1'] * 100:.2f}% | "
                f"Continuity: {metrics['route_continuity'] * 100:.2f}% | "
                f"Latency: {elapsed_ms:.1f} ms"
            )

        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

            results.append({
                "Trip ID": tid,
                "GPS Points": 0,
                "Reference Arcs": 0,
                "Predicted Arcs": 0,
                "Route Recall": float("nan"),
                "Route Precision": float("nan"),
                "Route F1": float("nan"),
                "Route Continuity": float("nan"),
                "Latency (ms)": float("nan"),
                "ms / 1k pts": float("nan"),
                "Status": f"Failed: {exc}",
            })

    results_df = pd.DataFrame(results)

    successful = results_df[results_df["Status"] == "Success"]

    if successful.empty:
        raise RuntimeError(
            "All trips failed. No benchmark report was generated."
        )

    # Summary uses the mean of successful trip-level metrics.
    summary = {
        "Trip ID": "AVERAGE / TOTAL",
        "GPS Points": int(total_points),
        "Reference Arcs": round(successful["Reference Arcs"].mean()),
        "Predicted Arcs": round(successful["Predicted Arcs"].mean()),
        "Route Recall": successful["Route Recall"].mean(),
        "Route Precision": successful["Route Precision"].mean(),
        "Route F1": successful["Route F1"].mean(),
        "Route Continuity": successful["Route Continuity"].mean(),
        "Latency (ms)": round(total_ms, 1),
        "ms / 1k pts": (
            round(total_ms / (total_points / 1000), 2)
            if total_points > 0 else 0.0
        ),
        "Status": f"{len(successful)}/{len(trip_dirs)} successful",
    }

    full_results = pd.concat(
        [results_df, pd.DataFrame([summary])],
        ignore_index=True
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prefix = (
        f"{target_dir.name}_"
        if target_dir.name != "student_test"
        else ""
    )

    csv_path = REPORTS_DIR / f"{prefix}end_to_end_evaluation_table.csv"
    md_path = REPORTS_DIR / f"{prefix}end_to_end_evaluation_table.md"

    full_results.to_csv(csv_path, index=False)

    markdown = (
        f"# PathSnap End-to-End Benchmark\n\n"
        f"Dataset: `{target_dir}`\n\n"
        f"Trips evaluated: {len(trip_dirs)}\n\n"
        f"Successful trips: {len(successful)}/{len(trip_dirs)}\n\n"
        f"{df_to_markdown_safe(full_results)}\n"
    )

    md_path.write_text(markdown, encoding="utf-8")

    print("\n" + "=" * 80)
    print("END-TO-END BENCHMARK SUMMARY")
    print("=" * 80)
    print(full_results.to_string(index=False))

    print("\nReports saved:")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")

    return full_results


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete PathSnap benchmark."
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Dataset folder containing trip directories."
    )

    args = parser.parse_args()
    run_all(args.data)


if __name__ == "__main__":
    main()
