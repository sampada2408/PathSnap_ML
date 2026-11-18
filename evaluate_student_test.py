
# Early import of lightgbm on Windows prevents OpenMP DLL conflict
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from common.contract import (
    ARTIFACTS_DIR,
    MODEL_ARTIFACT_PATH,
    FEATURE_COLS_ARTIFACT_PATH,
    extract_positive_probs,
)
from data_pipeline.io_utils import load_trip
from decoder.viterbi import ViterbiDecoder


# Paths
CANDIDATES_FILE = Path("data/processed/student_test_candidates.parquet")
REFERENCE_FILE = Path("data/processed/student_test_reference_arcs.json")

def _resolve_trip_root() -> Path:
    for p in [Path("data/raw/student_test"), Path("data/raw/Student folder/student_test"), Path("student_test")]:
        if p.exists() and any(p.iterdir()):
            return p
    return Path("data/raw/student_test")

TRIP_ROOT = _resolve_trip_root()
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    for path in [CANDIDATES_FILE, REFERENCE_FILE,
                 MODEL_ARTIFACT_PATH, FEATURE_COLS_ARTIFACT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading candidate data...")
    candidates_df = pd.read_parquet(CANDIDATES_FILE)

    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        references = json.load(f)

    model = joblib.load(MODEL_ARTIFACT_PATH)

    with open(FEATURE_COLS_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    config_path = ARTIFACTS_DIR / "hmm_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    decoder = ViterbiDecoder(
        beta=config.get("beta", 5.0),
        sigma_z=config.get("sigma_z", 4.07),
        unreachable_penalty_m=config.get(
            "unreachable_penalty_m", 350.0
        ),
        gap_threshold_s=config.get("gap_threshold_s", 10.0),
        dijkstra_cutoff_m=config.get("dijkstra_cutoff_m", 1000.0),
        clf_weight=config.get("clf_weight", 0.85),
        use_hybrid_emission=True,
    )

    trip_ids = sorted(
        set(candidates_df["trip_id"].astype(str))
        & set(references.keys())
    )

    print(f"Student-test trips: {len(trip_ids)}")

    all_true = []
    all_pred = []
    results = []
    prediction_frames = []

    for i, trip_id in enumerate(trip_ids, start=1):
        print(f"\n[{i}/{len(trip_ids)}] Trip {trip_id}")

        try:
            cands = candidates_df[
                candidates_df["trip_id"].astype(str) == trip_id
            ].copy().reset_index(drop=True)

            if cands.empty:
                raise ValueError("No candidate rows found")

            # Load observations and road network without loading .route.
            trip = load_trip(
                TRIP_ROOT / trip_id,
                with_route=False
            )

            reference = np.asarray(
                references[trip_id], dtype=int
            )

            point_indices = sorted(
                cands["point_idx"].astype(int).unique().tolist()
            )

            times_all = np.asarray(trip["t"], dtype=float)

            if any(p < 0 or p >= len(times_all) for p in point_indices):
                raise ValueError("Candidate point index out of range")

            # Prepare features using the official P2 feature contract.
            for col in feature_cols:
                if col not in cands.columns:
                    cands[col] = 0.0

            X = (
                cands[feature_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # P2 classifier probabilities.
            probabilities = extract_positive_probs(
                model.predict_proba(X)
            )

            # P3 official HMM + Viterbi decoder.
            # Times correspond to the sorted candidate observation points.
            predicted_arcs = decoder.decode(
                cands=cands,
                probs=probabilities,
                arcs=trip["arcs"],
                nodes=trip["nodes"],
                times=times_all[point_indices],
            )

            if len(predicted_arcs) != len(point_indices):
                raise ValueError("Prediction count mismatch")

            predictions = dict(zip(point_indices, predicted_arcs))

            common = [
                p for p in point_indices
                if p < len(reference)
            ]

            if not common:
                raise ValueError("No matching reference observations")

            y_true = np.asarray([reference[p] for p in common])
            y_pred = np.asarray([predictions[p] for p in common])

            accuracy = accuracy_score(y_true, y_pred)

            all_true.extend(y_true.tolist())
            all_pred.extend(y_pred.tolist())

            results.append({
                "trip_id": trip_id,
                "total_points": len(reference),
                "evaluated_points": len(common),
                "coverage_percent": 100 * len(common) / len(reference),
                "accuracy_percent": accuracy * 100,
                "status": "Success",
            })

            prediction_frames.append(pd.DataFrame({
                "trip_id": trip_id,
                "point_idx": common,
                "predicted_arc_id": y_pred,
                "reference_arc_id": y_true,
            }))

            print(f"Evaluated: {len(common)}")
            print(f"Accuracy: {accuracy * 100:.2f}%")

        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

            results.append({
                "trip_id": trip_id,
                "total_points": len(references[trip_id]),
                "evaluated_points": 0,
                "coverage_percent": 0.0,
                "accuracy_percent": np.nan,
                "status": f"Failed: {exc}",
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "student_test_results.csv",
        index=False
    )

    if prediction_frames:
        predictions_df = pd.concat(
            prediction_frames, ignore_index=True
        )
        predictions_df.to_csv(
            OUTPUT_DIR / "student_test_predictions.csv",
            index=False
        )

    successful = sum(r["status"] == "Success" for r in results)

    print("\n========== INTEGRATED P2 + P3 RESULTS ==========")
    print(f"Total trips: {len(trip_ids)}")
    print(f"Successful trips: {successful}")
    print(f"Failed trips: {len(trip_ids) - successful}")

    if all_true:
        overall = accuracy_score(all_true, all_pred)
        print(f"Evaluated observations: {len(all_true)}")
        print(f"Overall accuracy: {overall * 100:.2f}%")
    else:
        print("No successful predictions.")

    print("\nPer-trip results:")
    print(results_df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_DIR / "student_test_results.csv")
    print(OUTPUT_DIR / "student_test_predictions.csv")


if __name__ == "__main__":
    main()
