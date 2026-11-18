"""
PathSnap Person 2: Mock Candidate Data Generator
================================================
Generates realistic candidate parquet tables strictly complying with Contract 1.
Used by P2 (modeling) and P3 (decoder) to develop and validate the pipeline in
parallel without waiting for P1's data pipeline to finish.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from common.contract import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    LABEL_COLUMN,
    PROCESSED_DATA_DIR,
    validate_candidate_table,
)


def generate_mock_candidates(
    n_trips: int = 80,
    points_per_trip: int = 50,
    k_candidates: int = 6,
    miss_rate: float = 0.04,
    random_state: int = 42,
    partition_name: str = "train",
) -> pd.DataFrame:
    """
    Generate a mock candidate table adhering to Contract 1.

    Args:
        n_trips: Number of trips in the partition (80 for train, 10 for student_test)
        points_per_trip: Approximate number of GPS points per trip
        k_candidates: Number of candidate arcs evaluated per GPS point
        miss_rate: Probability that the ground-truth arc was missed by candidate search
        random_state: Random seed for reproducibility
        partition_name: Name of the partition ("train" or "student_test")

    Returns:
        DataFrame adhering strictly to Contract 1 schema.
    """
    rng = np.random.default_rng(random_state)
    records = []

    for trip_idx in range(1, n_trips + 1):
        trip_id = f"{partition_name[:2]}_{trip_idx:04d}"
        n_pts = rng.integers(max(20, points_per_trip - 15), points_per_trip + 20)

        # Base vehicle speed profile for this trip (m/s)
        base_speed = rng.uniform(8.0, 22.0)

        for pt_idx in range(n_pts):
            has_gt_in_candidates = rng.random() > miss_rate
            gt_cand_pos = rng.integers(0, min(3, k_candidates)) if has_gt_in_candidates else -1

            # Candidate arcs around this GPS point
            for c_idx in range(k_candidates):
                arc_id = rng.integers(1000, 90000)
                is_gt = c_idx == gt_cand_pos

                if is_gt:
                    # Realistic distribution for true road segment
                    dist_m = float(np.clip(rng.normal(loc=6.5, scale=4.0), 0.5, 35.0))
                    bearing_diff = float(np.clip(rng.normal(loc=12.0, scale=10.0), 0.0, 180.0))
                    dist_rank = int(rng.choice([0, 1, 2], p=[0.70, 0.22, 0.08]))
                    perp_dist_m = float(dist_m * np.sin(np.radians(max(1.0, bearing_diff))))
                    frac_along = float(np.clip(rng.uniform(0.1, 0.9), 0.0, 1.0))
                    label = 1
                else:
                    # Realistic distribution for false/distractor segments
                    dist_m = float(np.clip(rng.uniform(15.0, 120.0), 5.0, 200.0))
                    bearing_diff = float(np.clip(rng.uniform(25.0, 175.0), 0.0, 180.0))
                    dist_rank = int(min(k_candidates - 1, c_idx + rng.integers(0, 2)))
                    perp_dist_m = float(dist_m * rng.uniform(0.3, 0.98))
                    frac_along = float(np.clip(rng.uniform(0.0, 1.0), 0.0, 1.0))
                    label = 0

                speed = float(np.clip(base_speed + rng.normal(0, 1.5), 1.0, 35.0))
                arc_len_m = float(np.clip(rng.lognormal(mean=4.2, sigma=0.6), 15.0, 350.0))
                delta_time_s = float(rng.choice([1.0, 1.0, 1.0, 2.0, 3.0], p=[0.94, 0.03, 0.01, 0.01, 0.01]))

                records.append(
                    {
                        "trip_id": trip_id,
                        "point_idx": pt_idx,
                        "arc_id": arc_id,
                        "label": label,
                        "dist_m": round(dist_m, 3),
                        "bearing_diff": round(bearing_diff, 2),
                        "speed": round(speed, 2),
                        "frac_along": round(frac_along, 4),
                        "arc_len_m": round(arc_len_m, 2),
                        "dist_rank": dist_rank,
                        "delta_time_s": round(delta_time_s, 2),
                        "perp_dist_m": round(perp_dist_m, 3),
                    }
                )

    df = pd.DataFrame(records)

    # Sort deterministically by trip, point, and distance rank
    df = df.sort_values(by=["trip_id", "point_idx", "dist_rank"]).reset_index(drop=True)

    validate_candidate_table(df, require_label=True, feature_cols=FEATURE_COLUMNS)
    return df


def generate_and_save_mock_tables(
    output_dir: Optional[Path] = None,
    train_trips: int = 80,
    test_trips: int = 10,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """
    Generate both mock train and mock student-test parquet tables and save to disk.
    """
    out_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "mock_train.parquet"
    test_path = out_dir / "mock_student_test.parquet"

    print(f"[MockData] Generating {train_trips} mock training trips...")
    df_train = generate_mock_candidates(
        n_trips=train_trips,
        points_per_trip=50,
        k_candidates=6,
        miss_rate=0.04,
        random_state=seed,
        partition_name="train",
    )
    df_train.to_parquet(train_path, index=False)
    print(f"[MockData] Saved mock training table: {train_path} ({len(df_train):,} rows)")

    print(f"[MockData] Generating {test_trips} mock student-test trips...")
    df_test = generate_mock_candidates(
        n_trips=test_trips,
        points_per_trip=50,
        k_candidates=6,
        miss_rate=0.05,
        random_state=seed + 100,
        partition_name="student_test",
    )
    df_test.to_parquet(test_path, index=False)
    print(f"[MockData] Saved mock student-test table: {test_path} ({len(df_test):,} rows)")

    return train_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Contract 1 Mock Data for PathSnap")
    parser.add_argument("--train-trips", type=int, default=80, help="Number of train trips (default: 80)")
    parser.add_argument("--test-trips", type=int, default=10, help="Number of test trips (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_and_save_mock_tables(train_trips=args.train_trips, test_trips=args.test_trips, seed=args.seed)
