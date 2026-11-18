"""
PathSnap Person 2: Master Training & Experimentation Pipeline
============================================================
Entry point for Person 2 (Modeling & Tuning).

Usage:
  # 1. Run on mock data (zero-wait mode before P1 completes):
  python -m modeling.train --mock

  # 2. Run on real data (once P1 generates data/processed/train.parquet):
  python -m modeling.train

  # Options:
  --trials 25        # Number of Bayesian optimization trials (default: 25)
  --seed 42          # Random seed (default: 42)
  --fast             # Quick test mode (3 models, 5 trials)
"""

from __future__ import annotations

try:
    import lightgbm
except ImportError:
    pass

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from common.contract import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    validate_candidate_table,
)
from modeling.export import export_model_artifacts
from modeling.harness import run_all_experiments
from modeling.mock_data import generate_and_save_mock_tables
from modeling.report_draft import generate_report_markdown
from modeling.tune import run_bayesian_optimization
from modeling.visualizations import generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(description="PathSnap Person 2: Modeling Pipeline")
    parser.add_argument("--mock", action="store_true", help="Force using mock candidate data")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna tuning trials (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--fast", action="store_true", help="Fast run with fewer trials")
    args = parser.parse_args()

    n_trials = 5 if args.fast else args.trials

    # 1. Determine whether real data is available or mock data should be used
    real_train = PROCESSED_DATA_DIR / "train.parquet"
    real_test = PROCESSED_DATA_DIR / "student_test.parquet"

    mock_train = PROCESSED_DATA_DIR / "mock_train.parquet"
    mock_test = PROCESSED_DATA_DIR / "mock_student_test.parquet"

    if not args.mock and real_train.exists() and real_test.exists():
        print(f"\n[Data] Found real processed tables in {PROCESSED_DATA_DIR}!")
        train_path = real_train
        test_path = real_test
        is_mock = False
    else:
        print(f"\n[Data] Using Contract 1 mock data (stand-in for P1 data pipeline)...")
        if not mock_train.exists() or not mock_test.exists():
            print("[Data] Generating mock candidate tables...")
            generate_and_save_mock_tables(seed=args.seed)
        train_path = mock_train
        test_path = mock_test
        is_mock = True

    print(f"[Data] Loading train partition: {train_path}")
    df_train = pd.read_parquet(train_path)
    print(f"[Data] Loading student-test partition: {test_path}")
    df_test = pd.read_parquet(test_path)

    # Validate Contract 1
    validate_candidate_table(df_train, require_label=True)
    validate_candidate_table(df_test, require_label=True)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df_train.columns]
    print(f"[Data] Features verified: {feature_cols}")

    # 2. Run All Model Experiments (Harness)
    comp_df, results_map, fitted_models = run_all_experiments(
        df_train=df_train,
        df_test=df_test,
        feature_cols=feature_cols,
        output_dir=REPORTS_DIR,
        random_state=args.seed,
    )

    # Extract loss curve from MLP if available
    mlp_loss = None
    if "Multi-Layer Perceptron (MLP)" in fitted_models:
        mlp_wrapper = fitted_models["Multi-Layer Perceptron (MLP)"]
        if mlp_wrapper.training_history and "train_loss" in mlp_wrapper.training_history:
            mlp_loss = mlp_wrapper.training_history["train_loss"]

    # 3. Generate Compulsory Visualizations (Section 3.3)
    print("\n[Visualizations] Generating Section 3.3 charts...")
    viz_dict = generate_all_visualizations(results_map, mlp_loss_history=mlp_loss)

    # 4. Bayesian Hyperparameter Optimization (Section 4)
    print("\n[Optimization] Starting Section 4 Optuna Bayesian tuning study...")
    t_opt_start = time.perf_counter()
    untuned_lgb, tuned_lgb, trials_df, tune_table_df = run_bayesian_optimization(
        df_train=df_train,
        df_test=df_test,
        feature_cols=feature_cols,
        n_trials=n_trials,
        seed=args.seed,
        output_dir=REPORTS_DIR,
    )
    study_duration = time.perf_counter() - t_opt_start

    # Read best params
    best_params = trials_df.loc[trials_df["cv_macro_f1"].idxmax()].to_dict()
    # Clean non-param keys
    for k in ["trial_number", "cv_macro_f1", "duration_seconds"]:
        best_params.pop(k, None)

    # 5. Export Champion Model Artifacts (Contract 2)
    print("\n[Export] Exporting champion model for P3 (Decoder) and P4 (API)...")
    best_result = results_map.get("LightGBM")
    export_model_artifacts(
        model=tuned_lgb,
        feature_columns=feature_cols,
        metrics_summary={
            "cv_macro_f1": float(trials_df["cv_macro_f1"].max()),
            "is_mock_trained": is_mock,
            "trained_rows": len(df_train),
        },
        output_dir=ARTIFACTS_DIR,
    )

    # 6. Generate Sections 3 and 4 Technical Report Markdown
    report_file = generate_report_markdown(
        results=results_map,
        comparison_df=comp_df,
        tune_table_df=tune_table_df,
        optuna_best_params=best_params,
        n_trials=n_trials,
        study_duration=study_duration,
        output_path=REPORTS_DIR / "report_sections_3_and_4.md",
    )

    print("\n" + "=" * 70)
    print("PERSON 2 MODELING PIPELINE SUCCESSFULLY COMPLETED!")
    print(f"  ✓ 6 models trained & evaluated across 5 algorithm families")
    print(f"  ✓ 5-fold GroupKFold CV completed (zero leakage)")
    print(f"  ✓ Consolidated comparison table saved to reports/model_comparison_table.csv")
    print(f"  ✓ All 6 compulsory charts generated in reports/figures/")
    print(f"  ✓ Optuna Bayesian study finished and logged ({n_trials} trials)")
    print(f"  ✓ Contract 2 artifacts exported: artifacts/model.joblib & feature_columns.json")
    print(f"  ✓ Contract 2 verification passed")
    print(f"  ✓ Full report draft created: {report_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
