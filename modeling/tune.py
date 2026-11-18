"""
PathSnap Person 2: Bayesian Hyperparameter Optimization (Optuna)
===============================================================
Performs Bayesian Hyperparameter Optimization on the chosen model family
(LightGBM / Gradient Boosting) strictly adhering to Hackathon Section 4:
- GroupKFold on 80 training trips (0 evaluator leakage)
- Justified search space (tree depth, leaf count, shrinkage, regularization, feature subsampling)
- Objective: 5-fold CV Macro F1-score
- Records all trials to reports/optuna_trials.csv
- Compares untuned vs. tuned model across all compulsory metrics
- Generates optimization progression and parameter importance visualisations
"""

from __future__ import annotations

try:
    import lightgbm
except ImportError:
    pass

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from common.contract import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    LABEL_COLUMN,
    REPORTS_DIR,
    df_to_markdown_safe,
)
from modeling.harness import evaluate_single_model
from modeling.metrics import compute_all_metrics
from modeling.registry import ModelWrapper, create_lightgbm

# Suppress verbose Optuna logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimization(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_trials: int = 25,
    n_splits: int = 5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[ModelWrapper, ModelWrapper, pd.DataFrame, pd.DataFrame]:
    """
    Executes Bayesian Hyperparameter Optimization using Optuna TPESampler.
    """
    cols = feature_cols if feature_cols is not None else [
        c for c in FEATURE_COLUMNS if c in df_train.columns
    ]
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n======================================================================")
    print("4. BAYESIAN HYPERPARAMETER OPTIMISATION (Optuna TPE)")
    print(f"Target Model: LightGBM Classifier")
    print(f"Objective: 5-Fold GroupKFold Macro F1 on {df_train['trip_id'].nunique()} Training Trips")
    print(f"Number of Trials: {n_trials} | Random Seed: {seed}")
    print("======================================================================")

    # 1. Baseline / Untuned model
    print("\n[Optuna] Evaluating Untuned Baseline Model...")
    untuned_model = create_lightgbm(random_state=seed)
    untuned_res, untuned_fitted = evaluate_single_model(
        untuned_model, df_train, df_test, cols, n_splits=n_splits
    )

    # 2. Objective Function for Optuna
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_train["trip_id"].values
    X_train = df_train[cols].values
    y_train = df_train[LABEL_COLUMN].values

    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 220, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.25, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            df_val = df_train.iloc[val_idx]

            import lightgbm as lgb
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_val)
            m = compute_all_metrics(df_val, probs)
            fold_scores.append(m["macro_f1"])

        mean_f1 = float(np.mean(fold_scores))
        duration = time.perf_counter() - t0

        trial_records.append(
            {
                "trial_number": trial.number,
                "cv_macro_f1": mean_f1,
                "duration_seconds": round(duration, 3),
                **params,
            }
        )

        return mean_f1

    # 3. Create Study & Optimize
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t_study_start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    study_duration = time.perf_counter() - t_study_start

    print(f"\n[Optuna] Study complete in {study_duration:.2f}s!")
    print(f"[Optuna] Best Trial: #{study.best_trial.number} with CV Macro F1: {study.best_value:.4f}")
    print(f"[Optuna] Best Hyperparameters: {study.best_params}")

    # 4. Save Trial Log CSV
    trials_df = pd.DataFrame(trial_records)
    trials_csv = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"[Optuna] Saved trial log to {trials_csv}")

    # 5. Fit & Evaluate Tuned Model
    print("\n[Optuna] Evaluating Final Tuned Model on Student-Test...")
    tuned_model = create_lightgbm(random_state=seed, **study.best_params)
    tuned_res, tuned_fitted = evaluate_single_model(
        tuned_model, df_train, df_test, cols, n_splits=n_splits
    )

    # 6. Generate Section 4 Comparison Table
    comp_rows = [
        {
            "Metric": "Cross-Validation Macro F1-score",
            "Untuned Model": f"{untuned_res.cv_macro_f1_mean:.4f} ± {untuned_res.cv_macro_f1_std:.4f}",
            "Tuned Model": f"{tuned_res.cv_macro_f1_mean:.4f} ± {tuned_res.cv_macro_f1_std:.4f}",
            "Improvement": f"{(tuned_res.cv_macro_f1_mean - untuned_res.cv_macro_f1_mean):+.4f}",
        },
        {
            "Metric": "Student-Testing Top-1 Accuracy",
            "Untuned Model": f"{untuned_res.test_top1_accuracy:.4f}",
            "Tuned Model": f"{tuned_res.test_top1_accuracy:.4f}",
            "Improvement": f"{(tuned_res.test_top1_accuracy - untuned_res.test_top1_accuracy):+.4f}",
        },
        {
            "Metric": "Student-Testing Macro F1-score",
            "Untuned Model": f"{untuned_res.test_macro_f1:.4f}",
            "Tuned Model": f"{tuned_res.test_macro_f1:.4f}",
            "Improvement": f"{(tuned_res.test_macro_f1 - untuned_res.test_macro_f1):+.4f}",
        },
        {
            "Metric": "Student-Testing Candidate Log Loss",
            "Untuned Model": f"{untuned_res.test_log_loss:.4f}",
            "Tuned Model": f"{tuned_res.test_log_loss:.4f}",
            "Improvement": f"{(tuned_res.test_log_loss - untuned_res.test_log_loss):+.4f}",
        },
        {
            "Metric": "Inference Time (ms per 1k)",
            "Untuned Model": f"{untuned_res.inference_time_ms_per_1k:.2f}",
            "Tuned Model": f"{tuned_res.inference_time_ms_per_1k:.2f}",
            "Improvement": f"{(tuned_res.inference_time_ms_per_1k - untuned_res.inference_time_ms_per_1k):+.2f}",
        },
        {
            "Metric": "Training Time (s)",
            "Untuned Model": f"{untuned_res.train_time_seconds:.2f}",
            "Tuned Model": f"{tuned_res.train_time_seconds:.2f}",
            "Improvement": f"{(tuned_res.train_time_seconds - untuned_res.train_time_seconds):+.2f}",
        },
    ]
    tune_table_df = pd.DataFrame(comp_rows)
    tune_csv = out_dir / "bayesian_optimization_table.csv"
    tune_md = out_dir / "bayesian_optimization_table.md"
    tune_table_df.to_csv(tune_csv, index=False)

    with open(tune_md, "w", encoding="utf-8") as f:
        f.write("# 4. Bayesian Hyperparameter Optimisation Comparison\n\n")
        f.write(f"**Optimization Cost:** {study_duration:.2f} seconds across {n_trials} trials.\n\n")
        f.write(f"**Best Trial:** #{study.best_trial.number} (CV Macro F1 = {study.best_value:.4f})\n\n")
        f.write("```json\n" + json.dumps(study.best_params, indent=2) + "\n```\n\n")
        f.write(df_to_markdown_safe(tune_table_df))
        f.write("\n")

    print("\n======================================================================")
    print("4. UNTUNED VS. TUNED MODEL COMPARISON")
    print("======================================================================")
    print(tune_table_df.to_string(index=False))

    # 7. Generate Visualizations for Section 4
    plot_optimization_progression(trials_df, FIGURES_DIR / "optuna_optimization_history.png")
    plot_tuned_vs_untuned_bars(tune_table_df, FIGURES_DIR / "tuned_vs_untuned_comparison.png")

    return untuned_fitted, tuned_fitted, trials_df, tune_table_df


def plot_optimization_progression(trials_df: pd.DataFrame, save_path: Path):
    """
    Plot optimization progression over trials (best value vs trial value).
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    trials = trials_df["trial_number"].values
    scores = trials_df["cv_macro_f1"].values
    best_so_far = np.maximum.accumulate(scores)

    ax.scatter(trials, scores, color="#3a86ff", alpha=0.7, label="Trial Score (Macro F1)", s=45)
    ax.plot(trials, best_so_far, color="#e63946", linewidth=2.2, label="Best Score So Far")

    ax.set_title("Optuna Bayesian Optimization Progression (LightGBM)", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Trial Number", fontsize=10.5)
    ax.set_ylabel("5-Fold CV Macro F1-Score", fontsize=10.5)
    ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_tuned_vs_untuned_bars(comp_df: pd.DataFrame, save_path: Path):
    """
    Side-by-side bar chart of Untuned vs Tuned metrics.
    """
    plot_metrics = ["Student-Testing Top-1 Accuracy", "Student-Testing Macro F1-score"]
    sub_df = comp_df[comp_df["Metric"].isin(plot_metrics)].copy()
    if sub_df.empty:
        return

    labels = ["Top-1 Accuracy", "Macro F1-Score"]
    untuned_vals = [float(v) for v in sub_df["Untuned Model"].values]
    tuned_vals = [float(v) for v in sub_df["Tuned Model"].values]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    r1 = ax.bar(x - width / 2, untuned_vals, width, label="Untuned LightGBM", color="#457b9d")
    r2 = ax.bar(x + width / 2, tuned_vals, width, label="Tuned LightGBM (Optuna)", color="#2a9d8f")

    ax.set_title("Performance Gain: Untuned vs. Bayesian Tuned LightGBM", fontsize=12, weight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, facecolor="white")

    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9, weight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Bayesian Hyperparameter Optimization")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    args = parser.parse_args()

    from common.contract import PROCESSED_DATA_DIR

    train_file = PROCESSED_DATA_DIR / ("mock_train.parquet" if args.mock else "train.parquet")
    test_file = PROCESSED_DATA_DIR / ("mock_student_test.parquet" if args.mock else "student_test.parquet")

    if not train_file.exists():
        from modeling.mock_data import generate_and_save_mock_tables
        generate_and_save_mock_tables()
        train_file = PROCESSED_DATA_DIR / "mock_train.parquet"
        test_file = PROCESSED_DATA_DIR / "mock_student_test.parquet"

    df_tr = pd.read_parquet(train_file)
    df_te = pd.read_parquet(test_file)

    run_bayesian_optimization(df_tr, df_te, n_trials=args.trials)
