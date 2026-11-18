"""
PathSnap Person 2: Experiment Harness
=====================================
Orchestrates model evaluation with:
- GroupKFold by trip_id (5-fold, strict 0-leakage guarantee)
- Full metrics computation: Top-1 Accuracy, Macro F1, Log Loss, Candidate Miss Rate, Continuity
- Timing: Training time and Inference latency (ms per 1,000 candidates)
- Comprehensive comparison table generation adhering to Hackathon Section 3.1
- Detailed experiment logging (configs, random seeds, fold metrics, timings)
"""

from __future__ import annotations

try:
    import lightgbm
except ImportError:
    pass

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common.contract import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    LABEL_COLUMN,
    REPORTS_DIR,
    df_to_markdown_safe,
    validate_candidate_table,
)
from modeling.metrics import compute_all_metrics
from modeling.registry import ModelWrapper, get_all_registered_models


@dataclass
class ExperimentResult:
    model_name: str
    algorithm_family: str
    key_configuration: str
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    cv_top1_acc_mean: float
    cv_top1_acc_std: float
    cv_log_loss_mean: float
    cv_log_loss_std: float
    test_top1_accuracy: float
    test_macro_f1: float
    test_log_loss: float
    test_precision: float
    test_recall: float
    test_route_continuity: float
    train_time_seconds: float
    inference_time_ms_per_1k: float
    confusion_matrix: List[List[int]]
    hyperparameters: Dict[str, Any]
    random_state: int


def run_group_cv_for_model(
    model: ModelWrapper,
    df_train: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Perform 5-fold GroupKFold cross-validation grouped strictly by trip_id.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_train["trip_id"].values
    X = df_train[feature_cols].values
    y = df_train[LABEL_COLUMN].values

    fold_metrics = []
    fit_times = []
    inf_times = []

    print(f"\n--- Cross-Validation: {model.name} ({model.family}) ---")

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        df_val = df_train.iloc[val_idx].copy()

        # Fit model
        t0 = time.perf_counter()
        model.estimator.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t0
        fit_times.append(t_fit)

        # Inference
        t1 = time.perf_counter()
        raw_probs = model.estimator.predict_proba(X_val)
        t_inf = time.perf_counter() - t1
        inf_latency_per_1k = (t_inf / len(X_val)) * 1000.0 * 1000.0  # ms per 1,000 samples
        inf_times.append(inf_latency_per_1k)

        # Metrics
        m = compute_all_metrics(df_val, raw_probs)
        fold_metrics.append(m)
        print(
            f"  Fold {fold_idx}/{n_splits} | "
            f"Macro F1: {m['macro_f1']:.4f} | "
            f"Top-1 Acc: {m['top1_accuracy']:.4f} | "
            f"Log Loss: {m['log_loss']:.4f} | "
            f"Fit Time: {t_fit:.2f}s"
        )

    # Compute aggregate stats
    macro_f1s = [m["macro_f1"] for m in fold_metrics]
    top1_accs = [m["top1_accuracy"] for m in fold_metrics]
    log_losses = [m["log_loss"] for m in fold_metrics]

    return {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "top1_acc_mean": float(np.mean(top1_accs)),
        "top1_acc_std": float(np.std(top1_accs)),
        "log_loss_mean": float(np.mean(log_losses)),
        "log_loss_std": float(np.std(log_losses)),
        "avg_fit_time_s": float(np.mean(fit_times)),
        "avg_inf_ms_per_1k": float(np.mean(inf_times)),
        "fold_metrics": fold_metrics,
    }


def evaluate_single_model(
    model: ModelWrapper,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
) -> Tuple[ExperimentResult, ModelWrapper]:
    """
    Evaluates a model via GroupKFold CV on train partition,
    then fits on full train partition and evaluates on student-test partition.
    """
    # 1. Group CV
    cv_summary = run_group_cv_for_model(model, df_train, feature_cols, n_splits=n_splits)

    # 2. Fit on full training data
    print(f"  Fitting {model.name} on all {df_train['trip_id'].nunique()} training trips...")
    X_train_full = df_train[feature_cols].values
    y_train_full = df_train[LABEL_COLUMN].values

    t0 = time.perf_counter()
    model.fit(df_train[feature_cols], y_train_full)
    total_train_time = time.perf_counter() - t0

    # 3. Evaluate on student-test data
    X_test = df_test[feature_cols].values
    t1 = time.perf_counter()
    test_probs = model.predict_proba(df_test[feature_cols])
    inf_duration = time.perf_counter() - t1
    inf_latency_per_1k = (inf_duration / len(X_test)) * 1000.0 * 1000.0

    test_metrics = compute_all_metrics(df_test, test_probs)

    print(
        f"  Student-Test Results -> Top-1 Acc: {test_metrics['top1_accuracy']:.4f} | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | "
        f"Log Loss: {test_metrics['log_loss']:.4f} | "
        f"Route Continuity: {test_metrics['route_continuity']:.4f}"
    )

    # Format key config
    key_cfg = ", ".join(f"{k}={v}" for k, v in list(model.hyperparameters.items())[:3])

    result = ExperimentResult(
        model_name=model.name,
        algorithm_family=model.family,
        key_configuration=key_cfg,
        cv_macro_f1_mean=cv_summary["macro_f1_mean"],
        cv_macro_f1_std=cv_summary["macro_f1_std"],
        cv_top1_acc_mean=cv_summary["top1_acc_mean"],
        cv_top1_acc_std=cv_summary["top1_acc_std"],
        cv_log_loss_mean=cv_summary["log_loss_mean"],
        cv_log_loss_std=cv_summary["log_loss_std"],
        test_top1_accuracy=test_metrics["top1_accuracy"],
        test_macro_f1=test_metrics["macro_f1"],
        test_log_loss=test_metrics["log_loss"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        test_route_continuity=test_metrics["route_continuity"],
        train_time_seconds=round(total_train_time, 3),
        inference_time_ms_per_1k=round(inf_latency_per_1k, 2),
        confusion_matrix=test_metrics["confusion_matrix"],
        hyperparameters=model.hyperparameters,
        random_state=model.random_state,
    )
    return result, model


def run_all_experiments(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, ExperimentResult], Dict[str, ModelWrapper]]:
    """
    Run experiments across all registered models and build comparison table.
    """
    cols = feature_cols if feature_cols is not None else [
        c for c in FEATURE_COLUMNS if c in df_train.columns
    ]
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    models = get_all_registered_models(random_state=random_state)
    results_map: Dict[str, ExperimentResult] = {}
    fitted_models: Dict[str, ModelWrapper] = {}
    table_rows = []

    print("======================================================================")
    print(f"PATHSNAP MODEL EXPERIMENTATION HARNESS (P2)")
    print(f"Training Trips: {df_train['trip_id'].nunique()} ({len(df_train):,} candidates)")
    print(f"Student-Test Trips: {df_test['trip_id'].nunique()} ({len(df_test):,} candidates)")
    print(f"Features: {cols}")
    print(f"Models to evaluate: {len(models)}")
    print("======================================================================")

    for model in models:
        res, fitted = evaluate_single_model(model, df_train, df_test, cols)
        results_map[model.name] = res
        fitted_models[model.name] = fitted

        table_rows.append(
            {
                "Model": res.model_name,
                "Algorithm Family": res.algorithm_family,
                "Key Configuration": res.key_configuration,
                "CV Macro F1-score": f"{res.cv_macro_f1_mean:.4f} ± {res.cv_macro_f1_std:.4f}",
                "Top-1 Accuracy": f"{res.test_top1_accuracy:.4f}",
                "Route continuity": f"{res.test_route_continuity:.4f}",
                "Student-Test Macro F1-score": f"{res.test_macro_f1:.4f}",
                "Log Loss": f"{res.test_log_loss:.4f}",
                "Inference Time (ms/1k)": f"{res.inference_time_ms_per_1k:.2f}",
                "Train Time (s)": f"{res.train_time_seconds:.2f}",
            }
        )

    comparison_df = pd.DataFrame(table_rows)

    # Save outputs
    csv_path = out_dir / "model_comparison_table.csv"
    md_path = out_dir / "model_comparison_table.md"
    json_path = out_dir / "experiment_log.json"

    comparison_df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 3.1 Overall Model Comparison Table\n\n")
        f.write(df_to_markdown_safe(comparison_df))
        f.write("\n")

    serializable_results = {k: asdict(v) for k, v in results_map.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    print("\n======================================================================")
    print("3.1 CONSOLIDATED MODEL COMPARISON TABLE")
    print("======================================================================")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison table to {csv_path} and {md_path}")
    print(f"Saved full experiment logs to {json_path}")

    return comparison_df, results_map, fitted_models
