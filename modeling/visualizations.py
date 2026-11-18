"""
PathSnap Person 2: Visualizations Module
========================================
Generates all compulsory plots required by Hackathon Section 3.3:
1. Cross-validation Macro F1-score comparison across all models (with error bars)
2. Student-testing Top-1 Accuracy and Macro F1-score comparison
3. Log Loss comparison across models
4. Confusion matrices for each evaluated model
5. Training loss curves for models producing epoch-wise histories (e.g. MLP)
6. Latency vs. Accuracy Pareto trade-off plot

All figures are saved to reports/figures/ in high resolution (300 DPI PNG).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.contract import FIGURES_DIR
from modeling.harness import ExperimentResult


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_cv_macro_f1_comparison(
    results: Dict[str, ExperimentResult],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot 1: Cross-validation Macro F1-score comparison with standard deviation error bars.
    """
    set_plot_style()
    models = list(results.keys())
    means = [results[m].cv_macro_f1_mean for m in models]
    stds = [results[m].cv_macro_f1_std for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        models,
        means,
        yerr=stds,
        capsize=5,
        color="#2b5c8f",
        edgecolor="#1a365d",
        alpha=0.88,
        width=0.55,
    )

    ax.set_title("Cross-Validation Macro F1-Score (5-Fold GroupKFold)", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Macro F1-Score", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    plt.xticks(rotation=20, ha="right", fontsize=10)

    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + std + 0.02,
            f"{mean:.3f}\n±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "cv_macro_f1_comparison.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_student_test_comparison(
    results: Dict[str, ExperimentResult],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot 2: Student-testing Top-1 Accuracy and Macro F1-score side-by-side.
    """
    set_plot_style()
    models = list(results.keys())
    top1s = [results[m].test_top1_accuracy for m in models]
    f1s = [results[m].test_macro_f1 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width / 2, top1s, width, label="Top-1 Accuracy", color="#2a9d8f", edgecolor="#1e6b62")
    rects2 = ax.bar(x + width / 2, f1s, width, label="Macro F1-Score", color="#e76f51", edgecolor="#a8432b")

    ax.set_title("Student-Testing: Observation-Level Top-1 Accuracy vs. Macro F1", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "student_test_comparison.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_log_loss_comparison(
    results: Dict[str, ExperimentResult],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot 3: Log Loss comparison across models (lower is better probability calibration).
    """
    set_plot_style()
    models = list(results.keys())
    losses = [results[m].test_log_loss for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, losses, color="#e63946", edgecolor="#9b111e", alpha=0.85, width=0.5)

    ax.set_title("Student-Testing Candidate Log Loss (Lower is Better)", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Log Loss (Binary Cross-Entropy)", fontsize=11)
    plt.xticks(rotation=20, ha="right", fontsize=10)

    for bar, val in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            weight="bold",
        )

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "log_loss_comparison.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_all_confusion_matrices(
    results: Dict[str, ExperimentResult],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot 4: Diagnostic Confusion Matrices for all models in a 2x3 grid.
    """
    set_plot_style()
    models = list(results.keys())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows))
    axes = axes.flatten()

    for i, model_name in enumerate(models):
        cm = np.array(results[model_name].confusion_matrix)
        ax = axes[i]
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

        for (r, c), val in np.ndenumerate(cm):
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(c, r, f"{val:,}", ha="center", va="center", color=color, fontsize=11, weight="bold")

        ax.set_title(f"{model_name}", fontsize=11, weight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=9.5)
        ax.set_ylabel("True Label", fontsize=9.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["False Arc (0)", "True Arc (1)"])
        ax.set_yticklabels(["False Arc (0)", "True Arc (1)"])

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "confusion_matrices.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_mlp_training_loss(
    loss_history: List[float],
    save_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Plot 5: Training loss curve for models producing epoch-wise training history (MLP).
    Per hackathon instructions: "Training and validation loss plots only for models that produce epoch-wise training histories."
    """
    if not loss_history:
        return None

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", markersize=3, color="#1d3557", linewidth=2)
    ax.set_title("MLP Training Loss History Across Epochs", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Epoch / Iteration", fontsize=10.5)
    ax.set_ylabel("Loss (Cross-Entropy)", fontsize=10.5)

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "mlp_training_loss_curve.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_efficiency_tradeoff(
    results: Dict[str, ExperimentResult],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot 6: Computational Efficiency vs. Performance Pareto Frontier.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, res in results.items():
        lat = res.inference_time_ms_per_1k
        acc = res.test_top1_accuracy
        ax.scatter(lat, acc, s=120, edgecolors="black", linewidths=1.2, label=name)
        ax.annotate(
            name,
            (lat, acc),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9.5,
            weight="bold",
        )

    ax.set_title("Model Selection Trade-off: Top-1 Accuracy vs. Inference Latency", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Inference Latency (ms per 1,000 candidates)", fontsize=10.5)
    ax.set_ylabel("Observation-level Top-1 Accuracy", fontsize=10.5)

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "efficiency_tradeoff.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def generate_all_visualizations(
    results: Dict[str, ExperimentResult],
    mlp_loss_history: Optional[List[float]] = None,
) -> Dict[str, Path]:
    """
    Generate and save the entire suite of required charts.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated = {
        "cv_macro_f1": plot_cv_macro_f1_comparison(results),
        "student_test_comparison": plot_student_test_comparison(results),
        "log_loss_comparison": plot_log_loss_comparison(results),
        "confusion_matrices": plot_all_confusion_matrices(results),
        "efficiency_tradeoff": plot_efficiency_tradeoff(results),
    }
    if mlp_loss_history:
        mlp_path = plot_mlp_training_loss(mlp_loss_history)
        if mlp_path:
            generated["mlp_loss"] = mlp_path

    print(f"\n[Visualizations] Successfully saved {len(generated)} compulsory plots to {FIGURES_DIR}")
    return generated
