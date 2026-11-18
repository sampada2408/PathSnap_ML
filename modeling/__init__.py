"""
PathSnap Modeling Module (Person 2)
===================================
Contains:
- mock_data: Generator for Contract 1 stand-in candidate tables
- metrics: Evaluation metrics (Top-1 Acc, Miss Rate, Macro F1, Log Loss, etc.)
- registry: 6 models across 5 distinct algorithm families
- harness: 5-fold GroupKFold experiment harness & comparison table generator
- visualizations: Compulsory charts for Hackathon Section 3.3
- tune: Optuna Bayesian hyperparameter optimization for Hackathon Section 4
- export: Artifact export adhering to Contract 2
- report_draft: Complete technical report text for Sections 3 & 4
- train: Master CLI entry point
"""

from modeling.export import export_model_artifacts, verify_contract_2_artifact
from modeling.harness import evaluate_single_model, run_all_experiments
from modeling.metrics import (
    candidate_log_loss,
    candidate_miss_rate,
    compute_all_metrics,
    observation_top1_accuracy,
    route_continuity_proxy,
)
from modeling.mock_data import generate_and_save_mock_tables, generate_mock_candidates
from modeling.registry import (
    MODEL_REGISTRY,
    ModelWrapper,
    get_all_registered_models,
)

__all__ = [
    "generate_mock_candidates",
    "generate_and_save_mock_tables",
    "compute_all_metrics",
    "observation_top1_accuracy",
    "candidate_miss_rate",
    "candidate_log_loss",
    "route_continuity_proxy",
    "ModelWrapper",
    "MODEL_REGISTRY",
    "get_all_registered_models",
    "evaluate_single_model",
    "run_all_experiments",
    "export_model_artifacts",
    "verify_contract_2_artifact",
]
