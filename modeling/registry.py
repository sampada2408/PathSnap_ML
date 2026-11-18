"""
PathSnap Person 2: Model Registry
=================================
Implements model definitions across 5 distinct algorithm families:
1. Linear & Probabilistic: Logistic Regression (with StandardScaler pipeline)
2. Ensemble / Bagging: Random Forest
3. Boosting (Scikit-Learn): HistGradientBoosting
4. Boosting (LightGBM): LightGBM Classifier
5. Boosting (XGBoost): XGBoost Classifier
6. Neural Networks: Multi-Layer Perceptron (MLP with StandardScaler pipeline)

All models adhere strictly to Contract 2:
predict_proba(X) -> 1D float array of P(arc is correct).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from common.contract import extract_positive_probs


class ModelWrapper:
    """
    Standardized wrapper guaranteeing Contract 2:
    - Stores model instance, algorithm family name, hyperparameters, and feature columns
    - Implements predict_proba returning 1D array of positive class probabilities
    """

    def __init__(
        self,
        name: str,
        family: str,
        estimator: Any,
        hyperparameters: Dict[str, Any],
        random_state: int = 42,
    ):
        self.name = name
        self.family = family
        self.estimator = estimator
        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.feature_columns: Optional[List[str]] = None
        self.training_history: Optional[Dict[str, List[float]]] = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "ModelWrapper":
        if isinstance(X, pd.DataFrame):
            self.feature_columns = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)

        self.estimator.fit(X_mat, y)

        # Extract loss history if available
        if hasattr(self.estimator, "loss_curve_"):
            # e.g. MLPClassifier
            self.training_history = {"train_loss": list(self.estimator.loss_curve_)}
        elif hasattr(self.estimator, "named_steps") and hasattr(self.estimator.named_steps.get("mlp"), "loss_curve_"):
            self.training_history = {"train_loss": list(self.estimator.named_steps["mlp"].loss_curve_)}
        elif hasattr(self.estimator, "evals_result_"):
            # LightGBM or XGBoost
            self.training_history = {"evals": self.estimator.evals_result_}

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Contract 2 implementation:
        Returns 1D array of float probabilities representing P(arc is correct).
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_columns is not None:
                # Ensure column order matches
                X_mat = X[self.feature_columns].values
            else:
                X_mat = X.values
        else:
            X_mat = np.asarray(X)

        raw_probs = self.estimator.predict_proba(X_mat)
        return extract_positive_probs(raw_probs)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def create_logistic_regression(random_state: int = 42, **kwargs) -> ModelWrapper:
    params = {
        "C": kwargs.get("C", 1.0),
        "penalty": kwargs.get("penalty", "l2"),
        "solver": kwargs.get("solver", "lbfgs"),
        "max_iter": kwargs.get("max_iter", 1000),
        "class_weight": kwargs.get("class_weight", None),
        "random_state": random_state,
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**params)),
    ])
    return ModelWrapper(
        name="Logistic Regression",
        family="Linear & Probabilistic",
        estimator=pipe,
        hyperparameters=params,
        random_state=random_state,
    )


def create_random_forest(random_state: int = 42, **kwargs) -> ModelWrapper:
    params = {
        "n_estimators": kwargs.get("n_estimators", 100),
        "max_depth": kwargs.get("max_depth", 12),
        "min_samples_split": kwargs.get("min_samples_split", 5),
        "min_samples_leaf": kwargs.get("min_samples_leaf", 2),
        "max_features": kwargs.get("max_features", "sqrt"),
        "n_jobs": kwargs.get("n_jobs", -1),
        "random_state": random_state,
    }
    rf = RandomForestClassifier(**params)
    return ModelWrapper(
        name="Random Forest",
        family="Ensemble (Bagging)",
        estimator=rf,
        hyperparameters=params,
        random_state=random_state,
    )


def create_hist_gradient_boosting(random_state: int = 42, **kwargs) -> ModelWrapper:
    params = {
        "max_iter": kwargs.get("max_iter", 100),
        "learning_rate": kwargs.get("learning_rate", 0.08),
        "max_leaf_nodes": kwargs.get("max_leaf_nodes", 31),
        "min_samples_leaf": kwargs.get("min_samples_leaf", 20),
        "l2_regularization": kwargs.get("l2_regularization", 0.0),
        "random_state": random_state,
    }
    hgb = HistGradientBoostingClassifier(**params)
    return ModelWrapper(
        name="HistGradientBoosting",
        family="Boosting (Scikit-Learn)",
        estimator=hgb,
        hyperparameters=params,
        random_state=random_state,
    )


def create_lightgbm(random_state: int = 42, **kwargs) -> ModelWrapper:
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed. Install via pip install lightgbm")
    params = {
        "n_estimators": kwargs.get("n_estimators", 100),
        "learning_rate": kwargs.get("learning_rate", 0.08),
        "num_leaves": kwargs.get("num_leaves", 31),
        "max_depth": kwargs.get("max_depth", -1),
        "min_child_samples": kwargs.get("min_child_samples", 20),
        "subsample": kwargs.get("subsample", 0.85),
        "colsample_bytree": kwargs.get("colsample_bytree", 0.85),
        "reg_alpha": kwargs.get("reg_alpha", 0.0),
        "reg_lambda": kwargs.get("reg_lambda", 0.0),
        "random_state": random_state,
        "n_jobs": kwargs.get("n_jobs", -1),
        "verbose": -1,
    }
    lgbm = lgb.LGBMClassifier(**params)
    return ModelWrapper(
        name="LightGBM",
        family="Boosting (Gradient Boosting)",
        estimator=lgbm,
        hyperparameters=params,
        random_state=random_state,
    )


def create_xgboost(random_state: int = 42, **kwargs) -> ModelWrapper:
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. Install via pip install xgboost")
    params = {
        "n_estimators": kwargs.get("n_estimators", 100),
        "learning_rate": kwargs.get("learning_rate", 0.08),
        "max_depth": kwargs.get("max_depth", 6),
        "min_child_weight": kwargs.get("min_child_weight", 1),
        "subsample": kwargs.get("subsample", 0.85),
        "colsample_bytree": kwargs.get("colsample_bytree", 0.85),
        "reg_alpha": kwargs.get("reg_alpha", 0.0),
        "reg_lambda": kwargs.get("reg_lambda", 1.0),
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": kwargs.get("n_jobs", -1),
    }
    xgb_clf = xgb.XGBClassifier(**params)
    return ModelWrapper(
        name="XGBoost",
        family="Boosting (Gradient Boosting)",
        estimator=xgb_clf,
        hyperparameters=params,
        random_state=random_state,
    )


def create_mlp(random_state: int = 42, **kwargs) -> ModelWrapper:
    params = {
        "hidden_layer_sizes": kwargs.get("hidden_layer_sizes", (64, 32)),
        "activation": kwargs.get("activation", "relu"),
        "solver": kwargs.get("solver", "adam"),
        "alpha": kwargs.get("alpha", 0.0001),
        "learning_rate_init": kwargs.get("learning_rate_init", 0.001),
        "max_iter": kwargs.get("max_iter", 200),
        "early_stopping": kwargs.get("early_stopping", True),
        "validation_fraction": kwargs.get("validation_fraction", 0.1),
        "random_state": random_state,
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(**params)),
    ])
    return ModelWrapper(
        name="Multi-Layer Perceptron (MLP)",
        family="Neural Networks",
        estimator=pipe,
        hyperparameters=params,
        random_state=random_state,
    )


MODEL_REGISTRY: Dict[str, Callable[..., ModelWrapper]] = {
    "logistic_regression": create_logistic_regression,
    "random_forest": create_random_forest,
    "hist_gradient_boosting": create_hist_gradient_boosting,
    "lightgbm": create_lightgbm,
    "xgboost": create_xgboost,
    "mlp": create_mlp,
}


def get_all_registered_models(random_state: int = 42) -> List[ModelWrapper]:
    """
    Returns an instantiated list of all 6 models across 5 algorithm families.
    """
    models = []
    for key, factory in MODEL_REGISTRY.items():
        try:
            models.append(factory(random_state=random_state))
        except ImportError as e:
            print(f"[Registry] Skipping {key}: {e}")
    return models
