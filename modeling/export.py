"""
PathSnap Person 2: Model & Artifact Exporter
============================================
Exports the selected model strictly adhering to Contract 2:
- artifacts/model.joblib
- artifacts/feature_columns.json
- artifacts/model_metadata.json

Also includes an automated Contract 2 verification test to guarantee that
P3 (HMM decoder) and P4 (API predictor) can load and invoke the artifact
without runtime errors.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from common.contract import (
    ARTIFACTS_DIR,
    FEATURE_COLS_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    extract_positive_probs,
)
from modeling.registry import ModelWrapper


def export_model_artifacts(
    model: ModelWrapper,
    feature_columns: List[str],
    metrics_summary: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Saves the final trained model and associated feature columns to artifacts/.
    """
    out_dir = Path(output_dir) if output_dir else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    cols_path = out_dir / "feature_columns.json"
    meta_path = out_dir / "model_metadata.json"

    # 1. Save feature columns JSON
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    # 2. Save model estimator with joblib
    joblib.dump(model, model_path, compress=3)

    # 3. Save comprehensive metadata
    metadata = {
        "model_name": model.name,
        "algorithm_family": model.family,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns,
        "hyperparameters": model.hyperparameters,
        "random_state": model.random_state,
        "metrics_summary": metrics_summary or {},
        "contract_compliance": "Contract 2 (predict_proba returning 1D positive probabilities)",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Export] Model artifacts successfully exported:")
    print(f"  - Model: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - Feature Columns: {cols_path}")
    print(f"  - Metadata: {meta_path}")

    # 4. Verify Contract 2
    verify_contract_2_artifact(model_path, cols_path)

    return {
        "model_path": model_path,
        "cols_path": cols_path,
        "meta_path": meta_path,
    }


def verify_contract_2_artifact(
    model_path: Path,
    feature_cols_path: Path,
) -> bool:
    """
    Verification test for Contract 2:
    Simulates what P3 (decoder) and P4 (API) will do when loading the artifact.
    """
    print("\n[Contract 2 Verification] Running sanity check on saved model...")

    with open(feature_cols_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    loaded_model = joblib.load(model_path)

    # Create dummy batch of 10 observations
    dummy_data = pd.DataFrame(
        np.random.rand(10, len(features)),
        columns=features,
    )

    probs = loaded_model.predict_proba(dummy_data[features])
    pos_probs = extract_positive_probs(probs)

    assert isinstance(pos_probs, np.ndarray), "Output must be a numpy ndarray"
    assert pos_probs.shape == (10,), f"Expected shape (10,), got {pos_probs.shape}"
    assert np.all(pos_probs >= 0.0) and np.all(pos_probs <= 1.0), "Probabilities must be in [0.0, 1.0]"

    print(f"  ✓ Successfully verified Contract 2: predict_proba returned {pos_probs.shape} floats in [0, 1].")
    return True
