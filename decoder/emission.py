"""
PathSnap Person 3: Emission Probability Models
==============================================
Implements emission probability modeling for map-matching:
1. Classic Newson–Krumm (2009) distance-based Gaussian emission baseline
2. Machine-learning classifier-based emission model (Contract 2 probabilities)
3. Hybrid emission model balancing ML discrimination and physical distance guarantees
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import pandas as pd

from common.contract import extract_positive_probs

# Default standard deviation of GPS measurement error (meters), from Newson & Krumm 2009
DEFAULT_SIGMA_Z = 4.07


def gaussian_log_emission(
    distances_m: Union[np.ndarray, float],
    sigma_z: float = DEFAULT_SIGMA_Z,
) -> np.ndarray:
    """
    Computes log emission probability using the classic zero-mean Gaussian
    distribution on perpendicular GPS measurement error (Newson & Krumm 2009):
        p(z_t | s_t = a) = (1 / (sqrt(2*pi) * sigma_z)) * exp(-dist^2 / (2 * sigma_z^2))
        log p(z_t | s_t = a) = -0.5 * log(2*pi*sigma_z^2) - dist^2 / (2 * sigma_z^2)
    """
    d = np.asarray(distances_m, dtype=float)
    c = -0.5 * math.log(2.0 * math.pi * (sigma_z ** 2))
    return c - (d ** 2) / (2.0 * (sigma_z ** 2))


def classifier_log_emission(
    probs: Union[np.ndarray, list],
    eps: float = 1e-9,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Converts candidate probabilities P(arc is correct) from P2's classifier into
    log emission probabilities:
        log p(z_t | s_t = a) = log(max(P(a), eps)) / temperature
    """
    p = extract_positive_probs(np.asarray(probs, dtype=float))
    p = np.clip(p, eps, 1.0)
    if temperature != 1.0 and temperature > 0:
        return np.log(p) / temperature
    return np.log(p)


def hybrid_log_emission(
    distances_m: np.ndarray,
    probs: np.ndarray,
    sigma_z: float = DEFAULT_SIGMA_Z,
    clf_weight: float = 0.85,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Blends classifier confidence with physical Gaussian distance likelihood:
        log p_hybrid = clf_weight * log p_clf + (1 - clf_weight) * log p_gauss
    """
    log_gauss = gaussian_log_emission(distances_m, sigma_z=sigma_z)
    log_clf = classifier_log_emission(probs, eps=eps)

    # Normalize within each point's candidate set for numeric balance
    return clf_weight * log_clf + (1.0 - clf_weight) * log_gauss
