
import numpy as np
import pandas as pd


def gaussian_emission_probs(
    candidates: pd.DataFrame,
    sigma_m: float = 10.0
) -> np.ndarray:
    """
    Calculate Gaussian emission likelihoods for candidate road arcs.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Must contain the 'dist_m' column.
        Each row represents one GPS-point/candidate-arc pair.

    sigma_m : float
        Assumed GPS measurement noise standard deviation in metres.

    Returns
    -------
    numpy.ndarray
        Emission likelihood for every candidate row.
        Output order matches the input DataFrame.
    """

    if sigma_m <= 0:
        raise ValueError("sigma_m must be greater than zero")

    if "dist_m" not in candidates.columns:
        raise ValueError("Candidate table must contain 'dist_m'")

    distances = candidates["dist_m"].to_numpy(dtype=float)

    if not np.isfinite(distances).all():
        raise ValueError("dist_m contains invalid values")

    if np.any(distances < 0):
        raise ValueError("Distances cannot be negative")

    # Gaussian probability density
    coefficient = 1.0 / (
        np.sqrt(2.0 * np.pi) * sigma_m
    )

    probabilities = coefficient * np.exp(
        -(distances ** 2) / (2.0 * sigma_m ** 2)
    )

    return probabilities


def gaussian_emission_log_probs(
    candidates: pd.DataFrame,
    sigma_m: float = 10.0
) -> np.ndarray:
    """
    Calculate log Gaussian emission likelihoods.

    Log likelihoods are useful for the Viterbi algorithm
    because they avoid numerical underflow.
    """

    if sigma_m <= 0:
        raise ValueError("sigma_m must be greater than zero")

    if "dist_m" not in candidates.columns:
        raise ValueError("Candidate table must contain 'dist_m'")

    distances = candidates["dist_m"].to_numpy(dtype=float)

    if not np.isfinite(distances).all():
        raise ValueError("dist_m contains invalid values")

    if np.any(distances < 0):
        raise ValueError("Distances cannot be negative")

    log_coefficient = -np.log(
        np.sqrt(2.0 * np.pi) * sigma_m
    )

    log_probs = (
        log_coefficient
        - (distances ** 2) / (2.0 * sigma_m ** 2)
    )

    return log_probs
