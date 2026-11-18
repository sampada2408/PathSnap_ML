
import numpy as np


def viterbi_decode(
    emission_log_probs: list[np.ndarray],
    transition_log_probs: list[np.ndarray],
) -> list[int]:
    """
    Find the most likely sequence of candidate indices
    using the Viterbi algorithm.

    Parameters
    ----------
    emission_log_probs:
        One 1D array per GPS point.
        Each array contains log emission probabilities
        for that point's candidate arcs.

    transition_log_probs:
        One 2D array between each consecutive pair
        of GPS points.
        Shape: (previous_candidates, current_candidates).

    Returns
    -------
    list[int]:
        Selected candidate index for each GPS point.
    """

    n_steps = len(emission_log_probs)

    if n_steps == 0:
        return []

    if len(transition_log_probs) != n_steps - 1:
        raise ValueError(
            "There must be exactly one transition matrix "
            "between consecutive GPS points."
        )

    emissions = [
        np.asarray(values, dtype=float)
        for values in emission_log_probs
    ]

    for values in emissions:
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(
                "Every GPS point must have a non-empty "
                "1D emission array."
            )

        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("Invalid emission log probabilities.")

    # Store the best score for each candidate at each time step
    scores = emissions[0].copy()

    # Store the best previous candidate for backtracking
    backpointers = []

    for t in range(1, n_steps):

        transitions = np.asarray(
            transition_log_probs[t - 1],
            dtype=float
        )

        expected_shape = (
            len(emissions[t - 1]),
            len(emissions[t])
        )

        if transitions.shape != expected_shape:
            raise ValueError(
                f"Transition matrix at step {t} has shape "
                f"{transitions.shape}; expected {expected_shape}."
            )

        if np.isnan(transitions).any() or np.isposinf(transitions).any():
            raise ValueError("Invalid transition log probabilities.")

        # Previous score + transition score
        combined = scores[:, np.newaxis] + transitions

        # Find the best previous candidate for each current candidate
        best_previous = np.argmax(combined, axis=0)

        # Update scores using the best previous candidate
        scores = (
            combined[best_previous, np.arange(len(emissions[t]))]
            + emissions[t]
        )

        backpointers.append(best_previous)

    # Select the best candidate at the final GPS point
    if not np.isfinite(scores).any():
        raise ValueError(
            "No valid path exists through the candidate sequence."
        )

    best_final = int(np.argmax(scores))

    # Backtrack to recover the complete sequence
    path = [best_final]

    for pointers in reversed(backpointers):
        best_final = int(pointers[best_final])
        path.append(best_final)

    path.reverse()

    return path
