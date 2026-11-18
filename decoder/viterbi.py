"""
PathSnap Person 3: Viterbi Trellis Decoder (Contract 3 Implementation)
=====================================================================
Implements the Viterbi dynamic programming decoding algorithm for map-matching
strictly satisfying Contract 3:
    decode(cands, probs, arcs, nodes, times) -> list[int]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from common.contract import (
    KEY_COLUMNS,
    extract_positive_probs,
)
from decoder.emission import (
    classifier_log_emission,
    gaussian_log_emission,
    hybrid_log_emission,
)
from decoder.graph import RoadNetworkGraph, equirectangular_distance
from decoder.transition import (
    DEFAULT_BETA,
    DEFAULT_GAP_THRESHOLD_S,
    DEFAULT_UNREACHABLE_PENALTY_M,
    compute_log_transition_probability,
)


class ViterbiDecoder:
    """
    State-of-the-art Viterbi Trellis Map-Matching Decoder.
    """

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        sigma_z: float = 4.07,
        unreachable_penalty_m: float = DEFAULT_UNREACHABLE_PENALTY_M,
        gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
        dijkstra_cutoff_m: float = 1000.0,
        clf_weight: float = 0.85,
        use_hybrid_emission: bool = True,
    ):
        self.beta = beta
        self.sigma_z = sigma_z
        self.unreachable_penalty_m = unreachable_penalty_m
        self.gap_threshold_s = gap_threshold_s
        self.dijkstra_cutoff_m = dijkstra_cutoff_m
        self.clf_weight = clf_weight
        self.use_hybrid_emission = use_hybrid_emission

    def decode(
        self,
        cands: pd.DataFrame,
        probs: Optional[np.ndarray],
        arcs: np.ndarray,
        nodes: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Contract 3 implementation:
        Takes candidate observations, emission probabilities, and road network,
        returning the reconstructed sequence of arc IDs.
        """
        graph = RoadNetworkGraph(nodes, arcs)

        # Ensure candidates are sorted by point_idx
        df = cands.copy()
        if probs is not None:
            df["prob"] = extract_positive_probs(probs)
        else:
            df["prob"] = 1.0

        if "dist_m" not in df.columns:
            df["dist_m"] = 5.0
        if "frac_along" not in df.columns:
            df["frac_along"] = 0.5

        # Group candidates by point_idx
        unique_points = sorted(df["point_idx"].unique())
        if not unique_points:
            return []

        T = len(unique_points)
        point_groups = dict(tuple(df.groupby("point_idx")))

        # Approximate GPS coordinates per point from candidate projections or nodes
        # If timestamps not provided, assume 1.0 s nominal intervals
        if times is not None and len(times) == T:
            t_arr = np.asarray(times, dtype=float)
        else:
            t_arr = np.arange(T, dtype=float)

        # Precompute emission log-likelihoods per point
        trellis_states: List[List[int]] = []
        trellis_fracs: List[List[float]] = []
        trellis_emissions: List[List[float]] = []

        for pt_idx in unique_points:
            pt_df = point_groups[pt_idx]
            cand_arcs = pt_df["arc_id"].values.astype(int).tolist()
            cand_fracs = pt_df["frac_along"].values.astype(float).tolist()
            cand_dists = pt_df["dist_m"].values.astype(float)
            cand_probs = pt_df["prob"].values.astype(float)

            if probs is not None:
                if self.use_hybrid_emission:
                    log_emits = hybrid_log_emission(
                        distances_m=cand_dists,
                        probs=cand_probs,
                        sigma_z=self.sigma_z,
                        clf_weight=self.clf_weight,
                    )
                else:
                    log_emits = classifier_log_emission(cand_probs)
            else:
                log_emits = gaussian_log_emission(cand_dists, sigma_z=self.sigma_z)

            trellis_states.append(cand_arcs)
            trellis_fracs.append(cand_fracs)
            trellis_emissions.append(log_emits.tolist())

        # Viterbi Trellis Forward Pass
        V: List[np.ndarray] = [np.array(trellis_emissions[0], dtype=float)]
        backpointers: List[np.ndarray] = []

        for t in range(1, T):
            prev_states = trellis_states[t - 1]
            curr_states = trellis_states[t]
            prev_fracs = trellis_fracs[t - 1]
            curr_fracs = trellis_fracs[t]
            curr_emits = np.array(trellis_emissions[t], dtype=float)

            K_prev = len(prev_states)
            K_curr = len(curr_states)

            dt = max(0.1, t_arr[t] - t_arr[t - 1])

            # Approximate d_gps from candidate locations or central difference
            # Median distance between top-1 candidates
            d_gps = 15.0 * dt  # Realistic default: ~15 m/s

            v_prev = V[t - 1]
            v_curr = np.full(K_curr, -1e12, dtype=float)
            bp_curr = np.zeros(K_curr, dtype=int)

            for j in range(K_curr):
                arc_j = curr_states[j]
                frac_j = curr_fracs[j]
                best_val = -1e12
                best_idx = 0

                for i in range(K_prev):
                    arc_i = prev_states[i]
                    frac_i = prev_fracs[i]

                    log_trans = compute_log_transition_probability(
                        graph=graph,
                        arc_prev=arc_i,
                        arc_curr=arc_j,
                        d_gps=d_gps,
                        frac_prev=frac_i,
                        frac_curr=frac_j,
                        dt_seconds=dt,
                        beta=self.beta,
                        unreachable_penalty_m=self.unreachable_penalty_m,
                        gap_threshold_s=self.gap_threshold_s,
                        dijkstra_cutoff_m=self.dijkstra_cutoff_m,
                    )

                    score = v_prev[i] + log_trans
                    if score > best_val:
                        best_val = score
                        best_idx = i

                v_curr[j] = best_val + curr_emits[j]
                bp_curr[j] = best_idx

            V.append(v_curr)
            backpointers.append(bp_curr)

        # Viterbi Backtracking Pass
        best_last_idx = int(np.argmax(V[-1]))
        recovered_indices = [best_last_idx]

        for t in range(T - 1, 0, -1):
            best_idx = backpointers[t - 1][recovered_indices[-1]]
            recovered_indices.append(int(best_idx))

        recovered_indices.reverse()

        # Map back to arc IDs
        predicted_arcs = [
            trellis_states[t][idx] for t, idx in enumerate(recovered_indices)
        ]

        return predicted_arcs


def decode(
    cands: pd.DataFrame,
    probs: Optional[np.ndarray],
    arcs: np.ndarray,
    nodes: np.ndarray,
    times: Optional[np.ndarray] = None,
    decoder_config: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """
    Contract 3 Functional Interface:
    decode(cands, probs, arcs, nodes, times) -> list[int]
    """
    cfg = decoder_config or {}
    decoder_inst = ViterbiDecoder(
        beta=cfg.get("beta", DEFAULT_BETA),
        sigma_z=cfg.get("sigma_z", 4.07),
        unreachable_penalty_m=cfg.get("unreachable_penalty_m", DEFAULT_UNREACHABLE_PENALTY_M),
        gap_threshold_s=cfg.get("gap_threshold_s", DEFAULT_GAP_THRESHOLD_S),
        dijkstra_cutoff_m=cfg.get("dijkstra_cutoff_m", 1000.0),
        clf_weight=cfg.get("clf_weight", 0.85),
    )
    return decoder_inst.decode(cands=cands, probs=probs, arcs=arcs, nodes=nodes, times=times)
