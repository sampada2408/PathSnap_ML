"""
PathSnap Person 3: Transition Probability Model
================================================
Implements the Newson & Krumm (2009) exponential transition model:
- Compares road network distance d_net to straight-line GPS distance d_gps
- Handles unreachable / disconnected road segments with a robust penalty floor
- Dynamically scales parameters during large GPS temporal gaps (tunnels, dropouts)
- Also includes network_distance and transition_log_probability for RoadGraph compatibility.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from decoder.distance import ShortestPath
from decoder.graph import RoadGraph, RoadNetworkGraph

# Default transition scale parameter (beta), from Newson & Krumm 2009
DEFAULT_BETA = 5.0
# Penalty distance for unreachable / disconnected segments (meters)
DEFAULT_UNREACHABLE_PENALTY_M = 350.0
# GPS temporal gap threshold (seconds)
DEFAULT_GAP_THRESHOLD_S = 10.0


def compute_log_transition_probability(
    graph: RoadNetworkGraph,
    arc_prev: int,
    arc_curr: int,
    d_gps: float,
    frac_prev: float = 0.5,
    frac_curr: float = 0.5,
    dt_seconds: float = 1.0,
    beta: float = DEFAULT_BETA,
    unreachable_penalty_m: float = DEFAULT_UNREACHABLE_PENALTY_M,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
    dijkstra_cutoff_m: float = 1000.0,
) -> float:
    """
    Computes log transition probability log p(s_t = arc_curr | s_{t-1} = arc_prev).

    Formula (Newson & Krumm 2009):
        log p = -log(beta_eff) - |d_net - d_gps| / beta_eff
    """
    # 1. Handle large temporal GPS gaps
    if dt_seconds > 60.0:
        # Massive temporal gap: reset transition to neutral uninformative prior
        return -5.0

    if dt_seconds > gap_threshold_s:
        # Dynamically relax beta and expand search radius proportionally to time delta
        scale_factor = min(10.0, dt_seconds / gap_threshold_s)
        effective_beta = beta * scale_factor
        effective_cutoff = dijkstra_cutoff_m * scale_factor
    else:
        effective_beta = beta
        effective_cutoff = dijkstra_cutoff_m

    # 2. Compute road network driving distance
    d_net = graph.network_distance_between_arcs(
        arc_prev=arc_prev,
        arc_curr=arc_curr,
        frac_prev=frac_prev,
        frac_curr=frac_curr,
        cutoff_m=effective_cutoff,
    )

    # 3. Handle disconnected / unreachable segments
    if math.isinf(d_net):
        # Penalty floor preventing Viterbi path collapse
        log_prob = -math.log(effective_beta) - (unreachable_penalty_m / effective_beta)
        return float(log_prob)

    # 4. Standard exponential likelihood on routing detour |d_net - d_gps|
    diff = abs(d_net - d_gps)
    log_prob = -math.log(effective_beta) - (diff / effective_beta)
    return float(log_prob)


def network_distance(
    graph: RoadGraph,
    shortest_path: ShortestPath,
    prev_arc_id: int,
    curr_arc_id: int,
    prev_frac: float,
    curr_frac: float,
) -> float:
    """
    Calculate directed road-network distance between
    two candidate positions on road arcs using RoadGraph.
    """
    if not 0.0 <= prev_frac <= 1.0:
        raise ValueError("prev_frac must be between 0 and 1")
    if not 0.0 <= curr_frac <= 1.0:
        raise ValueError("curr_frac must be between 0 and 1")

    prev_length = graph.get_arc_length(prev_arc_id)
    curr_length = graph.get_arc_length(curr_arc_id)

    if prev_arc_id == curr_arc_id and curr_frac >= prev_frac:
        return (curr_frac - prev_frac) * prev_length

    _, prev_end = graph.get_arc_endpoints(prev_arc_id)
    curr_start, _ = graph.get_arc_endpoints(curr_arc_id)

    remaining_distance = (1.0 - prev_frac) * prev_length
    connecting_distance = shortest_path.distance(prev_end, curr_start)
    current_distance = curr_frac * curr_length

    if not np.isfinite(connecting_distance):
        return float("inf")

    return remaining_distance + connecting_distance + current_distance


def transition_log_probability(
    graph: RoadGraph,
    shortest_path: ShortestPath,
    prev_arc_id: int,
    curr_arc_id: int,
    prev_frac: float,
    curr_frac: float,
    gps_distance_m: float,
    beta: float = 20.0,
) -> float:
    """
    Calculate log transition probability between two candidate road positions using RoadGraph.
    """
    if beta <= 0:
        raise ValueError("beta must be greater than zero")
    if gps_distance_m < 0 or not np.isfinite(gps_distance_m):
        raise ValueError("gps_distance_m must be finite and non-negative")

    road_distance = network_distance(
        graph, shortest_path, prev_arc_id, curr_arc_id, prev_frac, curr_frac
    )
    if not np.isfinite(road_distance):
        return float("-inf")

    difference = abs(road_distance - gps_distance_m)
    return -np.log(beta) - difference / beta
