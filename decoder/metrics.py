"""
PathSnap Person 3: Route-Level Metrics
======================================
Computes route-level evaluation metrics:
1. Pointwise Accuracy: observation-level match against reference arc
2. Route Continuity: proportion of consecutive predicted transitions connected in the road network
3. Sequence-Level Precision, Recall, and F1: comparison against ground-truth .route sequence
4. Comparative Evaluator: Classifier-Only vs. Classifier + HMM/Viterbi
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from decoder.graph import RoadNetworkGraph


def compress_route(arc_sequence: List[int]) -> List[int]:
    """
    Compresses consecutive duplicate arcs into an ordered list of unique arcs traversed:
    e.g. [10, 10, 10, 25, 25, 30] -> [10, 25, 30].
    """
    if not arc_sequence:
        return []
    compressed = [arc_sequence[0]]
    for a in arc_sequence[1:]:
        if a != compressed[-1]:
            compressed.append(a)
    return compressed


def compute_route_continuity(
    predicted_arcs: List[int],
    graph: RoadNetworkGraph,
) -> float:
    """
    Route Continuity:
    Fraction of consecutive predicted arc pairs (a_{t-1}, a_t) that are physically
    continuous in the road network:
        - a_{t-1} == a_t (vehicle remains on same segment)
        - to_node(a_{t-1}) == from_node(a_t) (direct road connection)
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    continuous_count = 0
    total_transitions = len(predicted_arcs) - 1

    arcs_mat = graph.arcs

    for i in range(total_transitions):
        a_prev = predicted_arcs[i]
        a_curr = predicted_arcs[i + 1]

        if a_prev == a_curr:
            continuous_count += 1
        elif arcs_mat[a_prev, 1] == arcs_mat[a_curr, 0]:
            # to_node of prev connects to from_node of curr
            continuous_count += 1

    return float(continuous_count / total_transitions)


def compute_sequence_precision_recall_f1(
    predicted_arcs: List[int],
    ground_truth_route: List[int],
) -> Dict[str, float]:
    """
    Sequence-Level Metrics:
    Compares the compressed sequence of road segments predicted against
    the ground-truth .route sequence.
    """
    pred_compressed = compress_route(predicted_arcs)
    gt_compressed = compress_route(ground_truth_route)

    pred_set = set(pred_compressed)
    gt_set = set(gt_compressed)

    if not pred_set or not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = pred_set.intersection(gt_set)
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gt_set)

    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_unique_arcs": len(pred_set),
        "gt_unique_arcs": len(gt_set),
    }


def compute_pointwise_accuracy(
    predicted_arcs: List[int],
    reference_arcs: List[int],
) -> float:
    """
    Pointwise Observation Accuracy:
    Fraction of GPS observation points where the predicted arc exactly matches
    the per-point reference arc.
    """
    if not predicted_arcs or not reference_arcs:
        return 0.0
    matches = sum(p == r for p, r in zip(predicted_arcs, reference_arcs))
    return float(matches / len(predicted_arcs))


def compare_classifier_vs_hmm(
    classifier_arcs: List[int],
    hmm_arcs: List[int],
    reference_arcs: List[int],
    ground_truth_route: List[int],
    graph: RoadNetworkGraph,
) -> Dict[str, Any]:
    """
    Side-by-side comparison between Classifier-Only and Classifier + HMM decoding.
    """
    # 1. Classifier-Only Metrics
    clf_pointwise = compute_pointwise_accuracy(classifier_arcs, reference_arcs)
    clf_continuity = compute_route_continuity(classifier_arcs, graph)
    clf_seq = compute_sequence_precision_recall_f1(classifier_arcs, ground_truth_route)

    # 2. Classifier + HMM Metrics
    hmm_pointwise = compute_pointwise_accuracy(hmm_arcs, reference_arcs)
    hmm_continuity = compute_route_continuity(hmm_arcs, graph)
    hmm_seq = compute_sequence_precision_recall_f1(hmm_arcs, ground_truth_route)

    return {
        "classifier_only": {
            "pointwise_accuracy": clf_pointwise,
            "route_continuity": clf_continuity,
            "sequence_precision": clf_seq["precision"],
            "sequence_recall": clf_seq["recall"],
            "sequence_f1": clf_seq["f1"],
        },
        "classifier_plus_hmm": {
            "pointwise_accuracy": hmm_pointwise,
            "route_continuity": hmm_continuity,
            "sequence_precision": hmm_seq["precision"],
            "sequence_recall": hmm_seq["recall"],
            "sequence_f1": hmm_seq["f1"],
        },
        "gains": {
            "continuity_gain": hmm_continuity - clf_continuity,
            "pointwise_gain": hmm_pointwise - clf_pointwise,
            "sequence_f1_gain": hmm_seq["f1"] - clf_seq["f1"],
        },
    }
