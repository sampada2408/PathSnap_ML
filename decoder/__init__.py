"""
PathSnap Decoder Module (Person 3)
==================================
Contains:
- graph: RoadNetworkGraph, equirectangular distances, and Dijkstra routing
- emission: Gaussian (Newson-Krumm), classifier-based, and hybrid emissions
- transition: Exponential transition model, gap scaling, and unreachable floor
- viterbi: Contract 3 decode() implementation
- metrics: Pointwise accuracy, route continuity, sequence precision/recall/F1
- tune: Parameter grid search and artifacts/hmm_config.json exporter
- evaluate: 3-way evaluation (Newson-Krumm vs. Classifier-Only vs. Classifier+HMM)
- report_draft: Complete Section 5 report generator
"""

from decoder.emission import (
    classifier_log_emission,
    gaussian_log_emission,
    hybrid_log_emission,
)
from decoder.graph import RoadNetworkGraph, equirectangular_distance
from decoder.metrics import (
    compare_classifier_vs_hmm,
    compute_pointwise_accuracy,
    compute_route_continuity,
    compute_sequence_precision_recall_f1,
)
from decoder.transition import compute_log_transition_probability
from decoder.viterbi import ViterbiDecoder, decode

__all__ = [
    "RoadNetworkGraph",
    "equirectangular_distance",
    "gaussian_log_emission",
    "classifier_log_emission",
    "hybrid_log_emission",
    "compute_log_transition_probability",
    "ViterbiDecoder",
    "decode",
    "compute_pointwise_accuracy",
    "compute_route_continuity",
    "compute_sequence_precision_recall_f1",
    "compare_classifier_vs_hmm",
]
