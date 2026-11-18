
import numpy as np

from decoder.distance import ShortestPath
from decoder.graph import RoadGraph


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
    two candidate positions on road arcs.

    Parameters
    ----------
    graph : RoadGraph
        Directed road network.
    shortest_path : ShortestPath
        Dijkstra shortest-path calculator.
    prev_arc_id : int
        Previous candidate arc ID.
    curr_arc_id : int
        Current candidate arc ID.
    prev_frac : float
        Previous position along its arc (0 to 1).
    curr_frac : float
        Current position along its arc (0 to 1).

    Returns
    -------
    float
        Road-network distance in metres.
        Returns infinity if no directed path exists.
    """

    # Validate fractional positions
    if not 0.0 <= prev_frac <= 1.0:
        raise ValueError("prev_frac must be between 0 and 1")

    if not 0.0 <= curr_frac <= 1.0:
        raise ValueError("curr_frac must be between 0 and 1")

    # Get lengths of both arcs
    prev_length = graph.get_arc_length(prev_arc_id)
    curr_length = graph.get_arc_length(curr_arc_id)

    # Special case: movement forward on the same arc
    if prev_arc_id == curr_arc_id and curr_frac >= prev_frac:
        return (curr_frac - prev_frac) * prev_length

    # Get endpoints of both arcs
    _, prev_end = graph.get_arc_endpoints(prev_arc_id)
    curr_start, _ = graph.get_arc_endpoints(curr_arc_id)

    # Remaining distance on previous arc
    remaining_distance = (1.0 - prev_frac) * prev_length

    # Shortest directed distance between the two arcs
    connecting_distance = shortest_path.distance(
        prev_end,
        curr_start
    )

    # Distance travelled along current arc
    current_distance = curr_frac * curr_length

    # No valid directed connection
    if not np.isfinite(connecting_distance):
        return float("inf")

    # Total road-network distance
    return (
        remaining_distance
        + connecting_distance
        + current_distance
    )


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
    Calculate log transition probability between
    two candidate road positions.

    Uses the difference between road-network distance
    and estimated GPS travel distance.

    Parameters
    ----------
    beta : float
        Transition sensitivity parameter in metres.

    Returns
    -------
    float
        Log transition probability.
        Returns negative infinity for disconnected arcs.
    """

    if beta <= 0:
        raise ValueError("beta must be greater than zero")

    if gps_distance_m < 0 or not np.isfinite(gps_distance_m):
        raise ValueError(
            "gps_distance_m must be finite and non-negative"
        )

    # Calculate road-network distance
    road_distance = network_distance(
        graph,
        shortest_path,
        prev_arc_id,
        curr_arc_id,
        prev_frac,
        curr_frac
    )

    # Disconnected arcs are impossible transitions
    if not np.isfinite(road_distance):
        return float("-inf")

    # Difference between road distance and GPS distance
    difference = abs(road_distance - gps_distance_m)

    # Log transition probability
    return -np.log(beta) - difference / beta
