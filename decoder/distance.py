
import heapq
import math


class ShortestPath:
    """
    Calculates shortest distances in a directed road graph
    using Dijkstra's algorithm.
    """

    def __init__(self, graph):
        self.graph = graph
        self.cache = {}

    def distance(self, source, target, max_distance=float("inf")):
        """
        Returns shortest distance from source node to target node.

        Returns infinity if target is unreachable within max_distance.
        """

        if source == target:
            return 0.0

        cache_key = (source, target)

        # Reuse previously calculated distances
        if cache_key in self.cache:
            cached_distance = self.cache[cache_key]
            if cached_distance <= max_distance:
                return cached_distance

        # Priority queue: (distance_so_far, current_node)
        queue = [(0.0, source)]

        # Shortest known distance to each node
        distances = {source: 0.0}

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            # Ignore outdated queue entries
            if current_distance > distances[current_node]:
                continue

            # Stop when target is reached
            if current_node == target:
                self.cache[cache_key] = current_distance
                return current_distance

            # Stop exploring paths beyond the cutoff
            if current_distance > max_distance:
                continue

            # Explore outgoing directed arcs
            for neighbor, arc_id, arc_length in self.graph.get_neighbors(
                current_node
            ):
                new_distance = current_distance + arc_length

                if new_distance > max_distance:
                    continue

                if new_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = new_distance

                    heapq.heappush(
                        queue,
                        (new_distance, neighbor)
                    )

        return math.inf
