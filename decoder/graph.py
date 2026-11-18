
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer


@dataclass
class RoadGraph:
    """
    Represents the road network as a directed graph.

    Node IDs are row indices in the .nodes file.
    Arc IDs are row indices in the .arcs file.
    """

    nodes_xy: np.ndarray
    arcs: np.ndarray
    arc_lengths: np.ndarray
    adjacency: dict
    transformer: Transformer

    @classmethod
    def load(
        cls,
        nodes_path,
        arcs_path,
        projected_crs="EPSG:32632"
    ):
        # Read longitude, latitude from .nodes
        nodes_lonlat = np.loadtxt(nodes_path, dtype=float)
        nodes_lonlat = np.atleast_2d(nodes_lonlat)

        # Read from_node_id, to_node_id from .arcs
        arcs = np.loadtxt(arcs_path, dtype=int)
        arcs = np.atleast_2d(arcs)

        if nodes_lonlat.shape[1] != 2:
            raise ValueError(".nodes must contain exactly 2 columns")

        if arcs.shape[1] != 2:
            raise ValueError(".arcs must contain exactly 2 columns")

        if not np.isfinite(nodes_lonlat).all():
            raise ValueError(".nodes contains invalid coordinates")

        if arcs.min() < 0 or arcs.max() >= len(nodes_lonlat):
            raise ValueError("An arc refers to an invalid node ID")

        # Convert longitude/latitude into projected metre coordinates
        transformer = Transformer.from_crs(
            "EPSG:4326",
            projected_crs,
            always_xy=True
        )

        x, y = transformer.transform(
            nodes_lonlat[:, 0],
            nodes_lonlat[:, 1]
        )

        nodes_xy = np.column_stack((x, y))

        # Calculate the physical length of every directed arc
        start_xy = nodes_xy[arcs[:, 0]]
        end_xy = nodes_xy[arcs[:, 1]]

        arc_lengths = np.linalg.norm(
            end_xy - start_xy,
            axis=1
        )

        # Build adjacency list
        adjacency = {
            node_id: []
            for node_id in range(len(nodes_xy))
        }

        for arc_id, (u, v) in enumerate(arcs):
            adjacency[int(u)].append(
                (int(v), int(arc_id), float(arc_lengths[arc_id]))
            )

        return cls(
            nodes_xy=nodes_xy,
            arcs=arcs,
            arc_lengths=arc_lengths,
            adjacency=adjacency,
            transformer=transformer
        )

    def get_arc_length(self, arc_id):
        return float(self.arc_lengths[arc_id])

    def get_arc_endpoints(self, arc_id):
        return tuple(self.arcs[arc_id])

    def get_neighbors(self, node_id):
        return self.adjacency[node_id]
