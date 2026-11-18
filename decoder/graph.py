
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer


@dataclass
class RoadGraph:
    """
    Represents a directed road network.

    Node IDs are row indices in the .nodes file.
    Arc IDs are row indices in the .arcs file.
    All coordinates are projected into metres.
    """

    nodes_xy: np.ndarray
    arcs: np.ndarray
    arc_lengths: np.ndarray
    adjacency: dict
    transformer: Transformer
    epsg: int

    @classmethod
    def load(cls, nodes_path, arcs_path):
        """
        Load a road network from .nodes and .arcs files.

        Projection follows Person 1's io_utils.py:
        choose the UTM zone using mean longitude and latitude.
        """

        # Read longitude, latitude from .nodes
        nodes_lonlat = np.loadtxt(
            nodes_path,
            dtype=float,
            ndmin=2
        )

        # Read from_node_id, to_node_id from .arcs
        arcs = np.loadtxt(
            arcs_path,
            dtype=np.int64,
            ndmin=2
        )

        # Validate input shapes
        if nodes_lonlat.shape[1] != 2:
            raise ValueError(
                ".nodes must contain exactly 2 columns"
            )

        if arcs.shape[1] != 2:
            raise ValueError(
                ".arcs must contain exactly 2 columns"
            )

        if len(nodes_lonlat) == 0:
            raise ValueError(".nodes file is empty")

        if len(arcs) == 0:
            raise ValueError(".arcs file is empty")

        if not np.isfinite(nodes_lonlat).all():
            raise ValueError(
                ".nodes contains invalid coordinates"
            )

        # Validate longitude and latitude ranges
        if np.any(np.abs(nodes_lonlat[:, 0]) > 180):
            raise ValueError("Invalid longitude values")

        if np.any(np.abs(nodes_lonlat[:, 1]) > 90):
            raise ValueError("Invalid latitude values")

        # Validate node IDs used by arcs
        if arcs.min() < 0 or arcs.max() >= len(nodes_lonlat):
            raise ValueError(
                "An arc refers to an invalid node ID"
            )

        # Calculate UTM zone exactly as Person 1 does
        lon_mean = float(nodes_lonlat[:, 0].mean())
        lat_mean = float(nodes_lonlat[:, 1].mean())

        zone = int((lon_mean + 180) // 6) % 60 + 1

        epsg = (
            32600 if lat_mean >= 0 else 32700
        ) + zone

        # Convert longitude/latitude to projected metres
        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{epsg}",
            always_xy=True
        )

        x, y = transformer.transform(
            nodes_lonlat[:, 0],
            nodes_lonlat[:, 1]
        )

        nodes_xy = np.column_stack((x, y))

        # Calculate the length of every directed arc
        start_xy = nodes_xy[arcs[:, 0]]
        end_xy = nodes_xy[arcs[:, 1]]

        arc_lengths = np.linalg.norm(
            end_xy - start_xy,
            axis=1
        )

        # Build directed adjacency list
        adjacency = {
            node_id: []
            for node_id in range(len(nodes_xy))
        }

        for arc_id, (u, v) in enumerate(arcs):
            adjacency[int(u)].append(
                (
                    int(v),
                    int(arc_id),
                    float(arc_lengths[arc_id])
                )
            )

        return cls(
            nodes_xy=nodes_xy,
            arcs=arcs,
            arc_lengths=arc_lengths,
            adjacency=adjacency,
            transformer=transformer,
            epsg=epsg
        )

    def get_arc_length(self, arc_id):
        """Return the length of an arc in metres."""
        return float(self.arc_lengths[arc_id])

    def get_arc_endpoints(self, arc_id):
        """Return start and end node IDs of an arc."""
        return tuple(self.arcs[arc_id])

    def get_neighbors(self, node_id):
        """Return all outgoing connections from a node."""
        return self.adjacency[node_id]

    def project_coordinates(self, lon, lat):
        """Convert GPS longitude/latitude into projected metres."""
        x, y = self.transformer.transform(lon, lat)
        return x, y
