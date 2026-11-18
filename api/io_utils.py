"""Raw trip file readers. Only .track/.nodes/.arcs are read here.

The reference route file (ground truth) is deliberately NOT read in this module, so the
prediction pipeline can import it without ever touching labels.
"""
from pathlib import Path

import numpy as np


def load_trip(trip_dir):
    """Return (track[N,3] lon,lat,t ; nodes[M,2] lon,lat ; arcs[K,2] from,to)."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=int, ndmin=2)
    return track, nodes, arcs


def arc_coords(nodes, arcs, arc_ids):
    """Arc ids -> list of [[lat, lon], [lat, lon]] segments (Leaflet/Folium order)."""
    segs = []
    for a in arc_ids:
        if a < 0:
            continue
        u, v = arcs[a]
        segs.append([[float(nodes[u, 1]), float(nodes[u, 0])],
                     [float(nodes[v, 1]), float(nodes[v, 0])]])
    return segs
