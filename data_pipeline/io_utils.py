"""Raw-file readers + projection. Raw formats (verified on the real data):
.track  : lon lat t_seconds   (whitespace separated, no header, CRLF line ends)
.nodes  : lon lat             (row index = node_id)
.arcs   : from_node to_node   (row index = arc_id)
.route  : arc_id per line     (chronological reference route; train/student_test only)
Every trip ships its OWN nodes/arcs files (its own road-network extract).
"""
from pathlib import Path
import numpy as np
from pyproj import Transformer

# Trips are in DIFFERENT cities worldwide (e.g. lon 8.6, 4.9, 18.7, -78), so each trip
# is projected with its own UTM zone (chosen from the trip's mean longitude/latitude).
_CACHE = {}


def utm_epsg(lon_mean, lat_mean):
    zone = int((lon_mean + 180) // 6) % 60 + 1
    return (32600 if lat_mean >= 0 else 32700) + zone


def project(lon, lat, epsg):
    if epsg not in _CACHE:
        _CACHE[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = _CACHE[epsg].transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def list_trips(partition_dir):
    return sorted(p for p in Path(partition_dir).iterdir() if p.is_dir())


def load_trip(trip_dir, with_route=False):
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    out = {"trip_id": tid, "track": track, "nodes": nodes, "arcs": arcs}
    out["epsg"] = utm_epsg(float(nodes[:, 0].mean()), float(nodes[:, 1].mean()))
    out["tx"], out["ty"] = project(track[:, 0], track[:, 1], out["epsg"])
    out["nx"], out["ny"] = project(nodes[:, 0], nodes[:, 1], out["epsg"])
    out["t"] = track[:, 2]
    if with_route:
        out["route"] = np.loadtxt(trip_dir / f"{tid}.route", dtype=np.int64, ndmin=1)
    return out
