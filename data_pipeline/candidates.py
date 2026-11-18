"""Candidate generation: for every GPS point, the k nearest arcs within `radius_m`.
Uses only track + network (never .route), so it is safe at API inference time."""
import numpy as np
import shapely
from shapely import STRtree


def _seg_geom(tr):
    a = tr["arcs"]
    x1, y1, x2, y2 = tr["nx"][a[:, 0]], tr["ny"][a[:, 0]], tr["nx"][a[:, 1]], tr["ny"][a[:, 1]]
    return x1, y1, x2, y2


def point_seg(px, py, x1, y1, x2, y2):
    """Distance from points to segments (broadcasting) + fraction along segment in [0,1]."""
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    u = np.where(L2 > 0, ((px - x1) * dx + (py - y1) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    u = np.clip(u, 0, 1)
    return np.hypot(px - (x1 + u * dx), py - (y1 + u * dy)), u


def generate_candidates(tr, radius_m=60.0, k=15):
    """Returns DataFrame-ready dict: point_idx, arc_id, dist_m, frac_along (sorted by point, dist)."""
    x1, y1, x2, y2 = _seg_geom(tr)
    lines = shapely.linestrings(np.stack([np.stack([x1, y1], 1), np.stack([x2, y2], 1)], 1))
    tree = STRtree(lines)
    pts = shapely.points(tr["tx"], tr["ty"])
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius_m)  # (point_idx, arc_idx) pairs
    d, u = point_seg(tr["tx"][pi], tr["ty"][pi], x1[ai], y1[ai], x2[ai], y2[ai])
    order = np.lexsort((d, pi))
    pi, ai, d, u = pi[order], ai[order], d[order], u[order]
    # rank within each point, keep the k closest
    first = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    counts = np.diff(np.r_[first, len(pi)])
    rank = np.arange(len(pi)) - np.repeat(first, counts)
    keep = rank < k
    return dict(point_idx=pi[keep], arc_id=ai[keep], dist_m=d[keep], frac_along=u[keep], dist_rank=rank[keep])
