"""Feature engineering. All features use ONLY the track and the network."""
import numpy as np
import pandas as pd


def arc_geometry(tr):
    a = tr["arcs"]
    dx = tr["nx"][a[:, 1]] - tr["nx"][a[:, 0]]
    dy = tr["ny"][a[:, 1]] - tr["ny"][a[:, 0]]
    return np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 360  # length_m, bearing (0=N, clockwise)


def point_kinematics(tr):
    """Per-point speed (m/s) and heading (deg) from neighbours (central difference)."""
    x, y, t = tr["tx"], tr["ty"], tr["t"]
    n = len(x)
    lo, hi = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    dx, dy = x[hi] - x[lo], y[hi] - y[lo]
    dist = np.hypot(dx, dy)
    speed = dist / dt
    heading = np.degrees(np.arctan2(dx, dy)) % 360
    heading_valid = (dist > 2.0) & (dt <= 10)   # heading from <2 m movement or across a gap is noise
    dt_prev = np.r_[0.0, np.diff(t)]
    return speed, heading, heading_valid, dt_prev


def add_features(cands: dict, tr) -> pd.DataFrame:
    df = pd.DataFrame(cands)
    arc_len, arc_brg = arc_geometry(tr)
    speed, hd, hv, dtp = point_kinematics(tr)
    pi, ai = df.point_idx.values, df.arc_id.values
    diff = np.abs((hd[pi] - arc_brg[ai] + 180) % 360 - 180)
    df["bearing_diff"] = np.where(hv[pi], diff, 90.0)      # 90 = uninformative when heading unknown
    df["heading_valid"] = hv[pi].astype(np.int8)
    df["speed"] = speed[pi]
    df["dt_prev"] = dtp[pi]
    df["arc_len_m"] = arc_len[ai]
    arc_pos = df.frac_along.values * arc_len[ai]
    df["dist_to_end_m"] = arc_len[ai] - arc_pos
    g = df.groupby("point_idx")["dist_m"]
    df["dist_gap_to_best"] = df.dist_m - g.transform("min")
    df["n_cands"] = g.transform("size")
    df["has_reverse_arc"] = _has_reverse(tr)[ai].astype(np.int8)
    return df


def _has_reverse(tr):
    a = tr["arcs"]; nn = len(tr["nodes"])
    key = a[:, 0] * nn + a[:, 1]
    rev = a[:, 1] * nn + a[:, 0]
    return np.isin(rev, key)
