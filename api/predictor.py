"""predict_trip contract (frozen in common/contract.py).

    predict_trip(trip_dir) -> {"pred_arcs": list[int]}   # one arc id per GPS point

INTERIM predictor: nearest-arc candidates + Viterbi over road connectivity (distance-based
emissions, connected-arc transitions). It exists so the API/UI work end to end before the real
pipeline is ready. The real version is: load artifacts -> P1 features -> P2 predict_proba -> P3 decode.

Must never read the reference route file.
"""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from api.io_utils import load_trip

K = 15            # candidate arcs per GPS point
SIGMA = 20.0      # GPS noise in metres
BREAK_PEN = 12.0  # cost of jumping to an arc that is not reachable
STEP_PEN = 1.0    # cost of moving to a directly connected next arc
UTURN_PEN = 6.0   # cost of driving the same road backwards


def predict_trip(trip_dir):
    track, nodes, arcs = load_trip(Path(trip_dir))
    lat0 = track[:, 1].mean()
    scale = np.array([111320.0 * np.cos(np.radians(lat0)), 110540.0])
    P = track[:, :2] * scale
    N = nodes * scale
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]

    tree = cKDTree((A + B) / 2)
    k = min(K, len(arcs))
    _, idx = tree.query(P, k=k)
    idx = idx.reshape(len(P), k)

    a, b = A[idx], B[idx]
    ab = b - a
    ap = P[:, None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9), 0, 1)
    d = np.linalg.norm(P[:, None, :] - (a + t[..., None] * ab), axis=-1)
    emit = -(d ** 2) / (2 * SIGMA ** 2)                       # (n, k)

    src, dst = arcs[:, 0], arcs[:, 1]
    n = len(P)
    score = emit[0].copy()
    back = np.zeros((n, k), dtype=np.int32)
    for i in range(1, n):
        p, c = idx[i - 1], idx[i]
        same = p[:, None] == c[None, :]
        succ = dst[p][:, None] == src[c][None, :]
        rev = (dst[c][:, None] == src[p][None, :]).T & (src[c][None, :] == dst[p][:, None]) & ~same
        trans = np.full((k, k), -BREAK_PEN)
        trans[rev] = -UTURN_PEN
        trans[succ] = -STEP_PEN
        trans[same] = 0.0
        tot = score[:, None] + trans
        back[i] = tot.argmax(0)
        score = tot.max(0) + emit[i]
    path = np.empty(n, dtype=np.int64)
    j = int(score.argmax())
    for i in range(n - 1, -1, -1):
        path[i] = idx[i, j]
        j = back[i, j]
    return {"pred_arcs": [int(x) for x in path]}
