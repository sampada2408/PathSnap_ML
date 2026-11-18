"""Map the route (sequence of arc ids) to ONE reference arc per GPS point.
The route has fewer arcs than there are points, so a point's label is found by a
monotone alignment: choose route index j(i) non-decreasing in i that minimises the
sum of point->arc distances (exact DP, O(N*M)). Handles loops (same arc twice) and
stops a point from jumping to a far-away copy of a nearby parallel arc.
Only used for train / student_test. NEVER call at API inference time."""
import numpy as np
from .candidates import point_seg, _seg_geom


def align_points_to_route(tr):
    r = tr["route"]
    x1, y1, x2, y2 = _seg_geom(tr)
    x1, y1, x2, y2 = x1[r], y1[r], x2[r], y2[r]
    N, M = len(tr["tx"]), len(r)
    big = np.float32(1e9)
    dp = np.empty(M, dtype=np.float32)
    back = np.empty((N, M), dtype=np.int32)
    for i in range(N):
        d, _ = point_seg(tr["tx"][i], tr["ty"][i], x1, y1, x2, y2)
        d = d.astype(np.float32)
        if i == 0:
            dp = d
            back[0] = np.arange(M)
        else:
            # best previous state with index <= j: running min + its argmin
            run_min = np.minimum.accumulate(dp)
            is_new = np.r_[True, dp[1:] < run_min[:-1]]
            arg = np.maximum.accumulate(np.where(is_new, np.arange(M), 0))
            back[i] = arg
            dp = d + run_min
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(dp))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return j, r[j]  # route index per point, reference arc id per point
