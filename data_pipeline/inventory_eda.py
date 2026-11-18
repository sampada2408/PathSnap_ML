"""Dataset inventory + data-quality log (report section 2.2).
Usage: python -m data_pipeline.inventory_eda --data data/raw --out data/processed/eda
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .io_utils import list_trips, load_trip


def one_trip(d, part):
    tr = load_trip(d, with_route=(part != "evaluator"))
    t, n_pts = tr["t"], len(tr["t"])
    dt = np.diff(t)
    nn, na = len(tr["nodes"]), len(tr["arcs"])
    a = tr["arcs"]
    g = coo_matrix((np.ones(na), (a[:, 0], a[:, 1])), shape=(nn, nn))
    n_wcc, lab = connected_components(g, directed=True, connection="weak")
    n_scc, _ = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(a.ravel(), minlength=nn)
    seg = np.hypot(np.diff(tr["tx"]), np.diff(tr["ty"]))
    row = dict(
        partition=part, trip_id=tr["trip_id"], utm_epsg=tr["epsg"], n_points=n_pts, n_nodes=nn, n_arcs=na,
        n_route_arcs=len(tr["route"]) if "route" in tr else np.nan,
        nan_rows=int(np.isnan(tr["track"]).any(1).sum()),
        dup_rows=int(pd.DataFrame(tr["track"]).duplicated().sum()),
        non_increasing_t=int((dt <= 0).sum()),
        dt_eq_1_pct=float((dt == 1).mean() * 100), dt_max_s=float(dt.max()),
        n_gaps_gt_5s=int((dt > 5).sum()),
        max_step_m=float(seg.max()), speed_p99_mps=float(np.percentile(seg / np.maximum(dt, 1e-9), 99)),
        track_lon_min=tr["track"][:, 0].min(), track_lon_max=tr["track"][:, 0].max(),
        track_lat_min=tr["track"][:, 1].min(), track_lat_max=tr["track"][:, 1].max(),
        track_length_km=float(seg.sum() / 1000),
        isolated_nodes=int((deg == 0).sum()), n_weak_components=int(n_wcc), n_strong_components=int(n_scc),
        largest_weak_comp_pct=float(np.bincount(lab).max() / nn * 100),
        self_loop_arcs=int((a[:, 0] == a[:, 1]).sum()), bad_arc_node_ids=int(((a < 0) | (a >= nn)).sum()),
    )
    if "route" in tr:
        r = tr["route"]
        row["route_connected_pct"] = float((a[r[:-1], 1] == a[r[1:], 0]).mean() * 100) if len(r) > 1 else 100.0
        row["route_bad_arc_ids"] = int(((r < 0) | (r >= na)).sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="data/processed/eda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in ("train", "student_test"):
        pd_ = Path(args.data) / part
        if pd_.exists():
            for d in list_trips(pd_):
                rows.append(one_trip(d, part)); print(part, d.name, rows[-1]["n_points"], "pts")
    df = pd.DataFrame(rows)
    df.to_csv(out / "inventory.csv", index=False)
    summ = df.groupby("partition")[["n_points", "n_nodes", "n_arcs", "dt_max_s", "n_gaps_gt_5s", "n_weak_components"]].describe().T
    summ.to_csv(out / "summary_stats.csv")
    flags = {
        "trips_with_nan": df.loc[df.nan_rows > 0, "trip_id"].tolist(),
        "trips_with_non_increasing_time": df.loc[df.non_increasing_t > 0, "trip_id"].tolist(),
        "trips_with_gap_gt_30s": df.loc[df.dt_max_s > 30, "trip_id"].tolist(),
        "trips_with_disconnected_route": df.loc[df.get("route_connected_pct", 100) < 100, "trip_id"].tolist(),
        "trips_with_jump_gt_200m": df.loc[df.max_step_m > 200, "trip_id"].tolist(),
    }
    json.dump(flags, open(out / "data_quality_log.json", "w"), indent=2)
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
