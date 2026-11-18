"""EDA figures for report section 2.2 (matplotlib). Needs inventory.csv from inventory_eda
and (optionally) the candidate parquet + build_stats csv from build_tables.
Usage (repo root):
  python -m data_pipeline.eda_plots --data data/raw --proc data/processed --out data_pipeline/results/figures
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .io_utils import load_trip, list_trips


def save(fig, out, name):
    fig.tight_layout(); fig.savefig(Path(out) / name, dpi=150); plt.close(fig); print("wrote", name)


def fig_inventory(inv, out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        d = inv[inv.partition == part]
        ax[0, 0].hist(np.log10(d.n_points), bins=20, alpha=.7, color=c, label=f"{part} (n={len(d)})")
        ax[0, 1].hist(np.log10(d.n_arcs), bins=20, alpha=.7, color=c, label=part)
        ax[1, 0].hist(d.track_length_km, bins=20, alpha=.7, color=c, label=part)
        ax[1, 1].hist(d.n_weak_components, bins=20, alpha=.7, color=c, label=part)
    ax[0, 0].set(title="GPS points per trip", xlabel="log10(points)", ylabel="trips")
    ax[0, 1].set(title="Road arcs per trip network", xlabel="log10(arcs)")
    ax[1, 0].set(title="Track length", xlabel="km", ylabel="trips")
    ax[1, 1].set(title="Weakly connected components per network", xlabel="components")
    ax[0, 0].legend(); fig = ax[0, 0].figure
    fig.suptitle("Dataset inventory (partitions: train 80 trips, student_test 10 trips)")
    save(fig, out, "01_inventory.png")


def fig_time_gaps(data, out):
    dts = {}
    for part in ("train", "student_test"):
        p = Path(data) / part
        dts[part] = np.concatenate([np.diff(np.loadtxt(d / f"{d.name}.track", usecols=2, ndmin=1)) for d in list_trips(p)])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        x = dts[part]
        ax[0].hist(np.clip(x, 0, 20), bins=np.arange(0, 21) - .5, alpha=.7, color=c, label=f"{part}: {np.mean(x == 1) * 100:.1f}% exactly 1 s")
        ax[1].hist(np.log10(np.clip(x, 1, None)), bins=40, alpha=.7, color=c, label=f"{part}: max {x.max():.0f} s")
    ax[0].set(title="Time between consecutive points (clipped at 20 s)", xlabel="seconds", ylabel="count", yscale="log")
    ax[1].set(title="Same, log10 scale incl. long gaps", xlabel="log10(seconds)", yscale="log")
    ax[0].legend(); ax[1].legend()
    save(fig, out, "02_time_gaps.png")


def fig_missing(inv, out):
    checks = {"NaN track rows": "nan_rows", "Duplicate track rows": "dup_rows", "Non-increasing timestamps": "non_increasing_t",
              "Gap > 5 s": "n_gaps_gt_5s", "Isolated nodes": "isolated_nodes", "Self-loop arcs": "self_loop_arcs",
              "Arc refs invalid node": "bad_arc_node_ids", "Route refs invalid arc": "route_bad_arc_ids"}
    rows = [dict(check=k, trips_affected=int((inv[v] > 0).sum()), total_count=int(inv[v].fillna(0).sum()),
                 trips_total=len(inv)) for k, v in checks.items() if v in inv]
    rows.append(dict(check="Route not contiguous", trips_affected=int((inv.route_connected_pct < 100).sum()), total_count=np.nan, trips_total=len(inv)))
    rows.append(dict(check="Jump > 200 m between points", trips_affected=int((inv.max_step_m > 200).sum()), total_count=np.nan, trips_total=len(inv)))
    t = pd.DataFrame(rows); t.to_csv(Path(out) / "missing_values_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(t.check[::-1], t.trips_affected[::-1], color="tab:red")
    for i, v in enumerate(t.trips_affected[::-1]): ax.text(v + .3, i, str(v), va="center")
    ax.set(title=f"Data-quality checks: number of trips affected (of {len(inv)})", xlabel="trips")
    save(fig, out, "03_missing_and_quality.png")


def _route_xy(tr):
    a = tr["arcs"][tr["route"]]
    return np.stack([np.stack([tr["nx"][a[:, 0]], tr["ny"][a[:, 0]]], 1), np.stack([tr["nx"][a[:, 1]], tr["ny"][a[:, 1]]], 1)], 1)


def fig_trajectories(data, trips, out):
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (part, tid) in zip(axs.ravel(), trips):
        tr = load_trip(Path(data) / part / tid, with_route=True)
        ax.add_collection(LineCollection(_route_xy(tr), colors="limegreen", linewidths=3, label="reference route"))
        ax.scatter(tr["tx"], tr["ty"], s=3, c="tab:blue", label="GPS points")
        ax.set(title=f"{part}/{tid}: {len(tr['tx'])} pts, UTM {tr['epsg']}", aspect="equal", xlabel="easting (m)", ylabel="northing (m)")
        ax.autoscale(); ax.legend(loc="best", fontsize=8)
    fig.suptitle("GPS trajectories vs reference route (projected metres)")
    save(fig, out, "04_trajectories.png")


def fig_network(data, part, tid, out, margin=300):
    tr = load_trip(Path(data) / part / tid, with_route=True)
    a = tr["arcs"]
    segs = np.stack([np.stack([tr["nx"][a[:, 0]], tr["ny"][a[:, 0]]], 1), np.stack([tr["nx"][a[:, 1]], tr["ny"][a[:, 1]]], 1)], 1)
    x0, x1, y0, y1 = tr["tx"].min() - margin, tr["tx"].max() + margin, tr["ty"].min() - margin, tr["ty"].max() + margin
    m = ~((segs[:, :, 0].max(1) < x0) | (segs[:, :, 0].min(1) > x1) | (segs[:, :, 1].max(1) < y0) | (segs[:, :, 1].min(1) > y1))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(segs[m], colors="lightgray", linewidths=.6, label=f"road arcs ({m.sum()} of {len(a)} shown)"))
    ax.add_collection(LineCollection(_route_xy(tr), colors="limegreen", linewidths=2.5, label="reference route"))
    ax.scatter(tr["tx"], tr["ty"], s=4, c="tab:blue", label="GPS points")
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal", title=f"Road network around trip {part}/{tid}", xlabel="easting (m)", ylabel="northing (m)")
    ax.legend(loc="upper right")
    save(fig, out, f"05_network_{tid}.png")


def fig_label_quality(stats, out):
    s = stats.sort_values("pct_points_ref_far", ascending=False).head(15)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].barh(s.trip_id.astype(str)[::-1], s.pct_points_ref_far[::-1], color="tab:red")
    ax[0].set(title="Top 15 trips: % points far (>60 m) from their reference route", xlabel="%", ylabel="train trip")
    s2 = stats.sort_values("miss_rate_trusted_pts", ascending=False).head(15)
    ax[1].barh(s2.trip_id.astype(str)[::-1], s2.miss_rate_trusted_pts[::-1] * 100, color="tab:purple")
    ax[1].set(title="Top 15: candidate miss rate on trusted points", xlabel="%")
    save(fig, out, "06_label_quality_train.png")


def fig_features(parq, out):
    d = pd.read_parquet(parq, columns=["dist_m", "bearing_diff", "dist_rank", "label", "label_trusted", "heading_valid", "speed"])
    d = d[d.label_trusted == 1]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for lab, c, n in ((1, "tab:green", "correct arc"), (0, "tab:gray", "other candidates")):
        x = d[d.label == lab]
        ax[0, 0].hist(x.dist_m, bins=60, alpha=.6, color=c, density=True, label=n)
        ax[0, 1].hist(x[x.heading_valid == 1].bearing_diff, bins=60, alpha=.6, color=c, density=True, label=n)
    ax[0, 0].set(title="Distance point->arc", xlabel="metres"); ax[0, 0].legend()
    ax[0, 1].set(title="Heading difference (valid headings only)", xlabel="degrees")
    r = d[d.label == 1].dist_rank.value_counts(normalize=True).sort_index().head(15)
    ax[1, 0].bar(r.index, r.values * 100, color="tab:green"); ax[1, 0].set(title="Distance rank of the correct arc", xlabel="rank (0 = nearest)", ylabel="% of points")
    ax[1, 1].hist(np.clip(d[d.dist_rank == 0].speed, 0, 40), bins=60, color="tab:blue"); ax[1, 1].set(title="Speed (m/s, clipped at 40)", xlabel="m/s")
    fig.suptitle("Feature distributions (train, trusted labels)")
    save(fig, out, "07_feature_distributions.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw"); ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="data_pipeline/results/figures")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(Path(a.proc) / "eda" / "inventory.csv"); inv["trip_id"] = inv.trip_id.astype(str).str.zfill(8)
    fig_inventory(inv, a.out); fig_time_gaps(a.data, a.out); fig_missing(inv, a.out)
    tr = inv[inv.partition == "train"].sort_values("n_points")
    picks = [("train", tr.trip_id.iloc[i]) for i in (len(tr) // 10, len(tr) // 2, 9 * len(tr) // 10)] + [("student_test", inv[inv.partition == "student_test"].trip_id.iloc[0])]
    fig_trajectories(a.data, picks, a.out)
    fig_network(a.data, *picks[1], a.out)
    sp = Path(a.proc) / "train_build_stats.csv"
    if sp.exists(): fig_label_quality(pd.read_csv(sp), a.out)
    pq = Path(a.proc) / "train_candidates.parquet"
    if pq.exists(): fig_features(pq, a.out)


if __name__ == "__main__":
    main()
