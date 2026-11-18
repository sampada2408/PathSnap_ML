"""Build the candidate table parquet (Contract 1).
Usage:
  python -m data_pipeline.build_tables --data data/raw --out data/processed --partitions train student_test
  python -m data_pipeline.build_tables --data data/raw --out data/processed_v0 --partitions train --limit 10 --v0
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from .io_utils import list_trips, load_trip
from .candidates import generate_candidates
from .labels import align_points_to_route
from .candidates import point_seg, _seg_geom
from .features import add_features

V0_FEATURES = ["dist_m", "bearing_diff", "speed"]


def build_trip(d, radius, k, with_label=True):
    tr = load_trip(d, with_route=with_label)
    c = generate_candidates(tr, radius_m=radius, k=k)
    df = add_features(c, tr)
    df.insert(0, "trip_id", tr["trip_id"])
    ref = None
    stats = dict(trip_id=tr["trip_id"], n_points=len(tr["tx"]), n_cand_rows=len(df),
                 points_without_cand=int(len(tr["tx"]) - df.point_idx.nunique()))
    if with_label:
        j, ref = align_points_to_route(tr)
        ref_arc = ref[df.point_idx.values]
        df["label"] = (df.arc_id.values == ref_arc).astype(np.int8)
        # how far is each point from its own reference arc? (large => reference route does not cover this point)
        x1, y1, x2, y2 = _seg_geom(tr)
        ref_d, _ = point_seg(tr["tx"], tr["ty"], x1[ref], y1[ref], x2[ref], y2[ref])
        df["ref_dist_m"] = ref_d[df.point_idx.values]            # metadata, NOT a model feature
        df["label_trusted"] = (df.ref_dist_m <= radius).astype(np.int8)   # metadata, NOT a model feature
        hit = df.groupby("point_idx").label.max().reindex(range(len(tr["tx"])), fill_value=0)
        stats["miss_rate"] = float(1 - hit.mean())
        trusted = ref_d <= radius
        stats["pct_points_ref_far"] = float((~trusted).mean() * 100)
        stats["miss_rate_trusted_pts"] = float(1 - hit.values[trusted].mean()) if trusted.any() else float("nan")
        stats["top1_by_nearest"] = float(df[df.dist_rank == 0].label.mean())
    return df, stats, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw"); ap.add_argument("--out", default="data/processed")
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--radius", type=float, default=60.0); ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--v0", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for part in a.partitions:
        trips = list_trips(Path(a.data) / part)[: a.limit]
        frames, stats, refs = [], [], {}
        t0 = time.time()
        for d in trips:
            df, s, ref = build_trip(d, a.radius, a.k)
            if a.v0:
                df = df[["trip_id", "point_idx", "arc_id", "label", "label_trusted"] + V0_FEATURES]
            refs[d.name] = ref.tolist()  # per-point reference arc, for P3 route metrics
            frames.append(df); stats.append(s)
            print(f"{part} {d.name}: {s['n_points']} pts, miss={s.get('miss_rate', float('nan')):.4f}, "
                  f"miss_trusted={s.get('miss_rate_trusted_pts', float('nan')):.4f}, ref_far={s.get('pct_points_ref_far', 0):.1f}%, "
                  f"top1(nearest)={s.get('top1_by_nearest', float('nan')):.3f}")
        pd.concat(frames, ignore_index=True).to_parquet(out / f"{part}_candidates.parquet", index=False)
        pd.DataFrame(stats).to_csv(out / f"{part}_build_stats.csv", index=False)
        json.dump(refs, open(out / f"{part}_reference_arcs.json", "w"))
        print(f"[{part}] done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
