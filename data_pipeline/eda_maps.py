"""Interactive Folium EDA maps (report section 2.2): GPS points, reference route, road network,
and the candidate arcs of a few sample points. One HTML file per trip.
Usage (repo root):
  python -m data_pipeline.eda_maps --data data/raw --proc data/processed --out data_pipeline/results/maps
  python -m data_pipeline.eda_maps --trips train/00000095 student_test/00000000
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import folium
from .io_utils import load_trip


def _arc_latlon(tr, arc_ids):
    a = tr["arcs"][arc_ids]
    n = tr["nodes"]  # lon, lat
    return [[(n[f, 1], n[f, 0]), (n[t, 1], n[t, 0])] for f, t in a]


def build_map(data, part, tid, proc=None, max_arcs=4000, max_pts=4000, n_sample=3):
    tr = load_trip(Path(data) / part / tid, with_route=True)
    lon, lat = tr["track"][:, 0], tr["track"][:, 1]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="CartoDB positron", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    # NOTE: OpenStreetMap's own tile server blocks maps opened from a local file (HTTP 403), so use
    # CartoDB (default above) and offer Esri satellite as a second basemap.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite (Esri)").add_to(m)

    # road network, only arcs whose from-node lies near the track bounding box
    pad = 0.003
    nlon, nlat = tr["nodes"][:, 0], tr["nodes"][:, 1]
    fa = tr["arcs"][:, 0]
    inb = (nlon[fa] > lon.min() - pad) & (nlon[fa] < lon.max() + pad) & (nlat[fa] > lat.min() - pad) & (nlat[fa] < lat.max() + pad)
    ids = np.flatnonzero(inb)
    if len(ids) > max_arcs:
        ids = np.random.default_rng(0).choice(ids, max_arcs, replace=False)
    net = folium.FeatureGroup(name=f"Road network ({len(ids)} arcs shown)", show=False)
    for seg in _arc_latlon(tr, ids):
        folium.PolyLine(seg, color="#888888", weight=1, opacity=.6).add_to(net)
    net.add_to(m)

    ref = folium.FeatureGroup(name="Reference route (.route)")
    arcs = tr["route"]
    for arc_id, seg in zip(arcs, _arc_latlon(tr, arcs)):
        folium.PolyLine(seg, color="green", weight=5, opacity=.8, tooltip=f"arc {arc_id}").add_to(ref)
    ref.add_to(m)

    gps = folium.FeatureGroup(name="GPS points")
    step = max(1, len(lon) // max_pts)
    for i in range(0, len(lon), step):
        folium.CircleMarker((lat[i], lon[i]), radius=2, color="blue", fill=True, fill_opacity=.8,
                            tooltip=f"pt {i}  t={tr['t'][i]:.0f}s").add_to(gps)
    folium.Marker((lat[0], lon[0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(gps)
    folium.Marker((lat[-1], lon[-1]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(gps)
    gps.add_to(m)

    pq = Path(proc or "") / f"{part}_candidates.parquet"
    if proc and pq.exists():
        c = pd.read_parquet(pq, filters=[("trip_id", "==", tid)])
        if len(c):
            pts = np.linspace(0, len(lon) - 1, n_sample + 2, dtype=int)[1:-1]
            cg = folium.FeatureGroup(name="Candidates of 3 sample points (green=correct, red=other)", show=False)
            for p in pts:
                cc = c[c.point_idx == p]
                for arc_id, lab, d in zip(cc.arc_id, cc.label, cc.dist_m):
                    seg = _arc_latlon(tr, np.array([arc_id]))[0]
                    folium.PolyLine(seg, color="lime" if lab else "red", weight=6 if lab else 3,
                                    tooltip=f"pt {p} arc {arc_id} d={d:.1f} m label={lab}").add_to(cg)
                folium.CircleMarker((lat[p], lon[p]), radius=6, color="black", fill=True, tooltip=f"sample point {p}").add_to(cg)
            cg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ('<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px;'
              'border:1px solid #888;font-size:12px"><b>%s/%s</b> (%d pts)<br>'
              '<span style="color:blue">&#9679;</span> GPS points<br>'
              '<span style="color:green">&#9632;</span> reference route<br>'
              '<span style="color:#888">&#9632;</span> road network<br>'
              '<span style="color:red">&#9632;</span>/<span style="color:lime">&#9632;</span> candidates</div>') % (part, tid, len(lon))
    m.get_root().html.add_child(folium.Element(legend))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw"); ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="data_pipeline/results/maps")
    ap.add_argument("--trips", nargs="*", default=["train/00000095", "train/00000058", "student_test/00000000"])
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    for t in a.trips:
        part, tid = t.split("/")
        f = Path(a.out) / f"map_{part}_{tid}.html"
        build_map(a.data, part, tid, a.proc).save(str(f)); print("wrote", f)


if __name__ == "__main__":
    main()
