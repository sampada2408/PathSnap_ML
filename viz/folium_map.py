"""Folium map: GPS points, road network, reference route, predicted route (layer control + legend).

    python -m viz.folium_map --trip data\raw\student_test\00000056 --out reports\maps\00000056.html --predict

--predict runs api.predictor.predict_trip (stub now, real model later).
Reference routes are for OFFLINE analysis/report maps only; the evaluator UI never shows them.
"""
import argparse
from pathlib import Path

import folium
import numpy as np

from api.io_utils import arc_coords, load_trip

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
 padding: 10px 14px; border: 1px solid #999; border-radius: 4px; font: 13px 'Segoe UI', sans-serif;">
 <b>Trip {tid}</b><br>
 <span style="color:#2b6cb0">&#9679;</span> GPS observations<br>
 <span style="color:#999">&#9473;</span> Road network<br>
 <span style="color:#2f7d4f">&#9473;</span> Reference route<br>
 <span style="color:#d1495b">&#9473;</span> Predicted route
</div>"""


def build_map(trip_dir, pred_arcs=None, out_html="trip_map.html", margin_deg=0.004, max_points=6000):
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track, nodes, arcs = load_trip(trip_dir)
    lat, lon = track[:, 1], track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="&copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom",
        name="Esri World Street Map",
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.de/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, tile.openstreetmap.de",
        name="OpenStreetMap Mirror",
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Light",
        subdomains="abcd",
        control=True,
    ).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road network near the trajectory only (full networks can have 300k arcs)
    inside = ((nodes[:, 1] >= lat.min() - margin_deg) & (nodes[:, 1] <= lat.max() + margin_deg) &
              (nodes[:, 0] >= lon.min() - margin_deg) & (nodes[:, 0] <= lon.max() + margin_deg))
    net_ids = np.where(inside[arcs[:, 0]] & inside[arcs[:, 1]])[0]
    net = folium.FeatureGroup(name=f"Road network ({len(net_ids)} arcs)", show=False)
    folium.PolyLine(arc_coords(nodes, arcs, net_ids), color="#999999", weight=1.5, opacity=.7).add_to(net)
    net.add_to(m)

    pts = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(track) // max_points)
    for la, lo in zip(lat[::step], lon[::step]):
        folium.CircleMarker([la, lo], radius=2, color="#2b6cb0", fill=True, fill_opacity=.8, weight=0).add_to(pts)
    pts.add_to(m)

    route_file = trip_dir / f"{tid}.route"
    if route_file.exists():
        ref = np.loadtxt(route_file, dtype=int, ndmin=1)
        g = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(arc_coords(nodes, arcs, ref), color="#2f7d4f", weight=6, opacity=.8).add_to(g)
        g.add_to(m)

    if pred_arcs is not None:
        dedup = [a for i, a in enumerate(pred_arcs) if a >= 0 and (i == 0 or a != pred_arcs[i - 1])]
        g = folium.FeatureGroup(name="Predicted route")
        folium.PolyLine(arc_coords(nodes, arcs, dedup), color="#d1495b", weight=3, opacity=.95).add_to(g)
        g.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(tid=tid)))
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True)
    ap.add_argument("--out", default="reports/maps/trip_map.html")
    ap.add_argument("--predict", action="store_true", help="overlay predictions from api.predictor")
    a = ap.parse_args()
    pred = None
    if a.predict:
        from api.predictor import predict_trip
        pred = predict_trip(a.trip)["pred_arcs"]
    print("wrote", build_map(a.trip, pred, a.out))
