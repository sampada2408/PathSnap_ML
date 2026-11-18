"""PathSnap API (Laptop 1).

Run from the repo root (PowerShell):
    $env:TRIP_ROOT = "data\raw\student_test"
    uvicorn api.main:app --host 0.0.0.0 --port 8000

At evaluation time point TRIP_ROOT at the folder holding the 10 evaluator trips.
"""
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.evaluation import evaluate
from api.io_utils import arc_coords, load_trip
from api.predictor import predict_trip

def _resolve_trip_root() -> Path:
    env_root = os.getenv("TRIP_ROOT")
    if env_root:
        return Path(env_root)
    # Automatically prioritize evaluator_test if dropped into workspace
    for candidate in [Path("evaluator_test"), Path("data/raw/evaluator_test"), Path("data/raw/student_test")]:
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
            return candidate
    return Path("data/raw/student_test")

TRIP_ROOT = _resolve_trip_root()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    log.info("REQUEST from=%s %s %s -> %s (%.0f ms)", client, request.method, request.url.path, response.status_code, ms)
    return response


def _trip_ids():
    if not TRIP_ROOT.is_dir():
        return []
    return sorted(p.name for p in TRIP_ROOT.iterdir() if p.is_dir())


@app.get("/health")
def health():
    return {"status": "ok", "trip_root": str(TRIP_ROOT), "n_trips": len(_trip_ids())}


@app.get("/trips")
def trips():
    return {"trips": _trip_ids()}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in _trip_ids():  # also blocks path traversal
        raise HTTPException(status_code=404, detail=f"Unknown trip '{trip_id}'")
    trip_dir = TRIP_ROOT / trip_id

    t0 = time.perf_counter()
    result = predict_trip(trip_dir)            # never sees labels
    pred_arcs = result["pred_arcs"]
    infer_ms = (time.perf_counter() - t0) * 1000

    metrics = evaluate(trip_dir, pred_arcs)    # reads the reference route, server-side only
    track, nodes, arcs = load_trip(trip_dir)

    # de-duplicate consecutive repeats so the drawn route is compact
    route_arcs = [a for i, a in enumerate(pred_arcs) if a >= 0 and (i == 0 or a != pred_arcs[i - 1])]
    log.info("PREDICT trip=%s points=%d infer_ms=%.0f route_f1=%.3f", trip_id, len(track), infer_ms, metrics["route_f1"])

    return {
        "trip_id": trip_id,
        "n_points": int(len(track)),
        "gps_points": [[float(lat), float(lon)] for lon, lat in track[:, :2]],
        "pred_route": arc_coords(nodes, arcs, route_arcs),
        "metrics": metrics,
        "inference_ms": round(infer_ms, 1),
    }
