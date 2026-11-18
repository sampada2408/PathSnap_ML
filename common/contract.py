"""Frozen interfaces shared by all four people. Change only with agreement from everyone.

1. CANDIDATE TABLE  (P1 produces -> P2, P3 consume).  Parquet in data/processed/<partition>.parquet
   keys    : trip_id, point_idx, arc_id
   label   : 1 if arc_id is the reference arc for that GPS point else 0   (absent at API inference time)
   features: FEATURE_COLUMNS below (P1 fills in at the first checkpoint, then frozen)

2. MODEL  (P2 produces -> P3, P4 consume)
   artifacts/model.joblib + artifacts/feature_columns.json
   model.predict_proba(df[FEATURE_COLUMNS]) -> ndarray[n_rows, 2]; column 1 = P(arc is the correct one)

3. DECODER  (P3 produces -> P4 consumes)
   decode(cands: DataFrame, probs: ndarray, arcs: ndarray, nodes: ndarray, times: ndarray) -> list[int]
   returns one arc_id per GPS point (-1 if undecidable)

4. PREDICTOR  (P4 wires everything together)  api/predictor.py
   predict_trip(trip_dir) -> {"pred_arcs": list[int]}   # one arc id per GPS point
   MUST NOT read the .route file. Reference routes are read only in api/evaluation.py (server side).
"""

KEY_COLS = ["trip_id", "point_idx", "arc_id"]
LABEL_COL = "label"

# P1: fill in at the first checkpoint, e.g. ["dist_m", "bearing_diff", "speed", "frac_along", "arc_len_m", "dist_rank"]
FEATURE_COLUMNS: list = []

# Partition names on disk (data/raw/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route})
PARTITIONS = ["train", "student_test"]
