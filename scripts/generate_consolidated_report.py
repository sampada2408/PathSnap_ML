"""
PathSnap: Technical Report Generator (PDF/HTML Deliverable 2)
============================================================
Compiles the consolidated technical report adhering strictly to the PES University
Hackathon Guidelines:
- Section 2.2: Dataset Understanding and EDA
- Section 3: Model Experimentation and Comparison (6 models)
- Section 4: Bayesian Hyperparameter Optimisation (Optuna)
- Section 5: HMM and Viterbi Route Reconstruction
- Section 6: Geographical Visualisation Using Folium
- Section 7: API, UI and Deployment
"""

from __future__ import annotations

import base64
from pathlib import Path

import markdown

REPORTS_DIR = Path("reports")
DATA_RESULTS_DIR = Path("data_pipeline/results")
FIGURES_DIR = REPORTS_DIR / "figures"


def generate_markdown() -> str:
    # Read sections 3, 4, 5
    with open(REPORTS_DIR / "report_sections_3_and_4.md", "r", encoding="utf-8") as f:
        sec_3_and_4 = f.read()

    with open(REPORTS_DIR / "report_section_5.md", "r", encoding="utf-8") as f:
        sec_5 = f.read()

    with open(REPORTS_DIR / "end_to_end_evaluation_table.md", "r", encoding="utf-8") as f:
        student_test_eval_table = f.read()

    eval_test_table_path = REPORTS_DIR / "evaluator_test_end_to_end_evaluation_table.md"
    eval_test_eval_table = ""
    if eval_test_table_path.exists():
        with open(eval_test_table_path, "r", encoding="utf-8") as f:
            eval_test_eval_table = f.read()

    md = f"""# PATHSNAP (MAPSNAP): CONSOLIDATED TECHNICAL REPORT
**Distributed GPS Trajectory Map-Matching Engine**  
**Course:** UE24CS352A Machine Learning | **Institution:** PES University, RR Campus  
**Project Repository:** [https://github.com/sampada2408/PathSnap_ML](https://github.com/sampada2408/PathSnap_ML)  

---

## 1. Executive Summary & Team Contributions
PathSnap is a robust, distributed map-matching engine designed to reconstruct vehicle paths from sparse, noisy GPS coordinates across diverse global road networks. The system is split across 4 frozen, modular contracts:
- **Person 1 (Data Pipeline - `data_pipeline/`)**: Parsing WGS84 raw datasets, UTM projection, spatial candidate generation ($R=60\,\text{{m}}, K=15$) using `STRtree`, kinematic feature engineering, and ground-truth route labeling via monotone dynamic programming.
- **Person 2 (Modeling Pipeline - `modeling/`)**: Benchmarking 6 models across 5 distinct algorithm families using 5-fold `GroupKFold` cross-validation grouped by `trip_id`, 15-trial Bayesian optimization with Optuna, and exporting champion model artifacts (`artifacts/model.joblib`).
- **Person 3 (Decoder Pipeline - `decoder/`)**: Directed road graph shortest path routing via Dijkstra with LRU caching, Newson–Krumm exponential transition modeling, hybrid classifier-distance emission modeling, and Viterbi dynamic programming trellis decoder (`artifacts/hmm_config.json`).
- **Person 4 (API, UI & Deployment - `api/`, `ui/`, `viz/`)**: Production FastAPI backend, single-page Leaflet.js evaluator interface, multi-layer Folium visualizers, and two-laptop network deployment.

---

## 2. Dataset Understanding and Exploratory Data Analysis (Section 2.2)

### 2.2.1 How did you preprocess the data?
1. **Raw File Ingestion**: Each trip is self-contained with `.track` (longitude, latitude, timestamp in seconds), `.nodes` (WGS84 lon/lat coordinates), and `.arcs` (from-node, to-node directed road connections).
2. **Geodesic UTM Coordinate Transformation**: Because trips originate from multiple worldwide cities (e.g. lon 8.6, 4.9, 18.7, -78), applying a single global planar projection causes catastrophic distortion. We dynamically compute the local UTM zone from each trip's mean longitude/latitude ($\text{{zone}} = \lfloor(\text{{lon}} + 180) / 6\rfloor \pmod{{60}} + 1$; $\text{{EPSG}} = 32600 + \text{{zone}}$) and project coordinates using `pyproj.Transformer` into high-precision meters.
3. **Spatial Candidate Generation (`STRtree`)**: For each GPS point, road arcs within a spatial search radius of $R = 60\,\text{{m}}$ are queried using a 2D spatial R-Tree (`shapely.STRtree`). The nearest $K \le 15$ candidate segments are retained and sorted by perpendicular projection distance.
4. **Fractional Arc Projection**: For every candidate arc $(u, v)$, the GPS observation is orthogonally projected onto the segment, calculating perpendicular distance $d_{{\text{{perp}}}}$ and fractional progression $u \in [0, 1]$ along the segment vector.
5. **Monotone Route Alignment (Labels)**: For training and evaluation, reference routes (`.route`) are aligned to candidate observations using monotonic forward dynamic programming, ensuring that only plausible forward traversals are assigned positive labels ($y=1$).

### 2.2.2 What patterns did you observe during data exploration? What features were extracted?
#### Key Observed Patterns:
- **Spatial Measurement Noise**: Raw GPS coordinates exhibit lateral jitter with standard deviation $\sigma_z \approx 4 - 10\,\text{{m}}$, often placing points off the road or onto parallel side-streets.
- **Urban Canyon Multipath**: In dense environments, instantaneous headings fluctuate erratically when vehicle velocity drops below $2\,\text{{m/s}}$.
- **Variable Sampling Frequencies & Temporal Gaps**: Sampling intervals are nominally $\sim 1\,\text{{s}}$, but tunnel traversals and signal blockages introduce temporal gaps exceeding $10 - 60\,\text{{s}}$.
- **Candidate Class Imbalance**: With $K=15$ candidates per observation, true positive arcs account for only $\sim 6.67\%$ of candidates, requiring macro-averaged F1 and cross-entropy optimization.

#### Engineered Features:
1. `dist_m`: Orthogonal Euclidean distance from GPS point to candidate road arc (meters). Primary physical indicator.
2. `bearing_diff`: Angular divergence $| \theta_{{\text{{gps}}}} - \theta_{{\text{{arc}}}} | \in [0, 180^\circ]$. Identifies wrong-way and perpendicular cross-street candidates.
3. `speed`: Instantaneous velocity computed from adjacent GPS positions via central difference $(\Delta s / \Delta t)$.
4. `frac_along`: Fractional position $u \in [0, 1]$ along the directed arc.
5. `arc_len_m`: Physical length of the candidate road arc in meters.
6. `dist_rank`: Discrete ranking of candidates per GPS point ordered by proximity ($0$ = closest).
7. `delta_time_s`: Elapsed time since previous GPS observation, flagging signal dropouts.
8. `perp_dist_m`: Directional perpendicular offset $d_{{\text{{dist}}}} \times \sin(\theta_{{\text{{bearing}}}})$.

### 2.2.3 Missing-Value and Quality Analysis
| Feature Name | Missing Count | Data Type | Imputation & Handling Strategy |
| :--- | :---: | :---: | :--- |
| `dist_m` | 0 | float64 | Fully populated from geometric projection |
| `frac_along` | 0 | float64 | Clamped strictly to $[0.0, 1.0]$ |
| `speed` | 0 | float64 | Imputed to 0.0 at trajectory boundary points |
| `bearing_diff` | 0 | float64 | Replaced with uninformative default ($90^\circ$) when `speed < 2 m/s` |
| `heading_valid`| 0 | int8 | Binary flag ($1$ if displacement $> 2\,\text{{m}}$ and $\Delta t \le 10\,\text{{s}}$, else $0$) |

---

{sec_3_and_4}

---

{sec_5}

---

## 6. Geographical Visualisation Using Folium (Section 6)

Interactive multi-layer geographical visualisations were generated using Folium (`viz/folium_map.py`), displaying the real road network, raw GPS trajectory, ground-truth reference route, and predicted route.

### Map Design Principles:
1. **GPS Observations Layer**: Plotted as distinct blue circle markers (`#2b6cb0`, radius 2.5 px).
2. **Road Network Layer**: Rendered as a subtle grey wireframe (`#999999`, weight 1.5 px, opacity 0.7) covering the localized bounding box.
3. **Reference Ground-Truth Route**: Rendered as an emerald green polyline (`#2f7d4f`, weight 6 px, opacity 0.8).
4. **Predicted Reconstructed Route**: Overlaid as a bold ruby polyline (`#d1495b`, weight 3.5 px, opacity 0.95).
5. **Interactive Controls & Legend**: Built-in Leaflet Layer Control enables toggling individual layers; persistent floating legend identifies route components.

Generated interactive map artifacts:
- Student-Test Trip `00000056`: `reports/maps/00000056.html` (3.0 MB)
- Evaluator Test Trip `00000015`: `reports/maps/00000015.html`

---

## 7. API, UI and Deployment (Section 7)

### 7.1 Architecture & Endpoints
The production backend is built with FastAPI (`api/main.py`) and exposes three high-throughput REST endpoints:
- `GET /health`: System status, active `TRIP_ROOT` partition, and count of accessible trips.
- `GET /trips`: JSON array of all trip IDs available in the partition.
- `POST /predict/{{trip_id}}`: Executes the Contract 4 inference pipeline:
  $$\\text{{Raw Files}} \\xrightarrow{{\\text{{P1}}}} \\text{{Candidates}} \\xrightarrow{{\\text{{P2 Model}}}} P(\\text{{arc}}) \\xrightarrow{{\\text{{P3 Viterbi}}}} \\text{{Reconstructed Route}}$$
  Returns predicted arcs, GPS coordinates, route polyline coordinates, execution latency, and server-side metrics (Precision, Recall, F1, Continuity).

### 7.2 Two-Laptop Network Deployment Setup
- **Laptop 1 (Model Server)**:
  Hosts `artifacts/model.joblib`, candidate extraction, and Viterbi decoder.
  ```powershell
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
  ```
- **Laptop 2 (Evaluator UI)**:
  Opens `ui/index.html` in Chrome/Firefox. Evaluator inputs `http://<Laptop1_IP>:8080` into the API address bar and clicks **Match route**. The Leaflet map dynamically visualizes GPS dots, predicted route polylines, and real-time metric cards.

### 7.3 End-to-End System Evaluation Benchmarks

#### Partition A: Student-Test Dataset (10 Trips, 19,160 GPS Points)
{student_test_eval_table}

#### Partition B: Evaluator Test Dataset (`evaluator_test`, 10 Trips, 19,674 GPS Points)
{eval_test_eval_table}

---

## 8. Conclusion
By pairing high-throughput gradient boosting (Tuned LightGBM, 91.67% F1) with physical road topology constraints via Viterbi trellis dynamic programming, PathSnap delivers:
- **90.22% Route Recall** and **96.18% Physical Route Continuity** on student evaluation.
- **84.16% Route Recall** and **94.34% Physical Route Continuity** on the official evaluator test set.
- **Sub-second latency** (~966 ms per 1,000 GPS points), fully capable of real-time distributed deployment across two evaluation laptops.
"""
    return md


def main():
    md_content = generate_markdown()
    md_path = REPORTS_DIR / "CONSOLIDATED_TECHNICAL_REPORT.md"
    html_path = REPORTS_DIR / "CONSOLIDATED_TECHNICAL_REPORT.html"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Wrote Markdown report to: {md_path}")

    # Convert to HTML with tables, fenced code, and KaTeX math support
    html_body = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "toc"]
    )

    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PathSnap: Consolidated Technical Report (PES University ML Hackathon)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.65;
    color: #24292e;
    max-width: 1000px;
    margin: 40px auto;
    padding: 0 30px;
    background: #ffffff;
  }}
  h1, h2, h3, h4 {{ color: #0f172a; margin-top: 1.5em; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  h1 {{ font-size: 28px; border-bottom: 2px solid #0f172a; }}
  h2 {{ font-size: 22px; color: #1e293b; }}
  h3 {{ font-size: 18px; color: #334155; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 13.5px;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{
    background-color: #f1f5f9;
    font-weight: 600;
  }}
  tr:nth-child(even) {{ background-color: #f8fafc; }}
  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 14px 18px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 13px;
  }}
  code {{
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 5px;
    border-radius: 4px;
    font-family: Consolas, Monaco, "Courier New", monospace;
    font-size: 13px;
  }}
  pre code {{ background: none; color: inherit; padding: 0; }}
  blockquote {{
    border-left: 4px solid #3b82f6;
    margin: 0;
    padding: 8px 16px;
    background: #eff6ff;
    color: #1e3a8a;
  }}
  @media print {{
    body {{ max-width: 100%; margin: 0; padding: 15mm; }}
    h1, h2, h3 {{ page-break-after: avoid; }}
    table, pre {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)
    print(f"Wrote HTML report to: {html_path}")


if __name__ == "__main__":
    main()
