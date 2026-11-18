# PathSnap
Map-matching of noisy GPS trajectories (classifier + HMM/Viterbi).

## Layout
data_pipeline/ (P1)  modeling/ (P2)  decoder/ (P3)  api/ ui/ viz/ (P4)  common/ (shared contract)

## Setup (Windows)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Place the unzipped dataset in data/raw/ (not committed).
