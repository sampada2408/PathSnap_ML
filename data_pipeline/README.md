# data_pipeline (Person 1)

Run from the repo root, with the unzipped dataset at `data/raw/{train,student_test}/<trip_id>/`.

```bash
python -m data_pipeline.inventory_eda --data data/raw --out data/processed/eda
python -m data_pipeline.build_tables  --data data/raw --out data/processed_v0 --partitions train --limit 10 --v0   # v0 checkpoint
python -m data_pipeline.build_tables  --data data/raw --out data/processed --partitions train student_test       # full table
```
Modules: `io_utils` (readers, UTM32N projection) | `candidates` (STRtree radius+kNN) |
`labels` (monotone DP alignment of .route -> one arc per point) | `features` | `build_tables` (CLI).
`.route` is read ONLY in `labels.py` / `build_tables.py` for train and student_test, never at inference.
