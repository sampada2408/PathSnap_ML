# 4. Bayesian Hyperparameter Optimisation Comparison

**Optimization Cost:** 2.55 seconds across 5 trials.

**Best Trial:** #0 (CV Macro F1 = 0.9995)

```json
{
  "n_estimators": 120,
  "learning_rate": 0.2176307935254711,
  "num_leaves": 97,
  "max_depth": 7,
  "min_child_samples": 17,
  "subsample": 0.6967983561008608,
  "colsample_bytree": 0.6674250836504598,
  "reg_alpha": 1.1752647960576206,
  "reg_lambda": 0.10129197956845731
}
```

| Metric                             | Untuned Model   | Tuned Model     |   Improvement |
|:-----------------------------------|:----------------|:----------------|--------------:|
| Cross-Validation Macro F1-score    | 0.9995 ± 0.0003 | 0.9995 ± 0.0003 |        0      |
| Student-Testing Top-1 Accuracy     | 0.9574          | 0.9574          |        0      |
| Student-Testing Macro F1-score     | 0.9994          | 0.9994          |        0      |
| Student-Testing Candidate Log Loss | 0.0013          | 0.0014          |        0.0001 |
| Inference Time (ms per 1k)         | 1.57            | 0.94            |       -0.63   |
| Training Time (s)                  | 0.10            | 0.06            |       -0.05   |
