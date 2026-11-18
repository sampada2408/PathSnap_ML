# 4. Bayesian Hyperparameter Optimisation Comparison

**Optimization Cost:** 469.25 seconds across 15 trials.

**Best Trial:** #13 (CV Macro F1 = 0.9029)

```json
{
  "n_estimators": 180,
  "learning_rate": 0.07211087599241674,
  "num_leaves": 78,
  "max_depth": 7,
  "min_child_samples": 20,
  "subsample": 0.9411425339378616,
  "colsample_bytree": 0.9294043546147405,
  "reg_alpha": 0.14830180701382628,
  "reg_lambda": 0.12085550858926611
}
```

| Metric                             | Untuned Model   | Tuned Model     |   Improvement |
|:-----------------------------------|:----------------|:----------------|--------------:|
| Cross-Validation Macro F1-score    | 0.9021 ± 0.0195 | 0.9029 ± 0.0191 |        0.0008 |
| Student-Testing Top-1 Accuracy     | 0.8556          | 0.8568          |        0.0013 |
| Student-Testing Macro F1-score     | 0.9157          | 0.9167          |        0.001  |
| Student-Testing Candidate Log Loss | 0.0808          | 0.0793          |       -0.0014 |
| Inference Time (ms per 1k)         | 1.02            | 2.93            |        1.91   |
| Training Time (s)                  | 2.69            | 5.73            |        3.04   |
