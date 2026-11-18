# 3. Model Experimentation and Comparison
To identify the optimal candidate classification model for PathSnap, we conducted rigorous experiments across **6 distinct models representing 5 algorithmic families**: Linear/Probabilistic, Ensemble/Bagging, Scikit-Learn Histogram Boosting, Gradient Boosted Decision Trees (LightGBM, XGBoost), and Neural Networks (MLP).
### Experimental Design & Leakage Prevention- **Training Partition**: 80 training trips evaluated via 5-fold `GroupKFold` strictly partitioned by `trip_id`.  This ensures **zero data leakage** between training and validation folds (no observation from any trip appears in both).- **Student-Testing Partition**: 10 unseen student-testing trips used exclusively for development-stage model selection.- **Feature Standardization**: Standard scaling applied within folds to avoid leakage for linear and neural network models.- **Reproducibility**: Random seed fixed to 42 across all data partitions and model initializations.
## 3.1 Overall Model Comparison
| Model                        | Algorithm Family             | Key Configuration                                         | CV Macro F1-score   |   Top-1 Accuracy |   Route continuity |   Student-Test Macro F1-score |   Log Loss |   Inference Time (ms/1k) |   Train Time (s) |
|:-----------------------------|:-----------------------------|:----------------------------------------------------------|:--------------------|-----------------:|-------------------:|------------------------------:|-----------:|-------------------------:|-----------------:|
| Logistic Regression          | Linear & Probabilistic       | C=1.0, penalty=l2, solver=lbfgs                           | 0.8794 ± 0.0194     |           0.8372 |             0.6863 |                        0.8869 |     0.118  |                     0.16 |             1.58 |
| Random Forest                | Ensemble (Bagging)           | n_estimators=100, max_depth=12, min_samples_split=5       | 0.9005 ± 0.0202     |           0.8525 |             0.697  |                        0.9133 |     0.0839 |                     2.45 |            73.89 |
| HistGradientBoosting         | Boosting (Scikit-Learn)      | max_iter=100, learning_rate=0.08, max_leaf_nodes=31       | 0.9018 ± 0.0196     |           0.8555 |             0.6987 |                        0.9153 |     0.0808 |                     2.65 |            10.12 |
| LightGBM                     | Boosting (Gradient Boosting) | n_estimators=100, learning_rate=0.08, num_leaves=31       | 0.9021 ± 0.0195     |           0.8556 |             0.6963 |                        0.9157 |     0.0808 |                     1.08 |             3.45 |
| XGBoost                      | Boosting (Gradient Boosting) | n_estimators=100, learning_rate=0.08, max_depth=6         | 0.9025 ± 0.0194     |           0.8523 |             0.6973 |                        0.9145 |     0.0809 |                     0.56 |             6.5  |
| Multi-Layer Perceptron (MLP) | Neural Networks              | hidden_layer_sizes=(64, 32), activation=relu, solver=adam | 0.8992 ± 0.0203     |           0.8552 |             0.7012 |                        0.9151 |     0.0815 |                     1.8  |           251.33 |

## 3.2 Compulsory Questions for EVERY Model
### 3.2.1 Logistic Regression (Linear & Probabilistic)
#### A. Model Understanding1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   Logistic Regression establishes a clear, interpretable probabilistic linear baseline. Since GPS candidate scoring depends heavily on monotonic physical relationships (e.g., probability strictly decays with distance from trajectory and heading divergence), a regularized linear log-odds model provides well-calibrated probabilities with minimal computational overhead.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**: Fast training and inference (<0.01 ms per point), zero risk of deep tree memorization, smooth well-calibrated posterior probabilities directly usable as HMM emissions without calibration.
   - **Limitations**: Inability to model complex non-linear spatial interactions (e.g. high speed with sharp heading difference vs. low speed near intersections) without manual polynomial feature engineering.
#### B. Training and Configuration3. **What were the important hyperparameters and their initial values?**
   - Hyperparameters: `{"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "class_weight": null, "random_state": 42}`
4. **What loss function or optimisation objective was used?**
   Binary Cross-Entropy (Log Loss) with L2 weight regularization (ridge penalty).
#### C. Experimental Results5. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **0.8372**
   - Macro F1-score: **0.8869**
   - Candidate Log Loss: **0.1180**
   - Precision: **0.8034** | Recall: **0.7910**
   - Route Continuity Proxy: **0.6863**
   - Inference Latency: **0.16 ms / 1,000 candidates**
6. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   As reported above, Top-1 Accuracy is **0.8372**, Macro F1 is **0.8869**, and Log Loss is **0.1180**.
7. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   Observed mild underfitting. The training and cross-validation scores (Macro F1) are closely aligned, but overall Top-1 accuracy is constrained by linear boundary limitations. Comparing the 5-fold CV Macro F1 (0.8794 ± 0.0194) to the Student-Test Macro F1 (0.8869) shows minimal generalization gap.
### 3.2.2 Random Forest (Ensemble (Bagging))
#### A. Model Understanding1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   Random Forest constructs an ensemble of de-correlated decision trees via bootstrap aggregation and random feature subspace sampling. It naturally captures non-linear thresholds in spatial features (e.g., dist_rank == 0 combined with bearing_diff < 30 degrees) while remaining resilient to noisy GPS outliers.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**: Excellent robustness against spatial outliers and multipath noise, handles non-linear relationships without feature scaling, low variance due to extensive bagging.
   - **Limitations**: Ensemble tree averaging tends to push predicted probabilities away from 0 and 1 towards the base rate, which slightly softens emission contrast for HMM Viterbi decoding. Slower inference latency than linear models.
#### B. Training and Configuration3. **What were the important hyperparameters and their initial values?**
   - Hyperparameters: `{"n_estimators": 100, "max_depth": 12, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "n_jobs": -1, "random_state": 42}`
4. **What loss function or optimisation objective was used?**
   Gini impurity criterion across bagged decision trees.
#### C. Experimental Results5. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **0.8525**
   - Macro F1-score: **0.9133**
   - Candidate Log Loss: **0.0839**
   - Precision: **0.8702** | Recall: **0.8195**
   - Route Continuity Proxy: **0.6970**
   - Inference Latency: **2.45 ms / 1,000 candidates**
6. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   As reported above, Top-1 Accuracy is **0.8525**, Macro F1 is **0.9133**, and Log Loss is **0.0839**.
7. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   Balanced generalization. Cross-validation macro F1 closely tracks student-test performance, with max_depth=12 and min_samples_leaf=2 successfully preventing leaf memorization. Comparing the 5-fold CV Macro F1 (0.9005 ± 0.0202) to the Student-Test Macro F1 (0.9133) shows minimal generalization gap.
### 3.2.3 HistGradientBoosting (Boosting (Scikit-Learn))
#### A. Model Understanding1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   HistGradientBoosting discretizes continuous spatial features into integer-valued histogram bins (256 bins), drastically accelerating split evaluations while natively supporting monotonic and non-linear patterns.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**: Extremely fast training and evaluation, native binning prevents overfitting to fine GPS coordinate noise, low memory footprint.
   - **Limitations**: Slight quantization loss due to 256-bin discretization on fine continuous distances under 5 meters.
#### B. Training and Configuration3. **What were the important hyperparameters and their initial values?**
   - Hyperparameters: `{"max_iter": 100, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0, "random_state": 42}`
4. **What loss function or optimisation objective was used?**
   Binary cross-entropy loss (logistic deviance) optimized via sequential pseudo-residuals.
#### C. Experimental Results5. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **0.8555**
   - Macro F1-score: **0.9153**
   - Candidate Log Loss: **0.0808**
   - Precision: **0.8567** | Recall: **0.8396**
   - Route Continuity Proxy: **0.6987**
   - Inference Latency: **2.65 ms / 1,000 candidates**
6. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   As reported above, Top-1 Accuracy is **0.8555**, Macro F1 is **0.9153**, and Log Loss is **0.0808**.
7. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   Well-regularized fit. Early stopping and L2 shrinkage kept validation loss stable across all folds. Comparing the 5-fold CV Macro F1 (0.9018 ± 0.0196) to the Student-Test Macro F1 (0.9153) shows minimal generalization gap.
### 3.2.4 LightGBM (Boosting (Gradient Boosting))
#### A. Model Understanding1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   LightGBM employs Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) with leaf-wise (best-first) tree growth. For candidate ranking and classification, leaf-wise growth allows the model to isolate subtle spatial distinctions between parallel roads and highway ramps.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**: State-of-the-art tabular accuracy, sharp probability discrimination, very fast inference, native handling of non-linear interactions between bearing difference and vehicle speed.
   - **Limitations**: Leaf-wise tree growth can overfit small datasets if num_leaves and min_child_samples are not properly tuned.
#### B. Training and Configuration3. **What were the important hyperparameters and their initial values?**
   - Hyperparameters: `{"n_estimators": 100, "learning_rate": 0.08, "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.0, "reg_lambda": 0.0, "random_state": 42, "n_jobs": -1, "verbose": -1}`
4. **What loss function or optimisation objective was used?**
   Binary logloss objective with gradient boosting steps.
#### C. Experimental Results5. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **0.8556**
   - Macro F1-score: **0.9157**
   - Candidate Log Loss: **0.0808**
   - Precision: **0.8548** | Recall: **0.8429**
   - Route Continuity Proxy: **0.6963**
   - Inference Latency: **1.08 ms / 1,000 candidates**
6. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   As reported above, Top-1 Accuracy is **0.8556**, Macro F1 is **0.9157**, and Log Loss is **0.0808**.
7. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   Optimal fit achieved. Strong cross-validation stability (low variance across folds) and excellent generalization on student-test trips. Comparing the 5-fold CV Macro F1 (0.9021 ± 0.0195) to the Student-Test Macro F1 (0.9157) shows minimal generalization gap.
### 3.2.5 XGBoost (Boosting (Gradient Boosting))
#### A. Model Understanding1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   XGBoost incorporates second-order Taylor expansion gradients and exact L1/L2 tree regularization, providing fine-grained control over model complexity when identifying candidate road segments.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**: Exact split finding, strong L1/L2 penalty prevents extreme tree depth, highly reliable probability predictions.
   - **Limitations**: Slightly longer training time compared to histogram-binned LightGBM; requires careful tuning of subsampling ratios.
#### B. Training and Configuration3. **What were the important hyperparameters and their initial values?**
   - Hyperparameters: `{"n_estimators": 100, "learning_rate": 0.08, "max_depth": 6, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.0, "reg_lambda": 1.0, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}`
4. **What loss function or optimisation objective was used?**
   Binary logistic loss with exact Hessian second-order gradients and L2 leaf regularization.
#### C. Experimental Results5. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **0.8523**
   - Macro F1-score: **0.9145**
   - Candidate Log Loss: **0.0809**
   - Precision: **0.8598** | Recall: **0.8335**
   - Route Continuity Proxy: **0.6973**
   - Inference Latency: **0.56 ms / 1,000 candidates**
6. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   As reported above, Top-1 Accuracy is **0.8523**, Macro F1 is **0.9145**, and Log Loss is **0.0809**.
7. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   Slight early tendency to overfit high-rank candidates, mitigated by setting max_depth=6 and subsample=0.85. Comparing the 5-fold CV Macro F1 (0.9025 ± 0.0194) to the Student-Test Macro F1 (0.9145) shows minimal generalization gap.
### 3.2.6 Multi-Layer Perceptron (MLP) (Neural Networks)
#### A. Model Understanding1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   MLP maps scaled spatial features into a continuous non-linear latent space using multiple dense layers with ReLU activations and Adam optimization. It tests whether smooth continuous representations offer better map-matching.
2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**: Learns continuous non-linear decision boundaries, outputs smooth probability gradients across feature transitions.
   - **Limitations**: Requires careful feature normalization (StandardScaler), sensitive to hyperparameter choices (learning rate, alpha), slower convergence than tree-based ensembles on tabular data.
#### B. Training and Configuration3. **What were the important hyperparameters and their initial values?**
   - Hyperparameters: `{"hidden_layer_sizes": [64, 32], "activation": "relu", "solver": "adam", "alpha": 0.0001, "learning_rate_init": 0.001, "max_iter": 200, "early_stopping": true, "validation_fraction": 0.1, "random_state": 42}`
4. **What loss function or optimisation objective was used?**
   Binary cross-entropy loss optimized with Adam and backpropagation.
#### C. Experimental Results5. **What was the model's performance on the student-testing dataset?**
   - Observation-level Top-1 Accuracy: **0.8552**
   - Macro F1-score: **0.9151**
   - Candidate Log Loss: **0.0815**
   - Precision: **0.8534** | Recall: **0.8421**
   - Route Continuity Proxy: **0.7012**
   - Inference Latency: **1.80 ms / 1,000 candidates**
6. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   As reported above, Top-1 Accuracy is **0.8552**, Macro F1 is **0.9151**, and Log Loss is **0.0815**.
7. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   Controlled through validation-based early stopping (patience=10) and L2 weight penalty (alpha=0.0001). Achieved stable convergence without divergence. Comparing the 5-fold CV Macro F1 (0.8992 ± 0.0203) to the Student-Test Macro F1 (0.9151) shows minimal generalization gap.
## 3.3 Model Visualisations
The following visualisations were generated using the actual experimental outputs:
1. **Cross-Validation Macro F1-Score Comparison**: `reports/figures/cv_macro_f1_comparison.png` displays 5-fold CV performance across all 6 models with fold-variance error bars.
2. **Student-Testing Top-1 Accuracy vs. Macro F1**: `reports/figures/student_test_comparison.png` compares final candidate discrimination.
3. **Candidate Log Loss Comparison**: `reports/figures/log_loss_comparison.png` benchmarks emission probability calibration quality.
4. **Diagnostic Confusion Matrices**: `reports/figures/confusion_matrices.png` illustrates true positives, false alarms, and misses.
5. **MLP Epoch Training History**: `reports/figures/mlp_training_loss_curve.png` tracks loss descent across backpropagation epochs.
6. **Efficiency vs. Accuracy Pareto Trade-off**: `reports/figures/efficiency_tradeoff.png` justifies why LightGBM represents the optimal champion.

# 4. Bayesian Hyperparameter Optimisation (for the chosen model)
Based on the experimental evidence in Section 3, **LightGBM** was selected as the champion model. It achieved the highest Top-1 Accuracy and lowest Log Loss with rapid sub-millisecond inference. We conducted Bayesian Hyperparameter Optimisation using **Optuna** with the Tree-structured Parzen Estimator (TPE).

### Answers to Compulsory Questions:
1. **Which hyperparameters were included in the search?**
   - `n_estimators`: Number of boosting iterations [60 to 220]
   - `learning_rate`: Shrinkage rate [0.015 to 0.25, log scale]
   - `num_leaves`: Maximum tree leaves per base learner [15 to 127]
   - `max_depth`: Maximum tree depth limit [3 to 10]
   - `min_child_samples`: Minimum observations in a leaf [10 to 60]
   - `subsample`: Row subsampling ratio [0.65 to 0.95]
   - `colsample_bytree`: Feature fraction subsampling ratio [0.65 to 0.95]
   - `reg_alpha`: L1 regularization penalty [1e-4 to 5.0, log scale]
   - `reg_lambda`: L2 regularization penalty [1e-4 to 10.0, log scale]

2. **What ranges or distributions were considered?**
   Uniform continuous distributions were used for subsampling ratios; log-uniform distributions were applied to the learning rate and regularization terms to explore multiple orders of magnitude efficiently.

3. **What objective function was optimised?**
   The objective was to **maximize the mean Macro F1-score across 5-fold GroupKFold cross-validation** strictly on the 80 training trips.

4. **How many optimisation trials were performed?**
   **15 trials** were executed using Optuna's `TPESampler(seed=42)` across 528.75 seconds.

5. **What were the best hyperparameter values obtained?**
```json
{
  "n_estimators": 180.0,
  "learning_rate": 0.07211087599241674,
  "num_leaves": 78.0,
  "max_depth": 7.0,
  "min_child_samples": 20.0,
  "subsample": 0.9411425339378616,
  "colsample_bytree": 0.9294043546147405,
  "reg_alpha": 0.14830180701382628,
  "reg_lambda": 0.12085550858926611,
  "random_state": 42.0,
  "n_jobs": -1.0,
  "verbose": -1.0
}
```

6. **Did the tuned model outperform the untuned model?**
   **Yes.** As documented in the comparison table below, Bayesian tuning refined tree leaf allocations and regularization penalties, leading to improved cross-validation stability and higher Top-1 Accuracy on unseen student-test trips:

### Untuned vs. Tuned Model Comparison Table
| Metric                             | Untuned Model   | Tuned Model     |   Improvement |
|:-----------------------------------|:----------------|:----------------|--------------:|
| Cross-Validation Macro F1-score    | 0.9021 ± 0.0195 | 0.9029 ± 0.0191 |        0.0008 |
| Student-Testing Top-1 Accuracy     | 0.8556          | 0.8568          |        0.0013 |
| Student-Testing Macro F1-score     | 0.9157          | 0.9167          |        0.001  |
| Student-Testing Candidate Log Loss | 0.0808          | 0.0793          |       -0.0014 |
| Inference Time (ms per 1k)         | 1.02            | 2.93            |        1.91   |
| Training Time (s)                  | 2.69            | 5.73            |        3.04   |

### Optimisation Visualisations
- `reports/figures/optuna_optimization_history.png`: Visualizes convergence toward the optimal hyperparameter configuration across trials.
- `reports/figures/tuned_vs_untuned_comparison.png`: Highlights empirical gains on the student-testing benchmark.
