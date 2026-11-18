# 5. Route Reconstruction Comparison Table

| Method                              |   Pointwise Accuracy |   Route Continuity |   Sequence Precision |   Sequence Recall |   Sequence F1-Score |   Latency (ms / 1k pts) |
|:------------------------------------|---------------------:|-------------------:|---------------------:|------------------:|--------------------:|------------------------:|
| Newson-Krumm (Gaussian Baseline)    |               0.9279 |             0.973  |               0.9292 |            0.9236 |              0.9255 |                 2375.13 |
| Classifier-Only (Greedy Top-1)      |               0.5559 |             0.8307 |               0.5089 |            0.5604 |              0.5325 |                    5.82 |
| Classifier + HMM/Viterbi (Proposed) |               0.9004 |             0.9746 |               0.9349 |            0.9333 |              0.9333 |                 2398.38 |
