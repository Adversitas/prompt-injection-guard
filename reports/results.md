| Detector | Test P | Test R | Test F1 | Test FPR | Test AUC | CV F1 (train, 5-fold) | p50 latency |
|---|---|---|---|---|---|---|---|
| rules only | 1.000 | 0.283 | 0.442 | 0.000 | 0.642 | 0.571 ± 0.073 | 0.04 ms |
| tfidf word + logreg | 0.978 | 0.733 | 0.838 | 0.018 | 0.958 | 0.846 ± 0.019 | 1.18 ms |
| tfidf char + logreg | 0.981 | 0.867 | 0.920 | 0.018 | 0.979 | 0.898 ± 0.011 | 1.60 ms |
| tfidf word+char + logreg | 1.000 | 0.867 | 0.929 | 0.000 | 0.978 | 0.904 ± 0.020 | 4.05 ms |
| hybrid (word+char stacked with rules) | 1.000 | 0.867 | 0.929 | 0.000 | 0.978 | 0.917 ± 0.014 | 3.47 ms |
