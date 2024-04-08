# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.metrics import classification_metrics_label

CLASSIFICATION_METRICS_SCORE = {
    k: getattr(classification_metrics_score, k)
    for k in classification_metrics_score.__all__
}
CLASSIFICATION_METRICS_LABEL = {
    k: getattr(classification_metrics_label, k)
    for k in classification_metrics_label.__all__
}
METRICS = {
    **CLASSIFICATION_METRICS_SCORE,
    **CLASSIFICATION_METRICS_LABEL,
}