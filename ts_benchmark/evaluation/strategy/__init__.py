# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.strategy.anomaly_detect import FixedDetectScore, FixedDetectLabel, UnFixedDetectScore, \
    UnFixedDetectLabel, AllDetectScore, AllDetectLabel

STRATEGY = {
    "fixed_detect_score": FixedDetectScore,
    "fixed_detect_label": FixedDetectLabel,
    "unfixed_detect_score": UnFixedDetectScore,
    "unfixed_detect_label": UnFixedDetectLabel,
    "all_detect_score": AllDetectScore,
    "all_detect_label": AllDetectLabel,
}