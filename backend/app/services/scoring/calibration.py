"""
Piecewise linear score calibration.

Maps raw composite scores (0-100) to human-aligned scores (0-100)
using anchor points derived from the evaluation dataset.

Separate calibration for technical (claim_v2) and behavioral (STAR) paths.
"""

# Anchor points: (raw_composite, human_score_scaled_to_100)
# Derived from per_quality_analysis in evaluation/report.json
# human_score is on 1-10 scale, so multiply by 10 to get 0-100
TECHNICAL_ANCHORS = [
    (0.0, 0.0),        # floor
    (12.3, 21.7),      # poor: avg_composite_claim_4sig -> avg_human * 10
    (36.7, 51.8),      # average
    (91.6, 81.2),      # good
    (100.0, 100.0),    # ceiling
]

# Behavioral path doesn't have evaluation data yet, so use identity
BEHAVIORAL_ANCHORS = [
    (0.0, 0.0),
    (100.0, 100.0),
]


def _piecewise_linear(x: float, anchors: list[tuple[float, float]]) -> float:
    """Interpolate x through piecewise linear anchors."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]

    for i in range(len(anchors) - 1):
        x0, y0 = anchors[i]
        x1, y1 = anchors[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)

    return x  # shouldn't reach here


def calibrate(raw_score: float, is_behavioral: bool = False) -> float:
    """
    Calibrate a raw composite score (0-100) to a human-aligned score (0-100).
    """
    anchors = BEHAVIORAL_ANCHORS if is_behavioral else TECHNICAL_ANCHORS
    calibrated = _piecewise_linear(raw_score, anchors)
    return round(max(0.0, min(100.0, calibrated)), 1)
