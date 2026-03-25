"""Tests for piecewise linear score calibration."""

from app.services.scoring.calibration import calibrate


def test_floor():
    assert calibrate(0) == 0.0


def test_ceiling():
    assert calibrate(100) == 100.0


def test_poor_anchor():
    result = calibrate(12.3)
    assert abs(result - 21.7) < 0.2, f"Expected ~21.7, got {result}"


def test_average_anchor():
    result = calibrate(36.7)
    assert abs(result - 51.8) < 0.2, f"Expected ~51.8, got {result}"


def test_good_anchor():
    result = calibrate(91.6)
    assert abs(result - 81.2) < 0.2, f"Expected ~81.2, got {result}"


def test_monotonicity():
    s20 = calibrate(20)
    s50 = calibrate(50)
    s80 = calibrate(80)
    assert s20 < s50 < s80, f"Not monotonic: {s20}, {s50}, {s80}"


def test_behavioral_identity():
    for raw in [0, 25, 50, 75, 100]:
        result = calibrate(raw, is_behavioral=True)
        assert result == float(raw), f"Behavioral should be identity: {raw} -> {result}"


def test_below_floor():
    assert calibrate(-5) == 0.0


def test_above_ceiling():
    assert calibrate(110) == 100.0


def test_midpoint_interpolation():
    """A value between two anchors should interpolate linearly."""
    # Midpoint between poor (12.3, 21.7) and average (36.7, 51.8)
    mid_x = (12.3 + 36.7) / 2  # 24.5
    mid_y = (21.7 + 51.8) / 2  # 36.75
    result = calibrate(mid_x)
    assert abs(result - mid_y) < 0.5, f"Expected ~{mid_y}, got {result}"
