"""Tests for the exact geometry kernel."""
import random

from fractions import Fraction

from app.geometry import segment_distance_sq


def float_seg_dist(p1, p2, q1, q2):
    """Floating-point reference implementation of closed-segment distance."""
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    a = r[0] ** 2 + r[1] ** 2
    c = s[0] ** 2 + s[1] ** 2
    w = (p1[0] - q1[0], p1[1] - q1[1])
    best = float("inf")

    def f(u, v):
        dx = w[0] + u * r[0] - v * s[0]
        dy = w[1] + u * r[1] - v * s[1]
        return dx * dx + dy * dy

    if a == 0 and c == 0:
        return f(0, 0)
    if a == 0:
        t = min(1, max(0, (w[0] * s[0] + w[1] * s[1]) / c))
        return f(0, t)
    if c == 0:
        t = min(1, max(0, -(w[0] * r[0] + w[1] * r[1]) / a))
        return f(t, 0)

    b = r[0] * s[0] + r[1] * s[1]
    d = w[0] * r[0] + w[1] * r[1]
    e = w[0] * s[0] + w[1] * s[1]
    det = a * c - b * b
    if det != 0:
        u = (b * e - c * d) / det
        v = (a * e - b * d) / det
        if 0 <= u <= 1 and 0 <= v <= 1:
            best = min(best, f(u, v))
    # edges
    best = min(best,
               f(0, min(1, max(0, e / c))),
               f(1, min(1, max(0, (b + e) / c))),
               f(min(1, max(0, -d / a)), 0),
               f(min(1, max(0, (b - d) / a)), 1))
    return best


def test_known_cases():
    # parallel horizontal segments, gap 2
    d2, _, _ = segment_distance_sq((0, 0), (3, 0), (0, 2), (3, 2))
    assert d2 == 4
    # crossing X segments
    d0, u, v = segment_distance_sq((0, 0), (2, 2), (0, 2), (2, 0))
    assert d0 == 0 and u == Fraction(1, 2) and v == Fraction(1, 2)
    # shared endpoint
    assert segment_distance_sq((0, 0), (10, 0), (0, 0), (0, 10))[0] == 0
    # perpendicular gap 2
    assert segment_distance_sq((0, 5), (10, 5), (5, 0), (5, 3))[0] == 4
    # degenerate: point vs segment, foot on segment
    assert segment_distance_sq((5, 7), (5, 7), (0, 0), (10, 0))[0] == 49
    # point vs segment, closest is an endpoint
    d2, u, _ = segment_distance_sq((12, 3), (12, 3), (0, 0), (10, 0))
    assert d2 == 13 and u == 0
    # both points
    assert segment_distance_sq((1, 1), (1, 1), (4, 5), (4, 5))[0] == 25


def test_clearance_boundary_is_exact():
    # Two segments exactly 1 apart (perpendicular foot) compare exactly:
    # distance == clearance is allowed (closed segments, >= required).
    d2, _, _ = segment_distance_sq((0, 0), (4, 0), (2, 1), (2, 3))
    assert d2 == 1
    # Parallel lines with direction (3,4) (length 5); the second passes
    # through (4,-3), exactly 5 units away along normal (4,-3)/5.  Exact
    # rational computation must return 25, no float rounding.
    d2, _, _ = segment_distance_sq((0, 0), (3, 4), (4, -3), (7, 1))
    assert d2 == 25


def test_random_against_float_reference():
    rng = random.Random(7)
    for _ in range(2000):
        pts = [(rng.randint(-8, 8), rng.randint(-8, 8)) for _ in range(4)]
        exact, u, v = segment_distance_sq(*pts)
        ref = float_seg_dist(*pts)
        assert abs(float(exact) - ref) < 1e-9, pts
        assert 0 <= u <= 1 and 0 <= v <= 1


def test_witness_points_lie_on_segments():
    rng = random.Random(11)
    for _ in range(500):
        pts = [(rng.randint(-6, 6), rng.randint(-6, 6)) for _ in range(4)]
        (p1, p2, q1, q2) = pts
        d2, u, v = segment_distance_sq(*pts)
        px = Fraction(p1[0]) + u * (p2[0] - p1[0])
        py = Fraction(p1[1]) + u * (p2[1] - p1[1])
        qx = Fraction(q1[0]) + v * (q2[0] - q1[0])
        qy = Fraction(q1[1]) + v * (q2[1] - q1[1])
        got = (px - qx) ** 2 + (py - qy) ** 2
        assert got == d2
