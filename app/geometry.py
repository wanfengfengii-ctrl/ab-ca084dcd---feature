"""Exact 2-D closed-segment geometry used by the fibre-arm adjudicator.

All inputs are integers.  Every distance computation is carried out with
:class:`fractions.Fraction`, so clearance comparisons such as ``d >= s`` are
exact (no floating-point epsilon).  The public entry point is
:func:`segment_distance_sq`, which returns the *squared* minimum distance
between two closed segments together with the closest-point parameters.
"""

from fractions import Fraction
from typing import Tuple

# A point is a pair of ints; a squared distance is an exact Fraction.
Point = Tuple[int, int]
ClosestPoint = Tuple[Fraction, Fraction]


def _point_point(a: Point, b: Point) -> Fraction:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return Fraction(dx * dx + dy * dy)


def _point_segment(a: Point, c: Point, d: Point) -> Tuple[Fraction, Fraction]:
    """Squared distance from point ``a`` to closed segment ``c -> d``.

    Returns ``(distance_sq, t)`` where ``t`` is the clamped parameter of the
    closest point on the segment.
    """
    sx = d[0] - c[0]
    sy = d[1] - c[1]
    denom = sx * sx + sy * sy
    if denom == 0:  # degenerate segment: point-to-point
        return _point_point(a, c), Fraction(0)
    t = Fraction((a[0] - c[0]) * sx + (a[1] - c[1]) * sy, denom)
    if t < 0:
        t = Fraction(0)
    elif t > 1:
        t = Fraction(1)
    dx = a[0] - c[0] - t * sx
    dy = a[1] - c[1] - t * sy
    return dx * dx + dy * dy, t


def _clamp01(t: Fraction) -> Fraction:
    if t < 0:
        return Fraction(0)
    if t > 1:
        return Fraction(1)
    return t


def segment_distance_sq(
    p1: Point, p2: Point, q1: Point, q2: Point
) -> Tuple[Fraction, Fraction, Fraction]:
    """Exact squared distance between the closed segments p1p2 and q1q2.

    Uses the standard quadratic formulation

        f(u, v) = |(p1 + u*r) - (q1 + v*s)|**2,  u, v in [0, 1],

    and evaluates the interior stationary point (when it exists) together
    with the constrained minima on all four segment edges.  Degenerate
    (zero-length) segments fall back to point/segment distance.

    Returns ``(distance_sq, u, v)`` with the closest-point parameters.
    """
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    sx, sy = q2[0] - q1[0], q2[1] - q1[1]
    a = rx * rx + ry * ry
    c = sx * sx + sy * sy

    if a == 0 and c == 0:
        return _point_point(p1, q1), Fraction(0), Fraction(0)
    if a == 0:
        d2, v = _point_segment(p1, q1, q2)
        return d2, Fraction(0), v
    if c == 0:
        d2, u = _point_segment(q1, p1, p2)
        return d2, u, Fraction(0)

    wx, wy = p1[0] - q1[0], p1[1] - q1[1]
    b = rx * sx + ry * sy
    d = wx * rx + wy * ry
    e = wx * sx + wy * sy
    det = a * c - b * b

    def fval(u: Fraction, v: Fraction) -> Fraction:
        dx = wx + u * rx - v * sx
        dy = wy + u * ry - v * sy
        return dx * dx + dy * dy

    candidates = set()

    # Constrained minima on each of the four edges of the [0,1]^2 parameter
    # square, plus the re-optimised value along the other parameter so the
    # adjacent edge and every corner is covered.
    v0 = _clamp01(Fraction(e, c))
    candidates.add((Fraction(0), v0))
    candidates.add((_clamp01(Fraction(b * v0 - d, a)), v0))

    v1 = _clamp01(Fraction(b + e, c))
    candidates.add((Fraction(1), v1))
    candidates.add((_clamp01(Fraction(b * v1 - d, a)), v1))

    u0 = _clamp01(Fraction(-d, a))
    candidates.add((u0, Fraction(0)))
    candidates.add((u0, _clamp01(Fraction(b * u0 + e, c))))

    u1 = _clamp01(Fraction(b - d, a))
    candidates.add((u1, Fraction(1)))
    candidates.add((u1, _clamp01(Fraction(b * u1 + e, c))))

    # Unconstrained interior stationary point.
    if det != 0:
        ui = Fraction(b * e - c * d, det)
        vi = Fraction(a * e - b * d, det)
        if 0 <= ui <= 1 and 0 <= vi <= 1:
            candidates.add((ui, vi))

    best_val = None
    best_u = best_v = Fraction(0)
    for u, v in candidates:
        val = fval(u, v)
        if best_val is None or val < best_val:
            best_val, best_u, best_v = val, u, v
    return best_val, best_u, best_v


def fraction_point(point: Point, u: Fraction, direction: Tuple[int, int]) -> ClosestPoint:
    """Witness point ``point + u * direction`` as exact Fractions."""
    return (
        Fraction(point[0]) + u * direction[0],
        Fraction(point[1]) + u * direction[1],
    )
