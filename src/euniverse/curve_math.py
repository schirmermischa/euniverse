#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# euniverse.py - A program to display MER colour images created with eummy

# MIT License

# Copyright (c) [2026] [Mischa Schirmer]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
curve_math.py — monotone cubic Hermite spline utilities
=========================================================
Shared by curves_dialog.py (CurveWidget, for drawing/editing a curve) and
image_viewer.py (for composing every applied curve session against the
raw pixel data at full precision — see ImageViewer.current_display_lut).
Kept as a standalone module, rather than living in either of those, so
the image pipeline doesn't need to import a UI dialog module and vice
versa.

Uses the Fritsch-Carlson method: an initial tangent estimate (zero at any
local extremum, otherwise the average of the neighbouring secant slopes)
followed by a limiter that rescales tangents so no segment can overshoot
the y-range of its two endpoints. This is what keeps a hand-drawn S-curve
smooth without ringing, unlike a plain Catmull-Rom or natural cubic
spline.
"""

import numpy as np


def hermite_tangents(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Per-point tangent slopes for a monotone cubic Hermite spline through (xs, ys)."""
    n = len(xs)
    m = np.zeros(n)
    if n < 2:
        return m

    delta = np.diff(ys) / np.diff(xs)
    if n == 2:
        m[:] = delta[0]
        return m

    m[0]  = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] == 0 or delta[i] == 0 or (delta[i - 1] > 0) != (delta[i] > 0):
            m[i] = 0.0
        else:
            m[i] = (delta[i - 1] + delta[i]) / 2.0

    for i in range(n - 1):
        if delta[i] == 0:
            m[i]     = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i]     / delta[i]
            b = m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9.0:
                tau      = 3.0 / np.sqrt(s)
                m[i]     = tau * a * delta[i]
                m[i + 1] = tau * b * delta[i]
    return m


def hermite_eval(xs: np.ndarray, ys: np.ndarray, m: np.ndarray,
                  x_query: np.ndarray) -> np.ndarray:
    """
    Evaluate the Hermite spline defined by (xs, ys, m) at x_query. Callers
    are expected to have already clipped x_query into [xs[0], xs[-1]] —
    values outside that range fall back to the nearest segment's basis
    functions with t outside [0,1], which is not the flat-extension
    behaviour a curve tool needs beyond its endpoints.
    """
    idx = np.searchsorted(xs, x_query, side='right') - 1
    idx = np.clip(idx, 0, len(xs) - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    m0, m1 = m[idx], m[idx + 1]

    h = x1 - x0
    t = (x_query - x0) / h
    t2 = t * t
    t3 = t2 * t

    h00 =  2 * t3 - 3 * t2 + 1
    h10 =      t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 =      t3 -     t2

    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def eval_curve(xs: np.ndarray, ys: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper: tangents + eval + clip-to-domain in one call.
    x_query may extend beyond [xs[0], xs[-1]] — those points are clipped
    to the endpoints first, giving the usual "flat beyond the curve's
    black/white point" extension.
    """
    m = hermite_tangents(xs, ys)
    x_clipped = np.clip(x_query, xs[0], xs[-1])
    return hermite_eval(xs, ys, m, x_clipped)
