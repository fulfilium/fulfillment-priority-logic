"""The FPL operators as Warp functions (`@wp.func`), to call inside your own Warp kernels.

They compute what `fulfillment_priority_logic.math` computes, with the same slack, stabilizer, exact limits and
final clip. `p_mean` and `dual_p_mean` take anything with a length and an index: a Warp vector such as `wp.vec4`,
or a 1-D array such as a row `x[i]` of a 2-D array. Scalars passed alongside (p, slack, δ) must have the same type
as the elements, for example `wp.float32`. They compute values only; Warp's automatic differentiation is not
tested with them.

    @wp.kernel
    def score(x: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        i = wp.tid()
        out[i] = p_mean(x[i], -1.0, 1.0e-7)
"""

from __future__ import annotations

from typing import Any

import warp as wp


@wp.func
def _expm1(x: Any):
    # exp(x) - 1 without the cancellation near x = 0 (Kahan's trick); Warp has no expm1
    one = type(x)(1.0)
    u = wp.exp(x)
    r = u - one
    if u == one:
        r = x
    elif r != -one:
        r = r * x / wp.log(u)
    return r


@wp.func
def _log1p(x: Any):
    # log(1 + x) without the cancellation near x = 0 (Goldberg's trick); Warp has no log1p
    one = type(x)(1.0)
    u = one + x
    r = x
    if u != one:
        r = wp.log(u) * x / (u - one)
    return r


@wp.func
def _element(v: Any, i: int, flip: int):
    e = v[i]
    if flip != 0:
        e = type(e)(1.0) - e
    return e


@wp.func
def _p_mean(v: Any, p: Any, slack: Any, flip: int):
    # The steps of math.p_mean, on the elements of v, or on 1 - v when flip is 1 (for the dual).
    n = len(v)
    lo = _element(v, 0, flip)
    hi = lo
    for i in range(1, n):
        e = _element(v, i, flip)
        lo = wp.min(lo, e)
        hi = wp.max(hi, e)
    zero = lo - lo
    nf = type(lo)(float(n))
    y = lo
    if wp.isinf(p):
        if p > zero:
            y = hi
    elif p == zero:
        acc = zero
        for i in range(n):
            acc = acc + wp.log(_element(v, i, flip) + slack)
        y = wp.clamp(wp.exp(acc / nf) - slack, lo, hi)
    else:
        s = lo + slack
        if p > zero:
            s = hi + slack
        m = zero
        if s > zero:
            acc = zero
            for i in range(n):
                acc = acc + _expm1(p * wp.log((_element(v, i, flip) + slack) / s))
            m = s * wp.exp(_log1p(acc / nf) / p)
        y = wp.clamp(m - slack, lo, hi)
    return y


@wp.func
def p_mean(v: Any, p: Any, slack: Any):
    """The power mean of the elements of `v` (the "and"). p may be ±inf."""
    return _p_mean(v, p, slack, 0)


@wp.func
def dual_p_mean(v: Any, p: Any, slack: Any):
    """`1 − p_mean(1 − v)` (the "or")."""
    return type(p)(1.0) - _p_mean(v, p, slack, 1)


@wp.func
def offset(x: Any, delta: Any):
    """The priority offset `x·(1 − |δ|) + max(δ, 0)`."""
    return x * (type(x)(1.0) - wp.abs(delta)) + wp.max(delta, x - x)


@wp.func
def piecewise(x: Any, xs: Any, ys: Any):
    """Straight lines between landmarks `(xs[k], ys[k])`, with xs strictly increasing; flat outside them, NaN for
    NaN."""
    n = len(xs)
    y = ys[0]
    for k in range(n - 1):
        if x >= xs[k] and x < xs[k + 1]:
            y = ys[k] + (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]) * (x - xs[k])
    if x >= xs[n - 1]:
        y = ys[n - 1]
    if x != x:  # every comparison with NaN is false, so NaN would read as the first landmark
        y = x
    return y
