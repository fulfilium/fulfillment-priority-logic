"""The numerics of FPL as plain functions on arrays: the stable power mean, its dual, the offset and calibration.

Each function takes one numpy array, torch tensor or jax array, or anything numpy can turn into an array, and
returns an array of the same library, so gradients flow in torch and jax. The file imports only numpy and the
standard library, and not the standard `math` module, so you can copy it into another project under any name.

>>> from fulfillment_priority_logic.math import p_mean
>>> round(float(p_mean([0.25, 1.0], p=0)), 6)  # the geometric mean
0.5
"""

from __future__ import annotations

import contextlib
import itertools
import operator
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any, Literal, assert_never

import numpy as np

INF = float("inf")

DEFAULT_SLACK = 1e-7
"""The slack ε that `p_mean` adds before the mean and removes after."""

type Library = Literal["numpy", "torch", "jax"]
"""The array libraries that these functions work on."""

type Axis = int | Sequence[int] | None
"""The axis or axes to reduce, counted from the end when negative. None reduces every axis."""


def p_mean(x: Any, p: float, slack: float = DEFAULT_SLACK, axis: Axis = None) -> Any:
    """The power mean `((x₁ᵖ + … + xₙᵖ) / n)^(1/p)` of one array `x` along `axis`: the "and" of FPL.

    p = -inf gives the minimum, 0 the geometric mean, 1 the average and inf the maximum. The values of `x` must be
    >= 0. This function does not check that, because a check would read the values back from the device; the
    language's `evaluate` checks its inputs instead. Four things make the mean safe inside an optimizer.

    1. The slack ε. For p < 0 a single zero sends `xᵢᵖ` to infinity and the gradient with it. So every input is
       shifted up by ε, and the mean is shifted back down: `M_p(x + ε) − ε`. The thesis proves that this order
       keeps the result within [min x, max x]. A zero then drags the mean close to zero, as it should, but the
       gradient stays finite and still points somewhere useful.
    2. The stabilizer. Raising values to a large |p| overflows or underflows. The mean is homogeneous
       (`M_p(c·x) = c·M_p(x)`), so the inputs are divided by the smallest input when p < 0, or the largest when
       p > 0, and the result is multiplied back. Then every term `(xᵢ/s)ᵖ` lies in (0, 1] and one of them is
       exactly 1. Each reduced slice gets its own divisor. One divisor for a whole batch would let a tiny sample
       push the terms of every other sample to underflow.
    3. Small |p|. The mean is computed as `s·exp(log1p(mean(expm1(p·log(x/s)))) / p)`. In exact arithmetic this is
       the formula above. With `expm1` and `log1p` it stays accurate as p approaches 0, where the naive form
       divides a rounding error by p. At p = 0 itself it is the limit, the geometric mean `exp(mean(log x))`, so
       there is no threshold where the method switches.
    4. The final clip. Rounding can leave the result a hair outside [min x, max x]. The result is clipped back, but
       the gradient passes as if there were no clip, because the clip only removes rounding error.
    """
    if p != p:
        raise ValueError("p must be a number or ±inf, got nan")
    if not 0.0 <= slack < INF:
        raise ValueError(f"the slack ε must be a finite number >= 0, got {slack}")
    lib = library(x)
    x = _asarray(lib, x)
    if x.ndim == 0:
        x = x[None]
    shape = tuple(x.shape)
    axes = _axes(axis, len(shape))
    if any(shape[a] == 0 for a in axes):
        raise ValueError(f"p_mean over an empty axis has no value (input shape {shape}, axis {axis})")
    xp = _namespace(lib)
    # With ε = 0, an input of exactly 0 takes log(0) on purpose. The result is still right, so numpy stays quiet.
    with np.errstate(all="ignore") if lib == "numpy" else contextlib.nullcontext():
        lo, hi = _reduce(lib, "amin", x, axes), _reduce(lib, "amax", x, axes)
        if p == -INF:
            y = lo
        elif p == INF:
            y = hi
        else:
            shifted = x + slack
            if p == 0.0:
                m = xp.exp(_reduce(lib, "mean", xp.log(shifted), axes))
            else:
                # The divisor gets no gradient. Holding it fixed is exact, because M_p(x) = s·M_p(x/s) for any s.
                s = _stop_gradient(lib, lo + slack if p < 0 else hi + slack)
                # s is 0 only when ε = 0 and the min (p < 0) or every input (p > 0) is 0. The mean is 0 then.
                positive = s > 0
                safe = xp.where(positive, s, 1.0)
                t = _reduce(lib, "mean", xp.expm1(p * xp.log(shifted / safe)), axes)
                m = xp.where(positive, safe * xp.exp(xp.log1p(t) / p), 0.0)
            y = m - slack
            # The forward value is exactly the clipped one, because `y - sg(y)` is exactly 0. The gradient is y's.
            sg = _stop_gradient(lib, y)
            y = xp.clip(sg, _stop_gradient(lib, lo), _stop_gradient(lib, hi)) + (y - sg)
    return xp.squeeze(y, dim=axes) if lib == "torch" else xp.squeeze(y, axis=axes)


def dual_p_mean(x: Any, p: float, slack: float = DEFAULT_SLACK, axis: Axis = None) -> Any:
    """The "or" of FPL, `1 − p_mean(1 − x)` by De Morgan's law. It is high when any input is high.

    >>> round(float(dual_p_mean([0.95, 0.1], p=0)), 3)
    0.788
    """
    return 1.0 - p_mean(1.0 - _asarray(library(x), x), p, slack, axis)


def offset(x: Any, delta: float, *, pass_gradient: bool = False) -> Any:
    """The priority offset `x·(1 − |δ|) + max(δ, 0)`, for δ in [−1, 1].

    With δ > 0 it squeezes `x` into [δ, 1], and with δ < 0 into [0, 1 − |δ|]. Its gradient is 1 − |δ|. With
    `pass_gradient=True` the value is the same but the gradient is 1, so the offset moves the value without
    weakening its pull. That is a training device, not part of what the offset means.
    """
    if not -1.0 <= delta <= 1.0:
        raise ValueError(f"the offset δ must be in [-1, 1], got {delta}")
    lib = library(x)
    x = _asarray(lib, x)
    y = x * (1.0 - abs(delta)) + max(delta, 0.0)
    if not pass_gradient:
        return y
    # The forward value is exactly y, because `x - stop_gradient(x)` is exactly 0. The gradient with respect to x
    # is 1.
    return _stop_gradient(lib, y) + (x - _stop_gradient(lib, x))


def piecewise(x: Any, points: Sequence[tuple[float, float]]) -> Any:
    """Straight lines between landmarks `(x, y)`, flat beyond the first and the last one, and NaN for NaN.

    The landmarks must be finite and their x strictly increasing. Exactly at a landmark the slope is the slope of
    the segment to its right, and 0 at the last landmark.

    >>> round(float(piecewise(5.0, [(2, 1), (6, 0.1), (10, 0)])), 6)
    0.325
    """
    check_landmarks(points)
    lib = library(x)
    x = _asarray(lib, x)
    xp = _namespace(lib)
    # Each segment covers [x0, x1), so a landmark belongs to the segment on its right. Selecting with `where`
    # rather than summing clipped ramps gives every library the same one-sided slope there, and an infinite
    # input lands on a flat end instead of producing inf - inf.
    y = xp.zeros_like(x) + points[0][1]
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        y = xp.where((x >= x0) & (x < x1), y0 + (y1 - y0) / (x1 - x0) * (x - x0), y)
    y = xp.where(x >= points[-1][0], xp.zeros_like(x) + points[-1][1], y)
    return xp.where(x != x, x, y)  # every comparison with NaN is false, so NaN would read as the first landmark


def check_landmarks(points: Sequence[tuple[float, float]]) -> None:
    """Refuse landmarks that `piecewise` cannot use: none at all, a value that is not finite, or x not increasing."""
    if not points:
        raise ValueError("piecewise needs at least one landmark")
    if not all(abs(v) < INF for point in points for v in point):
        raise ValueError(f"landmarks must be finite, got {list(points)}")
    if any(x1 <= x0 for (x0, _), (x1, _) in itertools.pairwise(points)):
        raise ValueError(f"the landmarks' x must be strictly increasing, got {[a for a, _ in points]}")


def min_fulfillment(score: float, n: int, p: float) -> float:
    """The lowest any one of `n` inputs can be when their `p_mean` is `score` (Theorem IV.1 of the paper).

    It is `(n·scoreᵖ − n + 1)^(1/p)`, and 0 when that is undefined. For example, a harmonic mean (p = -1) of
    three requirements that scores 0.9 means that none of them is below 0.75. An average (p = 1) of 0.9 still
    allows one of them to be 0.7.

    >>> round(min_fulfillment(0.9, n=3, p=-1), 3)
    0.75
    """
    if not 0.0 <= score <= 1.0 or n < 1:
        raise ValueError(f"need a score in [0, 1] and n >= 1, got score={score}, n={n}")
    if n == 1 or p == -INF:
        return score
    if p == INF:
        return 0.0
    if p == 0.0:
        return score**n
    if score == 0.0:
        return 0.0
    inner = n * score**p - n + 1
    return inner ** (1 / p) if inner > 0 else 0.0


def stop_gradient(x: Any) -> Any:
    """`x` with no gradient flowing back through it: `detach` in torch, `lax.stop_gradient` in jax, `x` in numpy."""
    return _stop_gradient(library(x), x)


def library(x: object) -> Library:
    """The library that `x` comes from: torch or jax for their arrays, numpy for anything else.

    It looks a library up only if the program has already imported it, so it never imports one itself.
    """
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return "torch"
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(x, jax.Array):
        return "jax"
    return "numpy"


def _namespace(lib: Library) -> ModuleType:
    match lib:
        case "numpy":
            return np
        case "torch":
            return sys.modules["torch"]
        case "jax":
            return sys.modules["jax"].numpy
        case _:
            assert_never(lib)


def _asarray(lib: Library, x: Any) -> Any:
    xp = _namespace(lib)
    match lib:
        case "torch":
            t = x if isinstance(x, xp.Tensor) else xp.as_tensor(x)
            return t if t.is_floating_point() else t.to(xp.get_default_dtype())
        case "numpy" | "jax":
            a = xp.asarray(x)
            return a if xp.issubdtype(a.dtype, xp.floating) else a.astype(float)
        case _:
            assert_never(lib)


def _stop_gradient(lib: Library, x: Any) -> Any:
    match lib:
        case "numpy":
            return x
        case "torch":
            return x.detach()
        case "jax":
            return sys.modules["jax"].lax.stop_gradient(x)
        case _:
            assert_never(lib)


def _reduce(lib: Library, name: Literal["amin", "amax", "mean"], x: Any, axes: tuple[int, ...]) -> Any:
    """A reduction over `axes` that keeps them as size-1 dimensions. Torch names the arguments differently."""
    f = getattr(_namespace(lib), name)
    return f(x, dim=axes, keepdim=True) if lib == "torch" else f(x, axis=axes, keepdims=True)


def _axes(axis: Axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    out = set()
    for a in axis if isinstance(axis, Sequence) else [axis]:
        a = operator.index(a)  # an int or a numpy integer, never a float
        if not -ndim <= a < ndim:
            raise ValueError(f"axis {a} is out of range for an array with {ndim} dimensions")
        out.add(a % ndim)
    return tuple(sorted(out))
