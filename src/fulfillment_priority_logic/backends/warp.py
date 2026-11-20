"""The warp backend: the operators as Warp kernels on Warp arrays. It computes values only, no gradients.

It runs every node that computes only with the methods of `Ops`, which covers the package's nodes and every
derived node. A node that does its own array arithmetic is promised on numpy, torch and jax only: on Warp, `ops.xp`
raises, and arithmetic on Warp arrays raises Warp's own error.

The kernels run on the device of the input arrays, and plain numbers and numpy inputs go to that device in the
float type of those arrays. Limit: a batch can be broadcast only against single values, not (3, 1) against
(1, 4), and a single value comes back with shape (1,), because Warp has no 0-d arrays.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .. import warp_funcs as wf
from ..math import DEFAULT_SLACK

_FLOAT_DTYPES = (wp.float32, wp.float64)


def inputs(values: Sequence[Any]) -> list[Any]:
    arrays = [v for v in values if isinstance(v, wp.array)]
    floating = [a for a in arrays if a.dtype in _FLOAT_DTYPES]
    dtype = (wp.float64 if any(a.dtype == wp.float64 for a in floating) else wp.float32) if floating else None
    device = arrays[0].device if arrays else "cpu"
    out = []
    for v in values:
        if isinstance(v, wp.array):
            c = v if v.is_contiguous else v.contiguous()
            out.append(c if c.dtype in _FLOAT_DTYPES else _cast_array(c, dtype or wp.float32))
            continue
        a = np.atleast_1d(np.asarray(v))
        t = dtype or (wp.float64 if a.dtype == np.float64 else wp.float32)
        out.append(wp.array(np.ascontiguousarray(a, dtype=_numpy_dtype(t)), dtype=t, device=device))
    return out


def to_numpy(x: Any) -> np.ndarray:
    return x.numpy()


def value_range(x: Any) -> tuple[float, float] | None:
    a = x.numpy()
    return (float(a.min()), float(a.max())) if a.size else None


@dataclass(frozen=True)
class WarpOps:
    """The `Ops` of Warp: each method launches a kernel. Warp has no array namespace, so `xp` raises."""

    inputs: Mapping[str, Any]
    """The inputs of this evaluation, as Warp arrays."""

    @property
    def xp(self) -> Any:
        raise TypeError(
            "Warp has no array namespace, so a node that computes with ops.xp has no Warp kernel. "
            "Use numpy, torch or jax, or write the node as a derived node with expand()."
        )

    def input(self, name: str) -> Any:
        return self.inputs[name]

    def p_mean(self, values: Sequence[Any], p: float, slack: float = DEFAULT_SLACK) -> Any:
        rows, batch = _columns(values)
        return _reduce_rows(rows, p, slack, 0).reshape(batch)

    def dual_p_mean(self, values: Sequence[Any], p: float, slack: float = DEFAULT_SLACK) -> Any:
        rows, batch = _columns(values)
        return self.negate(_reduce_rows(rows, p, slack, 1).reshape(batch))

    def offset(self, x: Any, delta: float, *, pass_gradient: bool = False) -> Any:
        return _map(_offset, x, x.dtype(delta))  # values only, so pass_gradient changes nothing here

    def piecewise(self, x: Any, points: Sequence[tuple[float, float]]) -> Any:
        dtype = _numpy_dtype(x.dtype)
        xs, ys = (wp.array(np.array(v, dtype=dtype), device=x.device) for v in zip(*points, strict=True))
        return _map(_piecewise, x, xs, ys)

    def power(self, x: Any, w: float, *, pass_gradient: bool = False) -> Any:
        return _map(_power, x, x.dtype(w))  # values only, so pass_gradient changes nothing here

    def negate(self, x: Any) -> Any:
        return _map(_negate, x)

    def stop_gradient(self, x: Any) -> Any:
        return x


def ops(given: Mapping[str, Any]) -> WarpOps:
    return WarpOps(given)


@wp.kernel
def _p_mean_rows(x: wp.array2d(dtype=Any), p: Any, slack: Any, flip: int, out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = wf._p_mean(x[i], p, slack, flip)


@wp.kernel
def _gather_column(src: wp.array(dtype=Any), col: int, out: wp.array2d(dtype=Any)):
    i = wp.tid()
    k = i
    if src.shape[0] == 1:
        k = 0
    out[i, col] = out.dtype(src[k])


@wp.kernel
def _negate(x: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = x.dtype(1.0) - x[i]


@wp.kernel
def _offset(x: wp.array(dtype=Any), delta: Any, out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = wf.offset(x[i], delta)


@wp.kernel
def _power(x: wp.array(dtype=Any), w: Any, out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = wp.pow(x[i], w)


@wp.kernel
def _piecewise(x: wp.array(dtype=Any), xs: wp.array(dtype=Any), ys: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = wf.piecewise(x[i], xs, ys)


@wp.kernel
def _cast(src: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = out.dtype(src[i])


def _cast_array(x: Any, dtype: Any) -> Any:
    flat = x.flatten()
    out = wp.empty((flat.shape[0],), dtype=dtype, device=x.device)
    if flat.shape[0]:
        wp.launch(_cast, dim=flat.shape[0], inputs=[flat], outputs=[out], device=x.device)
    return out.reshape(x.shape)


def _numpy_dtype(dtype: Any) -> Any:
    return np.float64 if dtype == wp.float64 else np.float32


def _columns(xs: Sequence[Any]) -> tuple[Any, tuple[int, ...]]:
    """The operands as the columns of a (batch, operands) array, and the batch shape."""
    shapes = [tuple(x.shape) for x in xs]
    non_unit = [s for s in shapes if math.prod(s) != 1]
    batch = non_unit[0] if non_unit else shapes[0]
    if any(s != batch for s in non_unit):
        raise ValueError(f"the warp backend broadcasts only single values against a batch, got shapes {shapes}")
    dtype = wp.float64 if any(x.dtype == wp.float64 for x in xs) else wp.float32
    out = wp.empty((math.prod(batch), len(xs)), dtype=dtype, device=xs[0].device)
    if out.shape[0]:
        for col, x in enumerate(xs):
            wp.launch(_gather_column, dim=out.shape[0], inputs=[x.flatten(), col], outputs=[out], device=out.device)
    return out, batch


def _reduce_rows(rows: Any, p: float, slack: float, flip: int) -> Any:
    out = wp.empty((rows.shape[0],), dtype=rows.dtype, device=rows.device)
    if rows.shape[0]:
        args = [rows, rows.dtype(p), rows.dtype(slack), flip]
        wp.launch(_p_mean_rows, dim=rows.shape[0], inputs=args, outputs=[out], device=rows.device)
    return out


def _map(kernel: Any, x: Any, *args: Any) -> Any:
    flat = x.flatten()
    out = wp.empty((flat.shape[0],), dtype=x.dtype, device=x.device)
    if flat.shape[0]:
        wp.launch(kernel, dim=flat.shape[0], inputs=[flat, *args], outputs=[out], device=x.device)
    return out.reshape(x.shape)
