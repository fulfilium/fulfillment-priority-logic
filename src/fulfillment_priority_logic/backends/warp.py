"""The warp backend: the operators as Warp kernels on Warp arrays. It computes values only, no gradients.

The kernels run on the device of the input arrays, and plain numbers and numpy inputs go to that device in the
float type of those arrays. Limit: a batch can be broadcast only against single values, not (3, 1) against
(1, 4), and a single value comes back with shape (1,), because Warp has no 0-d arrays.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, assert_never

import numpy as np
import warp as wp

from .. import warp_funcs as wf
from ..nodes import And, Formula, Not, Offset, Or, Piecewise, Power, Var


def inputs(values: Sequence[Any]) -> list[Any]:
    arrays = [v for v in values if isinstance(v, wp.array)]
    dtype = (wp.float64 if any(a.dtype == wp.float64 for a in arrays) else wp.float32) if arrays else None
    device = arrays[0].device if arrays else "cpu"
    out = []
    for v in values:
        if isinstance(v, wp.array):
            out.append(v if v.is_contiguous else v.contiguous())
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


def apply(node: Formula, args: Sequence[Any], inputs: Mapping[str, Any]) -> Any:
    """The value of one node from the values of its children."""
    match node:
        case Var(name=name):
            return inputs[name]
        case And(p=p, slack=slack):
            rows, batch = _columns(args)
            return _reduce_rows(rows, p, slack, 0).reshape(batch)
        case Or(p=p, slack=slack):
            rows, batch = _columns(args)
            return _map(_negate, _reduce_rows(rows, p, slack, 1).reshape(batch))
        case Not():
            return _map(_negate, args[0])
        case Offset(delta=delta):
            return _map(_offset, args[0], args[0].dtype(delta))  # values only, so pass_gradient changes nothing
        case Power(w=w):
            return _map(_power, args[0], args[0].dtype(w))
        case Piecewise(points=points):
            x = args[0]
            landmarks = [
                wp.array(np.array(v, dtype=_numpy_dtype(x.dtype)), device=x.device) for v in zip(*points, strict=True)
            ]
            return _map(_piecewise, x, *landmarks)
        case _:
            assert_never(node)


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


def _numpy_dtype(dtype: Any) -> Any:
    return np.float64 if dtype == wp.float64 else np.float32


def _columns(xs: Sequence[Any]) -> tuple[Any, tuple[int, ...]]:
    """The operands as the columns of a (batch, operands) array, and the batch shape."""
    shapes = [tuple(x.shape) for x in xs]
    batch = max(shapes, key=math.prod)
    if any(math.prod(s) != 1 and s != batch for s in shapes):
        raise ValueError(f"the warp backend broadcasts only single values against a batch, got shapes {shapes}")
    dtype = wp.float64 if any(x.dtype == wp.float64 for x in xs) else wp.float32
    out = wp.empty((math.prod(batch), len(xs)), dtype=dtype, device=xs[0].device)
    for col, x in enumerate(xs):
        wp.launch(_gather_column, dim=out.shape[0], inputs=[x.flatten(), col], outputs=[out], device=out.device)
    return out, batch


def _reduce_rows(rows: Any, p: float, slack: float, flip: int) -> Any:
    out = wp.empty((rows.shape[0],), dtype=rows.dtype, device=rows.device)
    args = [rows, rows.dtype(p), rows.dtype(slack), flip]
    wp.launch(_p_mean_rows, dim=rows.shape[0], inputs=args, outputs=[out], device=rows.device)
    return out


def _map(kernel: Any, x: Any, *args: Any) -> Any:
    flat = x.flatten()
    out = wp.empty((flat.shape[0],), dtype=x.dtype, device=x.device)
    wp.launch(kernel, dim=flat.shape[0], inputs=[flat, *args], outputs=[out], device=x.device)
    return out.reshape(x.shape)
