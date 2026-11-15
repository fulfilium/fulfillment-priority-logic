"""The torch backend. Plain numbers and numpy inputs go to the device and float type of the tensors given."""

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ..ops import ArrayOps

_NUMPY_DTYPES = (torch.float16, torch.float32, torch.float64)


def inputs(values: Sequence[Any]) -> list[torch.Tensor]:
    tensors = [v for v in values if isinstance(v, torch.Tensor) and v.is_floating_point()]
    dtype = functools.reduce(torch.promote_types, (t.dtype for t in tensors)) if tensors else None
    device = tensors[0].device if tensors else None
    out = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=dtype, device=device) for v in values]
    return [t if t.is_floating_point() else t.to(dtype or torch.get_default_dtype()) for t in out]


def ops(given: Mapping[str, Any]) -> ArrayOps:
    return ArrayOps(torch, given, stack)


def stack(xs: Sequence[Any]) -> torch.Tensor:
    dtype = functools.reduce(torch.promote_types, (x.dtype for x in xs))
    return torch.stack(torch.broadcast_tensors(*(x.to(dtype) for x in xs)))


def to_numpy(x: Any) -> np.ndarray:
    t = x.detach().cpu()
    return (t if t.dtype in _NUMPY_DTYPES else t.float()).numpy()  # numpy has no bfloat16


def value_range(x: Any) -> tuple[float, float] | None:
    if x.numel() == 0:
        return None
    try:
        lo, hi = torch.aminmax(x.detach())  # on the device, in the tensor's own float type
        return float(lo.item()), float(hi.item())
    except RuntimeError:  # inside a torch.func transform the values cannot be read
        return None
