"""The numpy backend, the reference that the other backends are tested against."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..ops import ArrayOps


def inputs(values: Sequence[Any]) -> list[np.ndarray]:
    arrays = [np.asarray(v) for v in values]
    return [a if np.issubdtype(a.dtype, np.floating) else a.astype(np.float64) for a in arrays]


def ops(given: Mapping[str, Any]) -> ArrayOps:
    return ArrayOps(np, given, stack)


def stack(xs: Sequence[Any]) -> np.ndarray:
    return np.stack(np.broadcast_arrays(*xs))


def to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def value_range(x: Any) -> tuple[float, float] | None:
    return (float(np.min(x)), float(np.max(x))) if np.size(x) else None
