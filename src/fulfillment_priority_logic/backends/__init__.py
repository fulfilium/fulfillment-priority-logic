"""The backends: one module for each array library, which converts inputs to its arrays and reads results back.

A backend module has the functions of `Backend`. The numpy, torch and jax backends leave the numerics to
`fulfillment_priority_logic.math`; the warp backend runs kernels built from `fulfillment_priority_logic.warp_funcs`.
None of them knows the kinds of node: each node computes its value with the backend's `Ops`.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, cast

import numpy as np

from .. import math
from ..ops import Ops

type BackendName = Literal["numpy", "torch", "jax", "warp"]
"""The backends that `evaluate` can use."""


class Backend(Protocol):
    """The functions that a backend module defines."""

    def inputs(self, values: Sequence[Any]) -> list[Any]:
        """The inputs of one evaluation as this backend's arrays, on one device and in one float type."""
        ...

    def ops(self, given: Mapping[str, Any]) -> Ops:
        """The toolkit that the nodes compute with, over the converted inputs `given`."""
        ...

    def to_numpy(self, x: Any) -> np.ndarray:
        """An array read back as a numpy array."""
        ...

    def value_range(self, x: Any) -> tuple[float, float] | None:
        """The smallest and largest value of `x`, or None inside a transform that cannot read values."""
        ...


_LOADED: dict[BackendName, Backend] = {}


def load(name: BackendName) -> Backend:
    """The backend module called `name`, imported once, so that a traced or compiled function never imports."""
    if name not in _LOADED:
        try:
            _LOADED[name] = cast(Backend, importlib.import_module(f"{__name__}.{name}"))
        except ImportError as e:
            raise ImportError(
                f"the {name} backend needs {name}: pip install 'fulfillment-priority-logic[{name}]'"
            ) from e
    return _LOADED[name]


def infer(values: Iterable[Any]) -> BackendName:
    """The backend for the arrays among `values`. Plain numbers, lists and numpy arrays give numpy."""
    found = {_kind(v) for v in values} - {"numpy"}
    if len(found) > 1:
        raise TypeError(f"the inputs mix arrays from {sorted(found)}; convert them to one library first")
    return found.pop() if found else "numpy"


def _kind(x: object) -> BackendName:
    warp = sys.modules.get("warp")
    return "warp" if warp is not None and isinstance(x, warp.array) else math.library(x)
