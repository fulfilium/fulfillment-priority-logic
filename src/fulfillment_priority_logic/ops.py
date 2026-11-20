"""The toolkit that a node's `compute` receives: the operators of one backend, and the inputs of one evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

from . import math


class Ops(Protocol):
    """What a node can compute with. Every method takes and returns the backend's arrays, so a node written with
    these methods runs on every backend, with gradients in torch and jax."""

    @property
    def xp(self) -> ModuleType:
        """The array namespace (numpy, torch or jax.numpy) for any other arithmetic. Warp has none."""
        ...

    def input(self, name: str) -> Any:
        """The value of the input called `name`."""
        ...

    def p_mean(self, values: Sequence[Any], p: float, slack: float = math.DEFAULT_SLACK) -> Any:
        """The power mean of the operands' values, for each case of the batch. See `math.p_mean`."""
        ...

    def dual_p_mean(self, values: Sequence[Any], p: float, slack: float = math.DEFAULT_SLACK) -> Any:
        """`1 − p_mean(1 − values)`. See `math.dual_p_mean`."""
        ...

    def offset(self, x: Any, delta: float, *, pass_gradient: bool = False) -> Any:
        """See `math.offset`."""
        ...

    def piecewise(self, x: Any, points: Sequence[tuple[float, float]]) -> Any:
        """See `math.piecewise`."""
        ...

    def power(self, x: Any, w: float, *, pass_gradient: bool = False) -> Any:
        """See `math.power`."""
        ...

    def stop_gradient(self, x: Any) -> Any:
        """`x` with no gradient flowing back through it; the identity where there are no gradients."""
        ...

    def negate(self, x: Any) -> Any:
        """`1 − x`."""
        ...

    def scale_gradient(self, x: Any, k: Any) -> Any:
        """`x` with k times its gradient (`math.scale_gradient`). `Pull` computes with it."""
        ...

    def with_gradient_of(self, value: Any, climb: Any) -> Any:
        """`value` forward, the gradient of `climb` backward (`math.with_gradient_of`). `Surrogate` computes with
        it."""
        ...

    def ladder(self, stages: Sequence[Any], levels: Sequence[float], p: float) -> Any:
        """Ordered stages read on [0, 1] (`math.ladder`). `Ladder` computes with it."""
        ...


@dataclass(frozen=True)
class ArrayOps:
    """The `Ops` of numpy, torch and jax, which leave the numerics to `math.py`."""

    xp: ModuleType
    """The array namespace."""
    inputs: Mapping[str, Any]
    """The inputs of this evaluation, already converted to the backend's arrays."""
    stack: Callable[[Sequence[Any]], Any]
    """The backend's function that broadcasts arrays to one shape and stacks them along a new first axis."""

    def input(self, name: str) -> Any:
        return self.inputs[name]

    def p_mean(self, values: Sequence[Any], p: float, slack: float = math.DEFAULT_SLACK) -> Any:
        return math.p_mean(self.stack(values), p, slack, axis=0)

    def dual_p_mean(self, values: Sequence[Any], p: float, slack: float = math.DEFAULT_SLACK) -> Any:
        return math.dual_p_mean(self.stack(values), p, slack, axis=0)

    def offset(self, x: Any, delta: float, *, pass_gradient: bool = False) -> Any:
        return math.offset(x, delta, pass_gradient=pass_gradient)

    def piecewise(self, x: Any, points: Sequence[tuple[float, float]]) -> Any:
        return math.piecewise(x, points)

    def power(self, x: Any, w: float, *, pass_gradient: bool = False) -> Any:
        return math.power(x, w, pass_gradient=pass_gradient)

    def stop_gradient(self, x: Any) -> Any:
        return math.stop_gradient(x)

    def negate(self, x: Any) -> Any:
        return 1.0 - x

    def scale_gradient(self, x: Any, k: Any) -> Any:
        return math.scale_gradient(x, k)

    def with_gradient_of(self, value: Any, climb: Any) -> Any:
        return math.with_gradient_of(value, climb)

    def ladder(self, stages: Sequence[Any], levels: Sequence[float], p: float) -> Any:
        return math.ladder(tuple(self.stack(stages)), levels, p)
