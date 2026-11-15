"""Evaluating a formula on inputs, and the result that keeps the value of every node."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, assert_never, cast, overload

import numpy as np

from . import backends, math
from .nodes import And, Formula, Not, Offset, Or, Piecewise, Power, Var, fold_all
from .printing import render_line, render_tree

if TYPE_CHECKING:
    import jax
    import torch

type Number = float | np.floating[Any]
"""A plain number, which `evaluate` uses for the whole batch."""


# These overloads only tell a type checker which array type comes back for which inputs. Inputs that are all plain
# numbers match every overload, and the first one wins, which is also what happens at run time (numpy).
@overload
def evaluate(  # type: ignore[overload-overlap]
    formula: Formula,
    values: Mapping[str, np.ndarray | Number],
    backend: None = None,
    *,
    check: bool = True,
) -> Result[np.ndarray]: ...
@overload
def evaluate(  # type: ignore[overload-overlap]
    formula: Formula,
    values: Mapping[str, torch.Tensor | Number],
    backend: None = None,
    *,
    check: bool = True,
) -> Result[torch.Tensor]: ...
@overload
def evaluate(
    formula: Formula,
    values: Mapping[str, jax.Array | Number],
    backend: None = None,
    *,
    check: bool = True,
) -> Result[jax.Array]: ...
@overload
def evaluate(
    formula: Formula,
    values: Mapping[str, Any],
    backend: backends.BackendName | None = None,
    *,
    check: bool = True,
) -> Result[Any]: ...
def evaluate(
    formula: Formula, values: Mapping[str, Any], backend: backends.BackendName | None = None, *, check: bool = True
) -> Result[Any]:
    """Compute a formula and all of its parts on the given inputs.

    `values` maps each input name to a number or an array. Arrays must share one batch shape, and a plain number
    is used for the whole batch. The backend follows the arrays' library (numpy, torch, jax or warp) unless you
    name it, and plain numbers go to the device and float type of the arrays.

    With `check=True` every input must be free of NaN, and an input that is read as a fulfillment, not through
    `.piecewise(...)`, must lie in [0, 1]. The check reads the smallest and largest value back from the device. It
    is skipped where values cannot be read: inside `jax.jit`, `jax.vmap` or `jax.grad`, and inside `torch.func`
    transforms. Set `check=False` to skip it everywhere, for example in a training loop on a GPU.

    >>> result = evaluate(Var("a") & Var("b"), {"a": 0.25, "b": 1.0})
    >>> round(float(result.value), 6)
    0.5
    """
    expected = sorted(v.name for v in formula.leaves())
    missing = [n for n in expected if n not in values]
    if missing:
        raise ValueError(
            f"no input given for {', '.join(map(repr, missing))}. "
            f"The formula reads {expected}; the inputs given were {sorted(values)}."
        )
    name = backend or backends.infer(values[n] for n in expected)
    module = backends.load(name)
    inputs = dict(zip(expected, module.inputs([values[n] for n in expected]), strict=True))
    try:
        np.broadcast_shapes(*(tuple(x.shape) for x in inputs.values()))
    except ValueError:
        listed = ", ".join(f"{n} {tuple(x.shape)}" for n, x in inputs.items())
        raise ValueError(
            f"the inputs have batch shapes that do not fit together: {listed}. "
            "Give every input the same shape, or a plain number to use for the whole batch."
        ) from None
    if check:
        fulfillments = _fulfillment_inputs(formula)
        for n, x in inputs.items():
            _check_range(n, module.value_range(x), n in fulfillments)

    def step(node: Formula, args: tuple[Any, ...]) -> Any:
        try:
            match name:
                case "warp":
                    return cast(Any, module).apply(node, args, inputs)
                case "numpy" | "torch" | "jax":
                    return _apply(node, args, inputs, module)
                case _:
                    assert_never(name)
        except Exception as e:
            e.add_note(f"while evaluating {node!r} with the {name} backend")
            raise

    return Result(formula, name, fold_all(formula, step))


class Result[A](Mapping[Formula | str, A]):
    """The value of every node of a formula after `evaluate`, as arrays of type `A`.

    `result.value` is the value of the whole formula. `result["name"]` reads a named part or an input, and
    `result[node]` reads any part. `print(result)` shows the tree with values, `result.line()` shows them on one
    line, and `result.to_dict()` gives plain numbers. The values keep the backend's array type, so gradients flow
    through them.
    """

    __slots__ = ("_names", "_values", "backend", "formula")

    def __init__(self, formula: Formula, backend: backends.BackendName, values: Mapping[Formula, A]) -> None:
        self.formula = formula
        """The evaluated formula."""
        self.backend = backend
        """The backend that computed the values."""
        self._values = values
        self._names = {n.name: n for n in formula.nodes() if n.name is not None}

    @property
    def value(self) -> A:
        """The value of the whole formula."""
        return self._values[self.formula]

    def __getitem__(self, key: Formula | str) -> A:
        if isinstance(key, str):
            if key not in self._names:
                raise KeyError(f"no part of the formula is named {key!r}; the names are {sorted(self._names)}")
            key = self._names[key]
        return self._values[key]

    def __contains__(self, key: object) -> bool:
        return key in self._names or key in self._values

    def __iter__(self) -> Iterator[Formula]:
        return self.formula.nodes()

    def __len__(self) -> int:
        return len(self._values)

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """The value of every part as a float, or a numpy array for a batch, keyed by name or by formula."""
        out: dict[str, float | np.ndarray] = {}
        for node in self.formula.nodes():
            a = self._numpy(node)
            out.setdefault(node.name or node.formula(), float(a.item()) if a.size == 1 else a)
        return out

    def line(self, unicode: bool = True) -> str:
        """Every node and its value on one line. A batch shows its mean, marked with ~.

        >>> evaluate((Var("a") & Var("b")).named("both"), {"a": 0.25, "b": 1.0}).line()
        'both=∧₀:0.500⟨a:0.250 b:1.000⟩'
        """
        return render_line(self.formula, self._numpy, unicode)

    def tree(self, unicode: bool = True) -> str:
        """The formula as a tree with the value of each node beside it. `print(result)` shows this."""
        return render_tree(self.formula, unicode, self._numpy)

    def _numpy(self, node: Formula) -> np.ndarray:
        return backends.load(self.backend).to_numpy(self._values[node])

    def __repr__(self) -> str:
        return self.tree()


def _apply(node: Formula, args: Sequence[Any], inputs: Mapping[str, Any], module: backends.Backend) -> Any:
    """The value of one node from the values of its children, with numpy, torch or jax arrays."""
    match node:
        case Var(name=name):
            return inputs[name]
        case And(p=p, slack=slack):
            return math.p_mean(module.stack(args), p, slack, axis=0)
        case Or(p=p, slack=slack):
            return math.dual_p_mean(module.stack(args), p, slack, axis=0)
        case Not():
            return 1.0 - args[0]
        case Offset(delta=delta, pass_gradient=pass_gradient):
            return math.offset(args[0], delta, pass_gradient=pass_gradient)
        case Power(w=w):
            return args[0] ** w
        case Piecewise(points=points):
            return math.piecewise(args[0], points)
        case _:
            assert_never(node)


def _fulfillment_inputs(root: Formula) -> set[str]:
    """The inputs that are read directly as fulfillments, as opposed to measurements fed to `.piecewise`."""
    found = {root.name} if isinstance(root, Var) else set()
    for node in root.nodes():
        if not isinstance(node, Piecewise):
            found |= {c.name for c in node.children if isinstance(c, Var)}
    return found


def _check_range(name: str, bounds: tuple[float, float] | None, fulfillment: bool) -> None:
    if bounds is None:
        return
    lo, hi = bounds
    if lo != lo or hi != hi:
        raise ValueError(f"input {name!r} contains NaN")
    if fulfillment and (lo < 0.0 or hi > 1.0):
        raise ValueError(
            f"input {name!r} is not a fulfillment: its values range over [{lo:g}, {hi:g}], outside [0, 1]. "
            f"If it is a measurement, map it into [0, 1] with Var({name!r}).piecewise(...)."
        )
