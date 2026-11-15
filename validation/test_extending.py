"""A node added outside the package: a derived one and a primitive one, as in the README."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from conftest import BACKENDS, array, to_numpy

from fulfillment_priority_logic import And, Formula, Ops, Var, evaluate
from fulfillment_priority_logic.nodes import subscript


@dataclass(frozen=True)
class Forall(Formula):
    parts: tuple[Formula, ...]
    p: float = -4.0

    def expand(self) -> Formula:
        return And(*self.parts, p=self.p)

    def notation(self, *children: str) -> str:
        return f"∀{subscript(self.p)}({', '.join(children)})"


@dataclass(frozen=True)
class Polynomial(Formula):
    x: Formula
    coeffs: tuple[float, ...]
    measurements = ("x",)

    def compute(self, ops: Ops, *values: Any) -> Any:
        x = values[0]
        return ops.xp.clip(sum(c * x**k for k, c in enumerate(self.coeffs)), 0.0, 1.0)


a, b, c = Var("a"), Var("b"), Var("c")
X = np.random.default_rng(0).random((3, 16))


def test_a_derived_node_has_the_value_of_its_expansion_on_every_backend():
    forall = Forall((a, b.offset(0.2), ~c))
    assert repr(forall) == "∀₋₄(a, [b]₀.₂, ¬c)"
    for backend in BACKENDS:
        values = {k: array(backend, x) for k, x in zip("abc", X, strict=True)}
        derived = to_numpy(evaluate(forall, values).value)
        expanded = to_numpy(evaluate(forall.expand(), values).value)
        assert np.array_equal(derived, expanded), f"{backend}: {derived} != {expanded}"


def test_a_primitive_node_runs_with_gradients_on_the_array_backends():
    """It needs no change to the package. On Warp, which has no array namespace, it fails with a clear message."""
    poly = And(Polynomial(Var("x"), (0.2, 0.5, -0.1)), a, p=-1)
    x0 = np.array([0.5, 1.0, 2.0])
    expected = np.clip(0.2 + 0.5 * x0 - 0.1 * x0**2, 0, 1)
    assert np.allclose(to_numpy(evaluate(poly, {"x": x0, "a": 1.0})[poly.operands[0]]), expected)
    for backend in BACKENDS:
        if backend == "torch":
            import torch

            x = torch.tensor(x0, requires_grad=True)
            evaluate(poly, {"x": x, "a": 1.0}).value.sum().backward()
            assert x.grad is not None and (x.grad > 0).all(), f"torch: {x.grad}"
        if backend == "jax":
            import jax

            grad = jax.grad(lambda v: evaluate(poly, {"x": v, "a": 1.0}).value.sum())(jax.numpy.asarray(x0))
            assert (np.asarray(grad) > 0).all(), f"jax: {grad}"
        if backend == "warp":
            try:
                evaluate(poly, {"x": array("warp", x0), "a": 1.0})
            except TypeError as e:
                assert "write the node as a derived node" in str(e), str(e)
            else:
                raise AssertionError("warp: a node that uses ops.xp ran without an error")


@dataclass(frozen=True)
class Close(Formula):
    """Near a target: a measurement that its expansion turns into a fulfillment, beside a new input."""

    distance: Formula

    def expand(self) -> Formula:
        return self.distance.piecewise((0, 1), (1, 0.1), (5, 0)) & Var("visible")


def test_an_expansion_may_read_new_inputs_and_measurements():
    """A derived node is analysed after its expansion: the input that only the expansion reads is required, and a
    measurement that only the expansion calibrates is not held to [0, 1]."""
    close = Close(Var("d")).named("close")
    for backend in BACKENDS:
        values = {"d": array(backend, [0.5, 3.0]), "visible": array(backend, [1.0, 0.5])}
        derived = to_numpy(evaluate(close, values)["close"])
        expanded = to_numpy(evaluate(close.expand(), values).value)
        assert np.array_equal(derived, expanded), f"{backend}: {derived} != {expanded}"


def test_a_formula_hidden_from_children_is_refused():
    """A bug guard: a formula inside a tuple of pairs was not a child, so the node evaluated as if it were absent."""

    @dataclass(frozen=True)
    class Weighted(Formula):
        terms: tuple[tuple[Formula, float], ...]

        def compute(self, ops: Ops, *values: Any) -> Any:
            return values[0]

    shapes: list[Any] = [((a, 0.5),), [a]]
    for shape in shapes:
        try:
            Weighted(shape)
        except TypeError as e:
            assert "children() cannot see it" in str(e), str(e)
        else:
            raise AssertionError(f"{shape!r} was accepted")


def test_measurements_must_name_fields():
    try:

        @dataclass(frozen=True)
        class Typo(Formula):
            distance: Formula
            measurements = ("distnace",)

            def compute(self, ops: Ops, *values: Any) -> Any:
                return values[0]

    except TypeError as e:
        assert "distnace" in str(e), str(e)
    else:
        raise AssertionError("a measurements entry that names no field was accepted")
