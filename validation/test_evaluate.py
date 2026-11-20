"""The laws of evaluation: the backends agree, gradients are right, names and rewrites change nothing."""

from typing import Any

import numpy as np
from conftest import BACKENDS, array, to_numpy

from fulfillment_priority_logic import (
    And,
    Offset,
    Or,
    Piecewise,
    Var,
    check,
    evaluate,
    fold,
    math,
    rename,
    replace,
    transform,
)

a, b, c, d = Var("a"), Var("b"), Var("c"), Var("d")
SPEC = And(
    Or(a, Offset(b, 0.3), p=-2, name="either"),
    ~(c**1.5),
    Piecewise(d, ((0, 0), (1, 0.1), (3, 1))) & Offset(a, -0.2, pass_gradient=True),
    p=-1,
)


def inputs(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = {k: rng.random(n) for k in "abc"}
    x["a"][:3] = [0.0, 1.0, 1e-9]
    x["d"] = rng.uniform(-1, 4, n)
    return x


def test_every_backend_agrees_with_numpy_at_every_node():
    reference = evaluate(SPEC, {**inputs(64), "c": 0.25})  # a plain number is used for the whole batch
    cases = {"float64": (np.float64, 1e-12), "float32": (np.float32, 2e-6)}
    for backend in BACKENDS:
        for case, (dtype, tolerance) in cases.items():
            values = {k: array(backend, v, dtype) for k, v in inputs(64).items()}
            result = evaluate(SPEC, {**values, "c": 0.25})
            assert result.backend == backend, f"{backend}: inferred {result.backend}"
            for node in reference:
                error = np.abs(to_numpy(result[node]) - to_numpy(reference[node])).max()
                assert error < tolerance, f"{backend}, {case}, at {node!r}: off by {error}"


def gradient(backend: str, f, x: list[float]) -> np.ndarray:
    """The gradient of the scalar `f(x)` with respect to `x`, in torch or jax."""
    if backend == "torch":
        import torch

        t = torch.tensor(x, requires_grad=True)
        f(t).backward()
        assert t.grad is not None
        return t.grad.numpy()
    import jax

    return np.asarray(jax.grad(f)(jax.numpy.asarray(x)))


def test_gradients_match_finite_differences():
    """With the offset's true gradient. The offset that passes its gradient differs on purpose, by a known factor.

    Two more laws hold for the gradient of the mean itself. An input at 0 with p < 0 gets a finite, positive gradient,
    which is what the slack is for. Equal inputs each get 1/n, because the final clip passes the gradient through.
    """
    spec = And(Or(a, Offset(b, 0.3), p=-2), ~(c**1.5), Piecewise(d, ((0, 0), (1, 0.1), (3, 1))), p=-1)
    point = np.array([0.3, 0.6, 0.2, 1.7])  # d away from the landmarks, where the slope jumps

    def score(x: np.ndarray) -> float:
        return float(evaluate(spec, dict(zip("abcd", x, strict=True))).value)

    h = 1e-6
    expected = np.array([(score(point + h * e) - score(point - h * e)) / (2 * h) for e in np.eye(4)])
    passed = And(Offset(a, -0.6, pass_gradient=True), b, p=-1)
    for backend in [name for name in BACKENDS if name in ("torch", "jax")]:
        if backend == "torch":
            import torch

            x = torch.tensor(point, requires_grad=True)
            evaluate(spec, dict(zip("abcd", x, strict=True)), "torch").value.backward()
            assert x.grad is not None
            grad = x.grad.numpy()
            y = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
            evaluate(passed, {"a": y, "b": 0.7}, "torch").value.backward()
            assert y.grad is not None
            through = float(y.grad)
        else:
            import jax

            grad = np.asarray(jax.grad(lambda v: evaluate(spec, dict(zip("abcd", v, strict=True))).value)(point))
            through = float(jax.grad(lambda v: evaluate(passed, {"a": v, "b": 0.7}).value)(0.5))
        assert np.allclose(grad, expected, rtol=1e-5, atol=1e-8), f"{backend}: {grad} != {expected}"
        u = 0.5 * 0.4  # the offset's value, which the "and" sees
        d_and = (evaluate(And(Var("u"), b, p=-1), {"u": u + h, "b": 0.7}).value
                 - evaluate(And(Var("u"), b, p=-1), {"u": u - h, "b": 0.7}).value) / (2 * h)  # fmt: skip
        assert abs(through - d_and) < 1e-6, f"{backend}: the passed gradient {through} is not 1 · {d_and}"
        for p in (-8.0, -1.0, 0.0):
            g = gradient(backend, lambda v, p=p: math.p_mean(v, p), [0.0, 0.8, 0.9])
            assert np.isfinite(g).all() and g[0] > 0, f"{backend}, p = {p}: the gradient at 0 is {g[0]}"
        for p in (-3.0, 0.0, 0.7, 4.0):
            g = gradient(backend, lambda v, p=p: math.p_mean(v, p), [0.37] * 4)
            assert np.allclose(g, 0.25, rtol=1e-6), f"{backend}, p = {p}: equal inputs get {g}, not 1/4 each"


def test_names_and_rewrites_never_change_a_score():
    values = inputs(8)
    score = to_numpy(evaluate(SPEC, values).value)
    rewrites = {
        "a name": (SPEC.named("whole"), values),
        "a renamed input": (rename(SPEC, {"a": "z"}), {("z" if k == "a" else k): v for k, v in values.items()}),
        "swapped parent and child names": (
            rename(SPEC, {"a": "either", "either": "a"}),
            {("either" if k == "a" else k): v for k, v in values.items()},
        ),
        "the identity rewrite": (transform(SPEC, lambda n: n), values),
        "a part replaced by itself": (replace(SPEC, c, c), values),
    }
    for case, (formula, renamed) in rewrites.items():
        assert np.array_equal(to_numpy(evaluate(formula, renamed).value), score), case
    assert transform(SPEC, lambda n: n) == SPEC and hash(transform(SPEC, lambda n: n)) == hash(SPEC)
    assert Offset(a, 0.3, name="later") == Offset(a, 0.3).named("later"), "a name given to the class or by .named"


def test_fold_visits_a_shared_part_once():
    shared = a & b
    visited: list[object] = []
    fold(Or(shared, ~shared, Offset(shared, 0.1)), lambda node, below: visited.append(node))
    assert visited.count(shared) == 1 and len(visited) == len(set(visited)), visited


def test_plain_numbers_follow_the_arrays():
    """A bug guard: plain numbers went to the CPU in float64 beside a CUDA or float32 tensor, and Warp arrays went
    to Warp's default device, promoted float32 to float64, and refused non-contiguous arrays."""
    if "torch" in BACKENDS:
        import torch

        r = evaluate(And(a, b, c, p=-1), {"a": torch.full((3,), 0.5), "b": np.array([0.2, 0.4, 0.6]), "c": 0.9})
        assert r.value.dtype == torch.float32, f"torch: {r.value.dtype}"
        meta = evaluate(a & b, {"a": torch.ones(3, device="meta"), "b": 0.5}, check=False)
        assert meta.value.device.type == "meta", "torch: a plain number did not follow the tensor's device"
        meta_int = evaluate(
            Piecewise(d, ((0, 0), (10, 1))) & b,
            {"d": torch.tensor([2, 5], device="meta"), "b": 0.5},
            check=False,
        )
        assert meta_int.value.device.type == "meta", "torch: a plain number did not follow an integer tensor's device"
        vmapped = torch.func.vmap(lambda v: evaluate(a & b, {"a": v, "b": 0.5}).value)(torch.rand(4, 3))
        assert vmapped.shape == (4, 3), "torch: the check broke a torch.func transform"
    if "jax" in BACKENDS:
        import jax.numpy as jnp

        r = evaluate(
            Piecewise(d, ((0, 0), (10, 1))) & a & b,
            {"d": jnp.array([2, 5], dtype=jnp.int32), "a": jnp.array([0.5, 0.8], dtype=jnp.float32), "b": 1},
        )
        assert r.value.dtype == jnp.float32, f"jax: integer inputs promoted float32 to {r.value.dtype}"
    if "warp" in BACKENDS:
        import warp as wp

        x = wp.array(np.random.default_rng(0).random((3, 4)).astype(np.float32), device="cpu").transpose()
        r = evaluate(a & b, {"a": x, "b": 0.5})
        assert r.value.dtype == wp.float32 and r.value.device == x.device, "warp: device or float type changed"
        expected = evaluate(a & b, {"a": x.numpy(), "b": 0.5}).value
        assert np.abs(r.value.numpy() - expected).max() < 1e-6, "warp: a non-contiguous array was read wrongly"
        int_r = evaluate(Piecewise(d, ((0, 0), (10, 1))), {"d": wp.array([2, 5], dtype=wp.int32, device="cpu")})
        assert np.allclose(int_r.value.numpy(), [0.2, 0.5]), f"warp: integer input gave {int_r.value.numpy()}"
        empty = evaluate(a & b, {"a": wp.empty((0,), dtype=wp.float32, device="cpu"), "b": 0.5})
        assert empty.value.shape == (0,), f"warp: empty batch got shape {empty.value.shape}"


def test_a_number_where_a_formula_belongs_is_explained():
    """A bug guard: `Var("a") & 0.5` failed deep inside the name check, with an AttributeError about `.name`."""
    untyped: list[Any] = [0.5, "b", None]  # what code without a type checker can pass
    for case, build in {
        "a & 0.5": lambda: a & untyped[0],
        'And(a, "b")': lambda: And(a, untyped[1]),
        "Or(a, None)": lambda: Or(a, untyped[2]),
        "offset of 0.5": lambda: Offset(untyped[0], 0.3),
    }.items():
        try:
            check(build())
        except TypeError as e:
            assert "wrap inputs in Var('name')" in str(e), f"{case}: {e}"
        else:
            raise AssertionError(f"{case}: a number or string was accepted as an operand")


def test_check_pass_catches_invalid_formulas():
    """The check pass validates node parameters, names, and uniqueness across the tree without __post_init__."""
    invalid = {
        "offset delta out of range": Offset(a, 1.5),
        "non-positive power": a**0.0,
        "empty And": And(),
        "too few landmarks": Piecewise(d, ((0, 0),)),
        "invalid name": Var("not a name"),
        "duplicate name": And(Offset(a, 0.1, name="x"), Offset(b, 0.2, name="x")),
    }
    for case, formula in invalid.items():
        try:
            check(formula)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{case}: check() accepted an invalid formula")
