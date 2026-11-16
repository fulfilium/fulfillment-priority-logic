"""The laws of the operators, checked on every installed backend through `evaluate`."""

import ast
import itertools
from pathlib import Path

import numpy as np
from conftest import BACKENDS, array, mean_of, to_numpy

from fulfillment_priority_logic import And, Or, Var, evaluate, math, min_fulfillment

INF = float("inf")
PS = [-INF, -60.0, -8.0, -1.0, -0.3, 0.0, 0.5, 1.0, 2.0, 8.0, 60.0, INF]
ROWS = np.random.default_rng(0).random((32, 5))
ROWS[0] = [0.0, 0.4, 1.0, 1e-9, 0.5]


def test_the_named_means():
    x = [[0.2, 0.8]]
    cases = {
        "minimum": (-INF, 0.2),
        "harmonic": (-1.0, 2 / (1 / 0.2 + 1 / 0.8)),
        "geometric": (0.0, np.sqrt(0.16)),
        "arithmetic": (1.0, 0.5),
        "maximum": (INF, 0.8),
        "close to the minimum": (-1e4, 0.2),
        "close to the maximum": (1e4, 0.8),
    }
    for backend in BACKENDS:
        for name, (p, expected) in cases.items():
            got = mean_of(backend, x, p)[0]
            assert abs(got - expected) < 1e-4, f"{backend}, {name}: {got} != {expected}"


def test_range_preservation():
    """min x <= M_p(x) <= max x, which the slack must not break (thesis, range preservation of the slacked mean)."""
    for backend in BACKENDS:
        for dtype in (np.float32, np.float64):
            rows = ROWS.astype(dtype)
            for p in PS:
                got = mean_of(backend, rows, p, dtype)
                ok = (got >= rows.min(1)) & (got <= rows.max(1))
                assert ok.all(), f"{backend}, {dtype.__name__}, p = {p}"


def test_monotone_in_each_input_and_in_p():
    better = ROWS.copy()
    better[:, 2] = np.minimum(1.0, better[:, 2] + 0.2)
    for backend in BACKENDS:
        means = [mean_of(backend, ROWS, p) for p in PS]
        for p, m in zip(PS, means, strict=True):
            assert (mean_of(backend, better, p) >= m - 1e-12).all(), f"{backend}, p = {p}: not monotone in an input"
        assert all((a <= b + 1e-12).all() for a, b in itertools.pairwise(means)), f"{backend}: not monotone in p"


def test_idempotence():
    for backend in BACKENDS:
        for p in PS:
            got = mean_of(backend, [[0.37] * 5], p)[0]
            assert abs(got - 0.37) < 1e-12, f"{backend}, p = {p}: {got}"


def test_de_morgan():
    """Or(a, b) = ¬And(¬a, ¬b)."""
    a, b = Var("a"), Var("b")
    for backend in BACKENDS:
        values = {"a": array(backend, ROWS[:, 0]), "b": array(backend, ROWS[:, 1])}
        for p in (-2.0, 0.0, 1.0):
            either = to_numpy(evaluate(Or(a, b, p=p), values).value)
            dual = to_numpy(evaluate(~And(~a, ~b, p=p), values).value)
            assert np.abs(either - dual).max() < 1e-12, f"{backend}, p = {p}"


def test_continuous_through_p_zero():
    rows = ROWS * 0.9 + 0.05
    for backend in BACKENDS:
        geometric = mean_of(backend, rows, 0.0)
        for p in (-1e-3, -1e-6, 1e-6, 1e-3):
            assert np.abs(mean_of(backend, rows, p) - geometric).max() < 2 * abs(p), f"{backend}, p = {p}"


def test_offset_range_and_negation():
    """[a]_δ lies in [max(δ, 0), 1 − max(−δ, 0)], and ¬[a]_δ = [¬a]_{−δ}."""
    a = Var("a")
    x = np.linspace(0, 1, 11)
    for backend in BACKENDS:
        values = {"a": array(backend, x)}
        for delta in (-1.0, -0.35, 0.0, 0.2, 1.0):
            got = to_numpy(evaluate(a.offset(delta), values).value)
            assert np.abs(got - (x * (1 - abs(delta)) + max(delta, 0))).max() < 1e-12, f"{backend}, δ = {delta}"
            assert got.min() >= max(delta, 0) - 1e-12 and got.max() <= 1 - max(-delta, 0) + 1e-12, f"{backend}"
            flipped = to_numpy(evaluate((~a).offset(-delta), values).value)
            assert np.abs((1 - got) - flipped).max() < 1e-12, f"{backend}, δ = {delta}: ¬[a]_δ != [¬a]_-δ"


def test_powers_compose():
    """(a ** w1) ** w2 = a ** (w1·w2)."""
    a = Var("a")
    x = np.linspace(0, 1, 11)
    for backend in BACKENDS:
        values = {"a": array(backend, x)}
        for w1, w2 in ((0.5, 2.0), (3.0, 0.25), (1.7, 1.3)):
            twice = to_numpy(evaluate((a**w1) ** w2, values).value)
            once = to_numpy(evaluate(a ** (w1 * w2), values).value)
            assert np.abs(twice - once).max() < 1e-12, f"{backend}, w = {w1}, {w2}"


def test_min_fulfillment_bounds_every_input():
    """Theorem IV.1: no input is below the bound that its mean certifies, and the bound is reached."""
    rng = np.random.default_rng(1)
    for n, p in ((2, -4.0), (3, -1.0), (5, -0.5), (4, 0.0), (3, 1.0), (3, 2.0)):
        x = rng.random((2000, n)).clip(1e-3)
        scores = math.p_mean(x, p, axis=1)
        bounds = np.array([min_fulfillment(float(s), n, p) for s in scores])
        assert (x.min(1) >= bounds - 1e-6).all(), f"n = {n}, p = {p}"
        worst = np.array([0.4] + [1.0] * (n - 1))
        tight = min_fulfillment(float(math.p_mean(worst, p, slack=0.0)), n, p)
        assert abs(tight - 0.4) < 1e-9, f"n = {n}, p = {p}: the bound is not reached"


def test_extreme_values_stay_finite_and_in_their_own_slice():
    """In float32, at extreme p and with tiny inputs, the mean stays finite. Each slice is stabilized on its own, so a
    tiny value in one sample does not change the mean of another."""
    rows = np.array([[0.0, 1e-30, 1e-7, 1.0], [1e-30] * 4, [1.0, 1.0, 1.0, 0.0], [0.4, 0.6, 0.5, 0.7]], np.float32)
    for backend in BACKENDS:
        for p in (-1e3, -60.0, -8.0, 0.0, 8.0, 60.0, 1e3):
            got = mean_of(backend, rows, p, np.float32)
            assert np.isfinite(got).all() and (got >= rows.min(1)).all(), f"{backend}, p = {p}: {got}"
            alone = mean_of(backend, rows[3:], p, np.float32)[0]
            assert abs(alone - got[3]) <= 1e-6 * alone, f"{backend}, p = {p}: another sample changed this one's mean"


def test_piecewise_slope_at_a_landmark_is_the_one_on_its_right():
    """A bug guard: torch gave 0 exactly at a landmark and jax the average of both sides."""
    points = [(0, 0), (1, 0.1), (3, 1)]
    xs = [0.0, 1.0, 2.0, 3.0, 5.0]
    expected = [0.1, 0.45, 0.45, 0.0, 0.0]
    if "torch" in BACKENDS:
        import torch

        x = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
        math.piecewise(x, points).sum().backward()
        assert x.grad is not None and np.allclose(x.grad.numpy(), expected), f"torch: {x.grad}"
    if "jax" in BACKENDS:
        import jax

        grad = jax.grad(lambda v: math.piecewise(v, points).sum())(jax.numpy.asarray(xs))
        assert np.allclose(np.asarray(grad), expected), f"jax: {grad}"


def test_the_gradient_of_a_near_zero_input_keeps_its_digits():
    """A bug guard: in float32 at p = 1, the input at 5e-7 got a slope of 0.2459 where the mean's is 0.25, because
    the autodiff of expm1 adds 1 back to a value near -1."""
    x = np.array([5e-7, 0.3, 0.5, 0.9], np.float32)
    grads = {}
    if "torch" in BACKENDS:
        import torch

        t = torch.tensor(x, requires_grad=True)
        math.p_mean(t, 1.0).backward()
        grads["torch"] = None if t.grad is None else t.grad.numpy()
    if "jax" in BACKENDS:
        import jax

        grads["jax"] = np.asarray(jax.grad(lambda v: math.p_mean(v, 1.0))(jax.numpy.asarray(x)))
    for backend, g in grads.items():
        assert g is not None and np.allclose(g, 0.25, rtol=1e-5), f"{backend}: {g}"


def test_a_nan_measurement_is_never_a_score():
    """A bug guard: NaN went through the landmark comparisons and came out as the first landmark's value."""
    spec = Var("d").piecewise((0, 1), (10, 0.1), (60, 0))
    for backend in BACKENDS:
        got = to_numpy(evaluate(spec, {"d": array(backend, [np.nan, 5.0])}, check=False).value)
        assert np.isnan(got[0]) and not np.isnan(got[1]), f"{backend}: {got}"
        try:
            evaluate(spec, {"d": array(backend, [np.nan])})
        except ValueError as e:
            assert "NaN" in str(e)
        else:
            raise AssertionError(f"{backend}: the check let a NaN measurement through")


def test_math_py_can_be_copied_on_its_own():
    """A bug guard: math.py imports nothing from the package, and not the standard math module either, which a
    copy saved as math.py would import instead of itself."""
    tree = ast.parse(Path(math.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith("fulfillment_priority_logic")
        if isinstance(node, ast.Import):
            assert all(a.name not in ("math", "fulfillment_priority_logic") for a in node.names)
