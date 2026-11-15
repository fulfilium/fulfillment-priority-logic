"""What the tests share: the installed backends, and their arrays."""

import importlib.util
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from fulfillment_priority_logic import And, Var, evaluate
from fulfillment_priority_logic.backends import BackendName

ALL: tuple[BackendName, ...] = ("numpy", "torch", "jax", "warp")
BACKENDS = [b for b in ALL if b == "numpy" or importlib.util.find_spec(b)]
"""The installed backends. A law loops over them."""

if "jax" in BACKENDS:
    import jax

    jax.config.update("jax_enable_x64", True)  # compare in float64; the float32 cases pass float32 arrays
if "warp" in BACKENDS:
    import warp

    warp.config.log_level = warp.LOG_WARNING


def array(backend: str, x, dtype=np.float64):
    """`x` as an array of `backend`, on the CPU."""
    x = np.asarray(x, dtype=dtype)
    match backend:
        case "torch":
            import torch

            return torch.from_numpy(x)
        case "jax":
            import jax.numpy as jnp

            return jnp.asarray(x)
        case "warp":
            import warp as wp

            return wp.array(np.atleast_1d(x), dtype=wp.float64 if dtype == np.float64 else wp.float32, device="cpu")
        case _:
            return x


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x.numpy() if hasattr(x, "numpy") and not isinstance(x, np.ndarray) else x, dtype=np.float64)


def mean_of(backend: str, rows, p: float, dtype=np.float64, kind=And) -> np.ndarray:
    """`kind(...)` of the columns of `rows` with exponent p, evaluated by `backend`: one mean for each row."""
    rows = np.asarray(rows, dtype=dtype)
    names = [f"x{j}" for j in range(rows.shape[1])]
    values = {n: array(backend, rows[:, j], dtype) for j, n in enumerate(names)}
    return to_numpy(evaluate(kind(*map(Var, names), p=p), values).value)
