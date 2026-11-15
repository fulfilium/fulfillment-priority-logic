"""The jax backend. It also makes `Result` a pytree, so that a jitted function can return one."""

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from ..evaluate import Result
from ..ops import ArrayOps


def inputs(values: Sequence[Any]) -> list[jax.Array]:
    arrays = [jnp.asarray(v) for v in values]
    return [a if jnp.issubdtype(a.dtype, jnp.floating) else a.astype(float) for a in arrays]


def ops(given: Mapping[str, Any]) -> ArrayOps:
    return ArrayOps(jnp, given, stack)


def stack(xs: Sequence[Any]) -> jax.Array:
    return jnp.stack(jnp.broadcast_arrays(*xs))


def to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def value_range(x: Any) -> tuple[float, float] | None:
    if isinstance(x, Tracer):  # inside jit, vmap or grad the values are not known yet
        return None
    return (float(jnp.min(x)), float(jnp.max(x))) if x.size else None


def _flatten(r: Result[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    nodes = tuple(r)
    return tuple(r[n] for n in nodes), (r.formula, r.backend, nodes)


def _unflatten(aux: tuple[Any, ...], children: tuple[Any, ...]) -> Result[Any]:
    formula, backend, nodes = aux
    return Result(formula, backend, dict(zip(nodes, children, strict=True)))


jax.tree_util.register_pytree_node(Result, _flatten, _unflatten)
