"""Fulfillment Priority Logic: combine requirements, each met to a degree in [0, 1], into one score.

>>> from fulfillment_priority_logic import And, Var, evaluate
>>> print(evaluate(And(Var("warm"), Var("quiet"), p=-1), {"warm": 0.9, "quiet": 0.5}))
· ∧₋₁ ·   0.643
├─ warm   0.900
└─ quiet  0.500
"""

from .evaluate import Result, evaluate
from .math import DEFAULT_SLACK, dual_p_mean, min_fulfillment, p_mean
from .nodes import (
    And,
    Formula,
    Not,
    Offset,
    Or,
    Piecewise,
    Power,
    Var,
    check,
    children,
    fold,
    leaves,
    nodes,
    rename,
    replace,
    transform,
    with_children,
)
from .ops import Ops

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SLACK",
    "And",
    "Formula",
    "Not",
    "Offset",
    "Ops",
    "Or",
    "Piecewise",
    "Power",
    "Result",
    "Var",
    "check",
    "children",
    "dual_p_mean",
    "evaluate",
    "fold",
    "leaves",
    "min_fulfillment",
    "nodes",
    "p_mean",
    "rename",
    "replace",
    "transform",
    "with_children",
]
