"""The printouts of a formula with values: the tree view, and every node with its value on one line.

Both come from each node's `notation`, so a node that you add prints like the package's own.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .nodes import Formula, atom, children, fold_all


def render_tree(root: Formula, value_of: Callable[[Formula], np.ndarray] | None = None) -> str:
    """The formula as an indented tree, with the value of each node in a column on the right when given."""
    rows: list[tuple[str, str]] = []

    def visit(node: Formula, lead: str, joint: str) -> None:
        below = children(node)
        text = node.notation(*["·"] * len(below))
        label = f"{node.name}: {text}" if node.name and below else text
        rows.append((lead + joint + label, format_value(value_of(node)) if value_of else ""))
        indent = lead + ("│  " if joint == "├─ " else "   " if joint else "")
        for i, child in enumerate(below):
            visit(child, indent, "└─ " if i == len(below) - 1 else "├─ ")

    visit(root, "", "")
    width = max(len(text) for text, _ in rows)
    return "\n".join(f"{text:<{width}}  {value}".rstrip() for text, value in rows)


def render_line(root: Formula, value_of: Callable[[Formula], np.ndarray]) -> str:
    """The formula on one line with the value of every node after it, as `text:value`."""

    def step(node: Formula, below: tuple[str, ...]) -> str:
        name = f"{node.name}=" if node.name and below else ""
        return f"{name}{atom(node.notation(*below))}:{format_value(value_of(node), short=True)}"

    return fold_all(root, step, {})[root]


def format_value(a: np.ndarray, short: bool = False) -> str:
    """A value for printing: a number, a few numbers, or a summary of a batch. `short` gives a batch's mean, as ~m."""
    a = np.asarray(a, dtype=float)
    if a.size == 1:
        return _fmt(a.item())
    if short:
        return "~" + _fmt(a.mean())
    if a.size <= 4:
        return " ".join(_fmt(v) for v in a.ravel())
    return f"mean {_fmt(a.mean())}  min {_fmt(a.min())}  max {_fmt(a.max())}  ({a.size} values)"


def _fmt(v: float) -> str:
    return f"{v:.1e}" if v != 0 and abs(v) < 5e-4 else f"{v:.3f}"
