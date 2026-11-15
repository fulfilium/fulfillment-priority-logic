"""Printing formulas: the one-line formula, the tree view, and the one-line form with values."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_never

import numpy as np

from .math import DEFAULT_SLACK, INF
from .nodes import And, Formula, Not, Offset, Or, Piecewise, Power, Var

_SUB = str.maketrans("0123456789-+e", "₀₁₂₃₄₅₆₇₈₉₋₊ₑ")
_SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")

_UNICODE = {"and": "∧", "or": "∨", "not": "¬", "to": "→", "eps": "ε", "open": "⟨", "close": "⟩"}
_ASCII = {"and": "&", "or": "|", "not": "~", "to": "->", "eps": "eps", "open": "<", "close": ">"}
_TREE_UNICODE = ("├─ ", "└─ ", "│  ", "   ")
_TREE_ASCII = ("|- ", "`- ", "|  ", "   ")


def _num(x: float, unicode: bool = True) -> str:
    if abs(x) == INF:
        return ("-" if x < 0 else "") + ("∞" if unicode else "inf")
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:g}"


def _sub(x: float, unicode: bool) -> str:
    return _num(x).translate(_SUB) if unicode else "_" + _num(x, False)


def _op(node: And | Or, unicode: bool) -> str:
    s = _UNICODE if unicode else _ASCII
    text = s["and" if isinstance(node, And) else "or"] + _sub(node.p, unicode)
    if node.slack != DEFAULT_SLACK:
        text += f",{s['eps']}={_num(node.slack)}"
    return text


def _mark(node: Offset, unicode: bool) -> str:
    """The mark of an offset that passes its gradient through."""
    return ("↑" if unicode else "*") if node.pass_gradient else ""


def _power(node: Power, unicode: bool) -> str:
    return _num(node.w).translate(_SUP) if unicode and node.w == int(node.w) else f"^{_num(node.w)}"


def _landmarks(node: Piecewise, unicode: bool, sep: str = ", ") -> str:
    to = (_UNICODE if unicode else _ASCII)["to"]
    return sep.join(f"{_num(x)}{to}{_num(y)}" for x, y in node.points)


def formula(node: Formula, unicode: bool = True) -> str:
    """The formula on one line, with parentheses only where they are needed.

    A named part inside the formula prints as its name. The tree view shows what is inside it.
    """
    s = _UNICODE if unicode else _ASCII

    def part(n: Formula, also: tuple[type, ...] = ()) -> str:
        if n.name is not None:
            return n.name
        text = formula(n, unicode)
        multi = isinstance(n, And | Or) and len(n.operands) > 1
        return f"({text})" if multi or isinstance(n, also) else text

    match node:
        case Var():
            return node.name
        case And() | Or() if len(node.operands) == 1:
            return f"{_op(node, unicode)}({part(node.operands[0])})"
        case And() | Or():
            return f" {_op(node, unicode)} ".join(part(o) for o in node.operands)
        case Not():
            return s["not"] + part(node.operand)
        case Offset():
            inner = node.operand
            return f"[{inner.name or formula(inner, unicode)}]{_sub(node.delta, unicode)}{_mark(node, unicode)}"
        case Power():
            return part(node.operand, (Not, Power)) + _power(node, unicode)
        case Piecewise():
            return f"piecewise({part(node.operand)}; {_landmarks(node, unicode)})"
        case _:
            assert_never(node)


def label(node: Formula, unicode: bool = True, compact: bool = False) -> str:
    """What a node shows in the tree: its name, then its operator. `compact` leaves out spaces, for `render_line`."""
    s = _UNICODE if unicode else _ASCII
    match node:
        case Var():
            return node.name
        case And() | Or():
            op = _op(node, unicode)
        case Not():
            op = s["not"]
        case Offset():
            op = ("[·]" if unicode else "[.]") + _sub(node.delta, unicode) + _mark(node, unicode)
        case Power():
            op = ("(·)" if unicode else "(.)") + _power(node, unicode)
        case Piecewise():
            op = f"piecewise({_landmarks(node, unicode, ',')})" if compact else f"piecewise {_landmarks(node, unicode)}"
        case _:
            assert_never(node)
    return f"{node.name}{'=' if compact else ' '}{op}" if node.name else op


def _fmt(v: float) -> str:
    return f"{v:.1e}" if v != 0 and abs(v) < 5e-4 else f"{v:.3f}"


def format_value(a: np.ndarray, short: bool = False) -> str:
    """A value for printing: a number, a few numbers, or a summary of a batch."""
    a = np.asarray(a, dtype=float)
    if a.size == 1:
        return _fmt(a.item())
    if short:
        return "~" + _fmt(a.mean())
    if a.size <= 4:
        return " ".join(_fmt(v) for v in a.ravel())
    return f"mean {_fmt(a.mean())}  min {_fmt(a.min())}  max {_fmt(a.max())}  ({a.size} values)"


def render_tree(root: Formula, unicode: bool = True, value_of: Callable[[Formula], np.ndarray] | None = None) -> str:
    """The tree view, with each node's value in a column on the right when `value_of` is given."""
    branch, last, pipe, space = _TREE_UNICODE if unicode else _TREE_ASCII
    rows: list[tuple[str, str]] = []

    def visit(node: Formula, lead: str, joint: str) -> None:
        rows.append((lead + joint + label(node, unicode), format_value(value_of(node)) if value_of else ""))
        below = lead + (pipe if joint == branch else space if joint else "")
        for i, child in enumerate(node.children):
            visit(child, below, last if i == len(node.children) - 1 else branch)

    visit(root, "", "")
    width = max(len(text) for text, _ in rows)
    return "\n".join(f"{text:<{width}}  {value}".rstrip() for text, value in rows)


def render_line(root: Formula, value_of: Callable[[Formula], np.ndarray], unicode: bool = True) -> str:
    """Every node with its value on one line: `label:value`, and a group's operands inside angle brackets."""
    s = _UNICODE if unicode else _ASCII

    def item(node: Formula) -> str:
        text = f"{label(node, unicode, compact=True)}:{format_value(value_of(node), short=True)}"
        if node.children:
            text += s["open"] + " ".join(item(c) for c in node.children) + s["close"]
        return text

    return item(root)
