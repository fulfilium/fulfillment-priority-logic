"""The language: a formula is a tree of immutable nodes.

A formula only describes how requirements combine. It holds no numbers other than its parameters, so you can write
it once, print it, compare it with another, rewrite it, and evaluate it on any inputs with `evaluate`.

>>> upright, near = Var("upright"), Var("near")
>>> And(upright, near.offset(0.5), p=-1)
upright ∧₋₁ [near]₀.₅

Every node is a frozen dataclass, so two formulas built the same way are equal and hash the same. The tree
operations (`nodes`, `leaves`, `fold`, `transform`, `replace`, `rename`) are methods of every node.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Self, cast

from . import math

type Formula = Var | And | Or | Not | Offset | Power | Piecewise
"""Every kind of node. A `match` over a formula ends with `case _: assert_never(node)`."""


class Node:
    """The operators and tree operations that every kind of node shares. Build formulas from its subclasses."""

    __slots__ = ()
    name: str | None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls is Node or cls is _Group:
            raise TypeError(f"{cls.__name__} only holds shared methods; build formulas from Var, And, Or and operators")
        return super().__new__(cls)

    def __and__(self, other: Formula) -> And:
        """`a & b` is `And(a, b)` with p = 0. It is binary, as in the paper's grammar.

        The power mean is not associative, so `a & b & c`, which is `(a & b) & c`, differs from `And(a, b, c)`.
        """
        return And(_formula(self), other)

    def __or__(self, other: Formula) -> Or:
        """`a | b` is `Or(a, b)` with p = 0. It is binary; write `Or(a, b, c, p=...)` for several parts."""
        return Or(_formula(self), other)

    def __invert__(self) -> Not:
        """`~a` is the negation `1 − a`."""
        return Not(_formula(self))

    def __pow__(self, w: float) -> Power:
        """`a ** w` changes how demanding a requirement is without changing its range. See `Power`."""
        return Power(_formula(self), w)

    def offset(self, delta: float, *, pass_gradient: bool = False) -> Offset:
        """The priority offset `[a]_δ`, which lowers the priority of `a` when δ > 0. See `Offset`."""
        return Offset(_formula(self), delta, pass_gradient=pass_gradient)

    def piecewise(self, *points: tuple[float, float]) -> Piecewise:
        """A measurement mapped to a fulfillment through landmarks `(measurement, fulfillment)`. See `Piecewise`."""
        return Piecewise(_formula(self), points)

    def named(self, name: str) -> Self:
        """The same formula with a name, which printouts show and `result["name"]` reads."""
        if isinstance(self, Var):
            raise TypeError(f"a Var is already named by the input it reads ({self.name!r}); name the formulas above it")
        _check_name(name)
        return self._evolve(name=name)

    @property
    def children(self) -> tuple[Formula, ...]:
        """The formulas that this one is built from, in order."""
        return ()

    def with_children(self, children: Sequence[Formula]) -> Self:
        """The same node over new children, with the same parameters and name."""
        if len(children) != len(self.children):
            raise ValueError(f"{type(self).__name__} has {len(self.children)} children, got {len(children)}")
        if not children:
            return self
        return self._evolve(operands=tuple(children)) if isinstance(self, _Group) else self._evolve(operand=children[0])

    def nodes(self) -> Iterator[Formula]:
        """Every distinct node, parents before children, in the order that they first appear."""
        seen: set[Formula] = set()
        stack: list[Formula] = [_formula(self)]
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                yield node
                stack.extend(reversed(node.children))

    def leaves(self) -> tuple[Var, ...]:
        """The inputs that this formula reads, in the order that they first appear."""
        return tuple(n for n in self.nodes() if isinstance(n, Var))

    def fold[R](self, fn: Callable[[Formula, tuple[R, ...]], R]) -> R:
        """Combine the tree bottom-up with `fn(node, results of its children)`.

        A part that appears twice is folded once, so `fn` should depend only on its arguments.

        >>> Var("a").offset(0.1).fold(lambda node, below: 1 + max(below, default=0))  # the depth
        2
        """
        me = _formula(self)
        return fold_all(me, fn)[me]

    def transform(self, fn: Callable[[Formula], Formula]) -> Formula:
        """Rewrite the tree bottom-up: rebuild each node over its rewritten children, then apply `fn` to it."""
        return self.fold(lambda node, children: fn(node.with_children(children)))

    def replace(self, old: Formula, new: Formula) -> Formula:
        """The formula with every occurrence of the part `old` replaced by `new`."""
        return self.transform(lambda n: new if n == old else n)

    def rename(self, names: Mapping[str, str]) -> Formula:
        """The formula with its inputs and named parts renamed by `names`, which maps old names to new ones."""
        for new in names.values():
            _check_name(new)

        def step(n: Formula) -> Formula:
            if n.name is None or n.name not in names:
                return n
            return Var(names[n.name]) if isinstance(n, Var) else n._evolve(name=names[n.name])

        return self.transform(step)

    def formula(self, unicode: bool = True) -> str:
        """The formula on one line, in the paper's notation, or in ASCII with `unicode=False`."""
        from .printing import formula

        return formula(_formula(self), unicode)

    def tree(self, unicode: bool = True) -> str:
        """The formula as an indented tree, one node on each line."""
        from .printing import render_tree

        return render_tree(_formula(self), unicode)

    def __repr__(self) -> str:
        return self.formula()

    def __bool__(self) -> bool:
        raise TypeError("a formula has no truth value; combine formulas with & | ~ instead of and/or/not")

    def _evolve(self, **changes: Any) -> Self:
        """A copy with some fields changed, checked like a newly built node."""
        new = object.__new__(type(self))
        for f in fields(cast(Any, self)):
            object.__setattr__(new, f.name, changes.get(f.name, getattr(self, f.name)))
        new._check()
        return new

    def _check(self) -> None:
        _check_parts(_formula(self))


def _formula(node: Node) -> Formula:
    """A node as the union of its kinds. Every subclass of `Node` is a member of `Formula`."""
    return cast(Formula, node)


def fold_all[R](root: Formula, fn: Callable[[Formula, tuple[R, ...]], R]) -> dict[Formula, R]:
    """Fold the tree bottom-up, and return the result at every distinct node, not only at the root."""
    out: dict[Formula, R] = {}

    def go(node: Formula) -> R:
        if node not in out:
            out[node] = fn(node, tuple(go(c) for c in node.children))
        return out[node]

    go(root)
    return out


@dataclass(frozen=True, slots=True, repr=False)
class Var(Node):
    """An input: a fulfillment in [0, 1], or a raw measurement when it is read through `.piecewise(...)`."""

    name: str
    """The key of this input in the values given to `evaluate`."""

    def __post_init__(self) -> None:
        _check_name(self.name)


@dataclass(frozen=True, slots=True, repr=False, init=False)
class _Group(Node):
    operands: tuple[Formula, ...]
    """The parts that the mean combines, at least one."""
    p: float
    """How strict the mean is: -inf is the minimum, 0 the geometric mean, 1 the average and inf the maximum."""
    slack: float
    """The ε that the mean adds before and removes after, to keep gradients finite at 0."""
    name: str | None

    def __init__(self, *operands: Formula, p: float = 0.0, slack: float = math.DEFAULT_SLACK) -> None:
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "slack", slack)
        object.__setattr__(self, "name", None)
        self._check()

    def _check(self) -> None:
        kind = type(self).__name__
        if not self.operands:
            raise ValueError(f"{kind}() needs at least one operand: an empty group has no value")
        if self.p != self.p:
            raise ValueError(f"{kind}: p must be a number or ±inf, got nan")
        if not 0.0 <= self.slack < math.INF:
            raise ValueError(f"{kind}: the slack ε must be a finite number >= 0, got {self.slack}")
        _check_parts(_formula(self))

    @property
    def children(self) -> tuple[Formula, ...]:
        return self.operands


class And(_Group):
    """The conjunction `∧ₚ`: the power mean of the operands.

    `p` sets how strict the "and" is. At p = -inf the worst operand decides; at 0 it is the geometric mean, at 1
    the average and at inf the maximum. The paper uses p <= 0 for an "and". The slack ε keeps a single 0 from
    making the gradient infinite when p < 0, and changes the result by about ε.

    >>> And(Var("safe"), Var("fast"), Var("cheap"), p=-1)
    safe ∧₋₁ fast ∧₋₁ cheap
    """

    __slots__ = ()


class Or(_Group):
    """The disjunction `∨ₚ`, `1 − And(1 − a, 1 − b, ...)` by De Morgan's law. It is high when any operand is high."""

    __slots__ = ()


@dataclass(frozen=True, slots=True, repr=False)
class Not(Node):
    """The negation `¬a = 1 − a`. Write it `~a`."""

    operand: Formula
    """The negated formula."""
    name: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._check()

    @property
    def children(self) -> tuple[Formula, ...]:
        return (self.operand,)


@dataclass(frozen=True, slots=True, repr=False)
class Offset(Node):
    """The priority offset `[a]_δ = a·(1 − |δ|) + max(δ, 0)`. Write it `a.offset(δ)`.

    With δ > 0 the value is squeezed into [δ, 1], so a term at 0 no longer drags an `And` to 0, and the other
    terms can still be improved while this one is unmet: its priority is lower. With δ < 0 the value is squeezed
    into [0, 1 − |δ|], so the term can never be fully met. δ = 0 changes nothing.

    >>> Var("fuel").offset(-0.25)
    [fuel]₋₀.₂₅
    """

    operand: Formula
    """The formula whose priority changes."""
    delta: float
    """δ, in [−1, 1]."""
    pass_gradient: bool = field(default=False, kw_only=True)
    """If true, the gradient through the offset is 1 instead of 1 − |δ| in torch and jax, and the value is the same.
    It is a training device, not part of what the formula means. It prints as ↑: `[a]₀.₃↑`."""
    name: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._check()

    def _check(self) -> None:
        if not -1.0 <= self.delta <= 1.0:
            raise ValueError(
                f"offset of {self.operand!r}: δ must be in [-1, 1], got {self.delta}. "
                "δ > 0 raises the term's floor to δ, δ < 0 lowers its ceiling to 1 - |δ|, and δ = 0 changes nothing."
            )
        _check_parts(self)

    @property
    def children(self) -> tuple[Formula, ...]:
        return (self.operand,)


@dataclass(frozen=True, slots=True, repr=False)
class Power(Node):
    """Importance `a ** w`. Write it `a ** w`.

    0 stays 0 and 1 stays 1, but the values between move: w > 1 makes the requirement more demanding
    (0.8 ** 2 = 0.64), and w < 1 makes it more lenient. For w < 1 the slope at exactly 0 is infinite.
    """

    operand: Formula
    """The formula whose importance changes."""
    w: float
    """The power, a finite number > 0."""
    name: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._check()

    def _check(self) -> None:
        if not 0.0 < self.w < math.INF:
            raise ValueError(f"importance of {self.operand!r}: the power w must be a finite number > 0, got {self.w}")
        _check_parts(self)

    @property
    def children(self) -> tuple[Formula, ...]:
        return (self.operand,)


@dataclass(frozen=True, slots=True, repr=False)
class Piecewise(Node):
    """A measurement turned into a fulfillment by straight lines between landmarks. Write it `x.piecewise(...)`.

    For example, a delivery that is 0 minutes late is perfect, 10 minutes late is barely acceptable, and an hour
    late is worthless:

    >>> Var("minutes_late").piecewise((0, 1), (10, 0.1), (60, 0))
    piecewise(minutes_late; 0→1, 10→0.1, 60→0)
    """

    operand: Formula
    """The measurement."""
    points: tuple[tuple[float, float], ...]
    """The landmarks `(measurement, fulfillment)`: at least two, sorted when stored, no two with the same
    measurement, and each fulfillment in [0, 1]."""
    name: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(sorted(self.points)))
        self._check()

    def _check(self) -> None:
        where = f"piecewise({self.operand!r})"
        if len(self.points) < 2:
            raise ValueError(f"{where}: give at least two landmarks, got {len(self.points)}")
        math.check_landmarks(self.points)
        if not all(0.0 <= y <= 1.0 for _, y in self.points):
            raise ValueError(f"{where}: fulfillments must lie in [0, 1], got {[y for _, y in self.points]}")
        _check_parts(self)

    @property
    def children(self) -> tuple[Formula, ...]:
        return (self.operand,)


def _check_name(name: str) -> None:
    # Identifiers only, optionally joined by dots, so that a name can never be read as notation or as the printed
    # formula of an unnamed part.
    if not all(part.isidentifier() for part in name.split(".")):
        raise ValueError(f"a name must be a Python identifier, optionally with dots (like 'room.warm'), got {name!r}")


def _check_parts(root: Formula) -> None:
    """Refuse an operand that is not a formula, and a name given to two different parts.

    Code without a type checker builds formulas too, so a stray number or string is refused here, with a hint.
    """
    for operand in root.children:
        if not isinstance(operand, Node):
            hint = "wrap inputs in Var('name')"
            raise TypeError(f"{type(root).__name__}: an operand must be a formula, got {operand!r}; {hint}")
    seen: dict[str, Formula] = {}
    for n in root.nodes():
        if n.name is not None and seen.setdefault(n.name, n) is not n:
            raise ValueError(f"the name {n.name!r} is used for two different parts: {seen[n.name]!r} and {n!r}")
