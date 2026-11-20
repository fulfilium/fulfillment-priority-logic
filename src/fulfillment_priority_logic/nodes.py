"""The language: a formula is a tree of nodes, and its interpretations are folds.

A node is a frozen dataclass that subclasses `Formula`. Its children are the fields that hold a formula or a
tuple of formulas, in the order of the fields; a formula held in any other shape is refused. A node says how it
prints with `notation`, and what it computes in one of two ways: with `compute`, from the values of its children,
or with `expand`, as a formula made of other nodes. The nodes of the package are written this way, and a node
that you add is written the same way.

`fold` visits a formula bottom-up. Evaluation, the one-line text and the rewrites below are folds; the tree view
walks the formula top-down and shows the folded values. (For readers who know the term: the fields that hold
formulas make each node a functor, and `fold` is its catamorphism.)

>>> And(Var("upright"), Offset(Var("near"), 0.5), p=-1)
upright ∧₋₁ [near]₀.₅
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Self

from . import math

if TYPE_CHECKING:
    from .ops import Ops


@dataclass(frozen=True)
class Formula:
    """The base of every node. It gives the operators `&`, `|`, `~` and `**`, and the defaults below."""

    name: str | None = field(default=None, kw_only=True)
    """A name for this part, which printouts show and `result["name"]` reads. Every node takes it as `name=...`;
    `.named(...)` names a formula built with an operator."""

    measurements: ClassVar[tuple[str, ...]] = ()
    """The fields whose formulas are raw measurements, not fulfillments, so `evaluate` does not hold them to
    [0, 1]."""

    def notation(self, *children: str) -> str:
        """How the node prints on one line, given the text of its children. By default `Name(child, ...)`."""
        return f"{type(self).__name__}({', '.join(children)})"

    def compute(self, ops: Ops, *values: Any) -> Any:
        """The value of the node from the values of its children, computed with the backend's `ops`."""
        raise NotImplementedError(f"{type(self).__name__} defines neither compute nor expand")

    def expand(self) -> Formula | None:
        """A formula made of other nodes that this node stands for, or None if the node defines `compute`."""
        return None

    def __and__(self, other: Formula) -> And:
        """`a & b` is `And(a, b)` with p = 0. It is binary, as in the paper's grammar.

        The power mean is not associative, so `a & b & c`, which is `(a & b) & c`, differs from `And(a, b, c)`.
        """
        return And(self, other)

    def __or__(self, other: Formula) -> Or:
        """`a | b` is `Or(a, b)` with p = 0. It is binary; write `Or(a, b, c, p=...)` for several parts."""
        return Or(self, other)

    def __invert__(self) -> Not:
        """`~a` is the negation `1 − a`."""
        return Not(self)

    def __pow__(self, w: float) -> Power:
        """`a ** w` changes how demanding a requirement is without changing its range. See `Power`."""
        return Power(self, w)

    def check_naming(self) -> None:
        """Refuse a name that is not an identifier, or that two different parts below this node share. Every node
        runs it when it is built. A family of nodes with its own rule for names overrides it."""
        if self.name is not None:
            check_name(self.name)
            check_names(self)

    def named(self, name: str) -> Self:
        """The same formula with a name: `(a & b).named("both")`. A node built by its class takes `name=` instead."""
        if isinstance(self, Var):
            raise TypeError(f"a Var is already named by the input it reads ({self.name!r}); name the formulas above it")
        return _copy(self, {"name": name})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # This runs before @dataclass. Setting __repr__ keeps the decorator from writing its own, and the generated
        # __init__ calls the __post_init__ set here, which runs the node's own checks and then checks its fields.
        super().__init_subclass__(**kwargs)
        if cls.compute is Formula.compute and cls.expand is Formula.expand:
            raise TypeError(f"{cls.__name__} must define compute or expand")
        # Python 3.14 reads annotations lazily; FORWARDREF keeps a name that is not defined yet from raising.
        lazy = {"format": inspect.Format.FORWARDREF} if hasattr(inspect, "Format") else {}
        annotated = {n for c in cls.__mro__ for n in inspect.get_annotations(c, **lazy)}
        unknown = [m for m in cls.measurements if m not in annotated]
        if unknown:
            raise TypeError(f"{cls.__name__}.measurements names {unknown}, which are not fields of {cls.__name__}")
        if "__repr__" not in cls.__dict__:
            cls.__repr__ = Formula.__repr__
        own = getattr(cls, "__post_init__", None)

        def __post_init__(self: Formula) -> None:
            if own is not None:
                own(self)
            _check_fields(self)
            self.check_naming()

        cls.__post_init__ = __post_init__  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return formula_text(self)

    def __bool__(self) -> bool:
        raise TypeError("a formula has no truth value; combine formulas with & | ~ instead of and/or/not")


def children(node: Formula) -> tuple[Formula, ...]:
    """The children of a node: the formulas in its fields, in field order."""
    return tuple(child for _, group in child_fields(node) for child in group)


def _child_changes(node: Formula, new: Sequence[Formula]) -> dict[str, Any]:
    rest = iter(new)
    changes: dict[str, Any] = {}
    for name, group in child_fields(node):
        replaced = tuple(next(rest) for _ in group)
        if any(n is not o for n, o in zip(replaced, group, strict=True)):  # identity: `!=` compares whole subtrees
            changes[name] = replaced if isinstance(getattr(node, name), tuple) else replaced[0]
    return changes


def with_children[F: Formula](node: F, new: Sequence[Formula]) -> F:
    """The same node over new children, in the order of `children`, with the same parameters and name."""
    changes = _child_changes(node, new)
    return _copy(node, changes) if changes else node


def nodes(root: Formula) -> Iterator[Formula]:
    """Every distinct node, parents before children, in the order that they first appear."""
    seen: set[Formula] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            yield node
            stack.extend(reversed(children(node)))


def leaves(root: Formula) -> tuple[Var, ...]:
    """The inputs that a formula reads, in the order that they first appear."""
    return tuple(n for n in nodes(root) if isinstance(n, Var))


def fold[R](root: Formula, step: Callable[[Formula, tuple[R, ...]], R]) -> R:
    """Combine a formula bottom-up with `step(node, results of its children)`.

    A part that appears twice is folded once, so `step` should depend only on its arguments.

    >>> fold(Offset(Var("a"), 0.1), lambda node, below: 1 + max(below, default=0))  # the depth
    2
    """
    return fold_all(root, step, {})[root]


def fold_all[R](root: Formula, step: Callable[[Formula, tuple[R, ...]], R], memo: dict[Formula, R]) -> dict[Formula, R]:
    """`fold`, keeping the result of every distinct node in `memo`, which may already hold some of them."""

    def go(node: Formula) -> R:
        if node not in memo:
            memo[node] = step(node, tuple(go(c) for c in children(node)))
        return memo[node]

    go(root)
    return memo


def transform(root: Formula, fn: Callable[[Formula], Formula]) -> Formula:
    """Rewrite a formula bottom-up: rebuild each node over its rewritten children, then apply `fn` to it."""
    return fold(root, lambda node, new: fn(with_children(node, new)))


def replace(root: Formula, old: Formula, new: Formula) -> Formula:
    """The formula with every occurrence of the part `old` replaced by `new`."""
    return transform(root, lambda n: new if n == old else n)


def rename(root: Formula, names: Mapping[str, str]) -> Formula:
    """The formula with its inputs and named parts renamed by `names`, which maps old names to new ones."""
    for new in names.values():
        check_name(new)

    def step(node: Formula, below: tuple[Formula, ...]) -> Formula:
        changes = _child_changes(node, below)
        if node.name is not None and node.name in names:
            if isinstance(node, Var):
                return Var(names[node.name])
            changes["name"] = names[node.name]
        return _copy(node, changes) if changes else node

    renamed = fold(root, step)
    check_names(renamed)
    return renamed


def formula_text(root: Formula) -> str:
    """The formula on one line. A named part inside it prints as its name."""
    texts = fold_all(root, lambda n, below: n.name or n.notation(*below), {})
    return root.notation(*(texts[c] for c in children(root)))


def check_names(root: Formula) -> None:
    """Refuse a name that is given to two different parts, because `result["name"]` could not tell them apart."""
    seen: dict[str, Formula] = {}
    for n in nodes(root):
        if n.name is not None and seen.setdefault(n.name, n) is not n:
            raise ValueError(f"the name {n.name!r} is used for two different parts: {seen[n.name]!r} and {n!r}")


def check_operands(node: Formula, *operands: object) -> None:
    """Refuse an operand that is not a formula. Code without a type checker builds formulas too."""
    for operand in operands:
        if not isinstance(operand, Formula):
            hint = "wrap inputs in Var('name')"
            raise TypeError(f"{type(node).__name__}: an operand must be a formula, got {operand!r}; {hint}")


def subscript(x: float) -> str:
    """A number as subscript digits, for notation such as `∧₋₁`."""
    return _number(x).translate(str.maketrans("0123456789-+e", "₀₁₂₃₄₅₆₇₈₉₋₊ₑ"))


def superscript(x: float) -> str:
    """A number as superscript digits, for notation such as `a²`."""
    return _number(x).translate(str.maketrans("0123456789-+.", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺·"))


def atom(text: str) -> str:
    """The text in parentheses if it has a space outside brackets, so that it reads as one operand."""
    depth = 0
    for c in text:
        depth += (c in "([⟨") - (c in ")]⟩")
        if c == " " and depth == 0:
            return f"({text})"
    return text


@dataclass(frozen=True)
class Var(Formula):
    """An input: a fulfillment in [0, 1], or a raw measurement when a `Piecewise` reads it."""

    name: str  # type: ignore[assignment]
    """The key of this input in the values given to `evaluate`."""

    def __post_init__(self) -> None:
        check_name(self.name)

    def notation(self, *children: str) -> str:
        return self.name

    def compute(self, ops: Ops, *values: Any) -> Any:
        return ops.input(self.name)


@dataclass(frozen=True, init=False)
class _Mean(Formula):
    operands: tuple[Formula, ...]
    """The parts that the mean combines, at least one."""
    p: float
    """How strict the mean is: -inf is the minimum, 0 the geometric mean, 1 the average and inf the maximum."""
    slack: float
    """The ε that the mean adds before and removes after, to keep gradients finite at 0."""

    symbol: ClassVar[str] = ""
    dual: ClassVar[bool] = False

    def __init__(
        self, *operands: Formula, p: float = 0.0, slack: float = math.DEFAULT_SLACK, name: str | None = None
    ) -> None:
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "slack", slack)
        object.__setattr__(self, "name", name)
        self.__post_init__()

    def __post_init__(self) -> None:
        kind = type(self).__name__
        if not self.operands:
            raise ValueError(f"{kind}() needs at least one operand: an empty group has no value")
        check_operands(self, *self.operands)
        if self.p != self.p:
            raise ValueError(f"{kind}: p must be a number or ±inf, got nan")
        if not 0.0 <= self.slack < math.INF:
            raise ValueError(f"{kind}: the slack ε must be a finite number >= 0, got {self.slack}")

    def notation(self, *children: str) -> str:
        op = self.symbol + subscript(self.p) + ("" if self.slack == math.DEFAULT_SLACK else f",ε={self.slack:g}")
        return f"{op}({children[0]})" if len(children) == 1 else f" {op} ".join(map(atom, children))

    def compute(self, ops: Ops, *values: Any) -> Any:
        return (ops.dual_p_mean if self.dual else ops.p_mean)(values, self.p, self.slack)


@dataclass(frozen=True, init=False)
class And(_Mean):
    """The conjunction `∧ₚ`: the power mean of the operands.

    `p` sets how strict the "and" is. At p = -inf the worst operand decides; at 0 it is the geometric mean, at 1
    the average and at inf the maximum. The paper uses p <= 0 for an "and". The slack ε keeps a single 0 from
    making the gradient infinite when p < 0, and changes the result by about ε.
    """

    symbol: ClassVar[str] = "∧"


@dataclass(frozen=True, init=False)
class Or(_Mean):
    """The disjunction `∨ₚ`, `1 − And(1 − a, 1 − b, ...)` by De Morgan's law. It is high when any operand is high."""

    symbol: ClassVar[str] = "∨"
    dual: ClassVar[bool] = True


@dataclass(frozen=True)
class Not(Formula):
    """The negation `¬a = 1 − a`. Write it `~a`."""

    operand: Formula
    """The negated formula."""

    def __post_init__(self) -> None:
        check_operands(self, self.operand)

    def notation(self, *children: str) -> str:
        return "¬" + atom(children[0])

    def compute(self, ops: Ops, *values: Any) -> Any:
        return ops.negate(values[0])


@dataclass(frozen=True)
class Offset(Formula):
    """The priority offset `[a]_δ = a·(1 − |δ|) + max(δ, 0)`. Write it `Offset(a, δ)`.

    With δ > 0 the value is squeezed into [δ, 1], so a term at 0 no longer drags an `And` to 0, and the other
    terms can still be improved while this one is unmet: its priority is lower. With δ < 0 the value is squeezed
    into [0, 1 − |δ|], so the term can never be fully met. δ = 0 changes nothing.
    """

    operand: Formula
    """The formula whose priority changes."""
    delta: float
    """δ, in [−1, 1]."""
    pass_gradient: bool = field(default=False, kw_only=True)
    """If true, the gradient through the offset is 1 instead of 1 − |δ| in torch and jax, and the value is the same.
    It is a training device, not part of what the formula means. It prints as ↑: `[a]₀.₃↑`."""

    def __post_init__(self) -> None:
        check_operands(self, self.operand)
        if not -1.0 <= self.delta <= 1.0:
            raise ValueError(
                f"offset of {self.operand!r}: δ must be in [-1, 1], got {self.delta}. "
                "δ > 0 raises the term's floor to δ, δ < 0 lowers its ceiling to 1 - |δ|, and δ = 0 changes nothing."
            )

    def notation(self, *children: str) -> str:
        return f"[{children[0]}]{subscript(self.delta)}{'↑' if self.pass_gradient else ''}"

    def compute(self, ops: Ops, *values: Any) -> Any:
        return ops.offset(values[0], self.delta, pass_gradient=self.pass_gradient)


@dataclass(frozen=True)
class Power(Formula):
    """Importance `a ** w`. Write it `a ** w`.

    0 stays 0 and 1 stays 1, but the values between move: w > 1 makes the requirement more demanding
    (0.8 ** 2 = 0.64), and w < 1 makes it more lenient. Its gradient is the true slope, kept finite at 0 by the
    slack (see `math.power`); `Power(a, w, pass_gradient=True)` passes a gradient of 1 instead.
    """

    operand: Formula
    """The formula whose importance changes."""
    w: float
    """The power, a finite number > 0."""
    pass_gradient: bool = field(default=False, kw_only=True)
    """If true, the gradient through the power is 1 in torch and jax, and the value is the same. It is a training
    device, not part of what the formula means. It prints as ↑: `a²↑`."""

    def __post_init__(self) -> None:
        check_operands(self, self.operand)
        if not 0.0 < self.w < math.INF:
            raise ValueError(f"importance of {self.operand!r}: the power w must be a finite number > 0, got {self.w}")

    def notation(self, *children: str) -> str:
        base = children[0] if children[0].replace(".", "_").isidentifier() else f"({children[0]})"
        power = superscript(self.w) if self.w == int(self.w) else f"^{_number(self.w)}"
        return base + power + ("↑" if self.pass_gradient else "")

    def compute(self, ops: Ops, *values: Any) -> Any:
        return ops.power(values[0], self.w, pass_gradient=self.pass_gradient)


@dataclass(frozen=True)
class Piecewise(Formula):
    """A measurement turned into a fulfillment by straight lines between landmarks. Write it
    `Piecewise(x, points)`.

    For example, a delivery that is 0 minutes late is perfect, 10 minutes late is barely acceptable, and an hour
    late is worthless:

    >>> Piecewise(Var("minutes_late"), ((0, 1), (10, 0.1), (60, 0)))
    piecewise(minutes_late; 0→1, 10→0.1, 60→0)
    """

    operand: Formula
    """The measurement."""
    points: tuple[tuple[float, float], ...]
    """The landmarks `(measurement, fulfillment)`: at least two, sorted when stored, no two with the same
    measurement, and each fulfillment in [0, 1]."""

    measurements: ClassVar[tuple[str, ...]] = ("operand",)

    def __post_init__(self) -> None:
        check_operands(self, self.operand)
        object.__setattr__(self, "points", tuple(sorted(self.points)))
        where = f"piecewise({self.operand!r})"
        if len(self.points) < 2:
            raise ValueError(f"{where}: give at least two landmarks, got {len(self.points)}")
        math.check_landmarks(self.points)
        if not all(0.0 <= y <= 1.0 for _, y in self.points):
            raise ValueError(f"{where}: fulfillments must lie in [0, 1], got {[y for _, y in self.points]}")

    def notation(self, *children: str) -> str:
        landmarks = ", ".join(f"{_number(x)}→{_number(y)}" for x, y in self.points)
        return f"piecewise({children[0]}; {landmarks})"

    def compute(self, ops: Ops, *values: Any) -> Any:
        return ops.piecewise(values[0], self.points)


def child_fields(node: Formula) -> list[tuple[str, tuple[Formula, ...]]]:
    """The fields that hold children, with their children: a field that holds one formula, or a tuple of them."""
    out = []
    for f in fields(node):
        value = getattr(node, f.name)
        if isinstance(value, Formula):
            out.append((f.name, (value,)))
        elif isinstance(value, tuple) and value and all(isinstance(v, Formula) for v in value):
            out.append((f.name, value))
    return out


def _check_fields(node: Formula) -> None:
    """Refuse a formula held in a shape that `children` does not see, which would leave it out of every fold."""
    for f in fields(node):
        value = getattr(node, f.name)
        flat = value if isinstance(value, tuple) else (value,)
        if not all(isinstance(v, Formula) for v in flat) and _holds_formula(value):
            raise TypeError(
                f"{type(node).__name__}.{f.name} holds a formula inside a {type(value).__name__}, where children() "
                "cannot see it. Give each child its own field of type Formula, or use a field of type "
                "tuple[Formula, ...], and keep the numbers that go with them in another field."
            )


def _holds_formula(value: object) -> bool:
    if isinstance(value, Formula):
        return True
    if isinstance(value, Mapping):
        return any(map(_holds_formula, value.keys())) or any(map(_holds_formula, value.values()))
    return isinstance(value, tuple | list | set | frozenset) and any(map(_holds_formula, value))


def _copy[F: Formula](node: F, changes: Mapping[str, Any]) -> F:
    """A copy of a node with some fields changed, checked again by its `__post_init__`."""
    new = object.__new__(type(node))
    for f in fields(node):
        object.__setattr__(new, f.name, changes.get(f.name, getattr(node, f.name)))
    post_init = getattr(new, "__post_init__", None)
    if post_init is not None:
        post_init()
    return new


def _number(x: float) -> str:
    if abs(x) == math.INF:
        return "-∞" if x < 0 else "∞"
    return str(int(x)) if x == int(x) else f"{x:g}"


def check_name(name: str) -> None:
    """Refuse a name that is not a Python identifier, optionally joined by dots. Such a name could be read as
    notation, or as the printed formula of an unnamed part."""
    if not all(part.isidentifier() for part in name.split(".")):
        raise ValueError(f"a name must be a Python identifier, optionally with dots (like 'room.warm'), got {name!r}")
