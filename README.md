# Fulfillment Priority Logic

Fulfillment Priority Logic (FPL) is a small language for combining several requirements into one score. Each
requirement is a *fulfillment*: a number between 0 and 1 that says how well it is met right now, where 1 means
fully and 0 means not at all. FPL combines fulfillments with power means, so the result is again a number in
[0, 1] with the same meaning. You can use the score as an optimization objective, a loss, a way to rank
candidates, or anywhere else several requirements must become one judgment.

This package is a reference implementation of the language from the paper
[*Closing the Intent-to-Behavior Gap via Fulfillment Priority Logic*](https://arxiv.org/abs/2503.05818). It runs on
numpy, and also on PyTorch, JAX and Warp.

## Example

A thermostat should keep a room warm. It should also be cheap to run and quiet, but those matter less than warmth.

```python
from fulfillment_priority_logic import And, Offset, Piecewise, Var, evaluate

warm = Piecewise(Var("temperature"), ((14, 0), (18, 0.1), (21, 1)), name="warm")
cheap = Piecewise(Var("kwh"), ((10, 0), (6, 0.1), (2, 1)), name="cheap")
spec = And(warm, Offset(cheap & Var("quiet"), 0.3), p=-2)

print(spec)
print(evaluate(spec, {"temperature": 19.5, "kwh": 5.0, "quiet": 0.8}))
```

```text
warm ∧₋₂ [cheap ∧₀ quiet]₀.₃
· ∧₋₂ ·                                         0.596
├─ warm: piecewise(·; 14→0, 18→0.1, 21→1)       0.550
│  └─ temperature                               19.500
└─ [·]₀.₃                                       0.657
   └─ · ∧₀ ·                                    0.510
      ├─ cheap: piecewise(·; 2→1, 6→0.1, 10→0)  0.325
      │  └─ kwh                                 5.000
      └─ quiet                                  0.800
```

The room is at 19.5 °C, which counts as 0.55 warm. The heater uses 5 kWh, which counts as 0.325 cheap. The score
is 0.596, and the tree shows where it comes from, so you can see which requirement holds it down.

`evaluate` returns the value of every node. You can read one by name with `result["cheap"]`, get all of them with
`result.to_dict()`, or print them on one line with `result.line()`. Inputs can be plain numbers or arrays holding
a batch of cases, and then the formula is computed for every case at once.

## Concepts

| Construct | Written | Printed | What it does |
| --- | --- | --- | --- |
| fulfillment | `Var("quiet")` | `quiet` | An input in [0, 1]. |
| and | `And(a, b, p=-1)`, or `a & b` for p = 0 | `a ∧₋₁ b` | High only when all parts are high. `p` sets how strict the "and" is: at p = -∞ the worst part decides, at 0 it is the geometric mean, at 1 the average. |
| or | `Or(a, b, p=-1)`, or `a \| b` for p = 0 | `a ∨₋₁ b` | High when any part is high. It is `1 − And(1 − a, 1 − b)`. |
| not | `~a` | `¬a` | `1 − a`. |
| priority | `Offset(a, 0.3)` | `[a]₀.₃` | Lowers the priority of `a`. With δ > 0 it squeezes `a` into [δ, 1], so the other parts can still improve the score while `a` is 0. With δ < 0 it squeezes `a` into [0, 1 − \|δ\|]. δ is in [−1, 1]. |
| importance | `a ** 2` | `a²` | Makes `a` more demanding (w > 1) or more lenient (w < 1). 0 and 1 stay where they are. |
| calibration | `Piecewise(Var("kwh"), ((10, 0), (6, 0.1), (2, 1)))` | `piecewise(kwh; 2→1, 6→0.1, 10→0)` | Turns a measurement into a fulfillment, with straight lines between landmarks. |
| slack | `And(a, b, p=-1, slack=1e-3)` | `a ∧₋₁,ε=0.001 b` | A small number ε added before the mean and removed after. It keeps a single 0 from making gradients infinite. The default is 1e-7, which is not printed. |

Every node is a class, and every class takes a name: `Offset(a, 0.3, name="later")`. The operators `&`, `|`, `~`
and `**` are short forms of `And`, `Or`, `Not` and `Power` with their defaults; name what they build with
`(a & b).named("both")`. A named part prints as its name in the formula, and the tree shows what is
inside it. Names are Python identifiers, optionally joined by dots.

`a & b` and `a | b` are binary, as in the paper's grammar, so `a & b & c` means `(a & b) & c`. The power mean is
not associative, so that differs from `And(a, b, c)`. Write `And(a, b, c, p=...)` for an "and" of several parts.
The paper uses p ≤ 0 for "and" and "or"; the library accepts any p.

Formulas are frozen dataclasses, so they compare and hash by structure and work with `match`. The functions
`nodes`, `leaves`, `fold`, `transform`, `replace` and `rename` walk and rewrite them.

## Writing a good specification

1. **Split the goal into requirements.** Say what "done well" means, then split it into requirements that each
   answer one question: how satisfied is this right now? Each answer is a number in [0, 1].

2. **Calibrate each measurement.** A shape that works is three landmarks: the worst the quantity can be gives 0,
   the edge of acceptable gives 0.1, and good gives 1. Take "worst" to be the worst value the quantity can really
   take, not a nearer point. Take every landmark from a measurement or from numbers the task already gives you,
   rather than guessing. For example, say the heater draws at most 10 kWh and 6 kWh is the most you would accept.
   Then `1 - kwh / 10` calls 5 kWh half satisfied, but the calibrated version calls it 0.325, which is closer to
   "just acceptable":

   ```python
   cheap = Piecewise(Var("kwh"), ((10, 0), (6, 0.1), (2, 1)))
   print(round(float(evaluate(cheap, {"kwh": 5.0}).value), 3))  # 0.325
   ```

3. **Group the requirements and choose how strict each group is.** Things that must never fail, such as safety
   limits, combine strictly, with `p` near the minimum (-4 or -∞). Costs can be averaged (p = 1). The task itself
   usually sits in between (p = 0 or -1). No single value of `p` is right for every problem.

4. **Say what matters less with the right tool.** The three tools are not interchangeable. Use priority
   (`Offset`) when a requirement can wait, so that the others can improve while it is still 0. A power cannot do
   this, because `0 ** 0.5` is still 0 and still sends a strict "and" to 0. Use importance (`**`) when a value
   means too much or too little: `quiet ** 2` reads a fan at 0.8 as 0.64. An offset cannot do this, because it
   moves the floor, so `[quiet]₀.₅` reports a fan at full blast as half satisfied. Leave the slack small. It
   only keeps the numbers safe, and a large slack quietly turns every 0 into a positive score.

5. **Check that the score ranks cases the way you would.** Write down a few cases you can rank yourself: one that
   does everything, one that does half, one that does the wrong half, and one that does nothing. Evaluate the
   formula on each and check that the order matches yours. Then print the trees and check that a value like 0.5
   means something you could say in words.

   ```python
   cases = {
       "does everything": {"temperature": 21, "kwh": 2, "quiet": 1.0},
       "warm but costly": {"temperature": 21, "kwh": 9, "quiet": 0.9},
       "cheap but cold": {"temperature": 15, "kwh": 2, "quiet": 1.0},
       "does nothing": {"temperature": 12, "kwh": 10, "quiet": 0.0},
   }
   for name, values in cases.items():
       print(f"{name:16} {float(evaluate(spec, values).value):.3f}")
   ```

   ```text
   does everything  1.000
   warm but costly  0.531
   cheap but cold   0.035
   does nothing     0.000
   ```

## Adding your own node

A node is a frozen dataclass that subclasses `Formula`. Its fields that hold formulas are its children. It says
how it prints with `notation`, which gets the text of its children, and what it computes in one of two ways.

A derived node defines `expand`, which returns a formula made of other nodes. Its value and gradient are those of
that formula, but it prints as itself. A primitive node defines `compute`, which gets the values of its children
and `ops`, the operators of the current backend: `p_mean`, `dual_p_mean`, `offset` and `piecewise` from
`math.py`, `ops.stop_gradient`, and `ops.xp`, the array namespace (numpy, torch or jax.numpy). So a primitive node
runs on every one of those backends, with gradients. The package's own nodes are written the same way. A node's
children are the fields that hold a formula or a tuple of formulas; a formula held in any other shape is refused.
A node is plain data: building one checks nothing. A node with rules for its parameters states them in a `check`
method, and `check(formula)`, which `evaluate` runs, calls it on every node and checks the fields and names once.

```python
from dataclasses import dataclass

from fulfillment_priority_logic import And, Formula, Var, evaluate
from fulfillment_priority_logic.nodes import subscript


@dataclass(frozen=True)
class Forall(Formula):
    """Every part holds: a strict "and"."""

    parts: tuple[Formula, ...]
    p: float = -4.0

    def expand(self):
        return And(*self.parts, p=self.p)

    def notation(self, *parts):
        return f"∀{subscript(self.p)}({', '.join(parts)})"


@dataclass(frozen=True)
class Polynomial(Formula):
    """c₀ + c₁·x + c₂·x² + …, clipped to [0, 1]."""

    x: Formula
    coeffs: tuple[float, ...]
    measurements = ("x",)  # x is a measurement, not a fulfillment

    def compute(self, ops, x):
        return ops.xp.clip(sum(c * x**k for k, c in enumerate(self.coeffs)), 0.0, 1.0)

    def notation(self, x):
        return f"poly({x}; {', '.join(f'{c:g}' for c in self.coeffs)})"


fan = Polynomial(Var("rpm"), (1.0, -0.1, -0.02))  # rpm in thousands: quiet when slow
comfort = Forall((Var("warm"), Var("dry"), fan))
print(comfort)
print(evaluate(comfort, {"warm": 0.9, "dry": 0.8, "rpm": 3.0}))
```

```text
∀₋₄(warm, dry, poly(rpm; 1, -0.1, -0.02))
∀₋₄(·, ·, ·)                0.642
├─ warm                     0.900
├─ dry                      0.800
└─ poly(·; 1, -0.1, -0.02)  0.520
   └─ rpm                   3.000
```

## Installation

The package is not on PyPI yet. Install it from GitHub:

```sh
pip install "fulfillment-priority-logic @ git+https://github.com/fulfilium/fulfillment-priority-logic"
pip install "fulfillment-priority-logic[torch] @ git+https://github.com/fulfilium/fulfillment-priority-logic"
```

The package needs Python 3.12 or later, and numpy. The extras `torch`, `jax` and `warp` install the other
backends.

To work on the package itself, run `nix develop` if you use Nix, or install [uv](https://docs.astral.sh/uv/)
otherwise. Then `uv sync` sets up the environment (add `--extra torch`, `--extra jax`, `--extra warp` or
`--all-extras` for the other backends), and `uv run pytest` runs the tests. `AGENTS.md` lists the checks and the
rules for changes.

## Backends

The backend is chosen from the type of the input arrays, or by name with `evaluate(..., backend="torch")`.

| Backend | Gradients | Notes |
| --- | --- | --- |
| numpy | no | The reference. |
| torch | yes, with autograd | |
| jax | yes, and works under `jax.jit` | |
| warp | no, values only | Runs as Warp kernels on the device of the input arrays (tested on the CPU). It runs the package's nodes, derived nodes, and nodes that compute only with the methods of `ops`; a node that does its own arithmetic runs on numpy, torch and jax. A batch can only be broadcast against single values. |

With torch or jax the score is an ordinary tensor, so you can optimize it directly:

```python
import torch

kwh = torch.tensor(5.0, requires_grad=True)
score = evaluate(spec, {"temperature": 19.5, "kwh": kwh, "quiet": 0.8}).value
score.backward()
print(bool(kwh.grad < 0))  # True: using less energy raises the score
```

The numerics are in `math.py`, which you can use on arrays without the language. It has `p_mean(x, p, axis=...)`
for "and", `dual_p_mean` for "or", `offset`, `piecewise` and `min_fulfillment(score, n, p)`. The last one gives
the lowest value any one of `n` requirements can have when their "and" reaches a given score (Theorem IV.1 of the
paper). They work on numpy arrays, torch tensors and jax arrays alike. For Warp, the same operators exist as
`@wp.func` functions in `warp_funcs.py`, to call inside your own kernels.

## Notes on the implementation

The power mean is computed as `M_p(x + ε) − ε`, the form the thesis behind this work proves stays within the
range of its inputs. It is also stabilized for large |p|, exact at p = 0 and p = ±∞, and accurate as p approaches
0. The docstring of `p_mean` in `math.py` explains how.

The priority offset is `a·(1 − |δ|) + max(δ, 0)`. The paper writes it as `(a + max(δ, 0)) / (1 + δ)`. The two
give different values for every δ ≠ 0: for δ > 0 the paper's floor is δ/(1 + δ) and this one's is δ. They are the
same family, though: δ = δ_paper / (1 + δ_paper) here gives exactly the paper's values. For δ < 0 the paper's form
goes above 1, while this one stays in [0, 1] for every δ in [−1, 1].

The offset's gradient is 1 − |δ|. `Offset(a, δ, pass_gradient=True)`, printed `[a]_δ↑`, keeps the value but passes a
gradient of 1, so the offset moves the value without weakening its pull. It is a training device and not part of
what the formula means.

The power's gradient is its true slope `w·a^(w−1)`, kept finite at 0 for w < 1 by the same slack ε as the mean.
`Power(a, w, pass_gradient=True)`, printed `a^w↑`, keeps the value but passes a gradient of 1, so the power
changes what a value means without changing its pull.

## Citation

The paper applies FPL to reinforcement learning. Its code is at
[bmabsout/FPL-IROS25](https://github.com/bmabsout/FPL-IROS25).

```bibtex
@misc{elmabsout2025fpl,
  title         = {Closing the Intent-to-Behavior Gap via Fulfillment Priority Logic},
  author        = {El Mabsout, Bassel and Abdelgawad, Abdelrahman and Mancuso, Renato},
  year          = {2025},
  eprint        = {2503.05818},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2503.05818}
}
```

## License

Apache-2.0.
