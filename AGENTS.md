# AGENTS.md

This package implements Fulfillment Priority Logic, a small language that combines requirements, each met to a
degree in [0, 1], into one score. A formula is a tree of immutable nodes, and `evaluate` computes it on numpy,
torch, jax or Warp arrays.

## Before you push

- `uv run ruff check .` and `uv run ruff format --check .`
- `uv run ty check`
- `uv run pytest` (it also runs the docstring examples). Install every backend first with `uv sync --all-extras`.

Never skip, loosen or delete a test to make it pass. Find the cause.

## Rules

1. **The kinds of node are an open set.** A node is a frozen dataclass that subclasses `Formula` and defines
   `notation`, and either `compute` or `expand`. The package's own nodes (`Var`, `And`, ..., `Piecewise`) are
   written the same way, and no other code names them, except `Var` for the inputs. A node is built by its class,
   and every class takes `name=`. `Formula` gives the paper's operators (`&`, `|`, `~`, `**`) and `.named`, and no
   method that builds one kind of node (no `.offset(...)`): a method cannot be given to a node added later, so
   it would make the package's nodes read differently from everyone else's. Do not add a `match` over
   kinds of node; add a method to the nodes, or a fold.
2. **An interpretation is a fold.** Evaluation, the one-line text, `transform` and `rename` are folds over
   `children`, as `fold_all` in `evaluate` and `formula_text` show; the tree view walks top-down over folded
   values. A node's children are its fields that hold a formula or a tuple of formulas, and nothing else may hold
   one. `evaluate` expands derived nodes first, and reads the inputs and their roles from the expanded tree.
3. **Closed sets are `Literal`s, and a `match` over one ends with `case _: assert_never(x)`,** as the libraries in
   `math.py` do. Do not use `Enum`, and do not use a `bool` or `None` to mean a state.
4. **Check a value once, where it is made, and trust it afterwards.** `Offset.__post_init__` refuses a δ outside
   [−1, 1]; `Offset.compute` does not check it again. Do not check what the types already say, except where
   untyped code enters: `check_operands` refuses a number given as an operand, with a hint.
5. **The numerics live in `math.py`, and for Warp in `warp_funcs.py`, and nowhere else.** `math.py` imports only
   numpy and the standard library, and not the standard `math` module, so a reader can copy it alone.
6. **A backend is one file in `backends/`** that converts inputs to its arrays and reads results back
   (`inputs`, `to_numpy`, `value_range`), and gives the nodes their `Ops`, which call `math.py`.
7. **Use the type parameter syntax** (`def fold[R](...)`, `class Result[A]`, `type Library = ...`), not `TypeVar`
   or `Generic`. Use a `Protocol` for an interface such as `Ops` or `Backend`.
8. **Keep it simple.** Inline a one-line helper that has one caller. Do not add a parameter that no caller sets.
   Delete dead code.
9. **Few tests: one golden run and laws.** `validation/test_readme.py` runs the README on every installed backend.
   Every other test states one law, such as range preservation or De Morgan, or guards one bug that nothing else
   would catch. Loop over the cases and the backends inside the test, not with `parametrize`, and name the case in
   the assert message. Do not test what the types prove or what another test checks.
10. **Docstrings give the contract; comments say why.** Put a docstring on every public module, class, function,
    field and type alias. Write a comment only for a reason that the code cannot show, such as a numerical hazard.
    Never write history ("now", "used to", "moved from") in either.

## Writing

Write for a reader who does not know the code yet. Use short, plain sentences in the active voice, with a
concrete subject and a plain verb. Define a term of art the first time you use it. Do not drop "the" and "a",
and do not write slogans or riddles.

| Do not write | Write |
|---|---|
| the slack-subtracted-after-stabilizer reduction | The mean adds ε to every input, and subtracts ε from the result. |
| a name must not change a score | Naming a part does not change the value of the formula. |
| offset buys priority | An offset with δ > 0 raises the floor of a term, so the other terms are improved first. |

A commit has a title that names the change, then a short body that says what was wrong and what changed. It has
no trailers and no session links, and its author and committer use the GitHub noreply address.

The repository is public. Do not write private names, paths, accounts or data in code, tests, documents or
commit messages. Use neutral examples, such as a thermostat or a delivery.
