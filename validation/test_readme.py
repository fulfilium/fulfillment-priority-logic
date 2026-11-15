"""The golden run: the README's code runs as written, prints what the README shows, and prints the same on every
installed backend."""

import contextlib
import io
import re
import textwrap
from pathlib import Path

from conftest import BACKENDS

import fulfillment_priority_logic as fpl

README = (Path(__file__).parent.parent / "README.md").read_text()
BLOCK = re.compile(r"^( *)```(\w*)\n(.*?)^\1```", re.MULTILINE | re.DOTALL)


def test_the_readme_runs_and_prints_what_it_shows():
    blocks = [(lang, textwrap.dedent(body)) for _, lang, body in BLOCK.findall(README)]
    namespace: dict = {}
    for i, (lang, code) in enumerate(blocks):
        if lang != "python" or ("import torch" in code and "torch" not in BACKENDS):
            continue
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, f"README block {i}", "exec"), namespace)
        if i + 1 < len(blocks) and blocks[i + 1][0] == "text":
            assert out.getvalue() == blocks[i + 1][1], f"README block {i} printed something else"
        for comment in re.findall(r"print\(.*\)\s+# (\S+)", code):
            assert comment.rstrip(":") in out.getvalue(), f"README block {i} did not print {comment}"
    spec, cases = namespace["spec"], namespace["cases"]
    for backend in BACKENDS:
        for case, values in cases.items():
            shown = fpl.evaluate(spec, values, backend).tree()
            assert shown == fpl.evaluate(spec, values).tree(), f"{backend}, {case}"
