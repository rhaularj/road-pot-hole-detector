#!/usr/bin/env python
"""Convert Python scripts into Jupyter notebooks (.ipynb).

    python scripts/py_to_ipynb.py scripts/evaluate.py                 # -> notebooks/converted/evaluate.ipynb
    python scripts/py_to_ipynb.py scripts/*.py --out-dir notebooks/converted
    python scripts/py_to_ipynb.py scripts/evaluate.py --args "--weights best.pt --split test"

How a script is split into cells:
  * If it contains "# %%" cell markers (VS Code / Jupytext style), they are used as-is;
    "# %% [markdown]" starts a markdown cell whose "# " comment prefixes are removed.
  * Otherwise the module docstring becomes a markdown cell, the imports one cell, and
    every top-level function / class (with its comments and decorators) its own cell.

So that the notebook runs like the script:
  * Local modules it imports (e.g. utils.py next to it) are written to disk first with a
    %%writefile cell, so the notebook is self-contained.
  * `if __name__ == "__main__":` becomes an "Arguments" cell that sets sys.argv
    (edit ARGS there, or prefill it with --args) followed by the original call.
  * __file__ is defined, since notebooks do not have it.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUTURE_RE = re.compile(r"^from __future__ import .*\n?", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Notebook building blocks (also used by build_cloud_notebooks.py)
# --------------------------------------------------------------------------- #
def _source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def code_cell(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(text)}


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def writefile_cell(path: str, text: str) -> dict:
    return code_cell(f"%%writefile {path}\n{text.rstrip()}\n")


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save_notebook(nb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(nb["cells"]):
        cell.setdefault("id", f"cell-{i}")
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Script -> cells
# --------------------------------------------------------------------------- #
def local_imports(tree: ast.Module, folder: Path) -> list[Path]:
    """Modules imported by the script that live next to it (e.g. utils.py)."""
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module.split(".")[0]]
        elif isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        for name in names:
            candidate = folder / f"{name}.py"
            if candidate.is_file() and candidate not in found:
                found.append(candidate)
    return found


def dependency_cells(script: Path, seen: set[Path]) -> list[dict]:
    """%%writefile cells for local modules, dependencies first."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    cells = []
    for dep in local_imports(tree, script.parent):
        if dep in seen:
            continue
        seen.add(dep)
        cells += dependency_cells(dep, seen)
        cells.append(writefile_cell(dep.name, dep.read_text(encoding="utf-8")))
    return cells


def cells_from_markers(text: str) -> list[dict]:
    chunks = re.split(r"^# %%", text, flags=re.MULTILINE)
    cells = [code_cell(chunks[0])] if chunks[0].strip() else []  # code before the first marker
    for chunk in chunks[1:]:
        header, _, body = chunk.partition("\n")
        if "[markdown]" in header:
            md = "\n".join(re.sub(r"^# ?", "", line) for line in body.strip("\n").split("\n"))
            if md.strip():
                cells.append(markdown_cell(md))
        elif body.strip():
            cells.append(code_cell(body))
    return cells


def cells_from_ast(text: str, script_name: str, args: list[str]) -> list[dict]:
    tree = ast.parse(text)
    lines = text.split("\n")
    body = list(tree.body)
    cells = []

    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        doc = body.pop(0)
        title = f"# `{script_name}`\n\n"
        cells.append(markdown_cell(title + "```text\n" + doc.value.value.strip() + "\n```"))
        start_line = doc.end_lineno
    else:
        start_line = 0

    group_kind, group_text = None, []

    def flush():
        nonlocal group_kind, group_text
        chunk = FUTURE_RE.sub("", "\n".join(group_text)).strip("\n")
        if chunk.strip():
            cells.append(code_cell(chunk))
        group_kind, group_text = None, []

    prev_end = start_line
    for node in body:
        segment = "\n".join(lines[prev_end:node.end_lineno])  # includes comments and decorators above it
        prev_end = node.end_lineno

        is_main = (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                   and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
        if is_main:
            flush()
            cells.append(markdown_cell("## Arguments\n\nEdit `ARGS` (the command-line arguments of the "
                                       "script), then run the next cell."))
            cells.append(code_cell(f"import sys\n\nARGS = {args!r}\nsys.argv = [{script_name!r}] + ARGS"))
            inner = "\n".join(lines[node.body[0].lineno - 1:node.end_lineno])
            indent = len(inner) - len(inner.lstrip())
            cells.append(code_cell("\n".join(line[indent:] for line in inner.split("\n"))))
            continue

        kind = ("import" if isinstance(node, (ast.Import, ast.ImportFrom))
                else "def" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                else "stmt")
        if kind == "def" or kind != group_kind:
            flush()
        group_kind = kind
        group_text.append(segment)
        if kind == "def":
            flush()
    flush()
    return cells


def convert(script: Path, args: list[str]) -> dict:
    text = script.read_text(encoding="utf-8")
    cells = []
    deps = dependency_cells(script, {script})
    if deps:
        cells.append(markdown_cell("## Local modules\n\nThe script imports these files, so they are written "
                                   "next to the notebook first."))
        cells += deps
    if "__file__" in text:
        cells.append(code_cell(f"from pathlib import Path\n\n# notebooks have no __file__; "
                               f"pretend the script sits in the current folder\n"
                               f"__file__ = str(Path.cwd() / {script.name!r})"))
    if re.search(r"^# %%", text, flags=re.MULTILINE):
        cells += cells_from_markers(FUTURE_RE.sub("", text))
    else:
        cells += cells_from_ast(text, script.name, args)
    return notebook(cells)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scripts", nargs="+", help=".py files to convert")
    p.add_argument("--out-dir", default="notebooks/converted", help="where to write the notebooks")
    p.add_argument("--args", default="", help="command-line arguments to prefill in the Arguments cell")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    for name in a.scripts:
        script = Path(name)
        if script.suffix != ".py" or not script.is_file():
            sys.exit(f"Not a Python file: {script}")
        out = out_dir / f"{script.stem}.ipynb"
        save_notebook(convert(script, shlex.split(a.args)), out)
        print(f"{script} -> {out}")


if __name__ == "__main__":
    main()
