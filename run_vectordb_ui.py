#!/usr/bin/env python3
"""Launcher for the vector-DB Toga UI.

Convenience wrapper so you can start the GUI without remembering the
``poetry run`` incantation or which directory you're in:

    ./run_vectordb_ui.py
    python run_vectordb_ui.py
    poetry run python run_vectordb_ui.py

Any arguments are forwarded to ``systemd_mcp.vectordb_ui.cli:main`` (run
with ``--help`` to see them, e.g. ``--db``, ``--embed-host``,
``--default-host``).

If this file is run by an interpreter that can't see the project's
dependencies (e.g. plain ``/usr/bin/python`` instead of the Poetry
virtualenv), it re-executes itself under ``poetry run python`` so the
``systemd_mcp`` package and ``toga`` are importable. Set
``VECTORDB_UI_NO_REEXEC=1`` to disable that fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _reexec_under_poetry() -> "None":
    """Re-run this script via ``poetry run python`` from the project root.

    Guarded by an env var so the child process (which already has the
    deps) never loops back into another re-exec.
    """
    import shutil
    import subprocess

    poetry = shutil.which("poetry")
    if poetry is None:
        sys.stderr.write(
            "run_vectordb_ui: dependencies aren't importable and 'poetry' "
            "is not on PATH.\n"
            "  Install deps with `poetry sync`, then run "
            "`poetry run python run_vectordb_ui.py`.\n"
        )
        raise SystemExit(2)

    env = dict(os.environ)
    env["VECTORDB_UI_NO_REEXEC"] = "1"
    cmd = [poetry, "run", "python", str(Path(__file__).resolve()), *sys.argv[1:]]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    raise SystemExit(proc.returncode)


def main() -> int:
    # Make `systemd_mcp` importable when invoked as a loose script from
    # the project root (python run_vectordb_ui.py).
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from systemd_mcp.vectordb_ui.cli import main as ui_main
    except ModuleNotFoundError as exc:
        # Either the project package or a GUI dependency (toga) is missing
        # from the current interpreter. Try once under poetry.
        if os.environ.get("VECTORDB_UI_NO_REEXEC") == "1":
            sys.stderr.write(
                f"run_vectordb_ui: cannot import the UI ({exc}).\n"
                f"  Run `poetry sync` to install toga + the project, then "
                f"`poetry run python run_vectordb_ui.py`.\n"
            )
            return 2
        _reexec_under_poetry()

    return ui_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
