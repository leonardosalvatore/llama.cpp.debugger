"""``llama_debugger_vectordb_ui`` - launch the Toga control panel.

Parses configuration (which DB file, which embedding server, default
SSH target to pre-fill the journald form) into a :class:`UiConfig` and
hands it to :class:`systemd_mcp.vectordb_ui.app.VectorDbUiApp`.

Defaults line up with the rest of the project so the zero-flag case
"just works" against the standard demo setup:

    Terminal 1: ./start-llama-server.sh            (chat, port 53425)
    Terminal 2: ./start-llama-embedding-server.sh  (embeddings, 53426)
    Terminal 3: ./run_linux_in_qemu.sh             (SUT on ssh :2222)
    Terminal 4: poetry run llama_debugger_vectordb_ui
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

from systemd_mcp.vectordb.cli import (
    DEFAULT_DB,
    DEFAULT_EMBED_HOST,
    DEFAULT_EMBED_PORT,
)

# SUT defaults mirror systemd_mcp.server._TARGET (the QEMU box from
# run_linux_in_qemu.sh). Hard-coded rather than imported so launching
# the UI doesn't trigger the server module's import-time setup.
DEFAULT_SUT_HOST = "127.0.0.1"
DEFAULT_SUT_PORT = 2222
DEFAULT_SUT_USER = "debian"
DEFAULT_SUT_PASSWORD = "debian"


@dataclass
class UiConfig:
    """Everything the app needs that isn't a live widget value."""

    db_path: str = DEFAULT_DB
    embed_host: str = DEFAULT_EMBED_HOST
    embed_port: int = DEFAULT_EMBED_PORT
    embed_model: str = "nomic-embed-text-v1.5"
    embed_family: str = "nomic"
    sut_host: str = DEFAULT_SUT_HOST
    sut_port: int = DEFAULT_SUT_PORT
    sut_user: str = DEFAULT_SUT_USER
    sut_password: str = DEFAULT_SUT_PASSWORD
    # Source for the "Build from LVGL source" button on the Manage tab.
    build_source: str = "https://github.com/lvgl/lvgl.git"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llama_debugger_vectordb_ui",
        description="Toga desktop UI to view journald logs, search the LVGL "
        "vector DB, and manage the .db file.",
    )
    p.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"path to the sqlite-vec .db file (default: {DEFAULT_DB})",
    )
    p.add_argument("--embed-host", default=DEFAULT_EMBED_HOST,
                   help=f"embedding llama-server host (default: {DEFAULT_EMBED_HOST})")
    p.add_argument("--embed-port", type=int, default=DEFAULT_EMBED_PORT,
                   help=f"embedding llama-server port (default: {DEFAULT_EMBED_PORT})")
    p.add_argument("--embed-family", default="nomic", choices=("nomic", "generic"),
                   help="'nomic' adds search_document/search_query prefixes; "
                        "'generic' sends text verbatim (bge / e5).")
    p.add_argument("--default-host", default=DEFAULT_SUT_HOST,
                   help=f"pre-fill SSH host in the journald form (default: {DEFAULT_SUT_HOST})")
    p.add_argument("--default-port", type=int, default=DEFAULT_SUT_PORT,
                   help=f"pre-fill SSH port (default: {DEFAULT_SUT_PORT})")
    p.add_argument("--default-user", default=DEFAULT_SUT_USER,
                   help=f"pre-fill SSH username (default: {DEFAULT_SUT_USER})")
    p.add_argument("--default-password", default=DEFAULT_SUT_PASSWORD,
                   help="pre-fill SSH password (default: debian)")
    return p


def _config_from_args(args: argparse.Namespace) -> UiConfig:
    return UiConfig(
        db_path=args.db,
        embed_host=args.embed_host,
        embed_port=args.embed_port,
        embed_family=args.embed_family,
        sut_host=args.default_host,
        sut_port=args.default_port,
        sut_user=args.default_user,
        sut_password=args.default_password,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)

    # Import the app lazily so `--help` works even on a box without the
    # toga/toga-gtk extras installed.
    from .app import VectorDbUiApp

    VectorDbUiApp(config).main_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
