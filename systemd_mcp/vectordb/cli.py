"""``llama_debugger_vectordb`` - manage the LVGL docs vector store.

Subcommands:

    build    Clone the LVGL docs (or use a local tree), embed every chunk
             through the embedding llama-server, and write to a sqlite-vec
             ``.db`` file. Destructive: rebuilds from scratch.
    query    Embed a single query and print the top-k matching chunks.
    info     Print db stats (backend, dim, chunk count, model, source).
    delete   Remove the .db file from disk.

Defaults assume the demo flow:

    Terminal 1: ./start-llama-server.sh           (chat, port 53425)
    Terminal 2: ./start-llama-embedding-server.sh (embeddings, port 53426)
    Terminal 3: poetry run llama_debugger_vectordb build
                poetry run llama_debugger_vectordb query "..."

Override ``--embed-host`` / ``--embed-port`` / ``--db`` / ``--source`` to
point at a different setup.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from openai import APIConnectionError, APIStatusError

from .embed import EmbeddingClient
from .ingest import DEFAULT_SOURCE_URL, build as ingest_build
from .store import VectorStore, format_info


DEFAULT_DB = "systemd_mcp/vectordb/vector-database.db"
DEFAULT_EMBED_HOST = "127.0.0.1"
DEFAULT_EMBED_PORT = 53426


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_db_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"path to the sqlite-vec .db file (default: {DEFAULT_DB})",
    )


def _add_embed_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--embed-host",
        default=DEFAULT_EMBED_HOST,
        help=f"embedding llama-server host (default: {DEFAULT_EMBED_HOST})",
    )
    p.add_argument(
        "--embed-port",
        type=int,
        default=DEFAULT_EMBED_PORT,
        help=f"embedding llama-server port (default: {DEFAULT_EMBED_PORT})",
    )
    p.add_argument(
        "--embed-model",
        default="nomic-embed-text-v1.5",
        help="model name string to send (llama-server ignores this; "
             "useful for the meta record)",
    )
    p.add_argument(
        "--embed-family",
        default="nomic",
        choices=("nomic", "generic"),
        help="'nomic' adds search_document/search_query prefixes; "
             "'generic' sends text verbatim (use for bge / e5).",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llama_debugger_vectordb",
        description="Manage the LVGL-docs vector database used by "
                    "llama.cpp.debugger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Demo flow:
              ./start-llama-server.sh                # chat (terminal 1)
              ./start-llama-embedding-server.sh      # embeddings (terminal 2)
              llama_debugger_vectordb build          # ingest LVGL docs
              llama_debugger_vectordb info
              llama_debugger_vectordb query "How do I create an animation?"
              llama_debugger_vectordb delete --yes
            """
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="(re)build the vector DB from LVGL docs")
    _add_db_arg(pb)
    _add_embed_args(pb)
    pb.add_argument(
        "--source",
        default=DEFAULT_SOURCE_URL,
        help="git URL OR path to a clone OR path directly at docs/src "
             f"(default: {DEFAULT_SOURCE_URL})",
    )
    pb.add_argument(
        "--docs-subdir",
        default=None,
        help="override the subdir under the source repo to walk "
             "(default: docs/src)",
    )

    pq = sub.add_parser("query", help="search the vector DB")
    _add_db_arg(pq)
    _add_embed_args(pq)
    pq.add_argument("text", help="query string")
    pq.add_argument("-k", "--k", type=int, default=5, help="top-k (default: 5)")
    pq.add_argument(
        "--max-snippet",
        type=int,
        default=400,
        help="truncate displayed chunk text to this many chars (default: 400)",
    )

    pi = sub.add_parser("info", help="print DB stats")
    _add_db_arg(pi)

    pd = sub.add_parser("delete", help="delete the DB file")
    _add_db_arg(pd)
    pd.add_argument(
        "-y", "--yes",
        action="store_true",
        help="don't prompt for confirmation",
    )

    return p


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _make_embed_client(args: argparse.Namespace) -> EmbeddingClient:
    return EmbeddingClient(
        host=args.embed_host,
        port=args.embed_port,
        model=args.embed_model,
        model_family=args.embed_family,
    )


def _explain_embed_error(tag: str, embed: EmbeddingClient, exc: BaseException) -> None:
    """Translate openai/httpx errors into actionable next-step instructions."""
    if isinstance(exc, APIConnectionError):
        print(
            f"[{tag}] cannot reach the embedding server at {embed.base_url}.\n"
            f"  Start it in another terminal:\n"
            f"      ./start-llama-embedding-server.sh\n"
            f"  Or pass --embed-host / --embed-port to point somewhere else.",
            file=sys.stderr,
        )
        return
    if isinstance(exc, APIStatusError):
        body = ""
        try:
            body = exc.response.text  # type: ignore[union-attr]
        except Exception:
            pass
        hint = ""
        # llama-server returns 400 here when --pooling none is set.
        if "pooling" in body.lower() or "pooling" in str(exc).lower():
            hint = (
                "\n  Hint: the embedding server must be started with "
                "`--pooling mean` (or cls/last); --pooling none is rejected "
                "by /v1/embeddings."
            )
        print(
            f"[{tag}] embedding server returned {exc.status_code}: "
            f"{body or exc}{hint}",
            file=sys.stderr,
        )
        return
    print(f"[{tag}] embedding failed: {exc}", file=sys.stderr)


def cmd_build(args: argparse.Namespace) -> int:
    embed = _make_embed_client(args)
    print(
        f"[build] embedding via {embed.base_url} "
        f"(model={embed.model}, family={embed.model_family})",
        file=sys.stderr,
    )
    print(f"[build] writing to {args.db}", file=sys.stderr)
    print(f"[build] source: {args.source}", file=sys.stderr)

    # Probe the embedding server BEFORE doing any work (clone, chunking,
    # store init). A 2-3 MiB sparse clone of LVGL is cheap, but it's
    # still wasteful if the user just forgot to start the embedding
    # server in a second terminal. Probe also discovers the embedding
    # dim, which the store needs at first insert.
    print(
        f"[build] probing embedding server at {embed.base_url} ...",
        file=sys.stderr,
    )
    try:
        dim = embed.probe_dim()
    except Exception as exc:
        _explain_embed_error("build", embed, exc)
        return 2
    print(f"[build] embedding server OK (dim={dim})", file=sys.stderr)

    try:
        stats = ingest_build(
            db_path=args.db,
            source=args.source,
            embed_client=embed,
            docs_subdir=args.docs_subdir,
        )
    except Exception as exc:
        _explain_embed_error("build", embed, exc) if isinstance(
            exc, (APIConnectionError, APIStatusError)
        ) else print(f"[build] FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        f"[build] done: {stats.chunks_inserted} chunks from "
        f"{stats.files_seen} files (skipped {stats.skipped_empty} empty)",
        file=sys.stderr,
    )
    with VectorStore(args.db) as store:
        print(format_info(store.info()))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    if not Path(args.db).exists():
        print(
            f"[query] DB not found: {args.db} - run `build` first.",
            file=sys.stderr,
        )
        return 2

    embed = _make_embed_client(args)
    with VectorStore(args.db) as store:
        info = store.info()
        if info["chunk_count"] == 0:
            print("[query] DB is empty - run `build` first.", file=sys.stderr)
            return 2

        try:
            qvec = embed.embed_query(args.text)
        except Exception as exc:
            _explain_embed_error("query", embed, exc)
            return 2

        hits = store.search(qvec, k=args.k)

    if not hits:
        print("(no results)")
        return 0

    for i, hit in enumerate(hits, 1):
        snippet = hit.text.strip()
        if len(snippet) > args.max_snippet:
            snippet = snippet[: args.max_snippet].rstrip() + "..."
        heading = f" [{hit.heading}]" if hit.heading else ""
        title = f" - {hit.title}" if hit.title else ""
        print(f"#{i}  score={hit.score:.4f}  {hit.path}{title}{heading}")
        print(textwrap.indent(snippet, "    "))
        print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    if not Path(args.db).exists():
        print(f"[info] DB not found: {args.db}", file=sys.stderr)
        return 2
    with VectorStore(args.db) as store:
        print(format_info(store.info()))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    path = Path(args.db)
    if not path.exists():
        print(f"[delete] nothing to do: {path} does not exist.", file=sys.stderr)
        return 0
    if not args.yes:
        ans = input(f"Delete {path}? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("[delete] aborted.", file=sys.stderr)
            return 1
    try:
        path.unlink()
    except OSError as exc:
        print(f"[delete] failed: {exc}", file=sys.stderr)
        return 2

    # sqlite may also leave -shm / -wal sidecars (WAL mode); clean them up too.
    for sidecar in (
        path.with_suffix(path.suffix + "-shm"),
        path.with_suffix(path.suffix + "-wal"),
    ):
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass

    print(f"[delete] removed {path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_DISPATCH = {
    "build": cmd_build,
    "query": cmd_query,
    "info": cmd_info,
    "delete": cmd_delete,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH[args.cmd]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
