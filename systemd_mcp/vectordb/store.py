"""sqlite-vec backed vector store.

A single ``.db`` file holds three tables:

* ``chunks``      - row per document chunk: path, heading breadcrumb,
                    title, raw text, ISO-8601 created_at.
* ``chunks_vec``  - sqlite-vec virtual table with the float embedding,
                    rowid joins back to ``chunks.id``.
* ``meta``        - key/value strings: ``model_name``, ``model_family``,
                    ``dim``, ``source_url``, ``source_commit``,
                    ``built_at``. Lets ``info`` print a useful summary
                    without inspecting every row.

Why sqlite-vec?
  * One file, no daemon. Default DB path is
    ``systemd_mcp/vectordb/vector-database.db`` so the agent finds it
    without an env var; override with ``--db`` or
    ``$LLAMA_DEBUGGER_VECTORDB``.
  * Exact KNN over a few thousand chunks is plenty fast; LVGL's whole
    docs tree is well under that.
  * Pip-installable wheel, vector extension auto-loaded - no system-wide
    sqlite-vec install needed.

Falls back to a numpy brute-force search if ``sqlite-vec`` is not
importable (e.g. running on an exotic platform without a wheel). The
fallback is intentionally minimal - it loads every embedding into RAM
and computes dot products. Fine for the demo's ~thousand-chunk corpus.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import sqlite_vec  # type: ignore[import-untyped]

    _HAS_SQLITE_VEC = True
except Exception:  # pragma: no cover - exercised only on platforms without a wheel
    _HAS_SQLITE_VEC = False


@dataclass
class ChunkRow:
    path: str
    heading: str
    title: str
    text: str


@dataclass
class SearchHit:
    score: float
    path: str
    heading: str
    title: str
    text: str


class VectorStore:
    """Open or create a sqlite-vec backed store at ``db_path``."""

    def __init__(self, db_path: str, *, dim: Optional[int] = None) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        if _HAS_SQLITE_VEC:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)

        self._dim = dim
        self._init_schema()
        if self._dim is None:
            # Re-opening an existing store: pick up the dim that was
            # written to the meta table on the original build so search()
            # works without forcing the caller to repeat it.
            self._dim = self._dim_from_meta()
        if self._dim is None:
            # Older stores - and any store filled only via the UI
            # log-ingest path before it learned to write meta - have a
            # populated chunks_vec but no meta['dim']. The vec0 table
            # encodes its width in its schema, so recover it from there
            # rather than failing every search with an empty result.
            self._dim = self._dim_from_vec_table()

    # ---- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                path        TEXT NOT NULL,
                heading     TEXT NOT NULL DEFAULT '',
                title       TEXT NOT NULL DEFAULT '',
                text        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        # The vec0 virtual table needs to know the dimensionality up front.
        # If we don't have one yet (e.g. opening for `info` before any
        # build), defer creation until the first add_batch call.
        if _HAS_SQLITE_VEC and self._dim is not None:
            self._ensure_vec_table(self._dim)
        elif not _HAS_SQLITE_VEC:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS chunks_vec_fallback ("
                "    rowid INTEGER PRIMARY KEY,"
                "    embedding BLOB NOT NULL"
                ")"
            )
        self.conn.commit()

    def _dim_from_meta(self) -> Optional[int]:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT value FROM meta WHERE key = 'dim'")
            row = cur.fetchone()
        except sqlite3.Error:
            return None
        if not row or not row["value"]:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    def _dim_from_vec_table(self) -> Optional[int]:
        """Recover the embedding dim from an existing ``chunks_vec`` schema.

        The vec0 virtual table is declared as ``vec0(embedding
        float[N])``; parse N back out of ``sqlite_master.sql``. Works even
        when the table is empty, and needs no meta row.
        """
        if not _HAS_SQLITE_VEC:
            return None
        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'chunks_vec'"
            )
            row = cur.fetchone()
        except sqlite3.Error:
            return None
        if not row or not row["sql"]:
            return None
        m = re.search(r"float\[(\d+)\]", row["sql"])
        return int(m.group(1)) if m else None

    def _ensure_vec_table(self, dim: int) -> None:
        cur = self.conn.cursor()
        cur.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec "
            f"USING vec0(embedding float[{dim}])"
        )
        self.conn.commit()
        self._dim = dim

    # ---- mutation ---------------------------------------------------------

    def clear(self) -> None:
        """Wipe all rows and the vec table.

        The vec0 virtual table is bound to a fixed dim, so we drop and
        re-create it. If a dim was known on the current handle we re-bind
        it to the same dim immediately; otherwise the next ``add_batch``
        will create it fresh.
        """
        saved_dim = self._dim
        cur = self.conn.cursor()
        cur.executescript(
            """
            DELETE FROM chunks;
            DELETE FROM meta;
            """
        )
        if _HAS_SQLITE_VEC:
            cur.execute("DROP TABLE IF EXISTS chunks_vec")
            self._dim = None
            if saved_dim is not None:
                self._ensure_vec_table(saved_dim)
        else:
            cur.execute("DELETE FROM chunks_vec_fallback")
        self.conn.commit()

    def set_meta(self, **kv: Any) -> None:
        cur = self.conn.cursor()
        for k, v in kv.items():
            cur.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (k, "" if v is None else str(v)),
            )
        self.conn.commit()

    def add_batch(self, rows: Sequence[ChunkRow], vectors: np.ndarray) -> None:
        if len(rows) != vectors.shape[0]:
            raise ValueError(
                f"row count {len(rows)} != vector count {vectors.shape[0]}"
            )
        if vectors.size == 0:
            return

        dim = int(vectors.shape[1])
        if self._dim is None:
            if _HAS_SQLITE_VEC:
                self._ensure_vec_table(dim)
            else:
                self._dim = dim
        elif dim != self._dim:
            raise ValueError(
                f"vector dim {dim} != store dim {self._dim}; "
                f"clear() the store before re-embedding with a new model"
            )

        now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        cur = self.conn.cursor()
        for row, vec in zip(rows, vectors):
            cur.execute(
                "INSERT INTO chunks(path, heading, title, text, created_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (row.path, row.heading, row.title, row.text, now),
            )
            rowid = cur.lastrowid
            blob = np.ascontiguousarray(vec, dtype=np.float32).tobytes()
            if _HAS_SQLITE_VEC:
                cur.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES(?, ?)",
                    (rowid, blob),
                )
            else:
                cur.execute(
                    "INSERT INTO chunks_vec_fallback(rowid, embedding) VALUES(?, ?)",
                    (rowid, blob),
                )
        self.conn.commit()

    # ---- query ------------------------------------------------------------

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[SearchHit]:
        if self._dim is None:
            return []
        q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(
                f"query dim {q.shape[0]} != store dim {self._dim}"
            )

        cur = self.conn.cursor()
        if _HAS_SQLITE_VEC:
            cur.execute(
                """
                SELECT chunks.id, chunks.path, chunks.heading, chunks.title,
                       chunks.text, chunks_vec.distance AS distance
                FROM chunks_vec
                JOIN chunks ON chunks.id = chunks_vec.rowid
                WHERE chunks_vec.embedding MATCH ?
                  AND k = ?
                ORDER BY distance
                """,
                (q.tobytes(), k),
            )
            hits = []
            for row in cur.fetchall():
                # sqlite-vec returns L2 distance by default; with unit-norm
                # vectors that maps monotonically to cosine. Convert to a
                # similarity score in [0, 1] for nicer display.
                d = float(row["distance"])
                score = max(0.0, 1.0 - d / 2.0)
                hits.append(
                    SearchHit(
                        score=score,
                        path=row["path"],
                        heading=row["heading"],
                        title=row["title"],
                        text=row["text"],
                    )
                )
            return hits

        # numpy fallback
        cur.execute(
            "SELECT c.id, c.path, c.heading, c.title, c.text, f.embedding "
            "FROM chunks c JOIN chunks_vec_fallback f ON c.id = f.rowid"
        )
        rows = cur.fetchall()
        if not rows:
            return []
        mat = np.vstack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        )
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        return [
            SearchHit(
                score=float(sims[i]),
                path=rows[i]["path"],
                heading=rows[i]["heading"],
                title=rows[i]["title"],
                text=rows[i]["text"],
            )
            for i in order
        ]

    # ---- introspection ----------------------------------------------------

    def info(self) -> Dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM chunks")
        n_chunks = int(cur.fetchone()["n"])

        cur.execute("SELECT COUNT(DISTINCT path) AS n FROM chunks")
        n_files = int(cur.fetchone()["n"])

        cur.execute("SELECT key, value FROM meta")
        meta = {row["key"]: row["value"] for row in cur.fetchall()}

        try:
            db_size = os.path.getsize(self.db_path)
        except OSError:
            db_size = 0

        dim = self._dim
        if dim is None and meta.get("dim"):
            try:
                dim = int(meta["dim"])
            except ValueError:
                dim = None

        return {
            "db_path": self.db_path,
            "db_size_bytes": db_size,
            "backend": "sqlite-vec" if _HAS_SQLITE_VEC else "numpy-fallback",
            "dim": dim,
            "chunk_count": n_chunks,
            "file_count": n_files,
            "meta": meta,
        }

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # context-manager sugar
    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def format_info(info: Dict[str, Any]) -> str:
    """Pretty-print the info() dict for the CLI."""
    lines = [
        f"  db path      : {info['db_path']}",
        f"  db size      : {_human_bytes(info['db_size_bytes'])}",
        f"  backend      : {info['backend']}",
        f"  dim          : {info['dim']}",
        f"  chunks       : {info['chunk_count']}",
        f"  source files : {info['file_count']}",
    ]
    if info["meta"]:
        lines.append("  meta:")
        for k, v in sorted(info["meta"].items()):
            lines.append(f"    {k:14s}: {v}")
    return "\n".join(lines)


def _human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024.0:
            return f"{f:.1f} {unit}"
        f /= 1024.0
    return f"{f:.1f} TiB"


__all__ = ["VectorStore", "ChunkRow", "SearchHit", "format_info"]
