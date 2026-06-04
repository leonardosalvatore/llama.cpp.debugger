"""LVGL-aware ingestion: clone, walk, chunk, embed.

The corpus is the ``docs/``, ``src/`` and ``examples/`` trees of
https://github.com/lvgl/lvgl. Two file kinds, two chunking strategies:

* **Markdown / MDX** under ``docs/``: MDX layers React-style component
  tags (``<Callout>``, ``<LvglExample>``, ``<ApiLink>``, ``<Figure>``,
  ``<DirectoryIndex>``...) on top of Markdown. The chunker keeps the
  inner text and drops the JSX wrappers so embeddings see prose, not
  template syntax. Splits by ``##`` / ``###`` headings, then ~800-char
  windows with 120-char overlap.

* **C source / headers** under ``src/`` and ``examples/`` (and their
  ``.h`` counterparts): we don't try to be too clever - line-window
  splitting at ~50 lines with 8 lines of overlap. The "heading" of
  each chunk is the *enclosing* function name (best-effort regex
  match against the chunk plus a few preceding lines), so the agent
  can cite ``src/widgets/button/lv_button.c :: lv_button_create``.

Pipeline:

    clone_or_pull(url, dest)            # shallow + sparse git clone
        -> iter_source_files(root)      # docs|src|examples/**/*.{md,mdx,c,h}
        -> for each file (dispatched by ext):
              MDX:  parse_frontmatter -> clean_mdx -> split_by_headings
                    -> chunk_text(~800c)
              C/H:  chunk_c_source(text) -> ~50-line windows w/ func-name heading
        -> embed_client.embed_documents(batch)
        -> store.add_batch(rows, vecs)
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .embed import EmbeddingClient
from .store import ChunkRow, VectorStore

DEFAULT_SOURCE_URL = "https://github.com/lvgl/lvgl.git"
# Subdirectories we sparse-checkout AND walk for embedding. Kept separate
# from the file-extension filter so that adding e.g. "demos" later only
# needs one tuple change. Order matters: the build prints stats per file
# and the dirs end up roughly grouped in the DB.
DEFAULT_SPARSE_PATHS: Tuple[str, ...] = ("docs", "src", "examples")
SOURCE_FILE_EXTS = {".md", ".mdx", ".c", ".h"}
# Path prefixes (relative to the walk root) that are skipped during
# ingestion. These are LVGL trees that are mostly mechanical data
# rather than retrievable code/prose:
#   examples/assets/          - pre-encoded image/animation pixel arrays
#                               dumped as huge single-line C blobs
#   src/font/                 - bitmap font glyph arrays + their headers.
#                               The data dwarfs the API surface; users
#                               configure fonts via lv_font_t in their
#                               own code, not by reading these tables.
# Applied AFTER the extension filter so e.g. examples/assets/README.md
# would also be skipped. To re-include them, call iter_source_files with
# an empty exclude_prefixes tuple (or set DEFAULT_EXCLUDE_PREFIXES = ()).
# To trim further, src/libs/ (vendored thorvg / freetype / lz4 / etc.) is
# the next obvious candidate.
DEFAULT_EXCLUDE_PREFIXES: Tuple[str, ...] = (
    "examples/assets/",
    "src/font/",
)
CHUNK_TARGET_CHARS = 800
CHUNK_OVERLAP_CHARS = 120
# C-source line windows. ~50 lines * ~80 chars/line = ~4 KB per chunk
# in typical code, well under nomic-embed-text-v1.5's 2048-token
# training context (~3 chars/token => ~6000 chars).
C_CHUNK_LINES = 50
C_CHUNK_OVERLAP_LINES = 8
# Hard char cap applied AFTER line windowing - guards against pathological
# files where 50 lines is huge (Lottie JSON-as-C-string blobs in
# examples/widgets/lottie/, X-macro tables, dense generated code).
# Chunks past this cap are sliced into pieces of this size; their
# function-name heading is preserved across pieces.
#
# Sizing: nomic-embed-text-v1.5 has n_ctx_train=2048 tokens and
# start-llama-embedding-server.sh sets ubatch=2048, so any input over
# 2048 tokens is either rejected (HTTP 500) or silently truncated. The
# embedding client also prepends ``"search_document: "`` (17 chars,
# ~6 tokens) for nomic, so the actual payload sent to the server is
# C_CHUNK_MAX_CHARS + 17.
#
# Char->token density observed across the LVGL corpus:
#   * Prose (md/mdx):           ~4 chars/token
#   * Normal C code:            ~3 chars/token
#   * Dense draw / math code:   ~2 chars/token (lots of operators)
#   * JSON-as-C-string (Lottie): ~1.47 chars/token (every '{', '"', ','
#     becomes its own token; this was tighter than first measured and
#     is what triggered the 3017-char retries on the first build)
# Cap is sized for the actual worst observed density (~1.47) with
# margin for the prefix and a small safety pad:
#   2500 chars + 17 prefix = 2517 chars sent
#   2517 / 1.47 chars/token = 1712 tokens             -> 336 token margin
#   2517 / 1.50 chars/token = 1678 tokens (defensive) -> 370 token margin
# If you switch to a smaller-context embedding model (e.g.
# bge-small-en-v1.5 has n_ctx=512), drop this proportionally.
C_CHUNK_MAX_CHARS = 2500
EMBED_BATCH = 64


# ---------------------------------------------------------------------------
# Source acquisition
# ---------------------------------------------------------------------------


def clone_or_pull(
    url: str,
    dest: str,
    *,
    sparse_paths: Tuple[str, ...] = DEFAULT_SPARSE_PATHS,
) -> str:
    """Shallow + sparse + partial clone of ``url`` into ``dest``.

    The LVGL repo is large (full UI library: source, demos, examples,
    images, fonts, tests, ...). We only want the docs / src / examples
    subtrees, so the clone stacks four optimizations:

      --depth 1            : no commit history, just HEAD.
      --single-branch      : don't enumerate refs we won't use.
      --filter=blob:none   : partial clone - skip blob downloads up front.
      --sparse + sparse-checkout: only materialize blobs under the listed
                              paths, so the partial-clone fetch for the
                              working copy stays small too.

    Subsequent runs use ``fetch --depth 1`` + ``reset --hard FETCH_HEAD``
    instead of ``git pull``: that keeps the local repo at exactly one
    commit instead of slowly accumulating shallow history. They also
    re-apply the sparse pattern in case the caller widened it (e.g. an
    earlier docs-only clone now wants ``src/`` too).

    Falls back to re-using ``dest`` as-is if it's a non-git directory
    (lets the user point at a pre-extracted tree). If a previous full
    clone exists, we leave it alone and just refresh.
    """
    dest_path = Path(dest).expanduser().resolve()
    if dest_path.is_dir() and not (dest_path / ".git").exists():
        return str(dest_path)

    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "git not found on PATH; install git or pass --source pointing at "
            "a pre-cloned working tree"
        )

    sparse_paths = tuple(sparse_paths)

    if (dest_path / ".git").exists():
        print(
            f"[ingest] refreshing shallow clone at {dest_path} "
            f"(fetch --depth 1, sparse={list(sparse_paths)})",
            file=sys.stderr,
        )
        # Re-apply the sparse pattern in case it widened.
        subprocess.run(
            [git, "-C", str(dest_path),
             "sparse-checkout", "set", *sparse_paths],
            check=False,
        )
        subprocess.run(
            [git, "-C", str(dest_path),
             "fetch", "--depth", "1", "--no-tags", "origin", "HEAD"],
            check=False,
        )
        subprocess.run(
            [git, "-C", str(dest_path), "reset", "--hard", "FETCH_HEAD"],
            check=False,
        )
        return str(dest_path)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[ingest] sparse shallow-cloning {url} -> {dest_path} "
        f"(paths={list(sparse_paths)}, --depth 1, --filter=blob:none)",
        file=sys.stderr,
    )
    try:
        subprocess.run(
            [git, "clone",
             "--depth", "1",
             "--single-branch",
             "--filter=blob:none",
             "--no-checkout",
             "--sparse",
             url, str(dest_path)],
            check=True,
        )
        subprocess.run(
            [git, "-C", str(dest_path),
             "sparse-checkout", "set", *sparse_paths],
            check=True,
        )
        subprocess.run(
            [git, "-C", str(dest_path), "checkout"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        # Old git or a server that doesn't support partial clones: fall
        # back to a plain shallow clone of the whole tree. Still one
        # commit, just no blob filtering.
        print(
            f"[ingest] partial/sparse clone failed ({exc}); "
            f"retrying as a plain --depth 1 clone",
            file=sys.stderr,
        )
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)
        subprocess.run(
            [git, "clone", "--depth", "1", "--single-branch", url, str(dest_path)],
            check=True,
        )
    return str(dest_path)


def get_git_commit(repo_dir: str) -> str:
    git = shutil.which("git")
    if git is None or not (Path(repo_dir) / ".git").exists():
        return ""
    try:
        out = subprocess.check_output(
            [git, "-C", repo_dir, "rev-parse", "HEAD"], text=True
        )
        return out.strip()
    except subprocess.CalledProcessError:
        return ""


def resolve_walk_root(source: str) -> Tuple[str, str, str]:
    """Resolve ``--source`` to (working_tree, walk_root, source_origin).

    ``source`` may be a git URL, a path to a clone, or a path that already
    points directly at a ``docs/src`` tree (legacy). We accept all three.

    For URLs and full clones we walk the working tree itself - the file
    iterator filters by extension and only yields files under the sparse
    paths that actually exist on disk. For the legacy single-subdir case
    (someone passed ``--source path/to/docs/src``) we walk that subdir
    directly so the existing behavior keeps working.
    """
    cache_dir = Path(".cache/lvgl").resolve()
    if source.startswith("http://") or source.startswith("https://") or source.endswith(".git"):
        wt = clone_or_pull(source, str(cache_dir))
        return wt, wt, source

    p = Path(source).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"--source not found: {source}")

    # If the user pointed at a real working tree, walk the whole thing
    # (the iterator filters by extension).
    if (p / "docs").is_dir() or (p / "src").is_dir():
        return str(p), str(p), str(p)
    # Otherwise treat ``source`` as a leaf directory to walk directly
    # (e.g. someone passed ``--source path/to/docs/src``).
    return str(p), str(p), str(p)


# Kept as an alias for any external caller of the old name.
resolve_docs_root = resolve_walk_root


def iter_source_files(
    walk_root: str,
    *,
    exclude_prefixes: Tuple[str, ...] = DEFAULT_EXCLUDE_PREFIXES,
) -> Iterator[Path]:
    """Yield every *.md / *.mdx / *.c / *.h file under ``walk_root``,
    minus anything whose path-relative-to-``walk_root`` starts with a
    string in ``exclude_prefixes``.

    Sorted so the build is deterministic: the same source tree always
    produces chunks in the same order, which makes diffing two DB
    builds (e.g. before and after a chunker tweak) tractable.

    The path comparison uses POSIX-style separators (``/``) so the same
    excludes work on Windows; LVGL's tree is repo-relative either way.
    """
    root = Path(walk_root)
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in SOURCE_FILE_EXTS):
            continue
        if exclude_prefixes:
            rel = p.relative_to(root).as_posix()
            if any(rel.startswith(prefix) for prefix in exclude_prefixes):
                continue
        yield p


# Kept as an alias for any external caller of the old name.
iter_doc_files = iter_source_files


# ---------------------------------------------------------------------------
# MDX cleaning + chunking
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FRONT_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*?)\s*$", re.MULTILINE)
# Self-closing JSX: <Foo .../>   - drop it entirely (figures, examples, etc).
_JSX_SELF_RE = re.compile(r"<[A-Z][A-Za-z0-9]*\b[^>]*/>")
# Open/close JSX block: <Foo ...>inner</Foo> - keep the inner text.
_JSX_BLOCK_RE = re.compile(
    r"<([A-Z][A-Za-z0-9]*)\b[^>]*>(.*?)</\1>", re.DOTALL
)
# import / export statements at top of MDX files.
_MDX_IMPORT_RE = re.compile(
    r"^\s*(?:import|export)\s+[^\n]*\n", re.MULTILINE
)
# HTML comments.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Heading lines: capture level + text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class ParsedDoc:
    title: str
    description: str
    body: str  # cleaned, frontmatter stripped


def parse_frontmatter(text: str) -> ParsedDoc:
    title = ""
    description = ""
    body = text

    m = _FRONTMATTER_RE.match(text)
    if m:
        for km in _FRONT_KV_RE.finditer(m.group(1)):
            k, v = km.group(1).lower(), km.group(2).strip().strip("\"'")
            if k == "title":
                title = v
            elif k == "description":
                description = v
        body = text[m.end():]
    return ParsedDoc(title=title, description=description, body=body)


def clean_mdx(body: str) -> str:
    """Best-effort strip of MDX-specific syntax. Lossy by design."""
    out = _HTML_COMMENT_RE.sub("", body)
    out = _MDX_IMPORT_RE.sub("", out)
    out = _JSX_SELF_RE.sub("", out)

    # Multiple passes for nested blocks.
    for _ in range(4):
        new = _JSX_BLOCK_RE.sub(lambda m: m.group(2), out)
        if new == out:
            break
        out = new

    # Collapse runs of blank lines.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def split_by_headings(body: str) -> List[Tuple[str, str]]:
    """Return ``[(heading_breadcrumb, section_text), ...]``.

    The breadcrumb is ``"H1 > H2 > H3"`` joining whatever heading levels
    were last seen. Files with no headings produce a single section with
    an empty breadcrumb.
    """
    sections: List[Tuple[str, str]] = []
    stack: List[str] = []  # one slot per heading level 1..6
    cursor = 0

    headings = list(_HEADING_RE.finditer(body))
    if not headings:
        text = body.strip()
        return [("", text)] if text else []

    # Prelude before the first heading (if any).
    pre = body[: headings[0].start()].strip()
    if pre:
        sections.append(("", pre))

    for i, h in enumerate(headings):
        level = len(h.group(1))
        text = h.group(2).strip()

        # Resize stack to current level.
        if len(stack) < level:
            stack.extend([""] * (level - len(stack)))
        else:
            stack = stack[:level]
        stack[level - 1] = text

        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        chunk = body[h.end():end].strip()
        if not chunk:
            continue
        breadcrumb = " > ".join(s for s in stack if s)
        sections.append((breadcrumb, chunk))

    return sections


# ---------------------------------------------------------------------------
# C source / header chunking
# ---------------------------------------------------------------------------


# Match a function definition's opening line followed by its `{`. Greedy
# enough to handle qualifiers (``static``, ``inline``, ``LV_ATTRIBUTE_FAST_MEM``,
# pointer return types, multi-line argument lists). Doesn't try to parse
# every C oddity; misses are fine - the chunk just falls back to a generic
# "globals" heading.
_C_FUNC_DECL_RE = re.compile(
    r"""
    ^                                # line start
    (?:[A-Za-z_][\w\s\*]*?\s+)?      # optional return type / qualifiers (lazy)
    ([a-z_][a-z0-9_]*)               # function name (LVGL is snake_case)
    \s*\(                            # opening paren
    [^;{]*?                          # args (no ; or { on the way to close)
    \)                               # closing paren
    \s*                              #
    (?:LV_[A-Z_]+)?                  # optional trailing macro (LV_ATTRIBUTE_*)
    \s*\n?\s*                        #
    \{                               # function body opens
    """,
    re.MULTILINE | re.VERBOSE,
)


def chunk_c_source(text: str) -> List[Tuple[str, str]]:
    """Split a C source / header into ``[(heading, chunk_text), ...]``.

    Strategy: ~50-line windows with 8 lines of overlap. For each window
    the "heading" is the most-recent function name visible in or just
    before the window (so a chunk in the middle of ``lv_button_create``
    gets ``lv_button_create`` as its heading). For declarations-only
    files (most ``.h``) and code outside any function, the heading is
    just an empty string.

    Why not split on function boundaries? Because (a) C parsing in regex
    is hard, (b) some LVGL functions are 200+ lines, which would blow
    past the embedding model's context, and (c) LVGL has plenty of
    code-outside-functions: macros, typedefs, X-macro tables, static
    init arrays. Line windows handle all of these uniformly while the
    heading-extraction regex still gives the chunks useful provenance.
    """
    if not text.strip():
        return []
    lines = text.split("\n")
    chunks: List[Tuple[str, str]] = []
    n = len(lines)

    def _emit(heading: str, chunk: str) -> None:
        # Guard against pathologically dense files (font bitmap arrays,
        # X-macro tables): hard-slice anything over C_CHUNK_MAX_CHARS so
        # we never blow past the embedding model's training context or
        # the server's ubatch.
        if len(chunk) <= C_CHUNK_MAX_CHARS:
            chunks.append((heading, chunk))
            return
        i = 0
        while i < len(chunk):
            chunks.append((heading, chunk[i:i + C_CHUNK_MAX_CHARS]))
            i += C_CHUNK_MAX_CHARS

    if n <= C_CHUNK_LINES:
        heading = _enclosing_func(text, len(text))
        _emit(heading, text)
        return chunks

    step = max(1, C_CHUNK_LINES - C_CHUNK_OVERLAP_LINES)
    i = 0
    while i < n:
        end = min(i + C_CHUNK_LINES, n)
        chunk = "\n".join(lines[i:end])

        # Build the search context: this chunk plus ~30 lines preceding
        # it, so a function header that opens just above the window
        # still becomes the heading.
        ctx_start = max(0, i - 30)
        ctx = "\n".join(lines[ctx_start:end])
        heading = _enclosing_func(ctx, len(ctx))

        _emit(heading, chunk)

        if end == n:
            break
        i += step
    return chunks


def _enclosing_func(text: str, upto: int) -> str:
    """Return the last function name whose opening line appears in
    ``text[:upto]``, or '' if none."""
    last = ""
    for m in _C_FUNC_DECL_RE.finditer(text, 0, upto):
        last = m.group(1)
    return last


# ---------------------------------------------------------------------------
# Generic chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, *, target: int = CHUNK_TARGET_CHARS,
               overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Greedy paragraph-aware splitter with character-count caps."""
    if len(text) <= target:
        return [text]

    # Prefer breaking on blank lines, then sentences, then hard cuts.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= target:
            buf += "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    # Hard-split any single paragraph that's still too long.
    flat: List[str] = []
    for c in chunks:
        if len(c) <= target * 1.5:
            flat.append(c)
            continue
        i = 0
        while i < len(c):
            flat.append(c[i : i + target])
            i += max(1, target - overlap)
    return flat


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass
class BuildStats:
    files_seen: int = 0
    files_md: int = 0
    files_c: int = 0
    chunks_inserted: int = 0
    chunks_md: int = 0
    chunks_c: int = 0
    skipped_empty: int = 0


def _file_chunks(path: Path, raw: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Return ``(title, [(heading, chunk_text), ...])`` for a single file.

    Dispatches by suffix so this is the *only* place where the two
    chunking pipelines (MDX-aware vs. C-line-window) coexist.
    """
    ext = path.suffix.lower()
    if ext in {".md", ".mdx"}:
        parsed = parse_frontmatter(raw)
        body = clean_mdx(parsed.body)
        if not body.strip():
            return parsed.title, []
        chunks: List[Tuple[str, str]] = []
        for breadcrumb, section in split_by_headings(body):
            for chunk in chunk_text(section):
                chunks.append((breadcrumb, chunk))
        return parsed.title, chunks

    if ext in {".c", ".h"}:
        # No frontmatter or title for source - use the file basename
        # so the chat agent's tool result still has a recognizable
        # human-readable label.
        return path.name, chunk_c_source(raw)

    return "", []


def build(
    db_path: str,
    source: str,
    embed_client: EmbeddingClient,
    *,
    docs_subdir: Optional[str] = None,
) -> BuildStats:
    """Clone (if needed), embed, and store the LVGL corpus.

    The corpus is whatever the sparse-checkout materializes (defaults
    to ``docs/``, ``src/`` and ``examples/``) restricted by file
    extension to ``.md`` / ``.mdx`` / ``.c`` / ``.h``. A single
    destructive build replaces any prior store at ``db_path``.
    """
    working_tree, walk_root, origin = resolve_walk_root(source)
    if docs_subdir:
        walk_root = str(Path(working_tree) / docs_subdir)

    # Discover the embedding dim with one probe call so the store can size
    # its vec0 table correctly on the very first insert.
    dim = embed_client.probe_dim()

    store = VectorStore(db_path, dim=dim)
    store.clear()  # build is destructive: same name => fresh corpus.

    commit = get_git_commit(working_tree)
    store.set_meta(
        model_name=embed_client.model,
        model_family=embed_client.model_family,
        embed_endpoint=embed_client.base_url,
        dim=dim,
        source_url=origin,
        source_commit=commit,
        walk_root=walk_root,
        sparse_paths=",".join(DEFAULT_SPARSE_PATHS),
        file_exts=",".join(sorted(SOURCE_FILE_EXTS)),
        exclude_prefixes=",".join(DEFAULT_EXCLUDE_PREFIXES),
    )

    stats = BuildStats()
    pending_rows: List[ChunkRow] = []
    pending_texts: List[str] = []

    def flush() -> None:
        if not pending_rows:
            return
        vectors = embed_client.embed_documents(pending_texts)
        store.add_batch(pending_rows, vectors)
        stats.chunks_inserted += len(pending_rows)
        pending_rows.clear()
        pending_texts.clear()

    for path in iter_source_files(walk_root):
        stats.files_seen += 1
        rel = str(path.relative_to(working_tree))
        is_code = path.suffix.lower() in {".c", ".h"}
        if is_code:
            stats.files_c += 1
        else:
            stats.files_md += 1

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"[ingest] skip {rel}: {exc}", file=sys.stderr)
            continue

        title, file_chunks = _file_chunks(path, raw)
        if not file_chunks:
            stats.skipped_empty += 1
            continue

        for breadcrumb, chunk in file_chunks:
            pending_rows.append(
                ChunkRow(
                    path=rel,
                    heading=breadcrumb,
                    title=title,
                    text=chunk,
                )
            )
            pending_texts.append(chunk)
            if is_code:
                stats.chunks_c += 1
            else:
                stats.chunks_md += 1
            if len(pending_rows) >= EMBED_BATCH:
                flush()
                print(
                    f"[ingest] embedded {stats.chunks_inserted} chunks "
                    f"from {stats.files_seen} files "
                    f"(md/mdx={stats.files_md}, c/h={stats.files_c})",
                    file=sys.stderr,
                )

    flush()
    store.set_meta(
        built_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        chunks_md=stats.chunks_md,
        chunks_c=stats.chunks_c,
        files_md=stats.files_md,
        files_c=stats.files_c,
    )
    store.close()
    return stats


__all__ = [
    "build",
    "clone_or_pull",
    "resolve_walk_root",
    "resolve_docs_root",
    "iter_source_files",
    "iter_doc_files",
    "clean_mdx",
    "parse_frontmatter",
    "split_by_headings",
    "chunk_text",
    "chunk_c_source",
    "BuildStats",
    "DEFAULT_SOURCE_URL",
    "DEFAULT_SPARSE_PATHS",
    "DEFAULT_EXCLUDE_PREFIXES",
    "SOURCE_FILE_EXTS",
]
