"""Render a function-level CPU heatmap (treemap) PNG from perf data.

Used by the ``perf_heatmap`` MCP tool in :mod:`systemd_mcp.server`: given a
list of hot functions (each ``{"symbol", "dso", "percent", ...}`` as produced
by ``perf_top_functions``) it lays them out as a *squarified treemap* where a
tile's area is proportional to that function's CPU self-overhead and its color
runs a blue(cold) -> red(hot) heat scale.

The squarify layout is the classic Bruls/Huizing/van Wijk algorithm (same one
the ``squarify`` PyPI package implements); it is inlined here to avoid pulling
an extra dependency. Rendering uses matplotlib's Agg backend, so this works
headless. matplotlib is the only non-stdlib requirement; the caller
(``perf_heatmap``) turns a missing import into a soft-fail.

Standalone use::

    perf report -i perf.data --stdio -g none | ...   # or:
    python -m systemd_mcp.perf_heatmap functions.json out.png
"""

from __future__ import annotations

import json
import sys
from typing import Any, Sequence

# Heat gradient stops (cold -> hot), reused for both the tiles and the legend
# colorbar so they always agree. RGB in 0..1.
_HEAT_STOPS = [
    (0.13, 0.20, 0.52),  # deep blue   - cold
    (0.13, 0.56, 0.80),  # cyan-blue
    (0.15, 0.66, 0.38),  # green
    (0.98, 0.80, 0.20),  # amber
    (0.85, 0.15, 0.13),  # red         - hot
]


# --- squarified treemap layout ------------------------------------------------
# ``sizes`` are areas that already sum to ``dx * dy``. Each returned rect is a
# dict with ``x, y, dx, dy``. Ported from the MIT-licensed ``squarify`` module.


def _normalize_sizes(sizes: Sequence[float], dx: float, dy: float) -> list[float]:
    total_size = float(sum(sizes))
    total_area = float(dx * dy)
    if total_size <= 0:
        return [0.0 for _ in sizes]
    return [s * total_area / total_size for s in sizes]


def _layoutrow(sizes, x, y, dy):
    covered = sum(sizes)
    width = covered / dy
    rects = []
    for size in sizes:
        rects.append({"x": x, "y": y, "dx": width, "dy": size / width})
        y += size / width
    return rects


def _layoutcol(sizes, x, y, dx):
    covered = sum(sizes)
    height = covered / dx
    rects = []
    for size in sizes:
        rects.append({"x": x, "y": y, "dx": size / height, "dy": height})
        x += size / height
    return rects


def _layout(sizes, x, y, dx, dy):
    return _layoutrow(sizes, x, y, dy) if dx >= dy else _layoutcol(sizes, x, y, dx)


def _leftoverrow(sizes, x, y, dx, dy):
    width = sum(sizes) / dy
    return (x + width, y, dx - width, dy)


def _leftovercol(sizes, x, y, dx, dy):
    height = sum(sizes) / dx
    return (x, y + height, dx, dy - height)


def _leftover(sizes, x, y, dx, dy):
    return (
        _leftoverrow(sizes, x, y, dx, dy)
        if dx >= dy
        else _leftovercol(sizes, x, y, dx, dy)
    )


def _worst_ratio(sizes, x, y, dx, dy):
    return max(
        max(r["dx"] / r["dy"], r["dy"] / r["dx"])
        for r in _layout(sizes, x, y, dx, dy)
    )


def _squarify(sizes, x, y, dx, dy):
    sizes = [float(s) for s in sizes]
    if not sizes:
        return []
    if len(sizes) == 1:
        return _layout(sizes, x, y, dx, dy)

    i = 1
    while i < len(sizes) and _worst_ratio(sizes[:i], x, y, dx, dy) >= _worst_ratio(
        sizes[: i + 1], x, y, dx, dy
    ):
        i += 1

    current, remaining = sizes[:i], sizes[i:]
    lx, ly, ldx, ldy = _leftover(current, x, y, dx, dy)
    return _layout(current, x, y, dx, dy) + _squarify(remaining, lx, ly, ldx, ldy)


# --- rendering ----------------------------------------------------------------


def _short_dso(dso: str) -> str:
    """Trim a shared-object path/decoration to something legend-friendly."""
    dso = dso.strip()
    if dso.startswith("[") and dso.endswith("]"):
        return dso
    return dso.rsplit("/", 1)[-1]


def render_heatmap(
    functions: Sequence[dict[str, Any]],
    out_path: str,
    title: str = "perf function heatmap",
    width: int = 1600,
    height: int = 900,
    dpi: int = 100,
) -> str:
    """Render ``functions`` as a treemap heatmap PNG at ``out_path``.

    ``functions`` is a list of dicts with at least ``percent`` (float, CPU
    self-overhead %) and ``symbol`` (str); ``dso`` is used in labels when
    present. Returns ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    funcs = [
        f for f in functions if float(f.get("percent", 0) or 0) > 0
    ]
    if not funcs:
        raise ValueError("no functions with a positive percent to render")
    funcs.sort(key=lambda f: float(f["percent"]), reverse=True)

    cmap = LinearSegmentedColormap.from_list("perfheat", _HEAT_STOPS)
    percents = [float(f["percent"]) for f in funcs]
    max_pct = max(percents)
    norm = Normalize(vmin=0.0, vmax=max_pct)

    canvas_w, canvas_h = 100.0, 100.0
    areas = _normalize_sizes(percents, canvas_w, canvas_h)
    rects = _squarify(areas, 0.0, 0.0, canvas_w, canvas_h)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    for f, r in zip(funcs, rects):
        pct = float(f["percent"])
        color = cmap(norm(pct))
        ax.add_patch(
            Rectangle(
                (r["x"], r["y"]),
                r["dx"],
                r["dy"],
                facecolor=color,
                edgecolor="white",
                linewidth=1.2,
            )
        )
        # Only label tiles big enough to fit readable text.
        if r["dx"] < 7 or r["dy"] < 5:
            continue
        sym = str(f.get("symbol", "")).strip()
        if len(sym) > 34:
            sym = sym[:31] + "..."
        dso = _short_dso(str(f.get("dso", "")))
        label = f"{sym}\n{pct:.1f}%"
        if dso and r["dy"] >= 9:
            label = f"{sym}\n{dso}\n{pct:.1f}%"
        # White text on hot (dark red) tiles, near-black on the light amber
        # midrange; luminance of the tile color decides.
        r_, g_, b_ = color[:3]
        lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
        txt_color = "white" if lum < 0.55 else "#111111"
        fontsize = max(6.0, min(12.0, r["dx"] * 0.28))
        ax.text(
            r["x"] + r["dx"] / 2,
            r["y"] + r["dy"] / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=txt_color,
            family="monospace",
            wrap=True,
        )

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.invert_yaxis()  # first (hottest) tile at top-left
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("CPU self-overhead (%)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_functions(source: str) -> list[dict[str, Any]]:
    with open(source, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "functions" in data:
        return list(data["functions"])
    if isinstance(data, list):
        return list(data)
    raise ValueError("expected a JSON list or an object with a 'functions' key")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print(
            "usage: python -m systemd_mcp.perf_heatmap <functions.json> "
            "<out.png> [title]",
            file=sys.stderr,
        )
        return 2
    src, out = args[0], args[1]
    title = args[2] if len(args) > 2 else "perf function heatmap"
    funcs = _load_functions(src)
    path = render_heatmap(funcs, out, title=title)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
