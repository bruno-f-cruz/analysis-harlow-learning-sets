"""Presentation-ready matplotlib styling: dark background, copy/paste-to-slide figures.

Where :func:`analysis.plotting.a_lot_of_style` targets the production pipeline's
white-background figures, this module targets exploratory plots meant to be
pasted straight into a dark-background slide deck, exported via :func:`savefig`
as either an opaque-black PNG or a background-free SVG. The palette is the
dark-mode half of the data-viz skill's validated categorical palette (8 hues,
fixed order, never cycled), re-validated here against a *pure* black surface
(`#000000` rather than the skill's `#1a1a19` reference dark surface) since
that's what a slide background actually is. `ANIMAL_COLORS`/`animal_palette`
give a second, separately-validated fixed palette (Allen Institute brand
hues where they clear the same gates, on a per-animal basis) for whenever a
plot splits by subject_id.
"""

from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette -- dark-mode categorical hues from the data-viz skill's reference
# palette, order fixed (validated for adjacent + first-three-all-pairs CVD
# separation; see the skill's references/palette.md). Never cycle past this
# list -- fold extra series into "Other" or facet instead.
# ---------------------------------------------------------------------------

SURFACE = "#000000"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#333333"
DIVERGING_MIDPOINT = "#383835"

_HUE_NAMES = [
    "blue",
    "orange",
    "aqua",
    "yellow",
    "magenta",
    "green",
    "violet",
    "red",
]
CATEGORICAL_COLORS = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
]


def categorical(n: int | None = None) -> list[str]:
    """First `n` categorical colors, in the fixed (non-cyclic) validated order.

    Raises past 8 -- cycling back to slot 1 breaks the CVD-safe adjacent
    ordering the palette was validated against, so a 9th series belongs in
    "Other" or a facet, not a repeated color.
    """
    if n is None:
        return list(CATEGORICAL_COLORS)
    if n > len(CATEGORICAL_COLORS):
        raise ValueError(
            f"only {len(CATEGORICAL_COLORS)} validated categorical colors "
            f"available, got n={n} -- fold extra series into 'Other' or facet "
            "instead of cycling"
        )
    return CATEGORICAL_COLORS[:n]


# ---------------------------------------------------------------------------
# Per-animal identity palette -- Allen Institute brand hues (from the Brand
# Center palette), re-picked and re-validated the same way as
# `CATEGORICAL_COLORS`: most brand hues are too light to clear the dark
# lightness band, and once darkened enough to pass, violet/orange/lime turn
# out to be mutually CVD-unsafe with crimson and with each other on a pure
# black surface (no combination of them gets past 4 slots). The 4 that
# survive are Allen brand hues; the 5th (amber) is borrowed from
# `CATEGORICAL_COLORS` since no 5th brand hue clears the gate alongside the
# other four -- validated together as a set of 5, all-pairs, zero warnings.
#
# Hard-coded by subject_id rather than assigned by sort order: a plot over a
# subset of animals (a toggle, an exclusion like 864845 in the counterfactual
# cohort plots) must not shift everyone else's color -- 864845 is magenta
# whether or not it's the animal being dropped that day. Add new subjects
# here explicitly rather than growing this dynamically.
# ---------------------------------------------------------------------------

ANIMAL_COLORS = {
    "841299": "#01A49B",  # teal        (Allen brand teal, base)
    "841312": "#6464FF",  # periwinkle  (Allen brand periwinkle, base)
    "864845": "#BF00BF",  # magenta     (Allen brand magenta, mid shade -- base too light for black)
    "864846": "#CD0F55",  # crimson     (Allen brand crimson, base)
    "866063": "#c98500",  # amber       (borrowed from CATEGORICAL_COLORS's "yellow" slot)
}


def animal_palette(subject_ids=None) -> dict[str, str]:
    """Hard-coded color per animal, keyed by `subject_id`.

    Pass a subset of `ANIMAL_COLORS`'s keys to get just those animals' colors
    (e.g. for a plot that already filtered to certain subjects); omit it to
    get the full mapping. Raises on any subject_id without a hard-coded
    color -- add it to `ANIMAL_COLORS` above rather than falling back to an
    unvalidated color.
    """
    if subject_ids is None:
        return dict(ANIMAL_COLORS)
    ids = [str(s) for s in subject_ids]
    unknown = sorted(set(ids) - ANIMAL_COLORS.keys())
    if unknown:
        raise ValueError(
            f"no hard-coded color for subject_id(s) {unknown} -- add them to "
            "ANIMAL_COLORS in analysis.plotting_style"
        )
    return {sid: ANIMAL_COLORS[sid] for sid in ids}


def sequential_cmap(hue: str = "blue") -> LinearSegmentedColormap:
    """Single-hue colormap from `SURFACE` (low) to that categorical hue (high).

    The dark-background mirror of the skill's light-surface light->dark
    ramps: magnitude recedes toward black instead of toward white.
    """
    hex_by_hue = dict(zip(_HUE_NAMES, CATEGORICAL_COLORS))
    if hue not in hex_by_hue:
        raise ValueError(f"hue must be one of {_HUE_NAMES}, got {hue!r}")
    return LinearSegmentedColormap.from_list(
        f"presentation_{hue}", [SURFACE, hex_by_hue[hue]]
    )


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue<->red diverging colormap with a neutral gray midpoint.

    For polarity encodings (e.g. a p(leave) heatmap) -- never a hue at the
    midpoint, per the data-viz skill.
    """
    blue, red = CATEGORICAL_COLORS[0], CATEGORICAL_COLORS[-1]
    return LinearSegmentedColormap.from_list(
        "presentation_diverging", [blue, DIVERGING_MIDPOINT, red]
    )


# ---------------------------------------------------------------------------
# Consistent sizes / ratios -- every figure in a deck should share one of a
# few fixed scales rather than each plot inventing its own, so pasted PNGs
# line up.
# ---------------------------------------------------------------------------

FIGSIZE = {
    "standard": (7.0, 5.0),  # single chart
    "wide": (10.0, 4.5),  # full-width slide panel / timeseries
    "square": (6.0, 6.0),  # heatmaps, small multiples
    "tall": (5.5, 8.0),  # stacked panels
}


def new_figure(kind: str = "standard", **subplot_kwargs):
    """`plt.subplots` at one of the fixed `FIGSIZE` presentation sizes.

    Use inside :func:`presentation_style` so the sizing is paired with the
    dark styling. `subplot_kwargs` forwards to `plt.subplots` (e.g. `ncols`,
    `nrows`, `sharey`); `figsize` cannot be overridden this way -- pass a
    different `kind`, or scale a preset explicitly, instead.
    """
    if kind not in FIGSIZE:
        raise ValueError(f"kind must be one of {list(FIGSIZE)}, got {kind!r}")
    return plt.subplots(figsize=FIGSIZE[kind], **subplot_kwargs)


# ---------------------------------------------------------------------------
# Style context manager
# ---------------------------------------------------------------------------


@contextmanager
def presentation_style(
    font_scale: float = 1.3, line_width: float = 2.0, grid: bool = False
):
    """Dark-background rcParams for presentation/copy-paste PNGs.

    Mirrors :func:`analysis.plotting.a_lot_of_style`'s shape (a restorable
    rcParams context manager) but themed for a black slide background:
    opaque black figure/axes, light ink text, no gridlines by default (pass
    `grid=True` for the rare plot that needs one -- a recessive solid, never
    dashed, hairline when it's on), and the fixed `CATEGORICAL_COLORS` order
    wired in as the default color cycle.
    """
    old_params = plt.rcParams.copy()
    plt.style.use("default")
    plt.rcParams.update(
        {
            # Fonts
            "font.family": "DejaVu Sans",
            "font.size": 11 * font_scale,
            "axes.titlesize": 14 * font_scale,
            "axes.labelsize": 12 * font_scale,
            "xtick.labelsize": 10 * font_scale,
            "ytick.labelsize": 10 * font_scale,
            "legend.fontsize": 10 * font_scale,
            # Lines and markers
            "lines.linewidth": line_width,
            "lines.markersize": 8,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "axes.prop_cycle": plt.cycler(color=CATEGORICAL_COLORS),
            # Surfaces and ink
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.edgecolor": SURFACE,
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_PRIMARY,
            "axes.titlecolor": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.edgecolor": INK_MUTED,
            # Grid -- recessive, solid hairline (never dashed)
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": grid,
            "grid.color": GRIDLINE,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            # Legend -- no box; text in ink, identity carried by the mark color
            "legend.frameon": False,
            "legend.labelcolor": INK_PRIMARY,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    try:
        yield
    finally:
        plt.rcParams.update(old_params)


def savefig(fig, path, **kwargs):
    """Save `fig` as a presentation-ready PNG or SVG (picked from `path`'s suffix).

    PNG defaults to opaque black: `facecolor` is pinned to the figure's own
    (black) facecolor and `transparent` is forced off, guarding the common
    gotcha where a figure looks dark on screen but exports with a stray white
    background. SVG defaults the other way -- `transparent=True`, no
    facecolor -- since a vector figure is meant to drop onto whatever slide
    background already exists rather than carry its own black rectangle
    (data colors and white/gray ink are unaffected either way; pass
    `transparent=False` to force an opaque SVG instead).
    """
    kwargs.setdefault("dpi", plt.rcParams["savefig.dpi"])
    kwargs.setdefault("bbox_inches", "tight")
    if str(path).lower().endswith(".svg"):
        kwargs.setdefault("transparent", True)
    else:
        kwargs.setdefault("facecolor", fig.get_facecolor())
        kwargs.setdefault("transparent", False)
    fig.savefig(path, **kwargs)
