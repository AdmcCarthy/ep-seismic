"""Plot a :class:`~ep_seismic.volume.section.Section`.

Axes are labelled from the section's domain, so a time section reads TWT [ms] and
a depth section Depth [m]; the colour bar carries the quantity and unit. A caption
on the figure says how the domain was decided and how the section was made.
"""

from __future__ import annotations

import textwrap

import numpy as np

from ep_seismic.volume.section import Section

_XAXES = ("distance", "trace", "inline", "crossline")


def plot_section(
    section: Section,
    ax=None,
    *,
    xaxis: str = "distance",
    cmap: str | None = None,
    clip: float = 99.0,
    attribution: str | None = None,
    caption: bool = True,
):
    """Draw the section as an image and return the matplotlib ``Axes``.

    Amplitudes use a symmetric grey scale clipped at the ``clip`` percentile of
    ``|amplitude|``; other quantities (velocity, impedance, ...) use viridis
    between the ``100 - clip`` and ``clip`` percentiles. ``xaxis`` is
    ``"distance"``, ``"trace"``, ``"inline"`` or ``"crossline"``. ``attribution``
    (for example a data licence credit) is added to the caption.

    Raises
    ------
    ValueError
        For a single-trace section, an unknown ``xaxis`` or ``clip``, or an inline
        / crossline axis on a section where that number does not vary.
    """
    import matplotlib.pyplot as plt

    if section.n_traces < 2:
        raise ValueError("need at least two traces to plot a section")
    if xaxis not in _XAXES:
        raise ValueError(f"xaxis must be one of {_XAXES}")
    if not 50 < clip <= 100:
        raise ValueError("clip must be a percentile in (50, 100]")

    if xaxis == "distance":
        xs, xlabel = section.distance, "Distance along section [coordinate units]"
    elif xaxis == "trace":
        xs, xlabel = np.arange(section.n_traces, dtype=float), "Trace in section"
    else:
        xs = getattr(section, xaxis).astype(float)
        xlabel = xaxis.capitalize()
        if xs.min() == xs.max():
            raise ValueError(f"{xaxis} does not vary along this section")

    dom = section.domain
    finite = section.data[np.isfinite(section.data)]
    if finite.size == 0:
        raise ValueError("section has no finite samples")
    if dom.quantity == "amplitude":
        peak = float(np.percentile(np.abs(finite), clip)) or 1.0
        vmin, vmax = -peak, peak
        cmap = cmap or "gray"
    else:
        vmin, vmax = (float(v) for v in np.percentile(finite, [100 - clip, clip]))
        cmap = cmap or "viridis"

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    img = ax.imshow(
        section.data.T,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(xs[0], xs[-1], section.samples[-1], section.samples[0]),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(dom.vertical_label)
    ax.set_title(section.kind)
    cbar = ax.figure.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label(dom.quantity_label)
    if caption:
        text = section.caption()
        if attribution:
            text += f" | {attribution}"
        ax.figure.text(
            0.01,
            0.005,
            "\n".join(textwrap.wrap(text, 140)),
            fontsize=7,
            va="bottom",
            ha="left",
        )
        ax.figure.subplots_adjust(bottom=0.2)
    return ax
