import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

plt.rcParams['svg.fonttype'] = 'none'

def colored_lineplot(x, y, color, cmap="viridis", linewidth=2, ax=None, vmin=None, vmax=None):
    """
    Plot a line whose segments are individually colored.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the line.
    color : array-like or list
        - If numeric → interpreted through a colormap
        - If list of colors → used directly for each segment
    cmap : str or Colormap
        Colormap to use (if color is numeric)
    linewidth : float
        Width of line segments
    colorbar : bool
        Whether to draw a colorbar
    ax : matplotlib Axes (optional)
        Provide an existing axes object

    Returns
    -------
    lc : LineCollection
        The line collection object
    ax : matplotlib Axes
        The axes used
    """

    x = np.asarray(x)
    y = np.asarray(y)
    color = np.asarray(color)

    # Prepare Axes
    if ax is None:
        fig, ax = plt.subplots()

    # Build segments for LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Determine if "color" is numeric or actual color strings
    if np.issubdtype(color.dtype, np.number):
        vmin = vmin if not np.isnan(vmin) else color.min()
        vmax = vmax if not np.isnan(vmax) else color.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        lc = LineCollection(
            segments, cmap=cmap, norm=norm, linewidths=linewidth
        )
        lc.set_array(color[:-1])  # one color per segment

        ax.add_collection(lc)

    else:
        # Direct list of colors
        lc = LineCollection(
            segments, colors=color[:-1], linewidths=linewidth
        )
        ax.add_collection(lc)

    # Autoscale plot to fit line
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return lc


def rad2deg_ax(ax, n_ticks=5, axis='y'):
    # n_ticks is approximate and may slightly vary due to rounding
    ddeg = np.ceil(np.rad2deg(np.diff(ax.get_ylim())[0]) / n_ticks)
    if ddeg>5:
        ddeg = np.floor(ddeg/5)*5
    elif ddeg>25:
        ddeg = np.floor(ddeg/10)*10
    else:
        ddeg = np.ceil(ddeg)

    #TODO: Calc from values
    deg_range = (-360, 361)

    deg_ticks = np.arange(*deg_range, ddeg)  # degrees you want to show
    rad_ticks = np.radians(deg_ticks)

    if axis=='y':
        axis = ax.yaxis
    elif axis=='x':
        axis = ax.xaxis
    else:
        raise ValueError("axis must be either 'y' or 'x'")

    axis.set_major_locator(FixedLocator(rad_ticks))
    axis.set_major_formatter(FuncFormatter(lambda rad, _: f"{np.degrees(rad):.0f}"))


def scale_ax(ax, factor, axis='x'):
    if axis=='y':
        axis = ax.yaxis
    elif axis=='x':
        axis = ax.xaxis
    else:
        raise ValueError("axis must be either 'y' or 'x'")

    axis.set_major_formatter(FuncFormatter(lambda t, _: f"{t * factor:.1f}"))

def tidy_ax(ax,
            svg_fonttype='none',
            adjust_lims=True,  # Adjust xlim and ylim to full ticks
            remove_spines=('top', 'right'),
            box=True,  # Make axes square
            ):
    # Applies some modification that are usually used for preparing BBO figures for publications

    if svg_fonttype is not None:
        plt.rcParams['svg.fonttype'] = svg_fonttype

    if remove_spines is 'all':
        remove_spines = ax.spines.keys()
    for spine in remove_spines:
        ax.spines[spine].set_visible(False)

    if adjust_lims:
        if isinstance(adjust_lims, bool):
            adjust_lims = ((None, None),(None, None),)

        tickdiff = np.median(np.diff(ax.get_xticks()))
        xlim = np.array(ax.get_xlim())
        xlim[0] = np.floor(xlim[0]/tickdiff)*tickdiff if adjust_lims[0][0] is None else adjust_lims[0][0]
        xlim[1] = np.ceil(xlim[1]/tickdiff)*tickdiff if adjust_lims[0][1] is None else adjust_lims[0][1]
        if len(adjust_lims[0])>2:
            tickdiff = adjust_lims[0][2]
        ax.set_xticks(np.arange(xlim[0], xlim[1]+1, tickdiff))
        ax.set_xlim(xlim)

        tickdiff = np.median(np.diff(ax.get_yticks()))
        ylim = np.array(ax.get_ylim())
        ylim[0] = np.floor(ylim[0] / tickdiff) * tickdiff if adjust_lims[1][0] is None else adjust_lims[1][0]
        ylim[1] = np.ceil(ylim[1] / tickdiff) * tickdiff if adjust_lims[1][1] is None else adjust_lims[1][1]
        if len(adjust_lims[1]) > 2:
            tickdiff = adjust_lims[1][2]
        ax.set_yticks(np.arange(ylim[0], ylim[1]+1, tickdiff))
        ax.set_ylim(ylim)

    if box:
        ax.set_box_aspect(1)


def equalize_limits(ax=None, force_lower=None, force_upper=None):
    """
    Set xlim and ylim on the given axes so both cover the same span,
    using the larger of the two current spans. Keeps each axis centered.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if force_lower is not None:
        lower = force_lower
    else:
        lower = min(xmin, ymin)

    if force_upper is not None:
        upper = force_upper
    else:
        upper = max(xmax, ymax)

    ax.set_xlim((lower, upper))
    ax.set_ylim((lower, upper))

    return ax
