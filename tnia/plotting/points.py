"""Render point lists (for example single-molecule localizations) into images, so the projection and
slab plots in ``tnia.plotting.projections`` and ``tnia.plotting.plt_helper`` can show them.

Points come in as ``(N, 3)`` arrays in ``(x, y, z)`` order, the usual order for localization tables.
Volumes come out in tnia's image order, ``[z, y, x]``, and 2D images as ``[y, x]``. Each pixel holds
the number of points in it (a histogram); no blurring.

    from tnia.plotting.points import points_to_volume
    from tnia.plotting.projections import show_xyz_projection

    vol, bounds = points_to_volume(xyz, voxel_size=(20, 20))          # 20 nm in xy, 20 nm in z
    show_xyz_projection(vol, sxy=20, sz=20, projector=np.sum, gamma=0.5)
"""
import numpy as np


def _bounds(values, lo_hi):
    return (float(values.min()), float(values.max())) if lo_hi is None else (float(lo_hi[0]), float(lo_hi[1]))


def points_to_volume(xyz, voxel_size, bounds=None, weights=None):
    """Histogram 3D points into a volume indexed ``[z, y, x]``.

    Args:
        xyz (array (N, 3)): point positions in ``(x, y, z)`` order.
        voxel_size (float or (xy, z)): voxel edge length, in the units of ``xyz``.
        bounds (sequence of 3 (lo, hi) pairs, optional): the ``(x, y, z)`` ranges to render; ``None`` for
            any axis (or for all) uses the points' own min and max. Points outside are left out.
        weights (array (N,), optional): a weight per point instead of 1.

    Returns:
        volume (3d numpy array, ``[z, y, x]``), bounds (the three ``(lo, hi)`` ranges used).
    """
    xyz = np.asarray(xyz)
    sxy, sz = (voxel_size, voxel_size) if np.isscalar(voxel_size) else voxel_size
    bounds = [None] * 3 if bounds is None else list(bounds)
    rng = [_bounds(xyz[:, k], bounds[k]) for k in range(3)]
    edges = [np.arange(lo, hi + s, s) for (lo, hi), s in zip(rng, (sxy, sxy, sz))]
    h, _ = np.histogramdd(xyz[:, :3], bins=edges, weights=weights)
    return np.transpose(h, (2, 1, 0)), rng                      # (x, y, z) -> [z, y, x]


def points_to_image(xy, pixel_size, bounds=None, weights=None):
    """Histogram 2D points into an image indexed ``[y, x]``.

    Args:
        xy (array (N, 2) or more columns): point positions; the first two columns are used as ``(x, y)``.
        pixel_size (float or (x, y)): pixel edge length, in the units of ``xy``; two values for different x and y.
        bounds (sequence of 2 (lo, hi) pairs, optional): the ``(x, y)`` ranges; ``None`` uses the points'
            own min and max.
        weights (array (N,), optional): a weight per point instead of 1.

    Returns:
        image (2d numpy array, ``[y, x]``), bounds (the two ``(lo, hi)`` ranges used).
    """
    xy = np.asarray(xy)
    bounds = [None] * 2 if bounds is None else list(bounds)
    rng = [_bounds(xy[:, k], bounds[k]) for k in range(2)]
    px = (pixel_size, pixel_size) if np.isscalar(pixel_size) else pixel_size
    edges = [np.arange(lo, hi + p, p) for (lo, hi), p in zip(rng, px)]
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=edges, weights=weights)
    return h.T, rng


def range_to_slice(axis_bounds, voxel, lo, hi):
    """The voxel slice that covers ``[lo, hi)`` along one axis of an image made by ``points_to_volume`` or
    ``points_to_image`` (``axis_bounds`` is that axis's ``(lo, hi)`` from the returned bounds), for example
    to pick a slab for ``show_xyz_projection_slabs``."""
    i0 = int(np.floor((lo - axis_bounds[0]) / voxel))
    i1 = int(np.ceil((hi - axis_bounds[0]) / voxel))
    return slice(max(i0, 0), max(i1, 0))


# ---- wide lines on a projection (line profiles through points) ---------------------------------------------
from collections import namedtuple

# view -> (horizontal axis, vertical axis, slab axis), as columns of an (x, y, z) point array
_VIEW_AXES = {'xy': (0, 1, 2), 'xz': (0, 2, 1), 'yz': (1, 2, 0)}


class LineROI(namedtuple('LineROI', 'points index length line_width view slab')):
    """The points inside a wide line (see ``points_line_roi``), in the line's own frame."""
    __slots__ = ()

    def take(self, rows):
        """The same ROI with only these rows of ``points`` (for example the best N by some score)."""
        return self._replace(points=self.points[rows], index=self.index[rows])


def points_line_roi(points, start, end, line_width, view='xy', slab=None):
    """The points inside a wide line drawn on a projection, as a wide line in ImageJ, in the line's own frame.

    The line is drawn on one projection (``view``) of the points, from ``start`` to ``end`` (2D, in that view's
    coordinates), ``line_width`` wide (full width, in the view's plane). ``slab`` limits the axis the view projects
    over (the projection's thickness). Together they make a box: the line's length x line_width x slab.

        view    start / end    slab along
        'xy'    (x, y)         z
        'xz'    (x, z)         y
        'yz'    (y, z)         x

    Args:
        points (array (N, 3)): positions, ``(x, y, z)``.
        start, end (2 numbers each): the line's ends, in the view's coordinates.
        line_width (float): full width of the line, across it, in the view's plane.
        view (str): the projection the line is drawn on, ``'xy'``, ``'xz'`` or ``'yz'``.
        slab ((lo, hi), optional): range of the projected axis to keep; ``None`` keeps all.

    Returns:
        LineROI: ``points`` (M, 3) in the line's frame: along (0 at ``start``, ``length`` at ``end``), across
        (within +-line_width / 2) and the slab coordinate (unchanged); ``index`` (the rows of ``points`` kept);
        ``length``, ``line_width``, ``view``, ``slab``. ``roi.points`` is a point set like any other: a second line
        drawn on its ``'xz'`` view (along x slab coordinate) profiles the first line's side view.
    """
    p = np.asarray(points, dtype=float)
    h, v, s = _VIEW_AXES[view]
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.hypot(*(end - start)))
    u = (end - start) / length
    rel = p[:, [h, v]] - start
    along, across = rel @ u, rel @ np.array([-u[1], u[0]])
    keep = (along >= 0) & (along < length) & (np.abs(across) < line_width / 2)
    if slab is not None:
        keep &= (p[:, s] >= slab[0]) & (p[:, s] < slab[1])
    index = np.flatnonzero(keep)
    return LineROI(np.c_[along[index], across[index], p[index, s]], index, length, float(line_width), view, slab)


def _render(renderer, xy, pixel_size, bounds):
    out = renderer(xy, pixel_size, bounds=bounds)
    return out[0] if isinstance(out, tuple) else out


def points_line_profile(roi, bin_size, renderer=points_to_image, values=None, statistic='mean'):
    """Profile along a wide line (``points_line_roi``), from its start: the ROI rendered as an image (along x
    across, ``bin_size`` pixels), summed across the line. Returns (bin centres, values).

    Args:
        roi (LineROI): from ``points_line_roi``.
        bin_size (float): pixel size along and across the line.
        renderer (callable): turns points into an image, with the interface of ``points_to_image``:
            ``renderer(xy, pixel_size, bounds=((x lo, hi), (y lo, hi))) -> image [y, x]`` (or ``(image, bounds)``).
            The default, ``points_to_image``, is a histogram: each point counts 1, so the profile is raw counts.
            Any other (for example a Gaussian blur) gives a smoothed profile; Poisson statistics on it (such as
            ``two_peak_dip``'s sigma) then no longer hold.
        values (array (N,), optional): a value per point, aligned with the points ``roi`` was cut from (for
            example each localization's uncertainty). The profile is then that ``statistic`` of the values in each
            bin along the line, instead of the rendered image (NaN where a bin is empty).
        statistic (str or callable): for ``values``: 'mean', 'median', ... (``scipy.stats.binned_statistic``).
    """
    edges = np.arange(0, roi.length + bin_size, bin_size)
    if values is not None:
        from scipy.stats import binned_statistic
        v = np.asarray(values)[roi.index]
        return (edges[:-1] + edges[1:]) / 2, binned_statistic(roi.points[:, 0], v, statistic=statistic, bins=edges)[0]
    image = _render(renderer, roi.points[:, :2], bin_size,
                    ((0., roi.length), (-roi.line_width / 2, roi.line_width / 2)))
    return (edges[:-1] + edges[1:]) / 2, image.sum(axis=0)


def show_line_profiles(sets, lines, bin_size=20., view='xz', bounds=None, measure=None, smooth=1., z_stretch=1.,
                       gamma=0.6, cmap='gray', line_colors=('cyan', 'orange', 'lime', 'magenta'), height_ratios=None,
                       figsize=None, suptitle=None, renderer=points_to_image, values=None, statistic='median',
                       values_label='value', values_color='C3'):
    """A projection of each point set with wide lines drawn on it, and the profile along each line below it: one
    column per set, the projection on top, then one row per line.

    Args:
        sets (dict): ``{name: points (N, 3)}``, or ``{name: LineROI}`` (its points, in its own frame, are used).
        lines (dict): ``{label: (start, end, line_width)}``, drawn on ``view`` (coordinates as in ``points_line_roi``).
        bin_size (float): pixel size of the projection, and bin size of the profiles.
        view (str): the projection shown and drawn on, ``'xy'``, ``'xz'`` or ``'yz'``.
        bounds (((lo, hi), (lo, hi)), optional): horizontal and vertical range of the projection, in the view's
            coordinates. Default: a LineROI's box (along 0..length; its slab, if any, or the points' range), or the
            points' range.
        measure (callable, optional): ``f(positions, counts) -> dict or None``, for example
            ``tnia.metrics.profiles.two_peak_dip``; its ``'peaks'`` and ``'dip_at'`` are marked on the profile and
            its ``'separation'``, ``'dip'``, ``'sigma'`` go in the title.
        smooth (float): Gaussian sigma, in bins, of a smoothed line drawn over the raw counts; 0 for none.
        z_stretch (float): display-only stretch of the projection's vertical axis (z, in ``'xz'`` and ``'yz'``).
        gamma (float), cmap (str): display of the projection (counts / max, to the power gamma).
        line_colors (sequence): one colour per line.
        height_ratios (list, optional), figsize (tuple, optional), suptitle (str, optional): layout.
        renderer (callable): how points become an image, for the projection and the profiles (interface of
            ``points_to_image``, the default: a histogram, raw counts). See ``points_line_profile``.
        values (dict, optional): ``{name: array (N,)}``, a value per point of that set (aligned with its points),
            for example each localization's uncertainty: its ``statistic`` per bin along each line is drawn on a
            second axis (``values_label``, ``values_color``) over the counts.

    Returns:
        fig, axes (2D array), results ``{(name, label): dict(positions, counts, **measure(...))}``.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    h_ax, v_ax, _ = _VIEW_AXES[view]
    zs = 1. if view == 'xy' else z_stretch
    n_r, n_c = 1 + len(lines), len(sets)
    hr = height_ratios or [1.6] + [1.] * len(lines)
    fig, axes = plt.subplots(n_r, n_c, figsize=figsize or (4.75 * n_c, 4.3 * sum(hr) / 1.6 * 1.1),
                             squeeze=False, gridspec_kw={'height_ratios': hr})
    results = {}
    for c, (name, s) in enumerate(sets.items()):
        roi = s if isinstance(s, LineROI) else None
        p = roi.points if roi is not None else np.asarray(s, dtype=float)
        if bounds is not None:
            b = bounds
        elif roi is not None and view == 'xz':
            b = ((0., roi.length), roi.slab if roi.slab is not None else (p[:, 2].min(), p[:, 2].max()))
        elif roi is not None and view == 'xy':
            b = ((0., roi.length), (-roi.line_width / 2, roi.line_width / 2))
        else:
            b = ((p[:, h_ax].min(), p[:, h_ax].max()), (p[:, v_ax].min(), p[:, v_ax].max()))
        edges = [np.arange(lo, hi + bin_size, bin_size) for lo, hi in b]
        img = _render(renderer, p[:, [h_ax, v_ax]], bin_size, b)
        ax = axes[0, c]
        ax.imshow((img / max(img.max(), 1)) ** gamma, origin='lower', cmap=cmap, aspect='equal',
                  extent=[edges[0][0], edges[0][-1], edges[1][0] * zs, edges[1][-1] * zs])
        ax.set_title(f'{name}\n{len(p):,} points', fontsize=10)
        for (label, (start, end, width)), color in zip(lines.items(), line_colors):
            st, en = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
            u = (en - st) / np.hypot(*(en - st))
            off = np.array([-u[1], u[0]]) * width / 2
            corners = np.array([st - off, en - off, en + off, st + off, st - off])
            ax.plot(corners[:, 0], corners[:, 1] * zs, color=color, lw=0.8, ls=':')
            ax.set_xlim(edges[0][0], edges[0][-1]); ax.set_ylim(edges[1][0] * zs, edges[1][-1] * zs)
        for r, ((label, (start, end, width)), color) in enumerate(zip(lines.items(), line_colors), start=1):
            line_roi = points_line_roi(p, start, end, width, view)
            pos, cnt = points_line_profile(line_roi, bin_size, renderer)
            res = measure(pos, cnt) if measure is not None else None
            ax = axes[r, c]
            ax.plot(pos, cnt, color='0.6', lw=0.8, label='counts')
            if smooth:
                ax.plot(pos, gaussian_filter1d(cnt.astype(float), smooth), color='C0', lw=1.5, label='smoothed')
            title = label
            if res:
                for pk in res['peaks']:
                    ax.axvspan(pk - 1.5 * bin_size, pk + 1.5 * bin_size, color=color, alpha=0.25)
                ax.axvspan(res['dip_at'] - 1.5 * bin_size, res['dip_at'] + 1.5 * bin_size, color='red', alpha=0.2)
                title += f": peaks {res['separation']:.0f} apart, dip {res['dip']:.2f} ({res['sigma']:.1f} sigma)"
            elif measure is not None:
                title += ': no two peaks'
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(f'along the {label} line, from its start')
            ax.set_ylabel(f'points per {bin_size:g}')
            ax.legend(fontsize=7)
            results[(name, label)] = dict(positions=pos, counts=cnt, **(res or {}))
            if values is not None and name in values:
                _, val = points_line_profile(line_roi, bin_size, values=values[name], statistic=statistic)
                ax2 = ax.twinx()
                ax2.plot(pos, val, color=values_color, lw=1.5, marker='.', ms=4)   # markers: lone bins show too
                ax2.set_ylabel(values_label, color=values_color)
                ax2.tick_params(axis='y', colors=values_color)
                results[(name, label)]['values'] = val
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if suptitle is not None else None)
    return fig, axes, results


# ---- colour-coded renders (for example by depth) ---------------------------------------------------------

def points_to_color_image(xy, values, pixel_size, bounds=None, value_range=None, cmap='jet', clip=None, blur=0.):
    """Image of 2D points coloured by a value per point (for example depth).

    Each pixel's colour is the mean value of its points, through ``cmap`` over ``value_range``; its brightness is
    the number of points, clipped at ``clip`` (default: the 99.5th percentile of the non-empty pixels).

    Args:
        xy (array (N, 2)): positions, ``(x, y)``.
        values (array (N,)): the value per point that sets the colour.
        pixel_size (float or (x, y)), bounds: as ``points_to_image``.
        value_range ((lo, hi), optional): the values mapped onto the colour map; default their min and max.
        cmap (str): matplotlib colour map.
        clip (float, optional): count that gives full brightness.
        blur (float): Gaussian blur of the result, in pixels; 0 for none.

    Returns:
        image (array [y, x, 3], 0..1), bounds.
    """
    import matplotlib.pyplot as plt
    values = np.asarray(values, dtype=float)
    counts, b = points_to_image(xy, pixel_size, bounds)
    sums, _ = points_to_image(xy, pixel_size, b, weights=values)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    lo, hi = value_range if value_range is not None else (values.min(), values.max())
    rgb = plt.get_cmap(cmap)(np.clip((mean - lo) / (hi - lo), 0, 1))[..., :3]
    top = clip if clip is not None else (np.percentile(counts[counts > 0], 99.5) if counts.any() else 1.)
    rgb = rgb * np.clip(counts / top, 0, 1)[..., None]
    if blur:
        from scipy.ndimage import gaussian_filter
        rgb = gaussian_filter(rgb, (blur, blur, 0))
    return rgb, b


def show_points_overview(xyz, pixel_size, depth_range=(-1200., 1200.), side_range=None, bounds=None, clip=None,
                         clip_percentile=99.5, blur=0., cmap='jet', side_cmap='gray', title=None, figsize=None, unit='nm',
                         y_down=True):
    """First look at 3D point sets: top view coloured by depth, and a side view (x against z) below it; one column
    per set, all on one brightness scale.

    Args:
        xyz (array (N, 3) or dict): positions ``(x, y, z)`` in one unit, or ``{name: positions}`` for several sets.
        pixel_size (float): pixel size of every view, in that unit.
        depth_range ((lo, hi)): depths mapped onto ``cmap`` in the top views.
        side_range ((lo, hi), optional): depth range of the side views (default ``depth_range``).
        bounds (((x lo, hi), (y lo, hi)), optional): the field of the top views (default: all points of all sets).
        clip (float, optional): count for full brightness; default the ``clip_percentile`` of non-empty pixels,
            the largest over the sets (one scale for all).
        blur (float): Gaussian blur, in pixels; 0 for none.
        cmap, side_cmap (str): colour maps of the top and side views.
        title (str, optional), figsize (tuple, optional), unit (str): figure title, size, axis unit.
        y_down (bool): top views with y increasing downward (image convention); False puts y up.

    Returns:
        fig, axes (2 x n sets), images ``{name: dict(top=rgb [y, x, 3], counts=[y, x], side=[z, x], top_bounds,
        side_bounds)}`` plus ``images['clip']`` and ``images['side_clip']``; for one set (not a dict) the name is None.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    sets = xyz if isinstance(xyz, dict) else {None: xyz}
    sets = {k: np.asarray(v, dtype=float) for k, v in sets.items()}
    side_range = side_range or depth_range
    if bounds is None:
        allxy = np.concatenate([v[:, :2] for v in sets.values()])
        bounds = [(allxy[:, 0].min(), allxy[:, 0].max()), (allxy[:, 1].min(), allxy[:, 1].max())]
    pct = lambda h: np.percentile(h[h > 0], clip_percentile) if h.any() else 1.
    images = {}
    for name, v in sets.items():
        counts, tb = points_to_image(v[:, :2], pixel_size, bounds=bounds)
        keep = (v[:, 2] >= side_range[0]) & (v[:, 2] < side_range[1])
        side, sb = points_to_image(v[keep][:, [0, 2]], pixel_size, bounds=[tb[0], side_range])
        images[name] = dict(counts=counts, side=side, top_bounds=tb, side_bounds=sb)
    top_clip = clip if clip is not None else max(pct(im['counts']) for im in images.values())
    side_clip = clip if clip is not None else max(pct(im['side']) for im in images.values())
    n = len(sets)
    fig, axes = plt.subplots(2, n, figsize=figsize or (8 * n, 10), squeeze=False, gridspec_kw={'height_ratios': [4, 1]})
    for c, (name, v) in enumerate(sets.items()):
        im = images[name]
        im['top'], tb = points_to_color_image(v[:, :2], v[:, 2], pixel_size, bounds=bounds, value_range=depth_range,
                                              cmap=cmap, clip=top_clip, blur=blur)
        sb = im['side_bounds']
        side = gaussian_filter(np.clip(im['side'], 0, side_clip), blur) if blur else im['side']
        axes[0, c].imshow(im['top'], origin='upper' if y_down else 'lower',
                          extent=[tb[0][0], tb[0][1]] + ([tb[1][1], tb[1][0]] if y_down else [tb[1][0], tb[1][1]]))
        axes[0, c].set_title((f'{name}: ' if name else '') + f'top view, colour = depth {depth_range[0]:g} to {depth_range[1]:g} {unit}')
        axes[1, c].imshow(side, origin='lower', cmap=side_cmap, vmin=0, vmax=side_clip, aspect='auto',
                          extent=[sb[0][0], sb[0][1], sb[1][0], sb[1][1]])
        axes[1, c].set_title((f'{name}: ' if name else '') + 'side view (x against depth)')
        axes[1, c].set_xlabel(f'x ({unit})'); axes[1, c].set_ylabel(f'z ({unit})'); axes[0, c].set_ylabel(f'y ({unit})')
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if not isinstance(xyz, dict):
        images = dict(images[None], clip=top_clip, side_clip=side_clip)
        return fig, axes[:, 0], images
    images.update(clip=top_clip, side_clip=side_clip)
    return fig, axes, images
