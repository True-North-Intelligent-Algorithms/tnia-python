"""
3D Fourier shell correlation (FSC) of point sets: lateral and axial resolution from two halves of the
localizations, tiled so a whole field fits in memory.

Moved verbatim from decode_cryo_palm_examples/frc_analysis.py (2026-11-23), which re-exports it. The lateral
FSC of a one-voxel-deep volume equals the 2D FRC of ``frc.two_frc`` (tests/test_fsc.py). The optional ``frc``
package (pip) is needed only for the snapped 1/7 crossing.

    from tnia.metrics.fsc import fsc_points, split_halves
    r = fsc_points(xyz_nm, split_halves(frame_ix, 'index'), voxel=(10., 10.), tile_px=512)
    r['res_nm']['lateral'], r['res_nm']['axial']
"""
import os

import numpy as np


def split_halves(frame_ix, split, frames_per_iter=250, verbose=True):
    """
    Boolean mask for half A. Half B is its complement.

    ``index`` -- even/odd position. Correct only when each row is already one
    molecule, i.e. grouped data.

    ``iter`` -- whole iters alternately. Correct for ungrouped data with an iter
    structure: a molecule blinks for a frame or two and cannot span a 250-frame
    iter, so it lands in one half, and both halves still cover the whole
    acquisition so drift stays common-mode.

    The leak is not exactly zero. A blink of ``L`` frames straddles an iter
    boundary when it starts in the last ``L-1`` frames of one, so a fraction
    ``(L-1)/frames_per_iter`` of molecules do appear in both halves -- 0.4% for
    a 2-frame blink at 250. Measured at 1.19% for a synthetic 4-frame blink,
    which is 3/250 exactly. Negligible, but it is a floor on how independent
    the halves can be, not an absence of the effect.

    Splitting ungrouped data by index puts copies of the same molecule in both
    halves; the curve then never decays and there is no crossing to find.
    """
    n = len(frame_ix)
    if split == 'index':
        mask = np.zeros(n, dtype=bool)
        mask[::2] = True
        if verbose:
            print(f'split           : index (even/odd)   '
                  f'{mask.sum():,} / {n - mask.sum():,}')
        return mask

    if split == 'iter':
        it = frame_ix // frames_per_iter
        n_iters = len(np.unique(it))
        if n_iters < 2:
            raise ValueError(
                f'frame_ix spans {frame_ix.min()}..{frame_ix.max()}, which is '
                f'{n_iters} iter of {frames_per_iter} frames. This data has no '
                f'usable iter structure -- either frame_ix is not a global frame '
                f'index (27 rebases it to zero, see C14), or it came from '
                f'somewhere without iters. Use split="index" and treat the '
                f'result as a lower bound.')
        mask = (it % 2 == 0)
        share = mask.sum() / max(1, n)
        if verbose:
            print(f'split           : iter ({n_iters} iters of {frames_per_iter})   '
                  f'{mask.sum():,} / {n - mask.sum():,}   '
                  f'({share*100:.0f}% / {100-share*100:.0f}%)')
            if not 0.4 < share < 0.6:
                print('                  WARNING: halves are unbalanced')
        return mask

    raise ValueError(f'unknown split {split!r} -- use "index" or "iter"')


def available_bytes():
    """MemAvailable (psutil on Windows), or None where that is not readable."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


def render_xyz(xyz, origin, shape, voxel, sigma_blur=5., abs_clip=100.,
               dtype=np.float32):
    """
    3D histogram plus a constant gaussian blur -- :func:`render_xy` in 3D.

    ``origin`` is the nm position of the corner of voxel (0, 0, 0); ``voxel``
    is ``(xy, z)`` in nm. Points outside the box are dropped.

    No contrast clip, unlike :func:`render_xy`: that clips at the image's own
    max, and every tile has a different max, so tiles would not be rendered
    alike (0031 D1). ``abs_clip`` is per voxel and has no such problem.
    """
    from scipy.ndimage import gaussian_filter

    dxy, dz = voxel
    step = np.array([dxy, dxy, dz], dtype=np.float64)
    ijk = np.floor((np.asarray(xyz, dtype=np.float64)
                    - np.asarray(origin, dtype=np.float64)) / step).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(shape)), axis=1)

    vol = np.zeros(shape, dtype=dtype)
    if inside.any():
        flat = np.ravel_multi_index(ijk[inside].T, shape)
        u, c = np.unique(flat, return_counts=True)
        vol.flat[u] = c
    if abs_clip is not None:
        np.clip(vol, 0., abs_clip, out=vol)
    if sigma_blur:
        vol = gaussian_filter(vol, sigma=(sigma_blur / dxy, sigma_blur / dxy,
                                          sigma_blur / dz))
    return vol


def _fsc_geometry(shape, voxel, lateral_deg=3., axial_deg=30.):
    """
    Per-sector bin index for every rfftn voxel, and the rfft weights.

    Bins are ``floor(|k| / dk)`` -- DIPlib's ``RadialSum`` rule, so a lateral
    curve lines up bin for bin with ``frc.two_frc`` (test_flat_matches_frc).
    Lateral and sphere use dk = 1/(tile side); axial uses 1/(z depth) when
    that is coarser, because kz is only sampled that finely. Voxels outside a
    sector, or past its last full bin, go to an overflow bin that is dropped.

    The weights count each rfft voxel for itself and its conjugate twin, so
    the sums equal those over the full spectrum.
    """
    nx, ny, nz = shape
    dxy, dz = voxel
    kx = np.fft.fftfreq(nx, d=dxy)
    ky = np.fft.fftfreq(ny, d=dxy)
    kz = np.fft.rfftfreq(nz, d=dz)
    # float64: in float32 an exact integer radius such as (5, 12) -> 13 comes
    # out as 12.99999 and floors into the bin below (test_flat_matches_frc)
    k = np.sqrt((kx[:, None] ** 2 + ky[None, :] ** 2)[:, :, None]
                + kz[None, None, :] ** 2)
    kz3 = kz[None, None, :]

    dk_xy = 1. / (min(nx, ny) * dxy)
    dk_z = max(dk_xy, 1. / (nz * dz))
    sin_lat = np.sin(np.radians(lateral_deg))
    cos_ax = np.cos(np.radians(axial_deg))

    sectors = {
        'lateral': (dk_xy, 0.5 / dxy, kz3 <= k * sin_lat),  # near the kx-ky plane
        'axial': (dk_z, 0.5 / dz, kz3 >= k * cos_ax),       # near the kz axis
        'sphere': (dk_xy, min(0.5 / dxy, 0.5 / dz), None),
    }
    geo = {}
    for s, (dk, nyq, in_sector) in sectors.items():
        nbins = int(np.floor(nyq / dk + 1e-9))
        b = np.floor(k / dk + 1e-6).astype(np.int32)
        keep = b < nbins
        if in_sector is not None:
            keep &= in_sector
        b[~keep] = nbins
        geo[s] = {'idx': b.ravel(), 'nbins': nbins, 'freq': np.arange(nbins) * dk}
    del k

    w = np.full(len(kz), 2., dtype=np.float32)
    w[0] = 1.
    if nz % 2 == 0:
        w[-1] = 1.
    return geo, w


def _blur_transfer(shape, voxel, sigma_blur):
    """
    |G(k)|^2 of a gaussian blur of ``sigma_blur`` nm, as an (nx, ny) and an
    (nz_rfft,) factor. Multiplying the FFT products by these is the blur, done
    in Fourier space: much faster than ``gaussian_filter`` on the volume. It is
    applied after the window rather than before, and is the exact gaussian
    rather than ndimage's truncated kernel, so numbers move slightly against
    the real-space blur (ledger rows before 2026-10-24 evening).
    """
    nx, ny, nz = shape
    dxy, dz = voxel
    g2 = lambda k: np.exp(-4. * np.pi ** 2 * sigma_blur ** 2 * k ** 2).astype(np.float32)
    gxy = g2(np.fft.fftfreq(nx, d=dxy))[:, None] * g2(np.fft.fftfreq(ny, d=dxy))[None, :]
    return gxy, g2(np.fft.rfftfreq(nz, d=dz))


def _fsc_sums(a, b, geo, w, win=None, blur=None):
    """
    Per-sector bin sums of Re(Fa Fb*), |Fa|^2, |Fb|^2 for one tile.

    ``a`` and ``b`` are windowed in place. Pass them without keeping another
    reference and they are freed once transformed. ``blur`` is
    :func:`_blur_transfer`'s output, or None for no blur.
    """
    from scipy.fft import rfftn

    if win is not None:
        for v in (a, b):
            v *= win[0][:, None, None]
            v *= win[1][None, :, None]
            v *= win[2][None, None, :]
    fa = rfftn(a, workers=-1)
    del a
    fb = rfftn(b, workers=-1)
    del b

    num = fa.real * fb.real
    num += fa.imag * fb.imag
    pa = fa.real ** 2
    pa += fa.imag ** 2
    pb = fb.real ** 2
    pb += fb.imag ** 2
    del fa, fb
    wz = w if blur is None else w * blur[1]
    for arr in (num, pa, pb):
        if blur is not None:
            arr *= blur[0][:, :, None]
        arr *= wz

    out = {}
    for s, g in geo.items():
        m = g['nbins'] + 1
        out[s] = tuple(np.bincount(g['idx'], weights=arr.ravel(), minlength=m)[:g['nbins']]
                       for arr in (num, pa, pb))
    return out


def _fsc_crossing(freq, curve):
    """1/7 crossing via ``frc.frc_res`` -- the same call 38/38c use."""
    import frc

    ok = np.isfinite(curve)
    if ok.sum() < 3:
        return None, 'fewer than 3 populated bins'
    try:
        res, _, _ = frc.frc_res(freq[ok], curve[ok], int(ok.sum()))
    except Exception as e:
        tail = float(np.median(curve[ok][-max(1, ok.sum() // 10):]))
        side = 'HIGH -- halves not independent?' if tail > 0.3 else \
               'LOW -- too few localizations, or a sector too thin?'
        return None, f'{type(e).__name__}: no crossing, curve stays {side} (tail {tail:.3f})'
    if not np.isfinite(res):
        return None, 'crossing at zero frequency'
    return float(res), None


def _fsc_crossing_interp(freq, curve, near=None, threshold=1. / 7):
    """
    1/7 crossing placed by straight-line interpolation between the two samples either side of it.

    ``frc_res`` (:func:`_fsc_crossing`) snaps to the frequency samples. Laterally they are fine, but
    axially a 2,560 nm z box gives samples ~4-6 nm apart near 150 nm, so two runs a few nm apart can
    read identical (spec 0038). Same frequency axis (lower bin edges) as the snapped value.
    ``near`` (nm^-1): use the downward crossing closest to it -- the snapped crossing, so a noisy
    curve that dips below 1/7 early reads as ``frc_res`` reads it; None takes the first.
    """
    ok = np.isfinite(curve)
    f, c = np.asarray(freq, np.float64)[ok], np.asarray(curve, np.float64)[ok]
    if len(c) < 3:
        return None, 'fewer than 3 populated bins'
    if c[0] < threshold:
        return None, 'starts below 1/7'
    down = np.flatnonzero((c[:-1] >= threshold) & (c[1:] < threshold))
    if not len(down):
        return None, 'no crossing: the curve stays above 1/7'
    i = down[0] if near is None else down[np.argmin(np.abs(f[down] - near))]
    fx = f[i] + (threshold - c[i]) * (f[i + 1] - f[i]) / (c[i + 1] - c[i])
    return (float(1. / fx), None) if fx > 0 else (None, 'crossing at zero frequency')


def fft_cost_bytes_3d(shape):
    """
    Rough peak for one tile of :func:`fsc_points`, in bytes.

    Per rfftn voxel (about half the real-space voxels): three int32 bin
    indices held for the whole run (12 B), two complex64 transforms (16 B),
    three float32 products (12 B), and slack for scipy's and bincount's
    temporaries. An estimate, not a measurement.
    """
    nx, ny, nz = shape
    return int(nx) * int(ny) * (int(nz) // 2 + 1) * 48


def check_fft_memory_3d(shape, raise_at=0.9, warn_at=0.5):
    """:func:`check_fft_memory` for a tile."""
    need = fft_cost_bytes_3d(shape)
    avail = available_bytes()
    if avail is None:
        return need
    if need > raise_at * avail:
        raise MemoryError(
            f'FSC on a {shape} voxel tile needs roughly {need/1e9:.1f} GB and '
            f'only {avail/1e9:.1f} GB is available. Use a smaller tile_px or a '
            f'coarser z voxel.')
    if need > warn_at * avail:
        print(f'   NOTE: ~{need/1e9:.1f} GB per tile, {avail/1e9:.1f} GB available.')
    return need


def fsc_points(xyz, mask_a, voxel=(10., 10.), tile_px=1024, z_range=None,
               lateral_deg=3., axial_deg=30., sigma_blur=5., abs_clip=100., window=True,
               min_tile_points=100, workers=None, verbose=True, interpolate=False):
    """
    FSC of two halves of one point set, in tiles. The engine under
    :func:`fsc_one`; the tests drive it directly.

    Parameters
    ----------
    xyz : (N, 3) array, nm
    mask_a : (N,) bool, True for half A
    voxel : (xy, z) nm
    tile_px : int
        Tile side in xy voxels; every tile spans the full z box.
    z_range : (z0, z1) nm or None
        ``None`` takes the 0.1..99.9 percentile (0016: raw min/max lets one
        stray point blow up the grid). Points outside are dropped and counted.
        The box is padded to a power of 2 in z, with the data centred, so the
        z taper of the window falls mostly on empty voxels.
    lateral_deg : float
        Half-angle of the lateral sector around the kx-ky plane. Keep it
        narrow: at 30 deg the cone takes in kz up to |k|/2, where z error
        damps the curve, so a method with worse z reads worse laterally too
        (0031, synthetic: sigma_z 20 -> 120 nm moved "lateral" 65 -> 83 nm at
        30 deg, not at all at 3 deg). 3 deg is close to the kz = 0 plane, i.e.
        a projection FRC.
    axial_deg : float
        Half-angle of the axial sector around kz. Wider, because a 3 deg cone
        around one axis holds too few Fourier voxels to give a stable curve;
        the xy error it lets in is much smaller than the z error it measures.
    min_tile_points : int
        Tiles with fewer points are skipped.
    workers : int or None
        Tiles worked on at once. ``None`` fits as many as 75% of free memory
        allows, at most 4. The result does not depend on it.
    interpolate : bool
        False (default, as every result so far): the 1/7 crossing as ``frc_res`` gives it, snapped
        to the frequency samples. True: interpolated between the samples either side
        (:func:`_fsc_crossing_interp`). The snapped value is always in ``res_nm_snapped``.

    No render margin, unlike :func:`frc_one`: with ``window=True`` the Tukey
    taper is zero at every tile face, so blur that crosses a face is weighted
    to nothing either way.

    Returns
    -------
    dict with per-sector ``curve``, ``freq``, ``n_vox``, ``res_nm``, ``why``,
    and ``n_tiles``, ``n_used``, ``n_drop_z``, ``n_drop_tile``, ``shape``,
    ``z_range``.
    """
    import time
    from scipy.signal.windows import tukey

    dxy, dz = (float(v) for v in voxel)
    xyz = np.asarray(xyz, dtype=np.float32)
    mask_a = np.asarray(mask_a, dtype=bool)

    # --- z box --------------------------------------------------------------
    if z_range is None:
        z_range = tuple(float(v) for v in np.percentile(xyz[:, 2], [0.1, 99.9]))
    keep = (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
    n_drop_z = int((~keep).sum())
    xyz, mask_a = xyz[keep], mask_a[keep]

    pad = 3 * sigma_blur if sigma_blur else 0.
    n_z_data = max(1, int(np.ceil((z_range[1] - z_range[0] + 2 * pad) / dz)))
    nz = 1 << int(np.ceil(np.log2(n_z_data)))
    z0 = 0.5 * (z_range[0] + z_range[1]) - 0.5 * nz * dz
    shape = (int(tile_px), int(tile_px), nz)

    # --- tiles: sort once, slice per tile -----------------------------------
    tile_nm = tile_px * dxy
    x0, y0 = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    ntx = int(np.floor((xyz[:, 0].max() - x0) / tile_nm)) + 1
    nty = int(np.floor((xyz[:, 1].max() - y0) / tile_nm)) + 1
    tid = (np.floor((xyz[:, 0] - x0) / tile_nm).astype(np.int64) * nty
           + np.floor((xyz[:, 1] - y0) / tile_nm).astype(np.int64))
    order = np.argsort(tid, kind='stable')
    bounds = np.searchsorted(tid[order], np.arange(ntx * nty + 1))
    counts = np.diff(bounds)
    todo = np.flatnonzero(counts >= min_tile_points)
    n_drop_tile = int(counts.sum() - counts[todo].sum())
    del tid

    if verbose:
        print(f'voxel           : {dxy:g} nm xy, {dz:g} nm z')
        print(f'z               : {z_range[0]:.0f} .. {z_range[1]:.0f} nm kept '
              f'({n_drop_z:,} outside), box {nz} voxels from {z0:.0f} nm')
        print(f'tiles           : {shape[0]} x {shape[1]} x {shape[2]} voxels '
              f'= {tile_nm/1000:.2f} um square;  {len(todo)} of {ntx} x {nty} '
              f'used ({n_drop_tile:,} points in sparse tiles skipped)')

    check_fft_memory_3d(shape)
    geo, w = _fsc_geometry(shape, (dxy, dz), lateral_deg, axial_deg)
    blur = _blur_transfer(shape, (dxy, dz), sigma_blur) if sigma_blur else None
    win = None
    if window:
        win = (tukey(shape[0], 0.125).astype(np.float32),
               tukey(shape[1], 0.125).astype(np.float32),
               tukey(shape[2], 0.125).astype(np.float32))

    # --- accumulate ---------------------------------------------------------
    # Tiles run in threads (the blur, FFT and bincount release the GIL), but
    # are added up in tile order, so the sums are bit for bit those of a
    # serial run.
    if workers is None:
        avail = available_bytes()
        need = fft_cost_bytes_3d(shape)
        workers = 1 if avail is None else int(0.75 * avail // need)
        workers = max(1, min(4, workers, os.cpu_count() or 1))
    if verbose:
        print(f'workers         : {workers} tiles at a time')

    def _tile(t):
        sel = order[bounds[t]:bounds[t + 1]]
        tx, ty = divmod(int(t), nty)
        origin = (x0 + tx * tile_nm, y0 + ty * tile_nm, z0)
        pts, m = xyz[sel], mask_a[sel]
        return _fsc_sums(
            render_xyz(pts[m], origin, shape, (dxy, dz), 0., abs_clip),
            render_xyz(pts[~m], origin, shape, (dxy, dz), 0., abs_clip),
            geo, w, win, blur)

    from concurrent.futures import ThreadPoolExecutor
    acc = {s: [np.zeros(g['nbins']) for _ in range(3)] for s, g in geo.items()}
    t_start = time.time()
    with ThreadPoolExecutor(workers) as ex:
        tile_sums = ex.map(_tile, todo)
        for i, sums in enumerate(tile_sums):
            for s in geo:
                for j in range(3):
                    acc[s][j] += sums[s][j]
            if verbose and (i + 1) % 10 == 0:
                el = time.time() - t_start
                print(f'   tile {i+1}/{len(todo)}   {el/60:.1f} min, '
                      f'~{el/(i+1)*(len(todo)-i-1)/60:.1f} min left', flush=True)

    # --- curves -------------------------------------------------------------
    out = {k: {} for k in ('curve', 'freq', 'n_vox', 'res_nm', 'res_nm_snapped', 'why')}
    wfull = np.broadcast_to(w, (shape[0], shape[1], len(w))).ravel()
    for s, g in geo.items():
        num, da, db = acc[s]
        with np.errstate(invalid='ignore', divide='ignore'):
            curve = num / np.sqrt(da * db)
        out['curve'][s] = curve
        out['freq'][s] = g['freq']
        out['n_vox'][s] = np.bincount(g['idx'], weights=wfull,
                                      minlength=g['nbins'] + 1)[:g['nbins']]
        out['res_nm'][s], out['why'][s] = _fsc_crossing(g['freq'], curve)
        out['res_nm_snapped'][s] = out['res_nm'][s]
        if interpolate and out['res_nm'][s] is not None:
            res, why = _fsc_crossing_interp(g['freq'], curve, near=1. / out['res_nm'][s])
            if res is not None:
                out['res_nm'][s], out['why'][s] = res, why

    out.update(n_tiles=len(todo), n_used=int(len(xyz) - n_drop_tile),
               n_drop_z=n_drop_z, n_drop_tile=n_drop_tile, shape=shape,
               z_range=z_range)
    if verbose:
        print(f'done            : {(time.time() - t_start)/60:.1f} min')
    return out
