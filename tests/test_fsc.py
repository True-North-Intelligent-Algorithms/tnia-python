"""
Tests for the 3D FSC in tnia.metrics.fsc (moved from decode_cryo_palm_examples, spec 0031 there). Synthetic only.
"""
import sys

import numpy as np
import pytest

from tnia.metrics import fsc as fa


def _filaments(seed, n_fil, n_per, box, sig_xy, sig_z):
    """
    Molecules along random straight 2 um filaments inside ``box`` (nm), each
    jittered by the localization error. The same seed gives the same
    filaments whatever the sigmas, so only the error changes between calls.
    """
    rng = np.random.default_rng(seed)
    hi = np.asarray(box, dtype=float)
    p0 = rng.uniform(0., 1., (n_fil, 3)) * hi
    d = rng.normal(size=(n_fil, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t = rng.uniform(-1000., 1000., (n_fil, n_per))
    pts = (p0[:, None, :] + t[..., None] * d[:, None, :]).reshape(-1, 3)
    pts = pts[np.all((pts >= 0) & (pts < hi), axis=1)]
    pts = pts + rng.normal(size=pts.shape) * np.array([sig_xy, sig_xy, sig_z])
    return pts.astype(np.float32)


def _halves(n):
    mask = np.zeros(n, dtype=bool)
    mask[::2] = True
    return mask


def test_flat_matches_frc():
    """
    A volume one voxel deep has only kz = 0, so its lateral FSC must be the 2D
    FRC that 38/38c compute with frc.two_frc -- same bins, same frequency axis,
    same values. Catches binning and frequency-unit errors.
    """
    frc = pytest.importorskip('frc')
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    side, px = 128, 10.
    common = gaussian_filter(rng.random((side, side)), 2).astype(np.float32)
    a = common + 0.02 * rng.standard_normal((side, side)).astype(np.float32)
    b = common + 0.02 * rng.standard_normal((side, side)).astype(np.float32)

    ref = frc.two_frc(a, b)

    geo, w = fa._fsc_geometry((side, side, 1), (px, px), 30.)
    num, da, db = fa._fsc_sums(a[:, :, None].copy(), b[:, :, None].copy(),
                               geo, w)['lateral']
    ours = num / np.sqrt(da * db)

    assert len(ours) == len(ref), (len(ours), len(ref))
    np.testing.assert_allclose(geo['lateral']['freq'],
                               np.arange(side // 2) / side / px)
    np.testing.assert_allclose(ours, ref, atol=1e-3)


def test_axial_worse_and_sigma_monotone():
    """
    Filaments with sigma_z = 3 x sigma_xy: axial must read clearly worse than
    lateral, and doubling both sigmas must make both worse.
    """
    box = (2560., 2560., 1000.)
    res = {}
    for name, (sxy, sz) in {'base': (20., 60.), 'double': (40., 120.)}.items():
        xyz = _filaments(1, 150, 1000, box, sxy, sz)
        r = fa.fsc_points(xyz, _halves(len(xyz)), voxel=(10., 10.),
                          tile_px=256, verbose=False)
        res[name] = r['res_nm']
        for s in ('lateral', 'axial'):
            assert r['res_nm'][s] is not None, (name, s, r['why'][s])

    b, d = res['base'], res['double']
    print(f'   base   lateral {b["lateral"]:.1f}  axial {b["axial"]:.1f} nm')
    print(f'   double lateral {d["lateral"]:.1f}  axial {d["axial"]:.1f} nm')
    assert b['axial'] > 1.5 * b['lateral']
    assert d['lateral'] > 1.3 * b['lateral']
    assert d['axial'] > 1.3 * b['axial']


def test_tiling_matches_one_tile():
    """
    The same points as one tile and as 2 x 2 tiles must give the same lateral
    and axial resolution within 10% -- summing per-tile sums estimates the
    whole-volume FSC.
    """
    xyz = _filaments(2, 300, 1000, (4800., 4800., 1000.), 20., 60.)
    mask = _halves(len(xyz))
    one = fa.fsc_points(xyz, mask, tile_px=512, verbose=False)     # 5.12 um
    four = fa.fsc_points(xyz, mask, tile_px=256, verbose=False)    # 2.56 um

    assert one['n_tiles'] == 1 and four['n_tiles'] == 4, \
        (one['n_tiles'], four['n_tiles'])
    for s in ('lateral', 'axial'):
        r1, r4 = one['res_nm'][s], four['res_nm'][s]
        print(f'   {s:<8} 1 tile {r1:.1f}   2x2 tiles {r4:.1f} nm')
        assert abs(r4 - r1) / r1 < 0.10, (s, r1, r4)


if __name__ == '__main__':
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(name)
            try:
                fn()
                print('   PASS')
            except AssertionError as e:
                failed += 1
                print(f'   FAIL  {e!r}')
    sys.exit(1 if failed else 0)
