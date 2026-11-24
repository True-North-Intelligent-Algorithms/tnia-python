"""Measurements on 1D profiles (for example line profiles from ``tnia.plotting.points``)."""
import numpy as np


def two_peak_dip(positions, counts, min_separation, smooth=1., window=1):
    """Two peaks of a profile and the dip between them: is a gap (for example between two walls) resolved?

    The profile is smoothed (Gaussian, ``smooth`` bins) only to find the peaks: the two highest local maxima at
    least ``min_separation`` apart, and the lowest point between them. The dip is measured on the raw counts, summed
    over ``window`` bins either side of each point: dip = gap / mean of the peaks (smaller is a clearer gap), and
    its significance in sigma, Poisson on the counts: (mean peak - gap) / sqrt(gap + (peak1 + peak2) / 4).

    Args:
        positions (array): bin centres.
        counts (array): counts per bin.
        min_separation (float): the smallest distance between the two peaks, in the units of ``positions``.
        smooth (float): Gaussian sigma in bins, for finding the peaks only.
        window (int): bins either side summed for the counts at each peak and at the gap.

    Returns:
        dict(peaks=(p1, p2), dip_at, separation, dip, sigma), or None if there are not two peaks that far apart.
    """
    from scipy.ndimage import gaussian_filter1d
    positions, counts = np.asarray(positions), np.asarray(counts)
    hs = gaussian_filter1d(counts.astype(float), smooth)
    loc = np.flatnonzero((hs[1:-1] >= hs[:-2]) & (hs[1:-1] >= hs[2:])) + 1
    loc = loc[np.argsort(-hs[loc])]
    if len(loc) < 2:
        return None
    far = [p for p in loc[1:] if abs(positions[p] - positions[loc[0]]) >= min_separation]
    if not far:
        return None
    lo, hi = sorted((loc[0], far[0]))
    gap = lo + 1 + np.argmin(hs[lo + 1:hi]) if hi - lo > 1 else lo
    win = lambda i: counts[max(i - window, 0):i + window + 1].sum()
    w1, w2, g = win(lo), win(hi), win(gap)
    wm = (w1 + w2) / 2
    return dict(peaks=(positions[lo], positions[hi]), dip_at=positions[gap], separation=positions[hi] - positions[lo],
                dip=g / wm if wm else np.nan, sigma=(wm - g) / np.sqrt(g + (w1 + w2) / 4) if wm else np.nan)


def fwhm(profile, spacing=1.):
    """Full width at half maximum of a 1D profile with one peak, in units of ``spacing``.

    The profile's median is subtracted first (its background); the half-maximum crossings either side of the
    highest point are found by linear interpolation. Returns NaN if the peak is not above the background or a
    crossing is missing.
    """
    p = np.asarray(profile, dtype=float) - np.median(profile)
    pk = int(p.argmax())
    if p[pk] <= 0:
        return np.nan
    half = p[pk] / 2.0

    def cross(idxs):
        for a, b in zip(idxs[:-1], idxs[1:]):
            if (p[a] - half) * (p[b] - half) <= 0 and p[a] != p[b]:
                return a + (half - p[a]) / (p[b] - p[a]) * (b - a)
        return np.nan

    left, right = cross(list(range(pk, -1, -1))), cross(list(range(pk, len(p))))
    if not (np.isfinite(left) and np.isfinite(right)):
        return np.nan
    return abs(right - left) * spacing
