# 0005 — Documentation for people who need one function

**Status:** planned, not started. Written 2026-08-01.

## The problem

There is no API documentation. `docs/documentation/` contains one file,
`mkdocs.md`, which is a note about the tool rather than any output of it. The
README is honest about this — *"still lacking documentation"* — and says the
notebooks are the documentation.

For the reader the notebooks serve, that works. For the reader 0001 is aimed
at — someone who wants `show_xy_zy_slice` and nothing else — 442 notebooks is
not documentation, it is a haystack.

`docs/install.md` is separately stale: it recommends Python 3.9, devbio-napari,
and `clij2-fft` from test.pypi.

## What to write

Not autodoc. Three pages, hand-written, small.

### 1. `docs/subpackages.md` — the map

One table: subpackage, one line on what it is for, which extra installs it.
This is the page that makes 0001 legible — extras are meaningless until someone
can see which extra contains the thing they want.

| subpackage | for | extra |
| --- | --- | --- |
| `plotting` | xy/zy/xz slice and projection figures | `plotting` |
| `deconvolution` | Richardson-Lucy and PSF handling | `plotting` |
| `simulation` | phantoms and test images | `simulation` |
| `io` | tiff, raw, bioformats readers | `io` |
| `deeplearning` | patch extraction, augmentation, SAM/stardist helpers | `deeplearning` |
| `segmentation` | thresholding, separation, object filtering | `plotting` |
| … | | |

Fill in the rest from 0001's measured dependency table — the two documents
share a source of truth and should not disagree. Sixteen or so subpackages;
one line each; an afternoon.

Where a subpackage is a dead end (`gui/`, `napari/`, `wavelets/`), say so
plainly rather than omitting it. A blank in the table reads as an oversight; a
sentence reads as a decision.

### 2. A module docstring for `tnia.plotting.projections`

This module is the library's public face — it is the one thing scikit-ops
imports, and it exports **fifteen** `show_*` functions with overlapping names:

```
show_xy_zy          show_xy_zy_max      show_xy_zy_slice    show_xy_zy_slice_center
show_xy_xz          show_xy_xz_slice
show_xyz            show_xyz_slice      show_xyz_slice_center
show_xyz_max        show_xyz_sum        show_xyz_projection
show_xyz_max_slabs  show_xyz_projection_slabs
show_xyz_projection_peaks
```

There is a system in those names — `xy_zy` vs `xyz` is two panels vs three,
`_slice` vs `_max` is how the third axis is collapsed, `_center` means "pick the
middle for me", bare names take pre-computed panels. Nobody can see it from the
outside.

Write it as a module docstring:

> Two panels (`show_xy_zy`) or three (`show_xyz`). Choose how the remaining axis
> collapses: `_slice` at a coordinate you pass, `_slice_center` at the middle,
> `_max` or `_sum` by projection. The bare names take arrays you have already
> projected yourself.

Then a five-row "if you want X, call Y" table.

**Do not rename or consolidate anything.** scikit-ops pins
`show_xy_zy_slice`; these names are load-bearing in notebooks and in years of
forum posts. The naming is fine once explained — the missing piece is the
explanation, not a refactor.

### 3. Rewrite `docs/install.md`

0001 already owns this (its step 4). Two things to carry over that 0001 does
not spell out:

- Delete the devbio-napari / Python 3.9 / test.pypi recipe. It is from a
  different era and someone will try it.
- Keep the paragraph explaining *why* dependencies were left to the user. That
  reasoning was right; extras are simply the mechanism it was missing. 0001
  says this too — it is the sentence to preserve verbatim.

## Considered and rejected

*Sphinx or mkdocs autodoc over the whole library* — generates a page per module
whether or not the module deserves one, and the result is a wall of
`show_xyz_projection_slabs(image_to_show, x_slices, ...)` with the same missing
context it has today. Autodoc documents signatures; the problem here is
navigation. Revisit once the docstrings are worth generating from.

*Docstrings everywhere first* — 87 modules, and most of them nobody outside
this repo calls. The map plus one well-documented flagship module serves the
target reader; blanket docstring coverage serves nobody in particular.

## Gotchas

- `docs/` is a published Jekyll site: every page needs `layout: basic` front
  matter or it renders raw.
- The subpackage table will drift from `setup.cfg` the moment extras change.
  Cross-link it to 0001 in both directions so the next editor sees both.
- Numbers in this spec (15 functions, 87 modules) were measured 2026-08-01 and
  will be wrong after 0003 deletes things. Re-measure before publishing.

## Done when

- Someone who wants one projection figure can go from the README to a working
  `pip install` line and a function name without opening a notebook.
- `docs/install.md` contains no advice that predates the extras.
