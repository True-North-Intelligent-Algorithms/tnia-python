# tnia-python — notes for Claude

A personal collection of image-analysis utilities: plotting and projections,
deconvolution, deep learning helpers, simulation, segmentation. Imported by
other projects (see *Downstream* below), not just used from its own notebooks.

## State as of 2026-08-02

This file and `specs/` were written in a session that started in
../skop-napari, where the tnia-python dependency came up. They are committed
deliberately: this is a sole-authored repository, so the rules and motivations
here should travel to other machines and to any contributor.

0001 is implemented (packaging is now `pyproject.toml` + extras, version
0.2.0); 0002–0007 are still plans. Keep all of this out of the notebook churn
under `notebooks/imagesc/`, which is usually uncommitted and unrelated.

## Pending work

Read the relevant spec before touching the area it covers. Suggested order is
the numbering, except that **0003 should precede 0006** (deleting
`experimental/` removes 30 of the print statements).

- **[0001 — improve installation](specs/0001-improve-installation.md)** —
  **implemented 2026-08-02**, except the `_require` import helper and the
  README's dependency section. The base install is now `numpy` alone; every
  other dependency lives in an extra. Read the spec before touching packaging,
  and note the rule below about empty `__init__.py` files — it is what makes
  the extras work.
- **[0002 — clone weight](specs/0002-clone-weight.md)** — 675 MB of
  tracked notebooks against a 0.3 MB library. `nbstripout`, `--depth 1`, and a
  deferred decision about splitting `notebooks/` out.
- **[0003 — remove dead code](specs/0003-remove-dead-code.md)** — six
  unimported near-duplicate deconvolution modules, a ChatGPT-written test,
  `notebooks_/`, `experiments/`, test-named modules inside the package.
- **[0004 — CI and changelog](specs/0004-ci-and-changelog.md)** — there is
  no `.github/`. Two jobs: one runs the tests, one proves the minimal install
  is minimal. Plus a `CHANGELOG.md` starting at the 0001 release.
- **[0005 — documentation](specs/0005-documentation.md)** — a subpackage →
  extra map, a module docstring for `tnia.plotting.projections` (15 overlapping
  `show_*` names), and a rewrite of the stale `docs/install.md`.
- **[0006 — quiet the library](specs/0006-quiet-the-library.md)** — 153
  `print()` calls in `tnia/`. Delete, or convert to callbacks and warnings.
- **[0007 — build system](specs/0007-build-system.md)** — the packaging
  leftovers 0001 deliberately excludes: `requires-python`, classifiers and
  project URLs, `tnia.__version__`, the `MANIFEST.in`/`exclude_package_data`
  overlap, releasing from CI. Explicitly *after* 0001; blocks nothing.

**Prerequisite for 0001, not yet written into it:** four subpackages have no
`__init__.py` — `machinelearning`, `wavelets`, `gui/threads`,
`deconvolution/experimental` — so `find_packages()` does not ship them.
`from tnia.machinelearning.random_forest_helper import extract_features` works
in editable mode and fails from a wheel; three notebooks use it. Decide which
of the four should ship, then add empty `__init__.py` files for those.

## Layout and conventions

- `tnia/<subpackage>/` — the library. **Every `__init__.py` is empty**, and
  should stay that way: subpackages must not import each other at module
  scope, or scoped installs stop working.
- `pyproject.toml` holds all the packaging metadata. `setup.cfg` and `setup.py`
  were removed by 0001 — do not reintroduce them.
- `docs/` is a published Jekyll site — pages carry `layout: basic` front
  matter, and `docs/install.md` is the install page users actually read.
- `specs/` holds internal planning documents, **not** site content. They carry
  no front matter and `_config.yml` excludes the directory from the build, so
  they render on GitHub and are not published to the web.
- `notebooks/` is large, organised by date and topic, and usually has
  uncommitted edits in it. Do not sweep it up into unrelated commits.

## Facts worth not rediscovering

- `cv2` is imported nowhere in `tnia/`. It comes in only via albumentations,
  which only `tnia/deeplearning/augmentation.py` needs.
- `numpy` and `scikit-image` are imported almost everywhere but declared
  nowhere — they arrive transitively today.
- Published on PyPI as `tnia-python` (0.1.27 as of 2026-08-01).

## Downstream

`scikit-ops` (../scikit-ops) depends on tnia-python in its `examples`
dependency group and `pixi/examples/pixi.toml`, for
`tnia.plotting.projections.show_xy_zy_slice` in the deconvolution notebook.
That is a *host* dependency — no scikit-ops op imports tnia-python. The
deconvolution code there was migrated out of tnia-python rather than wrapping
it, which may be reversed later if several decon backends turn up.
