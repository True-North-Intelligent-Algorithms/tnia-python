# 0001 — Install only the parts you want

**Status:** implemented 2026-08-02, apart from two items. Written 2026-08-01.

Done: `setup.cfg`/`setup.py` replaced by `pyproject.toml`, extras table,
version 0.1.27 → 0.2.0, `docs/install.md` rewritten. Two subpackages that
`find_packages()` had never shipped — `machinelearning` and `wavelets` — now
have empty `__init__.py` files and extras of their own; `gui/threads` and
`deconvolution/experimental` were deliberately left unpackaged, nothing imports
them and 0003 may delete the latter.

**Not done:** step 5, the `_require` import helper. And the README still
describes the manual dependency approach and needs the `[all]` upgrade note.
The clean-environment check under *Done when* has not been run.

## The problem

`pip install tnia-python` installs albumentations, which installs OpenCV. A
user who wants `tnia.plotting.projections` — three functions that draw xy and
zy slices with matplotlib — gets a computer-vision stack and an augmentation
library they will never call.

This came up while wiring tnia-python into scikit-ops as a host dependency for
projection figures. It is the only thing that made that dependency awkward.

`docs/install.md` already describes the intent:

> We tend not to install many of the dependencies via setup.py. The
> dependencies are complex and not all are needed to run many examples. Thus we
> leave it up to the user to install dependencies manually, allowing them to
> potentially install a minimum set of dependencies for the specific code they
> are interested in running.

That is exactly right, and it is currently enforced by asking the reader to do
it by hand. Extras are how pip expresses the same thing.

## The term

**Extras** — `[project.optional-dependencies]` in PEP 621, written
`pip install tnia-python[plotting]`. It is what napari's `napari[all]` and
`napari[pyqt6]` are. "Features" is conda/pixi vocabulary and does not apply
here.

## What makes this cheap

Every `tnia/*/__init__.py` is empty (0–1 lines). So `import
tnia.plotting.projections` does not drag in `tnia.deeplearning`, and splitting
the dependencies does not require touching the code. Extras are worthless on a
package whose `__init__` eagerly imports its subpackages; this one does not.

**Keep it that way.** Anything added to an `__init__.py` that imports a
subpackage undoes this silently.

## What actually depends on what

Measured 2026-08-01, matching both `import x` and `from x import`:

| dependency | imported by |
| --- | --- |
| `albumentations` | `deeplearning` — one module, `augmentation.py` |
| `raster_geometry` | `deeplearning`, `simulation` |
| `tifffile` | `io` |
| `matplotlib` | `plotting`, `deeplearning`, `segmentation`, `background` |
| `scipy` | `plotting`, `fitting`, `morphology`, `segmentation` |
| `scikit-image` | 10 subpackages, `plotting` among them |
| `numpy` | 15 subpackages |

Two findings:

- **`cv2` is never imported directly anywhere in `tnia/`.** It arrives only
  through albumentations, which exactly one module needs. Moving albumentations
  to an extra removes OpenCV from every other install.
- **`numpy` and `scikit-image` are not declared at all**, despite being the two
  most widely imported. They arrive transitively today. So this work adds two
  declarations as well as pruning — current metadata is wrong in both
  directions.

## The plan

Base install becomes `numpy` alone. Everything else moves to an extra:

```toml
[project.optional-dependencies]
plotting     = ["matplotlib", "scipy", "scikit-image"]
io           = ["tifffile", "scikit-image"]
simulation   = ["raster-geometry", "scikit-image"]
deeplearning = ["albumentations", "raster-geometry", "matplotlib", "scikit-image"]
all          = ["tnia-python[plotting,io,simulation,deeplearning]"]
```

Note `plotting` is not matplotlib-only — it needs scipy and scikit-image too.
Worth knowing before promising a tiny install.

The subpackages not named above (`alignment`, `background`, `fitting`,
`metrics`, `morphology`, `nd`, `segmentation`, `utility`, …) draw from the same
pool of numpy / scipy / scikit-image / matplotlib. Either fold them into
`plotting`-style extras of their own later, or leave them covered by `all`. Do
not block the first iteration on naming every one.

### Considered and rejected

*Prune only the outliers* — keep `numpy, scipy, scikit-image, matplotlib` in
the base and make extras of just `deeplearning` and `io`. Smaller change,
breaks fewer people, kills OpenCV just as well. Rejected because it leaves
`pip install tnia-python[plotting]` meaningless, and the goal is a package
whose install can be scoped, not just one that is smaller by one library.

## Mechanics

1. `setup.cfg` `[metadata]` / `[options]` → `pyproject.toml` `[project]`.
   Keep setuptools as the build backend; `packages=find:` becomes
   `[tool.setuptools.packages.find]`. This is the whole modernization needed —
   a larger cleanup is a separate job.
2. Add the extras table above.
3. Bump the minor version. The default install changing is visible behaviour.
4. Update `docs/install.md`: replace the manual-dependency advice with the
   extras, keeping the reasoning, which was right all along.
5. Add an import helper so a missing extra explains itself:

```python
def _require(module, extra):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is needed for this; install it with "
            f"pip install tnia-python[{extra}]"
        ) from exc
```

Without it, someone who installed `[plotting]` and then calls a deeplearning
helper gets a bare `ModuleNotFoundError: albumentations` and no hint that
extras exist.

## Gotchas

- Extras cannot prune a *transitive* dependency. `[deeplearning]` still pulls
  OpenCV, because albumentations wants it. Check whether albumentations will
  accept `opencv-python-headless`, which at least drops the GUI libraries.
- Extras are not retroactive: upgrading in place keeps whatever is already
  installed. The pruning only shows in fresh environments.
- `pip install tnia-python` will stop being enough for existing users. The
  upgrade note is one line — *use `tnia-python[all]` for the old set* — but it
  needs to be in the release notes and the README, not only here.

## Done when

- `pip install tnia-python[plotting]` in a clean environment installs no
  albumentations and no OpenCV, and `from tnia.plotting.projections import
  show_xy_zy_slice` works.
- `pip install tnia-python[all]` matches what 0.1.27 installed, plus the two
  declarations that were missing.

## Downstream

scikit-ops carries `tnia-python` in its `examples` dependency group
(`pyproject.toml`) and in `pixi/examples/pixi.toml`, for `show_xy_zy_slice` in
`notebooks/deconvolution/bead-edge-handling.ipynb`. Both should become
`tnia-python[plotting]` once this ships — that is the change that proves it
worked.
