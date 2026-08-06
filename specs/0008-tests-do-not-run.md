# 0008 — The tests do not run on NumPy 2

**Status:** written 2026-08-04, not started.

## The problem

Every test in `tests/` fails at collection:

```
AttributeError: `np.float_` was removed in the NumPy 2.0 release.
```

Not one test runs. Not one fails, either — pytest never gets far enough to
call any of them, so the suite is silently absent rather than red.

Nothing in `tnia/` uses `np.float_`. It is `raster_geometry` 0.1.4.2, at
`raster/raster.py:2460`, in a default argument evaluated at import:

```python
def ...(..., dtype=np.float_):
```

Reproduced against `numpy` 2.4.6.

## Why all three modules go down together

`tests/helper.py` opens with `import raster_geometry as rg`, for
`make_random_spheres` and `make_3D_random_spheres`. All three test modules
import `helper`, so one unused import at the top of one file takes out:

- `test_add_small_to_large.py`
- `test_augmentation.py`
- `test_projections.py`

`first_test.py` is not collected at all — it does not match `test_*.py`, which
0003 already lists for deletion.

## How long it has been like this

Unknown, and that is the finding. NumPy 2.0 is from mid-2024. Nobody noticed,
because a suite that fails at collection looks much like a suite nobody ran —
and there is no CI to tell the difference ([0004](0004-ci-and-changelog.md)).

The `draw_boxes` work of 2026-08-04 hit it and verified that function by hand
instead. That is the wrong shape of fallback to repeat.

## Options

1. **Fix upstream.** `np.float_` → `np.float64` in raster_geometry. Correct,
   and outside our control; the project looks unmaintained.
2. **Pin `numpy<2` for tests.** Cheap and immediate, and wrong in the long
   run — tnia-python declares `numpy` unbounded and everything else in the
   ecosystem is on 2.x, so it would test a configuration nobody runs.
3. **Drop the dependency from `helper.py`.** The two functions it feeds want
   random spheres in a volume. `tnia.simulation` already draws spheres, and
   `skimage.draw` does too. If the helpers can be written without
   raster_geometry, three test modules come back for the cost of one function,
   and the fix survives whatever upstream does.

Prefer 3, with 2 as a stopgap only if 3 turns out to be more than an hour.

Note that raster-geometry stays a real dependency either way — `simulation`
and `deeplearning` both declare it, and `tnia/` imports it. This is about the
*tests* not being hostage to it.

## Done when

`pytest tests/` collects and runs on the numpy the library actually installs
with. Whether the tests then pass is a separate question, and worth knowing:
they have not run in long enough that some may have rotted.

## Related

- [0003](0003-remove-dead-code.md) — `first_test.py` is on its list already.
- [0004](0004-ci-and-changelog.md) — CI is what turns "the tests do not run"
  from a thing discovered by accident into a thing announced on push.
