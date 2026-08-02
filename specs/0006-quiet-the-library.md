# 0006 — Stop printing

**Status:** planned, not started. Written 2026-08-01.

## The problem

There are **153 `print()` calls in `tnia/`**. They fire during ordinary library
calls, into the caller's stdout, with no way to turn them off.

In a notebook this is invisible — it looks like progress. In a consumer like
scikit-ops, where a tnia function is one step inside someone else's pipeline,
it is a library talking over its host. It is also the difference between "a
utility I can depend on" and "a script someone packaged", which matters
specifically because 0001's goal is being depended upon.

Commit `74bc93b` ("Clean up print statements") already started this. This spec
is about finishing it deliberately rather than opportunistically.

## Where they are

Measured 2026-08-01, lines matching `^\s*print(`:

```
26  deconvolution/richardson_lucy_dask_cp.py
19  deconvolution/richardson_lucy.py
11  deconvolution/richardson_lucy_variable.py
10  deconvolution/experimental/richardson_lucy_gradient_consensus.py
10  deconvolution/experimental/..._latest_sandbox.py
10  deconvolution/experimental/..._latest.py
 8  gui/widgets/image_batch_widget.py
 8  deeplearning/dl_helper.py
 7  deconvolution/fftdeconv.py
 5  utility/memory.py
 4  machinelearning/random_forest_helper.py
 4  io/bioformats_helper.py
```

Deconvolution accounts for roughly half, and 30 of those sit in
`experimental/`, which 0003 may delete outright — **do 0003 first** and the
count drops before any work happens here.

## The plan

Not a find-and-replace. Each print is one of three things, and they get
different treatment:

**1. Debug output left behind.** Most of them. Delete. A print of an array
shape mid-iteration was a debugging session, not a feature.

**2. Genuine progress reporting** — the per-iteration lines in Richardson-Lucy,
where a long loop reasonably tells you where it is. These become a caller-owned
callback, not a logger:

```python
def richardson_lucy(image, psf, num_iters, callback=None):
    for i in range(num_iters):
        ...
        if callback is not None:
            callback(i, estimate)
```

A callback lets a notebook print, a GUI drive a progress bar, and a pipeline
ignore it. That is strictly better than a logger for this case, and it is what
the `gui/` code already needs — `image_batch_widget.py` prints because it has
no other channel.

**3. Warnings** — "falling back to numpy, cupy not available", and similar.
These become `warnings.warn(...)`, which is the standard channel and which
callers already know how to filter.

For anything left over that is genuinely informational, use a module logger:

```python
import logging
logger = logging.getLogger(__name__)
```

Never call `logging.basicConfig()` in library code — that is the library
configuring the application's logging, which is the same mistake as printing.

### Sequence

Per module, smallest first, one commit each. `utility/memory.py` (5) and
`io/bioformats_helper.py` (4) are warm-ups. `richardson_lucy.py` (19) and
`richardson_lucy_dask_cp.py` (26) are the real work and involve an API addition
(the callback), so they want their own pass and a changelog line.

## Considered and rejected

*A `verbose=True` keyword on every function* — spreads a flag through every
signature, and the caller still cannot redirect the output anywhere. Callbacks
and warnings both already solve this.

*Global `tnia.set_verbose(False)`* — module-level mutable state, and it makes
two callers in one process fight over a setting. `logging` exists for exactly
this and does it better.

*Sweep all 153 with a regex* — would delete the progress reporting people
actually rely on in the deconvolution loops, which is the one category worth
keeping.

## Gotchas

- Deleting prints from `richardson_lucy*` changes what notebooks show. That is
  the point, but check the deconvolution notebooks still read sensibly — if one
  depended on the printed iteration count, it now passes a `callback`.
- The callback parameter is an API addition: default `None`, no existing caller
  breaks. Still belongs in the changelog (0004).
- `gui/` and `napari/` may not be installed at all after 0001 settles which
  subpackages ship. Do not spend time there until that is decided.

## Done when

- `grep -rc "^\s*print(" tnia/` is in single digits, and each survivor is in a
  function whose job is to print.
- Importing and calling a `tnia.plotting` or `tnia.deconvolution` function from
  a script produces no output the caller did not ask for.
