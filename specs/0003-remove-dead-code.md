# 0003 — Delete what nothing uses

**Status:** planned, not started. Written 2026-08-01.

## The problem

A reader deciding whether to depend on this library reads the file listing
first. Right now that listing contains six variants of the same deconvolution
algorithm, a test written by ChatGPT that asserts `1 + 2 == 3`, and a directory
called `notebooks_` holding one markdown file.

None of this is broken. All of it costs credibility, and some of it costs the
next person an afternoon working out which file is the real one.

Git remembers everything deleted here. That is what makes this cheap.

## The list

Measured 2026-08-01.

### `tnia/deconvolution/experimental/` — six near-duplicates, 1053 lines

```
332  richardson_lucy_gradient_consensus.py
173  richardson_lucy_gradient_consensus_latest_sandbox.py
144  richardson_lucy_gradient_consensus_latest.py
138  richardson_lucy_gradient_consensus_2025_06_22.py
142  richardson_lucy_poisson_thinning.py
124  richardson_lucy_gradient_consensus_biggs.py
```

**Nothing in the repository imports any of them** — not `tnia/`, not
`notebooks/`, not `experiments/`. Grep for `deconvolution.experimental` returns
nothing.

Four of the six are the same algorithm at different moments in time; the
filenames (`_latest`, `_latest_sandbox`, a date) are version control done by
hand. Pick the one that is actually current, name it for what it does, delete
the other three. `_poisson_thinning` and `_biggs` are separate ideas — keep or
drop on their own merits.

Note this directory has no `__init__.py`, so it is not installed by
`find_packages()` today (see 0001's prerequisite). Deciding what survives here
also decides whether it ships at all.

### `tests/first_test.py`

```python
# stupid simple test made by chat-gpt

def add(x, y):
    return x + y

def test_addition():
    assert add(1, 2) == 3
```

Delete. `tests/` then holds three real tests — `test_projections.py`,
`test_augmentation.py`, `test_add_small_to_large.py` — plus `helper.py`, which
is a genuine fixture module.

### `notebooks_/`

One tracked file, `extract_psf.md`. The trailing underscore reads as a typo of
`notebooks/`. Move the content into `docs/` where prose belongs, and remove the
directory. `MANIFEST.in` has a `recursive-exclude notebooks_ *` line that goes
with it.

### `experiments/` — 23 tracked files, 10.3 MB

Scratch scripts, several in obvious series: `nd/slicers.py`, `slicers2.py`,
`slicers3.py`; `deconvolution/testbigspheres.py`, `testclijrl.py`. Some are
named `test_*` but are not tests — `experiments/io/test_nd_save.py` and friends
get collected by a bare `pytest` run from the repo root.

Per file: promote to `tnia/` if it earned it, move to `notebooks/` if it is an
example, otherwise delete. If the whole directory turns out to be scratch,
deleting it is the honest outcome — it is 10 MB and the history keeps it.

### Test-named modules inside the package

- `tnia/napari/deconvolution/test_restoration_deconvolution.py` — a test living
  inside the shipped package. Move to `tests/` or delete.
- `tnia/simulation/test_images.py` — **not a test.** It generates test *images*,
  and notebooks import from it. Pytest collects it anyway because of the name.

The second needs care: renaming to `sample_images.py` is a breaking change for
every notebook importing it. Do it with a shim —

```python
# tnia/simulation/test_images.py
from tnia.simulation.sample_images import *  # noqa: F401,F403 — deprecated alias
```

— and a note in the changelog (0004). Or defer it; the cost today is only a
confusing pytest run.

## Considered and rejected

*Move `experimental/` to a branch instead of deleting* — branches rot silently
and nobody looks at them. `git log` on a deleted path is the same information
with less pretending it will come back.

*Keep everything and add a README explaining what is current* — a file that
must be explained is a file that should be deleted. The explanation goes stale
faster than the code.

## Gotchas

- Check `notebooks/` before deleting anything from `experiments/`; a notebook
  may `sys.path` its way into one of those scripts. Grep by filename, not by
  import statement.
- Deleting `tnia/deconvolution/experimental/` is safe *today* — verify the grep
  again at the time, since notebooks change under this repo constantly.
- Do these as several small commits, one per bullet, not one sweep. And keep
  them out of the notebook churn (see `CLAUDE.md`).

## Done when

- `pytest` from the repo root collects only `tests/`, and every collected test
  is one someone meant to write.
- `grep -r "experimental" tnia/` returns nothing, or returns exactly one module
  with a name that describes the algorithm.
- The top-level directory listing contains nothing whose purpose has to be
  guessed.
