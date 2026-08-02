---
layout: basic
---

# 0004 — CI and a changelog, so 0001 stays true

**Status:** planned, not started. Written 2026-08-01.

## The problem

There is no `.github/` directory. Nothing runs the three tests in `tests/`, and
nothing checks what a `pip install` actually pulls in.

0001 defines success as *"`pip install tnia-python[plotting]` in a clean
environment installs no albumentations and no OpenCV."* That is a property of
the package metadata, and metadata regresses quietly — one `install_requires`
line added in a hurry and the extras become decorative. A claim nobody verifies
is a claim that will be false within a release or two.

Separately: 0001 changes what the default install gives you. Existing users
will find imports failing after an upgrade, and there is no file that tells
them why. Version `0.1.27` currently lives in exactly one place, `setup.cfg`,
with no record of what changed between it and `0.1.26`.

## The plan

### 1. A workflow that tests the extras, not just the code

```yaml
# .github/workflows/test.yml
name: test
on: [push, pull_request]

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install .[plotting,simulation]
      - run: pip install pytest
      - run: pytest tests/

  minimal-install:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install .[plotting]
      - name: no computer-vision stack in a plotting install
        run: |
          ! pip show albumentations 2>/dev/null
          ! pip show opencv-python 2>/dev/null
          ! pip show opencv-python-headless 2>/dev/null
      - name: the downstream import works
        run: python -c "from tnia.plotting.projections import show_xy_zy_slice"
```

The second job *is* 0001's acceptance criterion, executable. It is the reason
to do this before 0001 rather than after — write the test that fails, then make
it pass.

Note `tests/test_augmentation.py` needs albumentations, so the test job
installs more than `[plotting]`. That split is deliberate: one job proves the
code works, the other proves the install is small. Do not merge them.

### 2. An import-surface check

Cheap and catches the failure mode 0001 warns about — a subpackage `__init__.py`
that starts importing its neighbours, silently undoing the extras:

```yaml
      - name: __init__ files stay empty
        run: |
          find tnia -name __init__.py -size +1c -exec false {} +
```

Also worth asserting the four currently-unpackaged subpackages are or are not
shipped, once 0001's prerequisite settles that question:

```yaml
      - run: python -c "import tnia.machinelearning.random_forest_helper"
```

### 3. `CHANGELOG.md`

Keep-a-Changelog format, starting at the 0001 release. Do not backfill 933
commits of history — start where the record starts being useful, and say so:

```markdown
# Changelog

Kept from 0.2.0 onward. Earlier releases: see `git log`.

## [0.2.0] — unreleased

### Changed
- **Breaking:** the default install is now `numpy` only. Everything else moved
  to extras — `pip install tnia-python[all]` restores the 0.1.27 dependency
  set. See docs/specs/0001-improve-installation.md.

### Added
- Extras: `plotting`, `io`, `simulation`, `deeplearning`, `all`.
- `numpy` and `scikit-image` are now declared, having previously arrived
  transitively.
```

The upgrade line — *use `tnia-python[all]` for the old set* — belongs here, in
the README, and in the PyPI release notes. 0001 already says so; this is the
file it lands in.

## Considered and rejected

*A test matrix across Python versions and OSes* — more signal than this
library's problems justify, and slow enough that it stops being read. One
Python, one OS, two jobs. Widen it the first time a version-specific bug
actually appears.

*Backfilling the changelog from git history* — 933 commits, mostly notebooks,
written without release notes in mind. The result would be archaeology, not
documentation.

*Automating the release to PyPI from CI* — worth doing eventually, out of scope
here. Getting the metadata verified matters more than getting it published
faster.

## Gotchas

- `pip show` returning non-zero is the *pass* condition in the minimal-install
  job. Written as `! pip show x`, it is easy to invert by accident during a
  later edit — the job should be checked once by deliberately breaking it.
- The `__init__.py` size check as drafted **fails today**: `tnia/__init__.py`
  contains `# place holder` (15 bytes). Size is the wrong test — what matters
  is that these files contain no imports, not that they are byte-empty. Either
  exempt the top-level `tnia/__init__.py`, or make it a grep for
  `^\s*(import|from)\s` across `tnia/**/__init__.py`. The grep is the honest
  version of the rule stated in `CLAUDE.md`.
- CI on a repo this size: `actions/checkout@v4` clones the full 707 MB unless
  given `fetch-depth: 1` (the default is already shallow for v4, but confirm —
  see 0002).

## Done when

- A push runs both jobs, and the minimal-install job fails if albumentations is
  moved back into the base dependencies.
- `CHANGELOG.md` exists and its first entry describes 0001 in terms a user
  hitting an ImportError would recognise.
