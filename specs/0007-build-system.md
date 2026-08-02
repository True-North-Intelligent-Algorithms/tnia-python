# 0007 — Finish the packaging modernization

**Status:** planned, not started. Written 2026-08-02. **Deliberately after
0001** — do not pull this forward.

## Scope

0001 ports `setup.cfg` to `pyproject.toml` and adds the extras table, and says
of that port: *"This is the whole modernization needed — a larger cleanup is a
separate job."* This is that job. It exists so the leftovers are written down
rather than smuggled into 0001, where they would obscure the one change that
matters.

Nothing here blocks anything. It is worth doing when packaging is already open
in front of you, and not otherwise.

## What is missing today

Measured 2026-08-02, from `setup.cfg`:

- **No `requires-python`.** Nothing stops `pip install tnia-python` on 3.7,
  where it will fail obscurely. Pick a floor and declare it.
- **No classifiers, no keywords, no `[project.urls]`.** The PyPI page carries
  `README_pypi.md` and nothing else — no license classifier, no link to the
  repo or the docs site, no "Development Status" signal. This is the cheapest
  credibility item in the whole list.
- **The version lives in one place, `setup.cfg`, and `tnia/__init__.py` is the
  string `# place holder`.** There is no `tnia.__version__`, so a user who
  reports a bug cannot easily say which version they are on. Either set
  `__version__` and make `pyproject.toml` read it via `dynamic = ["version"]`,
  or keep `pyproject.toml` authoritative and expose
  `importlib.metadata.version("tnia-python")` from `tnia/__init__.py`. The
  second is less machinery and does not put an import in an `__init__.py`.
- **`MANIFEST.in` and `[options.exclude_package_data]` overlap and disagree.**
  MANIFEST does `recursive-include tnia *` then excludes the site directories;
  `setup.cfg` separately declares `exclude_package_data` for `notebooks/*`,
  `images/*`, `presentations/*` — paths that are not inside the package, so
  those patterns have never matched anything. After the port, express the
  intent once, in `[tool.setuptools]`, and delete whichever file ends up
  redundant.
- **Releases are manual.** `dist/` holds 17 artifacts going back to 0.1.25 —
  built locally, uploaded by hand. Worth moving to a tag-triggered CI job with
  PyPI trusted publishing, which removes the API token from the laptop and
  makes `dist/` deletable.

## Also worth deciding here

- **Build backend.** Keep setuptools, or move to hatchling. Setuptools is
  working and the extras in 0001 are backend-agnostic; hatchling buys a shorter
  config and better defaults. No urgency either way — this is a preference,
  and 0001 depends on neither.
- **Lint and format.** There is no ruff/flake8/black config. If one gets
  added, it should land as its own commit with a `# noqa`-free baseline, not as
  a diff on top of unrelated work — a formatter run across 87 modules is
  unreviewable next to anything else.
- **Where `pre-commit` config lives.** 0002 introduces
  `.pre-commit-config.yaml` for `nbstripout`. If lint hooks are wanted, they
  join that file rather than starting a second mechanism.

## Considered and rejected

*Doing all of this inside 0001* — the reason for the split. 0001 has one
externally visible claim to prove (`[plotting]` installs no OpenCV) and one
acceptance test in 0004 to prove it. Bundling classifiers, a backend swap, and
a release pipeline into the same change makes a regression there hard to
bisect and hard to review.

*`setuptools-scm` for versioning from git tags* — attractive, but this repo
tags irregularly and a shallow clone (0002) has no tags at all, which breaks
the build in exactly the environment 0002 is steering people toward.

## Gotchas

- Changing `requires-python` after the fact does not retract old releases; pip
  will still resolve to an ancient version for a too-old interpreter. Declare
  it once and leave it.
- `dynamic = ["version"]` reading from `tnia/__init__.py` means the build
  imports or parses that file. Parsing is fine; importing it at build time is
  the thing that quietly breaks scoped installs.
- Trusted publishing needs the PyPI project configured to accept it before the
  first CI release, and the first attempt fails confusingly if that step is
  skipped.

## Done when

- `pip show tnia-python` and the PyPI page both say what Python it needs, what
  license it is under, and where the source is.
- `python -c "import tnia; print(tnia.__version__)"` works.
- `dist/` can be deleted from the working tree without losing the ability to
  cut a release.
