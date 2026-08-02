---
layout: basic
---

# 0002 — Make the repo clonable

**Status:** planned, not started. Written 2026-08-01.

## The problem

`README.md` opens by telling the reader to `git clone`. That clone is **707 MB
of history for a 704 MB checkout**, and 675 MB of it is notebooks — 442 of
them, most carrying their rendered outputs.

This costs nothing to a PyPI consumer: wheels exclude `notebooks/` via
`MANIFEST.in`, so scikit-ops never sees it. It costs everything to the person
in a presentation audience who follows the README on conference wifi.

## What the weight actually is

Measured 2026-08-01, tracked files only:

| path | tracked size |
| --- | --- |
| `notebooks/` | 674.7 MB |
| `presentations/` | 24.5 MB |
| `experiments/` | 10.3 MB |
| `assets/` | 3.5 MB |
| `docs/` | 2.8 MB |
| `tnia/` (the actual library) | **0.3 MB** |

Inside `notebooks/`: 534 MB of `.ipynb`, 140 MB of committed data files (`.tif`
and friends, 135 of them). 291 notebooks are over 200 KB, which for a notebook
means embedded output images. The five worst are 120 MB between them:

```
36.9 MB  notebooks/misc/for_js.ipynb
32.6 MB  notebooks/imagesc/2024_02_06_tem_pores/12_augment_patches.ipynb
24.6 MB  notebooks/imagesc/2025_04_30_scale/20_augmentation.ipynb
16.3 MB  notebooks/imagesc/2024_01_12_plant_roots/12_augment_patches.ipynb
10.4 MB  notebooks/imagesc/2025_01_04_unet_practice/35_unet_semantic_sparse_rgb.ipynb
```

The library is 0.04% of what a clone downloads.

## The plan

Three steps, increasing in disruption. **Do 1 and 2 now; 3 is a judgement
call.**

### 1. Stop the bleeding — `nbstripout`

Add a `.pre-commit-config.yaml` with `nbstripout`, so committed notebooks carry
code and markdown but not output blobs.

```yaml
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

`pre-commit install` once per clone. Notebooks still *run* and still *display*
in a live session; they just do not carry pixels into git.

This changes how notebooks read on GitHub — an unexecuted notebook shows no
figures. That is the real cost of this decision, and it is why step 3 exists.
Weigh it before committing to stripping everything: the notebooks are partly a
gallery.

**Middle option, if the gallery matters:** strip by default, and keep outputs
in a small hand-picked set by adding those paths to a `nbstripout` exclude. The
point is that 291 notebooks are not deliberately a gallery — five of them might
be.

### 2. One line in the README

```
git clone --depth 1 https://github.com/True-North-Intelligent-Algorithms/tnia-python.git
```

Downloads the checkout without 933 commits of history. Costs nothing, helps
immediately, and does not require any decision about the notebooks.

### 3. Consider splitting the notebooks out

Move `notebooks/` to `tnia-python-notebooks`, keep ~10 curated examples here
under `examples/`. This is the only step that shrinks `.git`, and only if
paired with a history rewrite — otherwise the bytes stay in the pack forever.

Do not do this in the same week as the presentation. Every link into
`notebooks/` — from the README, from the docs site, from years of forum.image.sc
posts — breaks. That last one is the reason to hesitate: those links are the
library's actual user-facing documentation today.

## Considered and rejected

*Rewrite history now with `git filter-repo`* — the only thing that reclaims the
707 MB. Rejected for this iteration: it invalidates every existing clone and
every commit SHA ever pasted into a forum thread, for a benefit (`.git` size)
that `--depth 1` mostly delivers for free. Revisit if the notebooks are split
out anyway.

*Git LFS for the `.tif` files* — 140 MB, but it adds a hard requirement on
every cloner and does not touch the 534 MB of notebook outputs, which are the
real problem.

## Gotchas

- `nbstripout` rewrites files on commit. The working tree currently has
  ~30 modified notebooks (see `git status`); running the hook will touch them
  all at once. Commit or stash that churn *first*, so the stripping shows up as
  its own reviewable commit.
- Stripping is not retroactive. Existing history keeps its blobs — this only
  stops growth.
- `--depth 1` clones cannot push branches without a follow-up fetch. Fine for
  the audience, worth a note if a contributor ever appears.

## Done when

- A fresh `git clone --depth 1` is under ~750 MB and the README says so.
- `pre-commit run --all-files` is idempotent, and a newly committed notebook
  adds kilobytes rather than megabytes.
- A decision on step 3 is written down here, either way.
