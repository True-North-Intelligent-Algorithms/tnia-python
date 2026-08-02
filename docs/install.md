---
layout: basic
---

# Installing tnia-python

`tnia-python` covers a lot of ground — plotting, deconvolution, deep learning,
simulation, io — and most people need one corner of it. So the base install is
deliberately small, and you add only the parts you want.

```
pip install tnia-python
```

That gives you the library and `numpy`, and nothing else. On its own it is
enough for very little — pick the extras for the code you actually plan to run.

## Extras

```
pip install "tnia-python[plotting]"
```

(Quotes matter on zsh and macOS, which otherwise treat the brackets as a glob.)

| extra | installs | for |
| --- | --- | --- |
| `plotting` | matplotlib, scipy, scikit-image | `tnia.plotting` — xy/zy/xz slices and projections. Also what `tnia.deconvolution` and `tnia.segmentation` draw with |
| `io` | tifffile, scikit-image | `tnia.io` — tiff and raw readers |
| `simulation` | raster-geometry, scikit-image | `tnia.simulation` — phantoms and test images |
| `deeplearning` | albumentations, raster-geometry, matplotlib, scikit-image | `tnia.deeplearning` — patch extraction, augmentation, SAM and stardist helpers |
| `machinelearning` | scikit-image | `tnia.machinelearning` — random forest feature extraction |
| `wavelets` | PyWavelets | `tnia.wavelets` |

Combine them with commas:

```
pip install "tnia-python[plotting,io]"
```

## Everything at once

```
pip install "tnia-python[all]"
```

This is the right choice if you are following along with the notebooks, or
teaching from them, and would rather not work out which example needs what.

**If you are upgrading from 0.1.x, this is the one you want.** Before 0.2.0
every install behaved like `[all]`, so upgrading in place can leave imports
failing until you add the extras you need.

## Install in editable mode

Useful if you want to run and modify the notebooks, or change library code and
have the changes take effect immediately.

```
git clone --depth 1 https://github.com/True-North-Intelligent-Algorithms/tnia-python.git
```

Then navigate to the location where you cloned the code and run

```
pip install -e ".[all]"
```

`--depth 1` skips the project history, which is large — the notebooks in this
repository carry their output images.

## Install from git

```
pip install "tnia-python[plotting] @ git+https://github.com/True-North-Intelligent-Algorithms/tnia-python"
```

## Why extras, rather than installing everything

The dependencies of this library are complex, and not all of them are needed to
run any given example. Someone who wants three functions that draw slices with
matplotlib should not have to install an augmentation library and a computer
vision stack to get them.

That reasoning has been on this page for a long time. What changed is the
mechanism: it used to be a request that you install the dependencies by hand,
and extras are how `pip` expresses the same intent — so it is now a supported
install rather than a manual step.

Two things extras cannot do:

- **They cannot prune the dependencies of dependencies.** `[deeplearning]`
  still installs OpenCV, because albumentations requires it.
- **They are not retroactive.** Upgrading in an environment that already has
  everything will not remove anything. The smaller install only shows up in a
  fresh environment.

---

# Legacy — installing the dependencies by hand

**Kept for reference. You should not need any of this on 0.2.0 or later** — use
the extras above instead. This is what the page said before extras existed, and
parts of it are now well out of date (Python 3.9, a `clij2-fft` build on
test.pypi). It is here in case you are working in an older environment that was
set up this way.

The approach was to install almost nothing through `setup.py`, and to leave it
to the user to assemble an environment suiting the code they wanted to run:

```
    conda create --name decon-napari python=3.9
    conda activate decon-napari
    conda install -c conda-forge jupyterlab
    conda install -c conda-forge pyopencl==2021.2.6 hdbscan numba=0.55.1
    pip install devbio-napari
    conda install -c conda-forge fftw
    pip install napari-sdeconv
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python
    pip install --index-url https://test.pypi.org/simple/ --no-deps clij2-fft
    pip install stardist
    pip install raster-geometry
```

Mac-users please also install this:

```
    conda install -c conda-forge ocl_icd_wrapper_apple
```

Linux users please also install this:

```
    conda install -c conda-forge ocl-icd-system
```

If opencl is not working Linux users may need to install opencl with pip

```
pip install pyopencl
```
