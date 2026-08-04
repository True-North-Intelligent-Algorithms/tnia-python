# tnia-python

A collection of useful python utilities from True North Intelligent Algorithms

If you need support for the library please post a question on the [Image.sc Forum](https://forum.image.sc/).

This project started as a means for me to test ideas and try out new python libraries.  As the number of experiments grew I started organizing things into packages.  The library is still in the very early stages of development, and is still lacking documentation, however it's at the point where I find it useful for prototyping bio-image processing protocols. 

For now, the user-base of the library is mostly people I am working with directly, teaching to in a course, or discussing image processing problems with online.  The helper functions make it quick for me to provide examples of image processing issues I am discussing.  

There are useful helper function to

* show max, sum and slice projections of 3D arrays
* Generate PSFs with simple API to psfmodels and sdeconv
* Extract PSFs from a bead image
* Draw circles and spheres in images to create phantom images
* Apply a forward imaging model (convolution + noise)

In the coming months and years I plan to keep iterating on the library and making it more usable.  Please reach out to me if you have any questions. 

## Installation

The library covers a lot of ground and most people need one corner of it, so the base install is deliberately small and you add only the parts you want.

```
pip install tnia-python
```

That gives you the library and `numpy`, and nothing else.  Pick the extras for the code you actually plan to run:

```
pip install "tnia-python[plotting]"
```

(Quotes matter on zsh and macOS, which otherwise treat the brackets as a glob.)

| extra | installs | for |
| --- | --- | --- |
| `plotting` | matplotlib, scipy, scikit-image | `tnia.plotting` — xy/zy/xz slices and projections.  Also what `tnia.deconvolution` and `tnia.segmentation` draw with |
| `io` | tifffile, scikit-image | `tnia.io` — tiff and raw readers |
| `simulation` | raster-geometry, scikit-image | `tnia.simulation` — phantoms and test images |
| `deeplearning` | albumentations, raster-geometry, matplotlib, scikit-image | `tnia.deeplearning` — patch extraction, augmentation, SAM and stardist helpers |
| `machinelearning` | scikit-image | `tnia.machinelearning` — random forest feature extraction |
| `wavelets` | PyWavelets | `tnia.wavelets` |

Combine them with commas, or take everything at once:

```
pip install "tnia-python[plotting,io]"
pip install "tnia-python[all]"
```

`[all]` is the right choice if you are following along with the notebooks and would rather not work out which example needs what.

**If you are upgrading from 0.1.x, `[all]` is the one you want.**  Before 0.2.0 every install behaved like `[all]`, so upgrading in place can leave imports failing until you add the extras you need.

Some heavier optional pieces are still installed by hand, because they are platform specific and not on every user's path — `clij2-fft` and `psfmodels` for deconvolution and PSF generation, `stardist` and `cellpose` for segmentation.

Full instructions, including editable installs and installing from git, are in [docs/install.md](https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/docs/install.md).
