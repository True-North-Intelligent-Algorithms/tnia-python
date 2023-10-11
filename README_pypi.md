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

## Dependencies  

We tend not to install many of the dependencies via setup.py.  The dependencies are complex and not all are needed to run many examples.  Thus we leave it up to the user to install dependencies manually, allowing them to potentially install a minimum set of dependencies for the specific code they are interested in running.

Most of the deconvolution related functionality uses clij2-fft.  Some functionality uses clesperanto.  The current recommended steps to create a conda/mamba environment for tnia-python are as follows

```
mamba create --name decon-dl-env python=3.9 devbio-napari pyqt -c conda-forge -c pytorch
mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install cupy-cuda11x
pip install tnia-python
pip install clij2-fft
pip install psfmodels
pip install "tensorflow<2.11"
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
