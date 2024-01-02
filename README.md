---
layout: basic
---

# tnia-python

A collection of useful python utilities from True North Intelligent Algorithms

This project started as a means for me to test ideas and try out new python libraries.  As the number of experiments grew I started organizing things into packages.  The library is still in the very early stages of development, and is still lacking documentation, however it's at the point where I find it useful for prototyping bio-image processing protocols. 

For now, the user-base of the library is mostly people I am working with directly, teaching to in a course, or discussing image processing problems with online.  The helper functions make it quick for me to provide examples of image processing issues I am discussing. 

If you have stumbled upon this repo without prompting and instructions from me and are curious what it does, browse the [notebooks folder](https://github.com/True-North-Intelligent-Algorithms/tnia-python/tree/main/notebooks).  Most useful will be the notebook examples for deconvolution, plotting, and deep learning.  

In the coming months and years I plan to keep iterating on the library and making it more usable.  Please reach out to me if you have any questions. 

## Install TNIA-Python in editable mode

A useful way to install is to get the code from github and then install in editable mode.  THis allows one to run and modify the notebooks, and modify other code then have modifications available immediately. 

```
git clone https://github.com/True-North-Intelligent-Algorithms/tnia-python.git
```

Then navigate to the location where you cloned the code and run 

```
pip install -e .
```

## Install from git

```
pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python
```

## Dependencies  

We tend not to install many of the dependencies via setup.py.  The dependencies are complex and not all are needed to run many examples.  Thus we leave it up to the user to install dependencies manually, allowing them to potentially install a minimum set of dependencies for the specific code they are interested in running.

# Current recommended method to set up a bio-imaging environment for running tnia-python examples

## Using Mamba and devbio-napari

### Mamba

(Credit to Robert Haase https://twitter.com/haesleinhuepf, for these instructions)

Install mambaforge on your computer as explained in this [blog post](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).  Installing mambaforge will make installing other toolkits much faster.

If you already have some conda or anaconda installation on your computer, ***please install mambaforge*** anyway as explained in the blog post linked above. 

### Devbio-napari

Devbio-napari is a bundle of napari plugins useful for 3D+t image processing and analysis.  It's a big installation but installs jupyter notebooks, napari, opencl and several useful plugins and libraries all at once.  

Install [devbio-napari](https://github.com/haesleinhuepf/devbio-napari#installation) into a fresh conda environment, e.g. using this command:

```
mamba create --name decon-dl-env python=3.9 devbio-napari pyqt -c conda-forge -c pytorch
```

When you are done, you can test your setup by executing these commands from the command line:
```
mamba activate decon-dl-env

naparia
```

### Install tnia-python in editable mode

```
git clone https://github.com/True-North-Intelligent-Algorithms/tnia-python.git
```

Then navigate to the location where you cloned the code and run 

```
pip install -e .
```

### clij2-fft and psf-models

The tnia-python library is used for projections and some helper functions, clij2-fft is used for deconvolution, and psfmodels is used for diffractions based PSFs. 
SS
```
pip install tnia-python
pip install clij2-fft
pip install psfmodels
```

Optionally we can also install sdeconv as an alternative library for generating PSFs

pip install napari-sdeconv

### Tensoflow and Cuda

For Windows we need to install tensorflow<2.11 and a compatible earlier version of Cuda 

```
mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install cupy-cuda11x
```

```
pip install "tensorflow<2.11"
```

The tensorflow version (<2.11) is not required for Mac or Linux and for Linux we can install tensorflow and cuda together

```

```

### Stardist and CSB Deep 

Then install stardist which should also install CSBDeep

```
pip install stardist
```

### raster-geometry

Raster-Geometry is used to generate simulate images for deconvolution testing and deep learning training.

```
pip install raster-geometry
```

### Some older examples may use fftw

```
    conda install -c conda-forge fftw
```

### Additional hints for Mac and Linux users

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
