---
layout: basic
---

# tnia-python

A collection of useful python utilities from True North Intelligent Algorithms

This project started as a means for me to test ideas and try out new python libraries.  As the number of experiments grew I started organizing things into packages.  The library is still in the very early stages of development, and is still lacking documentation, however it's at the point where I find it useful for prototyping bio-image processing protocols. 

For now, the user-base of the library is mostly people I am working with directly, teaching to in a course, or discussing image processing problems with online.  The helper functions make it quick for me to provide examples of image processing issues I am discussing. 

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
