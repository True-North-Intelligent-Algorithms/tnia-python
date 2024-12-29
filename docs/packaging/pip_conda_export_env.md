---
layout: basic
---

# Creating repeatable environment with conda, pip, and pip-tools 

We need to create an ```environment.yml``` file and use ```pip-compile``` to collect the transitive dependencies.  

For example for a stardist/napari environment you can first make an ```environment.yml``` which contains the conda dependencies and the pip dependencies.  In this case we will be compiling the pip dependencies into a ```requirements.txt``` file.  The ```environment.yml``` will be similar to below. 

```
name: stardist_napari_windows
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - cudatoolkit=11.8
  - cudnn=8.1.0
  - pip:
    - -r requirements.txt
```

then ```pip install pip-tools``` and make the following ```requirement.in```

```
numpy==1.26
napari[all]
tensorflow<2.11
stardist
gputools==0.2.15
edt
```

Then ```pip-compile requirement.in```, which creates a massive ```requirements.txt``` [see example here](https://github.com/True-North-Intelligent-Algorithms/notebooks-and-napari-widgets-for-dl/blob/7e5d6e2c5b62970af98706f81d2b0ccd06cfe423/dependencies/windows_stardist/requirements.txt).  

End user just needs to get the ```environment.yml``` and ```requirements.txt``` and run

```
conda env create -f environment.yml

````
