---
layout: basic
---

## Dealing with extruciatingly slow Conda update

### If conda is very slow in the base environment

Common advice is to update conda with ```conda update conda``` or use [mamba](https://anaconda.org/conda-forge/mamba) ```conda install -c conda-forge mamba```.  

However if the base conda environment is messed up this can be super slow (hours).  In this case roll back the base environment to a state before the base environment became slow.  

```
conda list --revisions
````

then roll back to an earlier revision.  Often it makes sense to roll back base to revision 0, as we don't ussually want to install anything in base

To roll back to revision 0 (Note: if you have installed packages you use in base, then take an inventory of these packages before rolling back, and install them in a new environment). 

```
conda install --revision 0
```

### If conda is very slow in other environments

Often times it gets stuck at 'solving environment'. 

In this case (in my experience) it may make sense to take an inventory of all the dependencies in the environment and recreate it, taking care to look at warning messages and making sure all the versions of each package are compatible.  



