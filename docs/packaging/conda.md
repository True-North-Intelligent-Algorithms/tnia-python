---
layout: basic
---

## Dealing with extruciatingly slow Conda update

Common advice is to update conda with ```conda update conda``` or use [mamba](https://anaconda.org/conda-forge/mamba) ```conda install -c conda-forge mamba```.  

However if the base conda environment is messed up this can be super slow (hours).  In this case roll back the base environment to a state before the base environment became slow.  

```
conda list --revisions
````

then

```
conda install --revision N
```

