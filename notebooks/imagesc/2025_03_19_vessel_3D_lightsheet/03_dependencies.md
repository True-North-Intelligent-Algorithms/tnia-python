# Dependencies

For this example you need Napari, several pytorch related dependencies, tnia-python (my utility library), and napari-easy-augment-batch-dl. 

(as an aside, I should probably provide a environment file, however I often just list the dependencies one by one, because often users have an environment with most of this set up and just need to install one or two extras).

Please post on image.sc if setting up the dependencies is not working for you. 

Note 1: Exact version of Cuda is probably not important.  cu124 is simply what I've been testing with lately.  

Note 2: We use Python 3.11 because at the moment it plays nicer with Cellpose. 

## 1.  Create and activate a new environment
```
    conda create -n vessels_lightsheet python=3.11
    conda activate vessels_lightsheet
```

## 2.  Install Pytorch-GPU

The [conda-forge version](https://anaconda.org/conda-forge/pytorch) version seems to be smart enough to install the right cuda version on Windows/Linux and the CPU version on MAC

```
    conda install conda-forge::pytorch
```

Alternatively you can use the [pytorch pip](https://pytorch.org/get-started/locally/) version which requires slightly different instruction for Windows/Mac/Linux

Windows/Linux

```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Mac

```
    pip3 install torch torchvision torchaudio
```

## 3.  Install remaining dependencies

```
    pip install "napari[all]" # seems to roll back numpy to 2.1.3
    pip install albumentations matplotlib scipy tifffile czifile 
    pip install --upgrade git+https://github.com/Project-MONAI/MONAI 
    pip install --upgrade git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git
    pip install --upgrade git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
```

### 4. (Optional) Install Cellpose 

Cellpose is not needed for this example, however if you have setup the GPU version of Pytorch properly you should be able to add Cellpose to the environment to complete a useful DL environment. 

```
    pip install cellpose
```