# Dependencies

For this example you need Napari, several pytorch related dependencies, tnia-python (my utility library), and napari-easy-augment-batch-dl. 

(as an aside, I should probably provide a environment file, however I often just list the dependencies one by one, because often users have an environment with most of this set up and just need to install one or two extras).

Please post on image.sc if setting up the dependencies is not working for you. 

Note: Exact version of Cuda is probably not important.  cu124 is simply what I've been testing with lately.  

## Linux

```
    conda create -n vessels_lightsheet python=3.13
    conda activate vessels_lightsheet
    pip install "napari[all]" # requires quotation marks to parse the square brackets on Linux, not sure if generalizes
    pip install albumentations
    pip install matplotlib
    pip install "tensorflow[and-cuda]" # as above, requires quotation marks
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install pytorch-lightning
    pip install git+https://github.com/Project-MONAI/MONAI 
    pip install scipy
    pip install tifffile
    pip install czifile
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
```

## Mac M1

```
    conda create -n vessels_lightsheet python=3.13
    conda activate vessels_lightsheet
    pip install "napari[all]" # also requires quotes on Mac
    pip install albumentations
    pip install matplotlib
    pip install torch torchvision torchaudio # remove the index flag url
    pip install pytorch-lightning
    pip install git+https://github.com/Project-MONAI/MONAI 
    pip install scipy
    pip install tifffile
    pip install czifile
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
```

## Windows 

```
    conda create -n vessels_lightsheet python=3.13
    conda activate vessels_lightsheet 
    pip install "napari[all]"
    pip install albumentations
    pip install matplotlib
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning
    pip install git+https://github.com/Project-MONAI/MONAI 
    pip install scipy
    pip install tifffile
    pip install czifile
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git
```
