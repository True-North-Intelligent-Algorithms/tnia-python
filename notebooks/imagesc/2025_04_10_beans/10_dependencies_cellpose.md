# Dependencies

For this example we need a cellpose environment with ```tnia-python``` (my utility library) and ```napari-easy-augment-batch-dl```. 

Below are sets of suggested installation instructions for Linux, Mac M1, and Windows.

## Linux

```
    conda create -n easy_augment_cellposeenv python=3.11
    conda activate easy_augment_cellposeenv
    conda install pip
    pip install "napari[all]" # requires quotation marks to parse the square brackets on Linux, not sure if generalizes
    pip install albumentations matplotlib scipy tifffile 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning
    pip install git+https://github.com/Project-MONAI/MONAI # as of early spring 2025 need to get the github version of monai if using numpy 2.x 
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
```

## Mac M1

```
    conda create -n easy_augment_cellpose_env python=3.11
    conda activate easy_augment_cellpose_env
    conda install pip
    pip install "napari[all]" # also requires quotes on Mac
    pip install albumentations matplotlib scipy tifffile 
    pip install torch torchvision torchaudio # remove the index flag url
    pip install pytorch-lightning
    pip install git+https://github.com/Project-MONAI/MONAI # as of early spring 2025 need to get the github version of monai if using numpy 2.x 
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
```

## Windows 

```
    conda create -n easy_augment_cellpose_env=3.11
    conda activate easy_augment_cellpose_env
    conda install pip
    pip install "napari[all]"
    pip install albumentations matplotlib scipy tifffile 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning
    pip install git+https://github.com/Project-MONAI/MONAI # as of early spring 2025 need to get the github version of monai if using numpy 2.x 
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
```
