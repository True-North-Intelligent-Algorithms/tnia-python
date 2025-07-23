# Dependencies

You will need to install napari and for augmentation you will need albumentations library.  Also explicitly install numpy 1.26.  (We have not tested with numpy 2.0 so it is a good idea to explicitly install numpy 1.26 to avoid another dependency installing numpy 2.x)

You will also need cellpose.  

In addition to the popular deep learning toolkits these tutorials use 2 helper and widget libraries.  

**tnia-python:** General image processing and deep learning helpers.  
**napari-easy-augment-batch-dl:**  Napari widgets for labelling and running deep learning workflows on image series.  

## Linux

```
    conda create -n TNIA_cellpose python=3.10
    conda activate TNIA_cellpose
    pip install numpy==1.26 # 
    pip install "napari[all]" # requires quotation marks to parse the square brackets on Linux, not sure if generalizes
    pip install albumentations
    pip install matplotlib
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning
    pip install monai
    pip install scipy
    pip install tifffile
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
```

## Mac M1

```
    conda create -n TNIA_cellpose python=3.10
    conda activate TNIA_cellpose
    pip install numpy==1.26 # 
    pip install "napari[all]" # also requires quotes on Mac
    pip install torch torchvision torchaudio # remove the index flag url
    pip install pytorch-lightning
    pip install monai
    pip install scipy
    pip install tifffile
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
```

## Windows 

```
    conda create -n TNIA_cellpose python=3.10
    conda activate I2k2024_pytorch
    pip install numpy==1.26
    pip install "napari[all]"
    pip install albumentations
    pip install matplotlib
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning
    pip install monai
    pip install scipy
    pip install tifffile
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
```
