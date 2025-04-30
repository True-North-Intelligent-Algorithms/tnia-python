# Dependencies

Microsam/Cellpose environment should run example 15/32/35 and 70.  Stardist environment should run examples 30 and 31.


## Microsam/cellpose

Should work for all OS.  Microsam install takes care of Napari, Pytorch and Cuda. 

conda create -n microsam_cellpose -c conda-forge python=3.11
conda activate microsam_cellpose
conda install -c conda-forge micro_sam
conda install cellpose
pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git
pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git

## Stardist Mac/Linux

conda create -n stardist python=3.11
conda activate stardist
conda install pip
pip install "napari[all]" # requires quotation marks to parse the square brackets on Linux, not sure if generalizes
pip install "tensorflow[and-cuda]" # as above, requires quotation marks (only needed for stardist)
pip install stardist 
pip install gputools #==0.2.15 may no longer needed if we are happy with np 2
pip install edt # pip throws: numba v0.60.0, tensorflow v2.18.0 require lower versions of numpy. Should still work but iffy.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git

## Stardist Windows

conda create -n stardist_windows python=3.11
conda activate stardist_windows
pip install numpy==1.26.4 # start with numpy 1.26 and hope nothing upgrades it...
pip install "napari[all]"
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.1.0
pip install "tensorflow<2.11"
pip install stardist==0.8.5
pip install gputools==0.2.15
pip install edt
pip install reikna==0.8.0 
pip install numpy==1.26.4 # in case numpy got upgraded go back (hacky yes, but this is what people are reporting works)
pip install numba==0.59.1
pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git




