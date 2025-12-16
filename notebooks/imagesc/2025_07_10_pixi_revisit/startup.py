import numpy as np

import sys
print("Python executable:", sys.executable)
print("Startup script running in environment:", sys.prefix)

import napari
print("Napari version:", napari.__version__)

import torch

# Print PyTorch version
print("PyTorch version:", torch.__version__)

# Check if CUDA (GPU) is available
print("CUDA available:", torch.cuda.is_available())

# If available, print the name of the GPU
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("Running on CPU")

import cellpose
print("Cellpose version:", cellpose.version)

viewer = napari.Viewer()
napari.run()
