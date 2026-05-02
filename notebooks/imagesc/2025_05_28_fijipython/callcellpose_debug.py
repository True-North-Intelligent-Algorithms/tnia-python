#@ ImageJ ij
#@ ImagePlus img

import cellpose
from cellpose import models
from skimage import color
import numpy as np

print('ij is',type(ij))

# print cellpose version
print(cellpose.version)

# check if GPU is available
import torch

if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")

model = models.CellposeModel(gpu=True, model_type='cyto3')

print(type(img))
img_py=ij.py.from_java(img)
print(type(img_py), img_py.shape)

result = model.eval(img_py)

print(type(result[0]),result[0].shape)

result_imp = ij.py.to_imageplus(result[0])
result_imp.setTitle("result")

ij.ui().show(result_imp)

ij.IJ.setThreshold(result_imp, 1, 100000)
ij.py.run_plugin("Convert to Mask", imp=result_imp);

ij.py.run_plugin("Create Selection", imp=result_imp);
ij.py.run_plugin("Add to Manager"); #// adds the ROI(s) to the ROI Manager

ij.IJ.selectWindow(img.getTitle()); 
ij.py.run_plugin("Show Overlay", imp=img);

