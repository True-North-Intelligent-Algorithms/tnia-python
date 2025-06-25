#@ ImageJ ij
#@ ImgPlus img

import cellpose
from cellpose import models
from skimage import color
import numpy as np

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

try:
	overlay = color.label2rgb(result[0], img_py, bg_label=0., alpha=0.4)
except Exception as e:
	print("Caught an exception:", e)
	print("try copy img_py before call to label2rgb")
	copy = np.copy(img_py)
	overlay = color.label2rgb(result[0], copy, bg_label=0., alpha=0.4)

print(type(overlay), overlay.shape)

result_java = ij.py.to_java(result[0])

#back_overlay = ij.py.to_java(overlay)
print(overlay.dtype, overlay.min(), overlay.max())
overlay = (overlay*255).astype('uint8')
overlay_java = ij.py.to_dataset(overlay, dim_order=['row', 'col', 'ch'])

ij.ui().show(result_java)
ij.ui().show(overlay_java)
