#@ ImageJ ij
#@ ImagePlus img

import napari
import numpy as np

img_py=ij.py.from_java(img)
print(type(img_py))
img_np = np.copy(img_py)
print(type(img_np))


viewer = napari.Viewer()
viewer.add_image(img_np)
napari.run()
