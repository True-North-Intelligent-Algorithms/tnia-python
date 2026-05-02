#@ ImageJ ij
#@ ImagePlus img

import napari

img_py=ij.py.from_java(img)

viewer = napari.Viewer()
viewer.add_image(img_py)
viewer.run()
