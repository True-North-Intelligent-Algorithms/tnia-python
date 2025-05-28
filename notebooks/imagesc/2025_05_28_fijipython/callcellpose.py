#@ ImageJ ij

from scyjava import to_python as j2p
from scyjava import to_java as p2j

from skimage.io import imread
from cellpose import models
#print(cellpose.version)

model = models.Cellpose(gpu=True, model_type='cyto2')

data=ij.io().open("D:\\images\\tnia-python-images\\imagesc\\2024_03_27_SOTA_segmentation\\images\\cell_00003.bmp")
datanp = imread("D:\\images\\tnia-python-images\\imagesc\\2024_03_27_SOTA_segmentation\\images\\cell_00003.bmp")
print(type(data))
print(type(datanp))
ij.ui().show(data)
#ij.ui().show(datanp)
test=j2p(data)
print(type(test))

result = model.eval(test, diameter=100, niter=2000)

back = p2j(result)

ij.ui().show(back)
