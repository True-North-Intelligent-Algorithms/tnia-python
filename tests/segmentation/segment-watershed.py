import pyclesperanto_prototype as cle

from skimage.filters import threshold_local
from skimage.io import imread
import matplotlib.pyplot as plt
from tnia.segmentation.rendering import draw_label_outline
from tnia.segmentation.separate import separate_touching 
#import tnia.io.bioformats_helper 
#import os
from skimage.morphology import disk
from skimage.filters import median
from skimage.morphology import label
import napari.viewer
import numpy as np

parent_dir = "/home/bnorthan/elephasbio/data/2021-05-14 Day 3 VD2 EMT-6 fragments/fragments-001/";
file_name = "cropped.tif"

im=imread(parent_dir+file_name)

# initialize GPU
cle.select_device("RTX")
print("Used GPU: " + cle.get_device().name)


# push image to GPU memory
input = cle.push(im)
print("Image size in GPU: " + str(input.shape))

blurred = cle.gaussian_blur(input, sigma_x=1, sigma_y=1)
binary = cle.threshold_otsu(blurred)

segmented = np.empty_like(input)
labeled = np.empty_like(input).astype('uint16')

last_num=0

# loop through each 
for i in range(im.shape[0]):
    segmented[i,:,:] = separate_touching(binary[i,:,:], 10, 1)
    labeled[i,:,:], temp = label(segmented[i,:,:], return_num=True, connectivity=1)
    labeled[labeled>0]=labeled[labeled>0]+last_num 
    last_num=temp

viewer = napari.Viewer()

viewer.add_image(im, scale=[3,1,1])
viewer.add_image(blurred, scale=[3,1,1])
binary=binary.astype('float32')
viewer.add_image(binary, scale=[3,1,1])
viewer.add_labels(labeled, scale=[3,1,1])
'''
plt.imshow(blurred[10,:,:])
plt.imshow(binary[10,:,:])
plt.show()
'''