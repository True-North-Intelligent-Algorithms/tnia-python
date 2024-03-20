import pandas as pd
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import os

image_name = 'D:\\images\\ABRF LMRG Image Analysis Study\\calibration\\calibration.tiff'
out_dir = 'D:\\images\\ABRF LMRG Image Analysis Study\\nuclei_out\\'

im = imread(image_name)
thresh = im>threshold_otsu(im)
labeled = label(thresh)
object_list = regionprops(labeled,im)

stats=pd.DataFrame(columns=['x','y','z','intensity','volume'])

for o in object_list:
    c=o.centroid
    v=o.area
    i=v*o.mean_intensity

    stats.loc[len(stats)]=[c[2],c[1],c[0],i,v]

csv_name = out_dir+'northan_brian_calibration.csv'
stats.to_csv(csv_name, index=False)
