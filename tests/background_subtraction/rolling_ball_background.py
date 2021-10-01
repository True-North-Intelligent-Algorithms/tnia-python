import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from skimage.filters import rank
from skimage.morphology import disk
import helper
from skimage.transform import rescale, resize, downscale_local_mean

from skimage import (
    data, restoration, util
)

image = data.coins()
smoothed = image #rank.mean(image,disk(3))

ds=4
image_downscaled = downscale_local_mean(smoothed, (ds,ds))

r=50/ds
background_ds = restoration.rolling_ball(image_downscaled, radius=r)

background = resize(background_ds, image.shape)

helper.plot_result(image, background)

imsave('C:\\Users\\bnort\\work\\testing\\images\\diagnostics_coins\\background.tif',background)
imsave('C:\\Users\\bnort\\work\\testing\\images\\diagnostics_coins\\coins.tif',image)
