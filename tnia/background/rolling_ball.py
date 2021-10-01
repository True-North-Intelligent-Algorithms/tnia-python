import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from skimage.filters import rank
from skimage.morphology import disk
from skimage.transform import rescale, resize, downscale_local_mean

from skimage import (
    data, restoration, util
)

def rolling_ball_ds(image, radius, perform_mean=True, downsample=1, upper_background_percentile=-1):
    """
    wraps restoration.rolling_ball with optional smoothing and downsampling step to try and replicate the steps in the ImageJ 'Subtract Background' plugin

    Args:
        image (2d array): input image
        radius (2d array): radius of ball
        perform_mean (bool, optional): If true perform smmothing. Defaults to True.
        downsample (int, optional): If >1 downsample by this factor. Defaults to 1.
        upper_background_percentile (int, optional): If about 1 clip background at this percentile (ussually above 80 for clipping). Defaults to -1.

    Returns:
        2d array: The background image
    """

    if perform_mean:
        smoothed = rank.mean(image,disk(3))
    else:
        smoothed = image

    ds=downsample
    if ds>1:
        image_downscaled = downscale_local_mean(smoothed, (ds,ds))
    else:
        image_downscaled = image
        ds = 1
    
    r=radius/ds

    print('r',r)
    print('size',image_downscaled.shape)
    
    background_ds = restoration.rolling_ball(image_downscaled, radius=r)

    if ds>1:
        background = resize(background_ds, image.shape)
    else:
        background = background_ds

    if (upper_background_percentile>0):
        upper_background = np.percentile(background, upper_background_percentile)
        np.putmask(background, background>upper_background, upper_background)

    return background