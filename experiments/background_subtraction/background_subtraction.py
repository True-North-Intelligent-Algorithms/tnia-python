import helper

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.io import imsave

from tnia.background.dilate_subtract import dilate_subtract

# Convert to float: Important for subtraction later which won't work with uint8
image = img_as_float(data.coins())
image = gaussian_filter(image, 1)

p=80
bs, dilated, seed, mask = dilate_subtract(image, p)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(16, 5),
                                    sharex=True,
                                    sharey=True)

ax0.imshow(image, cmap='gray')
ax0.set_title('original image')
ax0.axis('off')

ax1.imshow(seed, vmin=seed.min(), vmax=seed.max(), cmap='gray')
ax1.set_title('seed')
ax1.axis('off')

ax2.imshow(mask, vmin=mask.min(), vmax=mask.max(), cmap='gray')
ax2.set_title('mask')
ax2.axis('off')

helper.plot_result(image, dilated)
