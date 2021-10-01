from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data
import matplotlib.pyplot as plt

image = data.coins()

image_downscaled = downscale_local_mean(image, (4,4))

image_resized = resize(image_downscaled, image.shape)

fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original')

ax[1].imshow(image_downscaled, cmap='gray')
ax[1].set_title('downscaled')

ax[2].imshow(image_resized, cmap='gray')
ax[2].set_title('resized')

plt.show()

