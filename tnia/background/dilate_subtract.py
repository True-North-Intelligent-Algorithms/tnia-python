import numpy as np
from skimage.morphology import reconstruction

def dilate_subtract(image, p):
    """ dilate a grayscale image then use the dilation as estimate of background and subtract

        see https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html

        Args:
            image (2d array): input image
            p (float or double): percentile used to calculate h factor, h factor is used to create seed image

        Returns:
            image-background, dilated image, seed image, mask image
    """
    h = np.percentile(image, p)
    seed=image-h
    np.putmask(seed, seed<0, 0)

    mask = image

    dilated = reconstruction(seed, mask, method='dilation')

    return image-dilated, dilated, seed, mask


