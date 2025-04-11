import numpy as np

def create_saturation_mask(image):
    """
    Create a mask for saturated pixels in an image.

    This is useful for masking out saturated pixels in an image before deconvolution.

    Parameters
    ----------
    image : numpy.ndarray
        Image to create a saturation mask for.
    
    Returns
    -------
    numpy.ndarray
        Mask for saturated pixels.
    """

    mask  = np.ones_like(image)
    dtype = image.dtype

    if dtype == np.uint8:
        mask[image==255] = 0

    elif dtype == np.uint16:
        mask[image==65535] = 0
        
    return mask

