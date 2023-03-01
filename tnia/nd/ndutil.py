import numpy as np

def centercrop(im, shape):
    """
    Center crops the input image to the desired output shape.

    Args:
        im (numpy.ndarray): The input image to crop.
        shape (tuple): The desired output shape. Must have the same number of dimensions as the input image.

    Returns:
        numpy.ndarray: The center-cropped image.
    """

    if len(im.shape) != len(shape):
        raise ValueError("Input image and desired output shape must have the same number of dimensions.")

    start_indices = [(im.shape[i] - shape[i]) // 2 for i in range(len(im.shape))]

    slices = [slice(start_indices[i], start_indices[i] + shape[i]) for i in range(len(im.shape))]

    return im[tuple(slices)]


def makergb(r,g,b):
    """given 2d r, g and b numpy arrays combine into one rgb array

    Args:
        r ([2d numpy array]): red
        g ([2d numpy array]): green
        b ([2d numpy array]): blue

    Returns:
        [numpy array]: rgb 
    """
    r=(255.*r/r.max()).astype('uint8')
    g=(255.*g/g.max()).astype('uint8')
    b=(255.*b/b.max()).astype('uint8')

    rgb=np.stack([r,g,b])
    
    return np.transpose(rgb,(1,2,0))

