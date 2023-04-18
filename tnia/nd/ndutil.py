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
    
def centercrop_bysize(img, size, spacing):
    """
    Center crops the input image to the desired size with respect to the spacing.

    This function mostly written by Tim-Oliver Buchholz, tibuch on github, as part of the https://github.com/fmi-faim/napari-psf-analysis plugin for napari.  

    napari-psf-analysis is licensed under the BDS 3-Clause License,
    
    tibuch

    
    Parameters:
    ----------
        img (numpy array): the image to crop
        size (tuple): the size of the crop in the same units as spacing
        spacing (tuple): the spacing of the image

    Returns:
    -------
        crop (numpy array): the cropped and centered PSF
    """
    z, y, x = np.unravel_index(np.argmax(img), img.shape)
    
    half_size = np.ceil(size / spacing / 2).astype(int)
    
    z_slice = slice(max(0, z - half_size[0]), min(z + half_size[0] + 1, img.shape[0]))
    y_slice = slice(max(0, y - half_size[1]), min(y + half_size[1] + 1, img.shape[1]))
    x_slice = slice(max(0, x - half_size[2]), min(x + half_size[2] + 1, img.shape[2]))
    
    
    crop = img[z_slice, y_slice, x_slice]
    return crop


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

