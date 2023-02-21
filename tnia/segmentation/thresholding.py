import numpy as np
from skimage.filters import threshold_local

def local_global_threshold(image, local_window_size, global_threshold):
    """
    Applies a local threshold and a global threshold to an image and returns the logical AND of the two.

    Parameters:
    -----------
    image : ndarray
        The input image to be thresholded. The image should have a shape of (N, M), where N and M are the dimensions of the image.
    local_window_size : int
        The size of the window to use for local thresholding.
    global_threshold : float
        The global threshold to use for thresholding.

    Returns:
    --------
    ndarray
        The logical AND of the local and global thresholded image, with the same shape as the input image.

    Notes:
    ------
    This function uses the threshold_local function from the skimage.filters module to apply a local threshold to the input image. Pixels in the input image with values greater than the local threshold and the global threshold are set to True in the output image; all other pixels are set to False.
    """
    return np.logical_and(image>threshold_local(image, local_window_size), image>global_threshold)

def local_global_threshold_3d(image, local_window_size, global_threshold):
    """
    Applies a local threshold and a global threshold to a 3D image and returns the logical AND of the two.

    Parameters:
    -----------
    image : ndarray
        The input 3D image to be thresholded. The image should have a shape of (N, M, L), where N, M, and L are the dimensions of the image.
    local_window_size : int
        The size of the window to use for local thresholding.
    global_threshold : float
        The global threshold to use for thresholding.

    Returns:
    --------
    ndarray
        The logical AND of the local and global thresholded image, with the same shape as the input image.

    Notes:
    ------
    This function calls the local_global_threshold function on each 2D slice of the input image along the first dimension (i.e., for each value of the first index). The output is a 3D array with the same shape as the input image, where each slice along the first dimension is the thresholded 2D slice at that index.
    """
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        out[i,:,:]=local_global_threshold(image[i,:,:], local_window_size, global_threshold)
    
    return out

