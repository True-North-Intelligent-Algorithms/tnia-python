import numpy as np
from skimage.filters import threshold_local

def local_global_threshold(image, local_window_size, global_threshold):
    """ applies a local threshold and a global threshold and returns the AND of the two

    Args:
        image (np array): input image
        local_window_size (number): size of window for local threshold
        global_threshold (number): global threshold

    Returns:
        [np array]: And of local and global thresholded image
    """
    return np.logical_and(image>threshold_local(image, local_window_size), image>global_threshold)
