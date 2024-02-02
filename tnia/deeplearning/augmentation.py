import numpy as np
from skimage import transform

"""
Note:  There are lots of augmentation libraries out there.  This either provides convenient wrappers for some, or 
implements niche augmentations that are not available in other libraries.  
"""

def random_fliprot(img, mask, axis=None):
    """
    copied from https://github.com/stardist/stardist/blob/master/examples/3D/2_training.ipynb
    Randomly flip and rotate the image and mask

    Args:
        img (np array): the image to be flipped and rotated
        mask (np array): the mask to be flipped and rotated
        axis (tuple): the axis to be flipped and rotated
    Returns:
        np array: the flipped and rotated image
    """ 
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    """
    copied from https://github.com/stardist/stardist/blob/master/examples/3D/2_training.ipynb

    Args:
        img : input image 

    Returns:
        np array : intensity adjusted image
    """
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def stardist_3d_augmenter(x, y):
    """
    copied from https://github.com/stardist/stardist/blob/master/examples/3D/2_training.ipynb

    Augmentation of a single input/label image pair.
    
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y

def stardist_2d_augmenter(x, y):
    """
    copied from https://github.com/stardist/stardist/blob/master/examples/3D/2_training.ipynb

    Augmentation of a single input/label image pair.
    
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    return x, y


def random_shift_slices_in_stack(img, shift_range=2):
    """
    Shifts each plane in a stack by a random amount in x and y
    
    Args:
        img (3d np array): the images to be shifted
    Returns:
        3d np array: the shifted images
        shift_range (float): the maximum amount to shift
    """
    # Initialize an array to store the shifted images
    shifted_images = np.empty((img.shape[0], img.shape[1]-2*shift_range, img.shape[2]-2*shift_range))

    # For each slice in the img image
    for i in range(img.shape[0]):
        # Get the slice
        slice = img[i, :, :]

        # Generate a random shift between 0 and 2 pixels
        shift_y, shift_x = np.random.uniform(0, shift_range, shift_range)

        # Create the transformation matrix
        transform_matrix = transform.AffineTransform(translation=(shift_y, shift_x))

        # Apply the transformation
        shifted = transform.warp(slice, transform_matrix)

        # Crop the shifted image to 256x256
        cropped = shifted[2:-2, 2:-2]

        # Store the cropped image
        shifted_images[i] = cropped

    return shifted_images
