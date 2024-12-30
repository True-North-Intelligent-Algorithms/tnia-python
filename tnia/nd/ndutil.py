import numpy as np
from pathlib import Path

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

def get_max_img_shape_2d(images):
    """
    Get the maximum shape of a list of 2D images.  Will return the maximum number of rows and columns and number channels

    Args:
        images (list): list of 2D images

    Returns:
        tuple: the maximum shape of the images
    """
    max_rows = max(image.shape[0] for image in images)
    max_cols = max(image.shape[1] for image in images)

    if len(images[0].shape) == 3:
        max_channels = max(image.shape[2] for image in images)
    else:
        max_channels = 1
    return max_rows, max_cols, max_channels

def pad_to_largest(images, force8bit=False, normalize_per_channel = False):
    """
    Pads a list of images to the largest dimensions in the list.

    This is useful for displaying images of different sizes as a sequence in Napari
    
    Args:
        images (list): list of images to pad
        force8bit (bool): whether to normalize the images to 8 bit
        normalize_per_channel (bool): whether to normalize the images per channel
        
        Returns:
        numpy.ndarray: The padded images
    """

    # TODO: this is actually a pretty complicated function, not only do we pad but 
    # we also normalize the images to 8 bit, and we also convert to rgb if the image is not 3 channel
    # will need continued work and refactoring, as we are essentially doing multi-time, multi-channel, multi-format 
    # conversion for display. 
    
    # Find the maximum dimensions
    max_rows = max(image.shape[0] for image in images)
    max_cols = max(image.shape[1] for image in images)
    
    # Create a list to hold the padded images
    padded_images = []
    
    for image in images:
        # Calculate the padding for each dimension
        pad_rows = max_rows - image.shape[0]
        pad_cols = max_cols - image.shape[1]
        
        if len(image.shape) == 3:
            # we occasionally hit rgba images, just use the first 3 channels
            image = image[:,:,:3]
            # Pad the array
            padded_image = np.pad(image, 
                                ((0, pad_rows), (0, pad_cols), (0,0)), 
                                mode='constant', 
                                constant_values=0)
        else:
            padded_image = np.pad(image, 
                                ((0, pad_rows), (0, pad_cols)), 
                                mode='constant', 
                                constant_values=0)
 
        if (len(padded_image.shape) > 2):

            if padded_image.shape[2] != 3:
                padded_image = remove_alpha_channel(padded_image)
       
        if force8bit:
            
            if (len(padded_image.shape) > 2) and normalize_per_channel:
                padded_image = normalize_per_channel(padded_image)
            else:
                min_ = np.min(padded_image)
                max_ = np.max(padded_image)
                padded_image = ((padded_image - min_) / (max_ - min_) * 255).astype(np.uint8)

        padded_images.append(padded_image)

    shapes = [image.shape for image in padded_images]

    # BN was toying with the idea of displaying 3 channel and 1 channel images together but it is a bit messy, so commented out for now
    '''
    if len(set(shapes)) > 1:
        for i in range(len(padded_images)):
            if len(padded_images[i].shape)==2:
                padded_images[i] = padded_images[i][:,:,np.newaxis]
                padded_images[i] = multi_channel_to_rgb(padded_images[i])
    '''
    
    # Stack the padded images along a new third dimension
    result = np.array(padded_images)

    if force8bit:
        result = result.astype(np.uint8)
    
    return result

def normalize_per_channel_8bit(padded_image):
    """
    Normalize the image per channel and convert to 8 bit

    Useful for displaying images with different intensity ranges
    
    Args:
        padded_image (numpy.ndarray): The image to normalize

    Returns:
        numpy.ndarray: The normalized image
    """
    normalized_image = np.zeros_like(padded_image, dtype=np.uint8)
    for c in range(padded_image.shape[2]):  # Assuming the last dimension is the channel
        channel = padded_image[:, :, c]
        min_ = channel.min()
        max_ = channel.max()
        normalized_image[:, :, c] = ((channel - min_) / (max_ - min_) * 255).astype(np.uint8)
    return normalized_image

def remove_alpha_channel(im):
    """
    Removes the alpha channel from an image if it exists.

    Args:
        im (numpy.ndarray): The input image
    
    Returns:
        numpy.ndarray: The image with the alpha channel removed
    """
    # if too many channels (sometimes an alpha channel) remove the last ones
    if im.shape[2] > 3:
        im_ = im[:,:,:3]
    if im.shape[2]<3:
        im_ = np.concatenate([im, np.zeros(im.shape[:2])[:, :, None]], axis=2)
    if im.shape[2]<3:
        im_ = np.concatenate([im_, np.zeros(im.shape[:2])[:, :, None]], axis=2)
    return im_

def unpad_to_original(images, original_images):
    """
    Unpads a list of images to the original dimensions of the images.

    Args:
        images (list): list of images to unpad
        original_images (list): list of original images
    """
    unpadded_images = []        
    for n in range(len(images)):
        height, width = original_images[n].shape[:2]

        image = images[n][:height, :width]
        unpadded_images.append(image)

    return unpadded_images

