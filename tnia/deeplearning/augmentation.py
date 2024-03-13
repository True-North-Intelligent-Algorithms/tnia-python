import numpy as np
from skimage import transform
from random import randint
from skimage.io import imsave
from tnia.deeplearning.dl_helper import generate_patch_names, make_patch_directory
import json

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

def uber_augmenter(im, mask, patch_path, patch_base_name, patch_size, num_patches, do_vertical_flip=True, do_horizontal_flip=True, do_random_rotate90=True, do_random_sized_crop=True, do_random_brightness_contrast=True, do_random_gamma=False):
    """
    This function performs a series of image augmentations on the input image and mask.

    Args:
        im (np.array): The input image to be augmented.
        mask (np.array): The corresponding mask to be augmented.
        patch_path (str): The path to the patches to be extracted from the augmented images.
        patch_size (tuple): The size of the patches to be extracted from the augmented images.
        num_patches (int): The number of patches to be extracted.
        do_vertical_flip (bool, optional): Whether to perform vertical flip. Defaults to True.
        do_horizontal_flip (bool, optional): Whether to perform horizontal flip. Defaults to True.
        do_random_rotate90 (bool, optional): Whether to perform random 90 degree rotations. Defaults to True.
        do_random_sized_crop (bool, optional): Whether to perform random crops of varying sizes. Defaults to True.
        do_random_brightness_contrast (bool, optional): Whether to perform random brightness and contrast adjustments. Defaults to True.
        do_random_gamma (bool, optional): Whether to perform random gamma adjustments. Defaults to False.

    Returns:
        np.array: The augmented image patches.
        np.array: The augmented mask patches.
    """
    
    import albumentations as A
    import os

    image_patch_path =  os.path.join(patch_path, 'input0')
    label_patch_path =  os.path.join(patch_path, 'ground truth0')
    
    make_patch_directory(1, 1, patch_path)
    
    # Load the existing JSON data which is created when making the patch directory and append addition information to it
    json_file = patch_path / "info.json"

    with open(json_file, 'r') as infile:
        data = json.load(infile)

    # add the sub_sample information to the JSON file
    # TODO: Make these parameters
    data['sub_sample'] = 1
    data['axes'] = 'YX'

    # Write the modified data back to the JSON file
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile)


    if not os.path.exists(image_patch_path):
        os.mkdir(image_patch_path)
    if not os.path.exists(label_patch_path):
        os.mkdir(label_patch_path)


    for i in range(num_patches):
        x=randint(0,im.shape[1]-patch_size-1)
        y=randint(0,im.shape[0]-patch_size-1)

        ind = np.s_[y:y+patch_size, x:x+patch_size]

        # Create a list of augmentations
        augmentations = []

        if do_vertical_flip:
            augmentations.append(A.VerticalFlip(p=0.5))

        if do_horizontal_flip:
            augmentations.append(A.HorizontalFlip(p=0.5))

        if do_random_rotate90:
            augmentations.append(A.RandomRotate90(p=0.5))

        if do_random_sized_crop:
            augmentations.append(A.RandomSizedCrop(min_max_height=(patch_size//2, patch_size), height=patch_size, width=patch_size, p=0.5))

        if do_random_brightness_contrast:
            augmentations.append(A.RandomBrightnessContrast(p=0.8))

        if do_random_gamma:
            augmentations.append(A.RandomGamma(p=0.8))

        # Create the augmenter
        aug = A.Compose(augmentations)
        
        augmented = aug(image=im[ind], mask=mask[ind])

        im_aug = augmented['image']
        label_aug = augmented['mask']

        print(im_aug.shape, label_aug.shape)

        image_name, patch_name = generate_patch_names(str(image_patch_path), str(label_patch_path), patch_base_name)
        print(image_name, patch_name)
        imsave(image_name, im_aug)
        imsave(patch_name, label_aug)

