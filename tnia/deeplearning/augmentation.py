import numpy as np
from skimage import transform
from random import randint
from skimage.io import imsave
from tnia.deeplearning.dl_helper import generate_patch_names, make_patch_directory, generate_next_patch_name
import json
import os

try:
    import albumentations as A
except ImportError:
    print("Albumentations is not installed. Please install it using pip install albumentations.")

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

def uber_augmenter(im, mask, patch_path, patch_base_name, patch_size, num_patches, do_vertical_flip=True, 
                   do_horizontal_flip=True, do_random_rotate90=True, do_random_sized_crop=True, do_random_brightness_contrast=True, 
                   do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False, sub_sample_xy=1):
    """
    This function performs a series of image augmentations on the input image and mask and saves the resulting patches to disk.

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
        do_color_jitter (bool, optional): Whether to perform color jitter. Defaults to False.
        do_elastic_transform (bool, optional): Whether to perform elastic transformations. Defaults to False.

    """
    
    image_patch_path =  os.path.join(patch_path, 'input0')
    
    if isinstance(mask, list):
        label_patch_path =  [os.path.join(patch_path, 'ground truth'+str(i)) for i in range(len(mask))]
        num_truths = len(mask)
    else:
        label_patch_path =  os.path.join(patch_path, 'ground truth0')
        num_truths = 1
    make_patch_directory(1, num_truths, patch_path)
    
    # Load the existing JSON data which is created when making the patch directory and append addition information to it
    json_file = patch_path / "info.json"

    with open(json_file, 'r') as infile:
        data = json.load(infile)

    # add the sub_sample information to the JSON file
    # TODO: Make these parameters
    data['sub_sample'] = sub_sample_xy

    # TODO: make logic to detect axis more complex
    if len(im.shape) == 3:
        data['axes'] = 'YXC'
    else:
        data['axes'] = 'YX'
        
    if data['axes'] == 'YX':
        if (sub_sample_xy>1):
            im = im[::sub_sample_xy, ::sub_sample_xy]
            for i in range(len(mask)):
                mask[i] = mask[i][::sub_sample_xy, ::sub_sample_xy]
    elif data['axes'] == 'YXC':
        if (sub_sample_xy>1):
            im = im[::sub_sample_xy, ::sub_sample_xy, :]
            for i in range(len(mask)):
                mask[i] = mask[i][::sub_sample_xy, ::sub_sample_xy]
        
    # Write the modified data back to the JSON file
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile)


    if not os.path.exists(image_patch_path):
        os.mkdir(image_patch_path)
    
    if isinstance(mask, list):
        for i in range(len(mask)):
            if not os.path.exists(label_patch_path[i]):
                os.mkdir(label_patch_path[i])
    else:
        if not os.path.exists(label_patch_path):
            os.mkdir(label_patch_path)


    for i in range(num_patches):
        if any(dim < patch_size for dim in mask[0].shape):
            continue
        
        im_aug, label_aug = uber_augmenter_im(im, mask, patch_size, do_vertical_flip, do_horizontal_flip,
                                        do_random_rotate90, do_random_sized_crop, do_random_brightness_contrast,
                                        do_random_gamma, do_color_jitter, do_elastic_transform)

        is_anynan = np.isnan(im_aug).any() or np.isnan(label_aug).any()

        if is_anynan:
            continue


        #image_name, patch_name = generate_patch_names(str(image_patch_path), str(label_patch_path), patch_base_name)
        next_patch_name = generate_next_patch_name(str(image_patch_path), patch_base_name)
        
        image_name = os.path.join(image_patch_path , next_patch_name+'.tif')
        
        imsave(image_name, im_aug)
        
        if isinstance(mask, list):
            for j in range(len(mask)):
                label_name = os.path.join(label_patch_path[j], next_patch_name+'.tif')
                imsave(label_name, label_aug[j])
        else:
            label_name = os.path.join(label_patch_path, next_patch_name+'.tif')
            imsave(label_name, label_aug)


def uber_augmenter_bb(im, bbs, classes, patch_path, patch_base_name, num_patches, do_vertical_flip=True, 
                   do_horizontal_flip=True, do_random_rotate90=True, do_random_sized_crop=True, do_random_brightness_contrast=True, 
                   do_random_gamma=False, do_color_jitter=False):
    """
    This function performs a series of image augmentations on the input image and bounding boxes and saves the resulting 
    augmented images and bounding boxes to disk.

    Args:
        im (np.array): The input image to be augmented.
        bbs (list):  The bounding boxes in YOLO format to be augmented.
        patch_path (str): The path to the patches to be extracted from the augmented images.
        patch_size (tuple): The size of the patches to be extracted from the augmented images.
        num_patches (int): The number of patches to be extracted.
        do_vertical_flip (bool, optional): Whether to perform vertical flip. Defaults to True.
        do_horizontal_flip (bool, optional): Whether to perform horizontal flip. Defaults to True.
        do_random_rotate90 (bool, optional): Whether to perform random 90 degree rotations. Defaults to True.
        do_random_sized_crop (bool, optional): Whether to perform random crops of varying sizes. Defaults to True.
        do_random_brightness_contrast (bool, optional): Whether to perform random brightness and contrast adjustments. Defaults to True.
        do_random_gamma (bool, optional): Whether to perform random gamma adjustments. Defaults to False.
        do_color_jitter (bool, optional): Whether to perform color jitter. Defaults to False.

    """

    image_patch_path =  os.path.join(patch_path, 'images')
    label_patch_path =  os.path.join(patch_path, 'labels')

    if not os.path.exists(image_patch_path):
        os.mkdir(image_patch_path)
    if not os.path.exists(label_patch_path):
        os.mkdir(label_patch_path)
    
    num_truths = 1
    
    # Load the existing JSON data which is created when making the patch directory and append addition information to it
    json_file = patch_path / "info.json"

    try:
        with open(json_file, 'r') as infile:
            data = json.load(infile)
    except:
        data = {}

    # add the sub_sample information to the JSON file
    # TODO: Make these parameters
    data['sub_sample'] = 1

    # TODO: make logic to detect axis more complex
    if len(im.shape) == 3:
        data['axes'] = 'YXC'
    else:
        data['axes'] = 'YX'
        
    # Write the modified data back to the JSON file
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile)

    if not os.path.exists(image_patch_path):
        os.mkdir(image_patch_path)
    
    if not os.path.exists(label_patch_path):
        os.mkdir(label_patch_path)


    for i in range(num_patches):
        im_aug, boxes_aug = uber_augmenter_im_bb(im, bbs, do_vertical_flip, do_horizontal_flip,
                                        do_random_rotate90, do_random_sized_crop, do_random_brightness_contrast,
                                        do_random_gamma, do_color_jitter)
        
        if im_aug is None or boxes_aug is None:
            continue

        is_anynan = np.isnan(im_aug).any() 

        if is_anynan:
            continue
        
        next_patch_name = generate_next_patch_name(str(image_patch_path), patch_base_name)
        
        image_name = os.path.join(image_patch_path , next_patch_name+'.tif')
        
        imsave(image_name, im_aug)
        
        boxes_name = os.path.join(label_patch_path, next_patch_name+'.txt')
        with open(boxes_name, 'w') as f:
            for box, class_ in zip(boxes_aug, classes):
                f.write(f"{class_} {box[0]} {box[1]} {box[2]} {box[3]}\n")



def uber_augmenter_im(im, mask, patch_size, do_vertical_flip=True, do_horizontal_flip=True, 
                     do_random_rotate90=True, do_random_sized_crop=True, do_random_brightness_contrast=True, 
                     do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False):
    """ single application of uber augmenter to label and image

    Args:
        im (_type_): _description_
        mask (_type_): _description_
        patch_size (_type_): _description_
        do_vertical_flip (bool, optional): _description_. Defaults to True.
        do_horizontal_flip (bool, optional): _description_. Defaults to True.
        do_random_rotate90 (bool, optional): _description_. Defaults to True.
        do_random_sized_crop (bool, optional): _description_. Defaults to True.
        do_random_brightness_contrast (bool, optional): _description_. Defaults to True.
        do_random_gamma (bool, optional): _description_. Defaults to False.
        do_color_jitter (bool, optional): _description_. Defaults to False.
        do_elastic_transform (bool, optional): _description_. Defaults to False.

    Returns:
        im_aug, label_aug: augmented image and label
    """
    
    x=randint(0,im.shape[1]-patch_size)
    y=randint(0,im.shape[0]-patch_size)

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
        # TODO: make more flexibility for resize
        augmentations.append(A.RandomSizedCrop(min_max_height=(3*patch_size//4, patch_size), height=patch_size, width=patch_size, p=0.5))

    if do_random_brightness_contrast:
        augmentations.append(A.RandomBrightnessContrast(p=0.8))

    if do_random_gamma:
        augmentations.append(A.RandomGamma(p=0.8))

    if do_color_jitter:
        # color jitter light
        # augmentations.append(A.ColorJitter(hue=0, brightness=0.5, saturation=0.1, p=0.5))
        # color jitter heavy
        augmentations.append(A.ColorJitter(hue=0.5, brightness=0.5, saturation=0.5, p=0.6))

    if do_elastic_transform:
        augmentations.append(A.ElasticTransform (alpha=.1, sigma=5, alpha_affine=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.5))
    # Create the augmenter

    if isinstance(mask, list):
        
        additional_targets = {}
        n=0
        for m in mask:
            additional_targets['mask'+str(n)] = 'mask'
            n+=1
        aug = A.Compose(augmentations, additional_targets=additional_targets)

        # create a dictionary with keys 'mask0', 'mask1', etc
        mask_dict = {'mask'+str(i): mask[i][ind] for i in range(len(mask))}

        # Call the function with argument unpacking
        augmented = aug(image=im[ind], **mask_dict)

        im_aug = augmented['image']        
        label_aug = [augmented['mask'+str(i)] for i in range(len(mask))]
    
    else:
        aug = A.Compose(augmentations)
        augmented = aug(image=im[ind], mask=mask[ind])

        im_aug = augmented['image']
        label_aug = augmented['mask']

    return im_aug, label_aug

def uber_augmenter_im_bb(im, bbs, do_vertical_flip=True, do_horizontal_flip=True, 
                     do_random_rotate90=True, do_random_scale=True, do_random_brightness_contrast=True, 
                     do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False):
    """ single application of uber augmenter for bounding boxes to boxes and image

    Args:
        im (_type_): _description_
        mask (_type_): _description_
        patch_size (_type_): _description_
        do_vertical_flip (bool, optional): _description_. Defaults to True.
        do_horizontal_flip (bool, optional): _description_. Defaults to True.
        do_random_rotate90 (bool, optional): _description_. Defaults to True.
        do_random_sized_crop (bool, optional): _description_. Defaults to True.
        do_random_brightness_contrast (bool, optional): _description_. Defaults to True.
        do_random_gamma (bool, optional): _description_. Defaults to False.
        do_color_jitter (bool, optional): _description_. Defaults to False.
        do_elastic_transform (bool, optional): _description_. Defaults to False.

    Returns:
        im_aug, label_aug: augmented image and label
    """

    im_size = (im.shape[0]+im.shape[1])//2

    # Create a list of augmentations
    augmentations = []

    if do_vertical_flip:
        augmentations.append(A.VerticalFlip(p=0.5))

    if do_horizontal_flip:
        augmentations.append(A.HorizontalFlip(p=0.5))

    if do_random_rotate90:
        augmentations.append(A.RandomRotate90(p=0.5))

    if do_random_scale:
        augmentations.append(A.RandomScale(scale_limit=(-0.3,0.3), p=0.5))

    if do_random_brightness_contrast:
        augmentations.append(A.RandomBrightnessContrast(p=0.8))

    if do_random_gamma:
        augmentations.append(A.RandomGamma(p=0.8))

    if do_color_jitter:
        # color jitter light
        augmentations.append(A.ColorJitter(hue=0, brightness=0.5, saturation=0.1, p=0.5))
        # color jitter heavy
        augmentations.append(A.ColorJitter(hue=0.5, brightness=0.5, saturation=0.5, p=0.6))

    # Create the augmenter

    aug = A.Compose(augmentations, bbox_params=A.BboxParams(format='yolo'))
    try:    
        augmented = aug(image=im, bboxes=bbs) 
    except Exception as e:
        print(e)
        return None, None
    
    im_aug = augmented['image']
    boxes_aug = augmented['bboxes']

    return im_aug, boxes_aug 

