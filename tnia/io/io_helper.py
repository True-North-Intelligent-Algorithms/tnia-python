import os
from glob import glob
import numpy as np
from skimage.io import imread

def get_file_names_from_dir(dir, extension):
    """ gets all file names with extension from directory

    Args:
        dir (string): directory to look for files
        extension (string): extension with no '.' ie 'tif', 'JPG'
    """
    return [y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.'+extension))]

def open_stack(dir, extension, opener):
    """
    Opens a stack of images from a directory using a specified opener function
    
    Args:
        dir (string): directory to look for files
        extension (string): extension with no '.' ie 'tif', 'JPG'
        opener (function): function to open the images
    """
    file_names = get_file_names_from_dir(dir, extension)

    stacked = []
    for f in file_names:
        im=opener(f)
        print(f)
        print(im.shape)
        stacked.append(im)
    
    return np.stack(stacked)

def open_zero_padded_numbered_stack(directory, extension, filter=None, splitter=None, length=3):
    """
    Opens a stack of images with zero-padded numbers in the file name
    
    Args:
        directory (pathlib.Path): directory to look for files
        extension (string): extension with no '.' ie 'tif', 'JPG'
        filter (string): filter to look for in file names
        splitter (string): character which splits the file at the number
        length (int): length of the zero-padded number
    """

    if filter is not None:
        image_files = list(directory.glob('*'+filter+'*.'+extension))
    else:
        image_files = list(directory.glob('*.'+extension))
    
    # Sort based on the numeric part of the indicator
    sorted_files = sorted(
        image_files,
        key=lambda x: int(str(x).split(splitter)[1][0:length])
    )

    images = []

    for image_file in sorted_files:
        image = imread(image_file)
        images.append(image)    

    images = np.array(images)

    return images