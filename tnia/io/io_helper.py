import os
from glob import glob
import numpy as np

def get_file_names_from_dir(dir, extension):
    """ gets all file names with extension from directory

    Args:
        dir (string): directory to look for files
        extension (string): extension with no '.' ie 'tif', 'JPG'
    """
    return [y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.'+extension))]

def open_stack(dir, extension, opener):
    file_names = get_file_names_from_dir(dir, extension)

    stacked = []
    for f in file_names:
        im=opener(f)
        print(f)
        print(im.shape)
        stacked.append(im)
    
    return np.stack(stacked)