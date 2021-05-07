import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import gray2rgb
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import closing, disk, erosion
from skimage.segmentation import find_boundaries, watershed

def separate_touching(segmented, min_distance, num_erosions):
    """ separates touching objects using a watershed process

    Args:
        segmented (2d np array): pre-segmented image
        min_distance (int):  minimum distance between separated
        num_erosions (int): number of eroisions to perform at end of process, in order to increase separation between split objects. 

    Returns:
        2d np array:  separated binary image
    """

    closed = closing(segmented, disk(2))
    eroded = erosion(closed, disk(2))
    distance = ndi.distance_transform_edt(eroded)
    coords = peak_local_max(distance, min_distance=min_distance, exclude_border=False)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=segmented, watershed_line=True)
    separated = labels
    separated[labels>0]=1

    for i in range(num_erosions):
        separated = erosion(separated, disk(1))

    return separated 


