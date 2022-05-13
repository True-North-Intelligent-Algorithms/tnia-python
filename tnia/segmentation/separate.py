import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import gray2rgb
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import closing, disk, erosion
from skimage.segmentation import find_boundaries, watershed
from skimage.filters import gaussian, threshold_otsu
from tnia.viewing.napari_helper import show_image_and_label as sil
from skimage.morphology import remove_small_holes
from tnia.morphology.fill_holes import fill_holes_3d_slicer

def separate_touching(segmented, min_distance, num_erosions):
    """ separates touching objects using a watershed process

    Args:
        segmented (2d np array): pre-segmented image
        min_distance (int):  minimum distance between separated
        num_erosions (int): number of eroisions to perform at end of process, in order to increase separation between split objects. 

    Returns:
        2d np array:  separated binary image
    """

    #closed = closing(segmented, disk(1))
    #eroded = erosion(closed, disk(2))
    distance = ndi.distance_transform_edt(segmented)
    coords = peak_local_max(distance, min_distance=min_distance, exclude_border=False)
    #print(coords.shape)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=segmented, watershed_line=True)
    separated = labels
    separated[labels>0]=1

    for i in range(num_erosions):
        separated = erosion(separated, disk(1))

    return separated, labels, distance

def separate_touching2(im, segmented, min_distance, spot_sigma, distance_sigma):
    """ This algorithm uses a strategy similar to the one described here in
    https://clij.github.io/clij2-docs/md/voronoi_otsu_labeling/

    The main difference is instead of calling Voronoi labeling we create
    a distance map and use it as the input to skimage.segmentation.watershed

    Args:
        im (numpy array): input image 
        segmented (binary numpy array): segmented input
        min_distance (number): min distance between separated objects 
        spot_sigma (array): sigma in each dimension for blur applied before spot detection
        distance_sigma ([type]): sigma in each dimension for blur applied before distance map
    """
    blurred_spot = gaussian(im, spot_sigma)

    # calculate peaks from blurred image
    coords = peak_local_max(blurred_spot, min_distance)
    print(coords.shape)
    mask = np.zeros(im.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    # calculate distance map from second blurred image
    blurred_distance = gaussian(im, distance_sigma)
    binary_distance = blurred_distance>threshold_otsu(blurred_distance)
 
    # TODO:  Address fill holes step, only valid for 3D
    #fill_holes_3d_slicer(binary_distance)
    
    distance = ndi.distance_transform_edt(binary_distance)

    labels = watershed(-distance, markers, mask=segmented,watershed_line=True)

    return labels,distance