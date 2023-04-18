from skimage.measure import label
from skimage.segmentation import find_boundaries
from skimage.color import gray2rgb
from skimage.measure import regionprops
import numpy as np

def draw_label_outline(img, segmented):
    """ draw the outlines of object in segmented onto img

    Args:
        img (2d np array): img to draw outlines on
        segmented (2d np array): segmented image

    Returns:
        2d rgp np array: img with outlines of segmented
    """
    label_ = label(segmented)
    overlay = find_boundaries(label_,mode='inner')

    # TODO: handle rgp input
    rgb = gray2rgb(img)

    rgb[overlay==True,0] = 255
    rgb[overlay==True,1] = 0
    rgb[overlay==True,2] = 0

    return rgb

def draw_centroids(segmented, output=None, img=None):
    """Draws centroids of connected objects in a binary image onto another image or a blank image.

    Args:
        segmented (numpy.ndarray): A binary pre-segmented image.
        output (numpy.ndarray, optional): An image to draw centroids onto. If None, centroids are drawn on an empty image of the same size as segmented. Defaults to None.
        img (numpy.ndarray, optional): An optional gray scale image to use as a scaling factor for the centroids. If not None, the centroids will be scaled with the pixel value in img. Defaults to None.

    Returns:
        numpy.ndarray: An image with centroids drawn on it.
    """

    if output is None:
        output = np.zeros_like(segmented)

    labels=label(segmented)

    objects = regionprops(labels)

    if img is None:
        for o in objects:
            idx = tuple(int(c) for c in o.centroid)
            output[idx] = 1
    else:
        for o in objects:
            idx = tuple(int(c) for c in o.centroid)
            output[idx] = img[idx]

    return output

    

