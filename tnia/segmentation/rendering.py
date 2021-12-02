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

def draw_centroids(segmented, img=None):
    """ draws centroids of connected objects in segmented onto img (if img is None draws cetroids on a blank image)
    
    Args:
        segmented (numpy array): binary pre-segmented image
        img (numpy array, optional): Image to draw centroids onto if None centroids are drawn on empty iamge. Defaults to None.
    """

    if img is None:
        img = np.zeros_like(segmented)

    labels=label(segmented)

    objects = regionprops(labels)

    for o in objects:
        img[int(o.centroid[0]),int(o.centroid[1]),int(o.centroid[2])]=1

    return img

    

