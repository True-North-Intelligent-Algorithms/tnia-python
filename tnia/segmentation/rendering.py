from skimage.measure import label
from skimage.segmentation import find_boundaries
from skimage.color import gray2rgb

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

