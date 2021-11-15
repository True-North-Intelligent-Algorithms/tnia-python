import napari

def show_image(im, title, viewer=None, label=False):
    """ convenience helper function to show image in Napari

    Args:
        im (numpy array):  image to show
        title (string): title of image
        viewer (napari instance, optional): pre-existing Napari
        label (bool, optional): True if a label

    Returns:
        [napari instance]: a Napari instance
    """
    if viewer==None:
        viewer = napari.Viewer()
    
    if label==False:
        viewer.add_image(im,name=title)
    else:
        viewer.add_labels(im,name=title)

    return viewer

def show_image_and_label(im, label, title, viewer=None):
    """[summary]

    Args:
        im (numpy array):  image to show
        label (numpy integer array): labels to show
        title (string): title of image
        viewer (napari instance, optional): pre-existing Napari

    Returns:
        [napari instance]: a Napari instance
    """
    viewer = show_image(im, title, viewer, False)
    show_image(label, title+' label', viewer, True)

    return viewer