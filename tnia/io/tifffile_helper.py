import tifffile

def save_zcyx(file_name, img):
    """ save zcyx file with imagej metadata and convenient defaults (set composite mode truje)

    Args:
        file_name:  
        img: numpy array containing contiguous image data in zcyx order
    """
    tifffile.imwrite(file_name, img, imagej=True, metadata={'axes': 'ZCYX', 'mode':'composite'})

