import tifffile
from tifffile import TiffFile
from tifffile.tifffile import imagej_description_metadata

def open_ij3D(filename):
    """ open imagej 3D image and read spacings from meta data

    Args:
        filename (string): Full locaton of file to open

    Returns:
        [numpy array, float, float, float]: returns numpy array with data and x, y and z voxel size
    """
    tif = TiffFile(filename)
    tags = tif.pages[0].tags
    x_resolution = tags['XResolution'].value
    y_resolution = tags['YResolution'].value

    x_size = x_resolution[1]/x_resolution[0]
    y_size = y_resolution[1]/y_resolution[0]

    ij_description = tags['ImageDescription'].value
    ij_description_metadata = imagej_description_metadata(ij_description)
    z_size=ij_description_metadata['spacing']

    return tif.asarray(), x_size, y_size, z_size


def save_zcyx(file_name, img):
    """ save zcyx file with imagej metadata and convenient defaults (set composite mode truje)

    Args:
        file_name:  
        img: numpy array containing contiguous image data in zcyx order
    """
    tifffile.imwrite(file_name, img, imagej=True, metadata={'axes': 'ZCYX', 'mode':'composite'})

