import numpy as np

def loadraw(filename, dtype, shape):
    """open a raw file

    Args:
        filename (string): name of raw file
        dtype (string): type of data ('uint16' ect)
        shape (list): dimensions of data  

    Returns:
        [numby array]: reads file and returns data in numpy array
    """
    # todo byteswap to handle endian should be optional
    data=np.fromfile(filename, dtype).byteswap()

    return  data.reshape(shape)