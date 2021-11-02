import numpy as np
import rawpy

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

def open_arw(file_name, output_bps=16):
    raw = rawpy.imread(file_name)
    return raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=output_bps)
