import numpy as np

def makergb(r,g,b):
    """given 2d r, g and b numpy arrays combine into one rgb array

    Args:
        r ([2d numpy array]): red
        g ([2d numpy array]): green
        b ([2d numpy array]): blue

    Returns:
        [numpy array]: rgb 
    """
    r=(255.*r/r.max()).astype('uint8')
    g=(255.*g/g.max()).astype('uint8')
    b=(255.*b/b.max()).astype('uint8')

    rgb=np.stack([r,g,b])
    
    return np.transpose(rgb,(1,2,0))

