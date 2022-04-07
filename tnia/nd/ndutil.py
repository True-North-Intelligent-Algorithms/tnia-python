import numpy as np

def centercrop(im, shape):
    startz=int(im.shape[0]/2)-int(shape[0]/2)
    starty=int(im.shape[1]/2)-int(shape[1]/2)
    startx=int(im.shape[2]/2)-int(shape[2]/2)

    return im[startz:startz+shape[0],starty:starty+shape[1],startx:startx+shape[2]]


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

