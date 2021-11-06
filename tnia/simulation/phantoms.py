import numpy as np
import raster_geometry as rg
from random import uniform, seed
import math

def sphere3d(size, radius):
    return rg.sphere(size, radius).astype(np.float32)

def ramp2d(shape,min,max):
    """ create an image of a 2D ramp to simulate uneven background

    Args:
        shape (list of 2 integers): Size of ramp image
        min (number): min of ramp
        max (number): max of ramp

    Returns:
        [2d numpy array]: ramp image 
    """

    ramp=np.empty(shape)

    for y in range(shape[0]):
        for x in range(shape[1]):
            ramp[y,x]=(max-min)*(y+x)/(shape[0]+shape[1])+min
    return ramp

def random_circles(im, num, min_r, max_r, min_intensity, max_intensity, seed_val=-1):
    """ draws random circles in an image useful for testing segmentation, background subtraction etc.

    Args:
        im (2d numpy array): input image, will be changed
        num (number): number of circles to draw
        min_r (number): min radius of circles
        max_r (number): max radius of circles
        min_intensity (number): max intensity of circles
        max_intensity (number): min intensity of circles
        seed_val (int, optional): use seed if you need to replicate the same random image defalts to -1 (no seed used).
    """
    if seed_val!=-1:
        seed(seed_val)

    for i in range(num):
        r=round(uniform(min_r, max_r))
        cx=round(uniform(r,im.shape[1]-r))
        cy=round(uniform(r,im.shape[0]-r))
        intensity=round(uniform(min_intensity, max_intensity))
        print(r,cx,cy,intensity)
        temp1=rg.circle([r*2,r*2],r)
        temp2=np.zeros_like(im)
        temp2[cy-r:cy+r,cx-r:cx+r]=temp1
        im[temp2>0]=im[temp2>0]+intensity
