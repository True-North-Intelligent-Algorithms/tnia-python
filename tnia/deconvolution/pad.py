import numpy as np
import math

def handle_prime(p,x,a):
    log = math.log(p)
    power=p

    while power <= x + a.shape[0]:
        j=x%power
        if j>0:
            j=power-j

        while j < a.shape[0]:
            a[j]+=log
            j+=power

        power*=p
    

def next_smooth(x):
    """[summary]
    author Johannes Schindelin
    author Brian Northan

    A class to determine the next smooth number (a number divisable only)
    by prime numbers up to k (in this case we fix k at 7).
    
    Based on A. Granville, Finding smooth numbers computationally.
    
    Args:
        x ([type]): number to test

    Returns:
        [type]: next smooth number larger than x
    """
    z = int(4*math.log2(x))
    delta = 0.000001

    a = np.zeros(z)

    handle_prime(2,x,a)
    handle_prime(3,x,a)
    handle_prime(5,x,a)
    handle_prime(7,x,a)

    log = math.log(x)
    for i in range(a.shape[0]):
        if a[i] >=log-delta:
            return x+i

    return -1

def get_next_smooth(size):
    """ for an nd tuple compute the next smooth size for each element

    Args:
        size ([type]): nd input tuple 

    Returns:
        [type]: tuple containing the next smooth size for each input element
    """
    return tuple(map(lambda i: next_smooth(i), size))

def get_pad_size(img, psf):
    """ given an image and psf return the extended size needed to avoid circular calculations
    during convolution and/or deconvolution 

    Args:
        img ([type]): nd image
        psf ([type]): nd psf 

    Returns:
        [tuple]: extended size to use to avoid circular calculations
    """
    return tuple(map(lambda i,j: i+2*math.floor(j/2),img.shape,psf.shape))


def pad(img, paddedsize, mode):
    """ pad image to paddedsize

    Args:
        img ([type]): image to pad 
        paddedsize ([type]): size to pad to 
        mode ([type]): one of the np.pad modes

    Returns:
        [nd array]: padded image
    """
    padding = tuple(map(lambda i,j: ( math.ceil((i-j)/2), math.floor((i-j)/2) ),paddedsize,img.shape))
    return np.pad(img, padding,mode), padding


def unpad(padded, imgsize):
    """Crop padded back to imgsize.

    Args:
        padded (numpy.ndarray or cupy.ndarray): The padded array.
        imgsize (tuple): The target size of the unpadded array.

    Returns:
        numpy.ndarray or cupy.ndarray: The unpadded array.
    """
    padding = tuple(map(lambda i,j: (math.ceil((i-j)/2), math.floor((i-j)/2)), padded.shape, imgsize))
    slices = tuple(slice(p[0], p[0]+s) for p, s in zip(padding, imgsize))

    return padded[slices]