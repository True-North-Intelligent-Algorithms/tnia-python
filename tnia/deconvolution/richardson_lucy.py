import numpy as np
from numpy.fft import fftn, ifftn, fftshift 
import cupy as cp
from tnia.deconvolution.pad import pad, unpad

def richardson_lucy_cp(image, psf, num_iters, noncirc=False, mask=None):
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option and option to mask bad pixels, uses cupy
    
    Note: Cupy FFT behavior is different than numpy.  Cupy FFT always returns real arrays of type float32 and complex of type complex64.
    So using 64 bit inputs for more precision will not work. 

    Args:
        image [numpy float array]: the image to be deconvolved 
        psf [numpy float array]: the point spread function
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.
        mask (numpy float array, optional): If not None, use this mask to mask image pixels that should not be considered in the deconvolution. Defaults to None.
            'bad' pixels will be zeroed during the deconvolution and then replaced with the original value after the deconvolution.

    Returns:
        [numpy float array]: the deconvolved image
    """
    
    # if noncirc==False and (image.shape != psf.shape) then pad the psf
    if noncirc==False and (image.shape != psf.shape):
        print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')
    
    HTones = np.ones_like(image)

    if (mask is not None):
        HTones = HTones * mask
        mask_values = image*(1-mask)
        image=image*mask
    
    # if noncirc==True then pad the image, psf and HTOnes array to the extended size
    if noncirc:
        # compute the extended size of the image and psf
        extended_size = [image.shape[i]+2*int(psf.shape[i]/2) for i in range(len(image.shape))]

        # pad the image, psf and HTOnes array to the extended size computed above
        original_size = image.shape
        image,_=pad(image, extended_size, 'constant')
        HTones,_=pad(HTones, extended_size, 'constant')
        psf,_=pad(psf, extended_size, 'constant')
    
    image = cp.array(image)
    psf = cp.array(psf)
    HTones = cp.array(HTones)

    otf = cp.fft.fftn(cp.fft.ifftshift(psf))
    otf_ = cp.conjugate(otf)

    if noncirc:
        estimate = cp.ones_like(image)*cp.mean(image)
    else:
        estimate = image

    HTones = cp.real(cp.fft.ifftn(cp.fft.fftn(HTones) * otf_))
    HTones[HTones<1e-6] = 1

    print()
    for i in range(num_iters):
        if i % 10 == 0:
            print(i, end =" ")
        
        reblurred = cp.real(cp.fft.ifftn(cp.fft.fftn(estimate) * otf))

        ratio = image / (reblurred + 1e-12)
        correction=cp.real((cp.fft.ifftn(cp.fft.fftn(ratio) * otf_)))
        estimate = estimate * correction/HTones 
    print()        
    
    estimate = cp.asnumpy(estimate)

    if noncirc:
        estimate = unpad(estimate, original_size)

    if (mask is not None):
        estimate = estimate*mask + mask_values
    
    return estimate


# WIP - version of RL that uses real FFT.  Seems to work and uses less memory, but is slower than the above version
def richardson_lucy_cp_rfft(image, psf, num_iters, noncirc=False, mask=None):
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option and option to mask bad pixels, uses cupy

    Note: Cupy FFT behavior is different than numpy.  Cupy FFT always returns real arrays of type float32 and complex of type complex64.
    So using 64 bit inputs for more precision will not work. 

    Args:
        image [numpy float array]: the image to be deconvolved 
        psf [numpy float array]: the point spread function
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.
        mask (numpy float array, optional): If not None, use this mask to mask image pixels that should not be considered in the deconvolution. Defaults to None.
            'bad' pixels will be zeroed during the deconvolution and then replaced with the original value after the deconvolution.

    Returns:
        [numpy float array]: the deconvolved image
    """
    
    # if noncirc==False and (image.shape != psf.shape) then pad the psf
    if noncirc==False and (image.shape != psf.shape):
        print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')
    
    HTones = np.ones_like(image)

    if (mask is not None):
        HTones = HTones * mask
        mask_values = image*(1-mask)
        image=image*mask

    # if noncirc==True then pad the image, psf and HTOnes array to the extended size
    if noncirc:
        # compute the extended size of the image and psf
        extended_size = [image.shape[i]+2*int(psf.shape[i]/2) for i in range(len(image.shape))]

        # pad the image, psf and HTOnes array to the extended size computed above
        original_size = image.shape
        image,_=pad(image, extended_size, 'constant')
        HTones,_=pad(HTones, extended_size, 'constant')
        psf,_=pad(psf, extended_size, 'constant')
    
    image = cp.array(image)
    psf = cp.array(np.fft.ifftshift(psf))
    HTones = cp.array(HTones)
    
    otf = cp.fft.rfftn(psf)
    psf = None
    otf_ = cp.conjugate(otf)

    if noncirc:
        estimate = cp.ones_like(image)*cp.mean(image)
    else:
        estimate = image

    HTones = cp.fft.irfftn(cp.fft.rfftn(HTones) * otf_, image.shape)
    HTones[HTones<1e-6] = 1

    print()
    for i in range(num_iters):
        if i % 10 == 0:
            print(i, end =" ")
        
        reblurred = cp.fft.irfftn(cp.fft.rfftn(estimate) * otf, image.shape)

        ratio = image / (reblurred + 1e-12)
        correction=cp.fft.irfftn(cp.fft.rfftn(ratio) * otf_, image.shape)
        estimate = estimate * correction/HTones 
    print()        
    
    estimate = cp.asnumpy(estimate)

    if noncirc:
        estimate = unpad(estimate, original_size)

    if (mask is not None):
        estimate = estimate*mask + mask_values
    
    return estimate
  