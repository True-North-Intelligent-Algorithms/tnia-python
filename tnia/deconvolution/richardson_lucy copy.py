import numpy as np
from numpy.fft import fftn, ifftn, fftshift 
import cupy as cp
from tnia.deconvolution.pad import pad, unpad

def richardson_lucy_cp(image, psf, num_iters, noncirc=False):
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option, uses cupy

    Args:
        image [numpy 32 bit float array]: the image to be deconvolved 
        psf [numpy 32 bit float array]: the point spread function
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.

    Returns:
        [numpy 32 bit float array]: the deconvolved image
    """
    
    # if noncirc==False and (image.shape != psf.shape) then pad the psf
    if noncirc==False and (image.shape != psf.shape):
        print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')
    
    HTones = np.ones_like(image)

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

    HTones = cp.fft.irfftn(cp.fft.rfftn(HTones) * otf_)
    HTones[HTones<1e-6] = 1



    print()
    for i in range(num_iters):
        if i % 10 == 0:
            print(i, end =" ")
        
        reblurred = cp.fft.irfftn(cp.fft.rfftn(estimate) * otf)

        ratio = image / (reblurred + 1e-12)
        correction=cp.fft.irfftn(cp.fft.rfftn(ratio) * otf_)
        estimate = estimate * correction/HTones 
    print()        
    
    estimate = cp.asnumpy(estimate)

    if noncirc:
        estimate = unpad(estimate, original_size)
    
    return estimate
 
# WIP code for numpy version 
def richardson_lucy_np(image, psf, num_iters):
    if (image.shape != psf.shape):
        print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')
   
    otf = fftn(fftshift(psf))
    otf_ = np.conjugate(otf)    
    estimate = image#np.ones(image.shape)/image.sum()

    print()
    for i in range(num_iters):
        print(i, end =" ")
        reblurred = np.real(ifftn(fftn(estimate) * otf))
        ratio = image / (reblurred + 1e-30)
        correction =  np.real((ifftn(fftn(ratio) * otf_)).astype(float))
        estimate = estimate * correction
    print()
    return estimate

   