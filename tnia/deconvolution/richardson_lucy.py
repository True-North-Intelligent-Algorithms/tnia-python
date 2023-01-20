import numpy as np
from numpy.fft import fftn, ifftn, fftshift 
import cupy as cp
from tnia.deconvolution.pad import pad

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

def richardson_lucy_cp(image, psf, num_iters):
    if (image.shape != psf.shape):
        print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')
    
    image = cp.array(image)
    psf = cp.array(psf)
    
    otf = cp.fft.fftn(cp.fft.fftshift(psf))
    otf_ = cp.conjugate(otf)
    estimate = image#np.ones(image.shape)/image.sum()

    print()
    for i in range(num_iters):
        print(i, end =" ")
        reblurred = np.real(cp.fft.ifftn(cp.fft.fftn(estimate) * otf))

        ratio = image / (reblurred + 1e-30)
        correction=cp.real((cp.fft.ifftn(cp.fft.fftn(ratio) * otf_)))
        estimate = estimate * correction 
    print()        
    return cp.asnumpy(estimate)
    