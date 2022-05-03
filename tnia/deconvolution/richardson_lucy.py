import numpy as np
from numpy.fft import fftn, ifftn, fftshift 
import cupy as cp

def richardson_lucy_np(image, psf, num_iters):
    
    otf = fftn(fftshift(psf))
    otf_ = np.conjugate(otf)    
    estimate = image#np.ones(image.shape)/image.sum()

    for i in range(num_iters):
        print(i)
        
        reblurred = ifftn(fftn(estimate) * otf)
        ratio = image / (reblurred + 1e-30)
        estimate = estimate * (ifftn(fftn(ratio) * otf_)).astype(float)
        

    return estimate

def richardson_lucy_cp(image, psf, num_iters):
    print('1')
    image = cp.array(image)
    print('2')
    psf = cp.array(psf)
    
    print('3')
    otf = cp.fft.fftn(cp.fft.fftshift(psf))
    print('4')
    estimate = image#np.ones(image.shape)/image.sum()

    print('iiiiiii')
    for i in range(num_iters):
        print('cupy',i)
       
        reblurred = cp.fft.ifftn(cp.fft.fftn(estimate) * otf)
        ratio = image / (reblurred + 1e-30)
        estimate = estimate * (cp.fft.ifftn(cp.fft.fftn(ratio) * otf)).astype(float)
        
    return cp.asnumpy(estimate)