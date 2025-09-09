import numpy as np
import cupy as cp
from tnia.deconvolution.pad import pad, unpad, get_next_smooth
from tnia.metrics.errors import RMSE

def richardson_lucy_cp(image, psf, num_iters, noncirc=False, mask=None, truth=None, print_diagnostics=False ):
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option and option to mask bad pixels, uses cupy
    
    Note: Cupy FFT behavior is different than numpy.  Cupy FFT always returns real arrays of type float32 and complex of type complex64.
    So using 64 bit inputs for more precision will not work. 

    Some background on the method used to handle edges: https://www.aanda.org/articles/aa/pdf/2005/25/aa2717-05.pdf

    Note: Note the option to mask bad pixels is not simply a matter of masking bad pixels.  Instead the HTOnes array is multiplied by the mask.
    This means bad pixels are handled the same way as edges.  Both edges and bad pixels are considered locations where data was not acquired. 
    This is likely not a novel approach, but I have not found any references to it in the literature.  If you know of any references please let 
    me know.

    Args:
        image [numpy float array]: the image to be deconvolved 
        psf [numpy float array]: the point spread function
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.
        mask (numpy float array, optional): If not None, use this mask to set invalid (saturated or other) image pixels to zero in the HTOnes array.

    Returns:
        [numpy float array]: the deconvolved image
    """

    
    mempool = cp.get_default_memory_pool()
    total_gpu_memory = mempool
    bpg=(1024**3)
 
    if print_diagnostics:
        available_gpu_memory = cp.cuda.Device(0).mem_info[0]
        total_gpu_memory = cp.cuda.Device(0).mem_info[1]
        print("Total GPU memory = {}".format(total_gpu_memory/bpg))
        print("Available GPU memory = {}".format(available_gpu_memory/bpg))
        print("At beginning, used = {}".format(mempool.used_bytes()/bpg))
        
    # if truth is not none we will be calculating the RMSE at each iteration
    if truth is not None:
        stats = {'rmse':[]}
    
    # if noncirc==False and (image.shape != psf.shape) then pad the psf
    if noncirc==False and (image.shape != psf.shape):
        if print_diagnostics:
            print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')
   
    HTones = np.ones_like(image)

    if (mask is not None):
        HTones = HTones * mask
        image=image*mask
    
    # if noncirc==True then pad the image, psf and HTOnes array to the extended size
    if noncirc:
        # compute the extended size of the image and psf
        extended_size = [image.shape[i]+2*int(psf.shape[i]/2) for i in range(len(image.shape))]
        extended_size = get_next_smooth(extended_size)

        # pad the image, psf and HTOnes array to the extended size computed above
        original_size = image.shape
        image,_=pad(image, extended_size, 'constant')
        HTones,_=pad(HTones, extended_size, 'constant')
        psf,_=pad(psf, extended_size, 'constant')

        if truth is not None:
            truth,_=pad(truth, extended_size, 'constant')   
      
    image = image.astype(np.float32)
    image = cp.array(image)

    if print_diagnostics:    
        print("After image, used = {}".format(mempool.used_bytes()/bpg))

    psf = psf.astype(np.float32)
    psf = cp.array(psf)

    if print_diagnostics:    
        print("After psf, used = {}".format(mempool.used_bytes()/bpg))

    HTones = cp.array(HTones, cp.float32)

    if print_diagnostics:    
        print("After HTones, used = {}".format(mempool.used_bytes()/bpg))

    if truth is not None:
        truth = cp.array(truth)

    otf = cp.fft.fftn(cp.fft.ifftshift(psf))
    otf_ = cp.conjugate(otf)

    if noncirc or mask is not None:
        if print_diagnostics:
            print('using flat sheet')
        estimate = cp.ones_like(image)*cp.mean(image)
    else:
        estimate = image

    delta = 1e-6

    if truth is not None:
        # if truth is not None we will be calculating the RMSE at each iteration
        # we need to keep track of the invalid pixels so make a copy of HTones (which is the invalid pixel mask at this point)
        # before it is correlated with the OTF
        mask_extended = HTones.copy()

    HTones = cp.real(cp.fft.ifftn(cp.fft.fftn(HTones) * otf_))
    HTones[HTones<delta] = 1

    print()
    for i in range(num_iters):
        if i % 10 == 0:
            print(i, end =" ")
        
        reblurred = cp.real(cp.fft.ifftn(cp.fft.fftn(estimate) * otf))

        ratio = image / (reblurred + delta)
        correction=cp.real((cp.fft.ifftn(cp.fft.fftn(ratio) * otf_)))
        correction[correction<0] = delta 
        estimate = estimate * correction/HTones 

        if truth is not None:
            rmse = RMSE(truth, estimate, mask_extended, cp)
            stats['rmse'].append(rmse)
    print()        
    
    estimate = cp.asnumpy(estimate)

    if noncirc:
        estimate = unpad(estimate, original_size)

    if (mask is not None):
        mask = cp.asnumpy(mask)
        estimate = estimate*mask
        estimate[ (1-mask) == 1] = estimate.max() 
    
    if truth is not None:
        stats['rmse'] = [s.get() for s in stats['rmse']]
        return estimate, stats
    else:
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
  