import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from tnia.deconvolution.pad import pad, unpad, get_next_smooth

def richardson_lucy_np(image, psf, num_iters, noncirc=False, mask=None, use_mkl=False, print_diagnostics=False):
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option and option to mask bad pixels, uses numpy

    Note: NumPy FFT functions always cast 32 bit arrays to float64, so passing in 32 bit arrays to save memory will not work. 

    Args:
        image [numpy float array]: the image to be deconvolved 
        psf [numpy float array]: the point spread function
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.
        mask (numpy float array, optional): If not None, use this mask to mask image pixels that should not be considered in the deconvolution. Defaults to None.
            'bad' pixels will be zeroed during the deconvolution and then replaced with the original value after the deconvolution.
        use_mkl (bool, optional): If true, explicitly try to use MKL for FFTs. Defaults to False.
    Returns:
        [numpy float array]: the deconvolved image
    """

    if use_mkl:
        try:
            import mkl_fft
            fftn = mkl_fft.fftn
            ifftn = mkl_fft.ifftn
            if print_diagnostics:
                print("Using MKL FFT for Richardson-Lucy deconvolution.")
        except ImportError:
            print("MKL FFT not available, using NumPy FFT instead.")
            use_mkl = False

    # if MKL FFT is not available or use_mkl is False, use NumPy FFT functions
    if use_mkl == False:
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn
        
    # if noncirc==False and (image.shape != psf.shape) then pad the psf
    if noncirc==False and (image.shape != psf.shape):
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
    else:
        # TODO: consider option to use get_next_smooth for non-circulant case as well
        extended_size = image.shape #get_next_smooth(extended_size)

    # pad the image, psf and HTOnes array to the extended size computed above
    original_size = image.shape
    image,_=pad(image, extended_size, 'constant')
    HTones,_=pad(HTones, extended_size, 'constant')
    psf,_=pad(psf, extended_size, 'constant')

    otf = fftn(ifftshift(psf))
    otf_ = np.conjugate(otf)

    if noncirc:
        estimate = np.ones_like(image)*np.mean(image)
    else:
        estimate = image
    
    delta = 1e-6

    HTones = np.real(ifftn(fftn(HTones) * otf_))
    HTones[HTones<delta] = 1

    for i in range(num_iters):
        if i % 10 == 0:
            print(i, end =" ")
        
        reblurred = np.real(ifftn(fftn(estimate) * otf))

        ratio = image / (reblurred + delta)
        correction=np.real((ifftn(fftn(ratio) * otf_)))
        correction[correction<0] = delta 
        estimate = estimate * correction/HTones 
    
    print()        

    if estimate.shape != original_size:
        estimate = unpad(estimate, original_size)

    if (mask is not None):
        estimate = estimate*mask + mask_values
    
    return estimate

  