
def forward_variable(image, psfs):
    forward = np.zeros_like(image)

    for psf in psfs:

    return forward
'''
 # WIP - version of RL that uses real FFT.  Seems to work and uses less memory, but is slower than the above version
def richardson_lucy_cp_rfft_variable(image, psfs, num_iters, noncirc=False, mask=None):
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option and option to mask bad pixels, uses cupy

    Args:
        image [numpy 32 bit float array]: the image to be deconvolved 
        psfs [list of numpy 32 bit float array]: the point spread functions all should be the same size
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.
        mask (numpy 32 bit float array, optional): If not None, use this mask to mask image pixels that should not be considered in the deconvolution. Defaults to None.
            'bad' pixels will be zeroed during the deconvolution and then replaced with the original value after the deconvolution.

    Returns:
        [numpy 32 bit float array]: the deconvolved image
    """

    
    # if noncirc==False and (image.shape != psf.shape) then pad the psf
    if noncirc==False and (image.shape != psf.shape):
        print('padding psf')
        
        for i in range(len(psfs)):
            psfs[i],_=pad(psfs[i], image.shape, 'constant')
    
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
        
        for i in range(len(psfs)):
            psfs[i],_=pad(psfs[i], extended_size, 'constant')
    
    image = cp.array(image)
    
    for i in range(len(psfs)):
        psfs[i] = cp.array(np.fft.ifftshift(psfs[i])) 
    
    HTones = cp.array(HTones)

    otfs = []
    otfs_ = []
    for i in range(len(psfs)):
        otfs.append(cp.fft.rfftn(psfs[i]))
        psfs[i] = None
        otfs_.append(cp.conjugate(otfs[i])) 

    if noncirc:
        estimate = cp.ones_like(image)*cp.mean(image)
    else:
        estimate = image

    HToness = []
    for i in range(len(psfs)):
        HToness.append(cp.fft.irfftn(otfs_[i] * HTones))
        HToness[i][HToness[i]<1e-6] = 1

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
'''

def ramp_3d(array, a, b, reverse=False):
    # Get the slice index range to apply the ramp to
    slice_range = range(a, b + 1)

    # Calculate the ramp values
    if reverse:
        ramp_values = np.linspace(1, 0, len(slice_range))
    else:
        ramp_values = np.linspace(0, 1, len(slice_range))

    # Set the ramp values in the slice range
    for i in range(a, b):
        array[i, :, :] = ramp_values[i]


 # WIP - version of RL that uses real FFT.  Seems to work and uses less memory, but is slower than the above version
def richardson_lucy_cp_rfft_variable_2(image, psft, psfb, num_iters, noncirc=False, mask=None):
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option and option to mask bad pixels, uses cupy

    Args:
        image [numpy 32 bit float array]: the image to be deconvolved 
        psfs [list of numpy 32 bit float array]: the point spread functions all should be the same size
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.
        mask (numpy 32 bit float array, optional): If not None, use this mask to mask image pixels that should not be considered in the deconvolution. Defaults to None.
            'bad' pixels will be zeroed during the deconvolution and then replaced with the original value after the deconvolution.

    Returns:
        [numpy 32 bit float array]: the deconvolved image
    """

    HTones = np.ones_like(image)

    if (mask is not None):
        HTones = HTones * mask
        mask_values = image*(1-mask)
        image=image*mask

    # if noncirc==True then pad the image, psf and HTOnes array to the extended size
    if noncirc:
        # compute the extended size of the image and psf
        extended_size = [image.shape[i]+2*int(psfb.shape[i]/2) for i in range(len(image.shape))]

        # pad the image, psf and HTOnes array to the extended size computed above
        original_size = image.shape
        image,_=pad(image, extended_size, 'constant')
        HTones,_=pad(HTones, extended_size, 'constant')
        psft,_=pad(psft, extended_size, 'constant')    
        psfb,_=pad(psfb, extended_size, 'constant')
    
    image = cp.array(image)
    
    psft = cp.array(np.fft.ifftshift(psft))
    psfb = cp.array(np.fft.ifftshift(psfb))
    
    HTones = cp.array(HTones)

    otfb=cp.fft.rfftn(psfb)
    otft=cp.fft.rfftn(psft)
    psfst = None
    psfsb = None
    otft_=cp.conjugate(otft)
    otfb_=cp.conjugate(otfb)

    if noncirc:
        estimate = cp.ones_like(image)*cp.mean(image)
    else:
        estimate = image

        HTonest=cp.fft.irfftn(otft_ * HTones)
        HTonest[HTonest<1e-6] = 1
        HTonestb=cp.fft.irfftn(otfb_ * HTones)
        HTonestb[HTonestb<1e-6] = 1


    rampt=np.zeros_like(estimate)
    rampb=np.zeros_like(estimate)

    ramp_3d(rampt, 0, rampt.shape[0], True)
    ramp_3d(rampb, 0, rampb.shape[0])

    print()
    for i in range(num_iters):
        if i % 10 == 0:
            print(i, end =" ")
        
        reblurredt = cp.fft.irfftn(cp.fft.rfftn(estimate*rampt) * otft, image.shape)
        reblurredb = cp.fft.irfftn(cp.fft.rfftn(estimate*rampt) * otft, image.shape)
        reblurred = reblurredt + reblurredb

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
 
 