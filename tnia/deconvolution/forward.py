from numpy.fft import fftn, ifftn, ifftshift, rfftn, irfftn
import numpy as np
from tnia.deconvolution.pad import pad, unpad, get_next_smooth

def forward(field, psf, background_level, add_poisson=True, gpu=False):
    '''
    Perform forward imaging on an input field using a point spread function (PSF).

    Parameters:
    -----------
    field : ndarray
        The input field to be imaged. 
    psf : ndarray
        The point spread function (PSF) to be used for imaging. 
    background_level : float
        The background level to be added to the image.
    add_poisson : bool, optional
        Whether or not to add Poisson noise to the image. Default is True.
    gpu : bool, optional
        Whether or not to use the GPU for computation. Default is False.

    Returns:
    --------
    ndarray
        The forward-imaged field, with the same shape as the input field.

    Notes:
    ------
    This function is partly inspired by code from https://github.com/jdmanton/rl_positivity_sim by James Manton.
    '''

    # compute the extended size of the image and psf
    extended_size = [field.shape[i]+2*int(psf.shape[i]/2) for i in range(len(field.shape))]
    extended_size = get_next_smooth(extended_size)
    original_size = field.shape
    
    # pad the image and psf to the extended size computed above
    field,_=pad(field, extended_size, 'constant')
    psf,_=pad(psf, extended_size, 'constant')

    try:
        import cupy as cp
    except:
        gpu = False

    if gpu==False:
        # perform forward imaging 
        otf = rfftn(ifftshift(psf))
        field_imaged = irfftn(rfftn(field)*otf)
        field_imaged = field_imaged+background_level
        
        # unpad before poisson noise is added
        field_imaged=unpad(field_imaged, original_size)

        # add poisson noise   
        if add_poisson:
            # set small negative values that may occur to 0
            field_imaged[field_imaged<0]=0
            field_imaged = np.random.poisson(field_imaged.astype(float))

        # return field as numpy array
        return field_imaged    
    else:
        # perform forward imaging on GPU, using cupy, del and mempool.free_all_blocks() is used to delete variables and free memory
        temp = ifftshift(psf)
        
        mempool=cp.get_default_memory_pool()
        psf_shift_cp = cp.array(temp)
        otf_cp = cp.fft.rfftn(psf_shift_cp)
        del psf_shift_cp
        mempool.free_all_blocks()
        
        field_cp = cp.array(field)
        field_fft_cp = cp.fft.rfftn(field_cp)
        del field_cp

        forward_fft_cp = field_fft_cp*otf_cp
        del field_fft_cp
        del otf_cp

        field_imaged_cp = cp.fft.irfftn(forward_fft_cp, field.shape)
        del forward_fft_cp
        
        field_imaged_cp_plus_background = field_imaged_cp+background_level
        del field_imaged_cp
        
        field_imaged = cp.asnumpy(field_imaged_cp_plus_background)
        del field_imaged_cp_plus_background
        
        # unpad before poisson noise is added
        field_imaged=unpad(field_imaged, original_size)
        
        # add poisson noise   
        if add_poisson:
            field_imaged_cp = cp.array(field_imaged)
            field_imaged_poisson_cp = cp.random.poisson(field_imaged_cp)
            del field_imaged_cp
            field_imaged = cp.asnumpy(field_imaged_poisson_cp)
            del field_imaged_poisson_cp
        else:
            field_imaged = cp.asnumpy(field_imaged)

        # explicitly clear the plan cache to avoid memory leak
        cache = cp.fft.config.get_plan_cache()
        cache.clear()
        
        cp.get_default_memory_pool().free_all_blocks()
        # return field as numpy array
        
        return field_imaged.astype('float32')


