from numpy.fft import fftn, ifftn, ifftshift 
from cupy.random import poisson
import cupy as cp
from tnia.deconvolution.pad import pad, unpad, get_next_smooth
from tnia.utility.cupy_checker import get_platform


def add_poisson_noise(img):
    img = cp.array(img)
    img = poisson(img.astype(float))
    return cp.asnumpy(img).astype('float32')

def forward(field, psf, background_level, add_poisson=True):
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

    Returns:
    --------
    ndarray
        The forward-imaged field, with the same shape as the input field.

    Notes:
    ------
    This function is inspired by code from https://github.com/jdmanton/rl_positivity_sim by James Manton.
    '''

    # compute the extended size of the image and psf
    extended_size = [field.shape[i]+2*int(psf.shape[i]/2) for i in range(len(field.shape))]
    extended_size = get_next_smooth(extended_size)
    original_size = field.shape
    
    # pad the image and psf to the extended size computed above
    field,_=pad(field, extended_size, 'constant')
    psf,_=pad(psf, extended_size, 'constant')
    
    # perform forward imaging 
    otf = fftn(ifftshift(psf))
    field_imaged = ifftn(fftn(field)*otf)
    field_imaged = field_imaged+background_level
    
    # unpad before poisson noise is added
    field_imaged=unpad(field_imaged, original_size)

    field_imaged = cp.array(field_imaged)
    
    # add poisson noise   
    if add_poisson:
        field_imaged = poisson(field_imaged.astype(float))

    # return field as numpy array
    return cp.asnumpy(field_imaged).astype('float32')


def convolve(x, h):
    pl = get_platform(x)
    return pl.fft.irfftn(pl.fft.rfftn(x) * pl.fft.rfftn(pl.fft.ifftshift(h)), x.shape)

