from numpy.fft import fftn, ifftn, fftshift 
from numpy.random import poisson

def forward(field, psf, max_photons, background_level, add_poisson=True):
    '''
        Note this function inspired by code from https://github.com/jdmanton/rl_positivity_sim by James Manton
    '''
    otf = fftn(fftshift(psf))
    field_imaged = ifftn(fftn(field)*otf)
    field_imaged = field_imaged/field_imaged.max()
    field_imaged = field_imaged*max_photons+background_level

    if add_poisson:
        return poisson(field_imaged.astype(float))
    else:
        return field_imaged.astype(float)
