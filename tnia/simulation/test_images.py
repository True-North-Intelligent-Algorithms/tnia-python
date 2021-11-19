from tnia.simulation.phantoms import ramp2d, random_circles, grid_of_circles
from tnia.deconvolution import psfs
from tnia.deconvolution import forward

def random_circles_on_ramp_background():
    ramp=ramp2d([1000,1000],100,1000).astype('float32')
    random_circles(ramp,5,100,150,1200,1400,354)
    
    n=1000
    wavelength = 500
    na=1.4
    pixel_size = 20

    psf=psfs.paraxial_psf(n, wavelength, na, pixel_size)
    img = forward.forward(ramp, psf,1000,0)

    return ramp, img

def grid_of_circles_on_ramp_background(radius,space,border):  
    """ draws a grid of circles on a background image

    Args:
        radius ([type]): radius of circles 
        space ([type]): space between circles 
        border (): border around circles
    """
   
    ramp=ramp2d([1000,1000],100,1000).astype('float32')
    grid_of_circles(ramp,radius,space,border,1000)
    
    n=1000
    wavelength = 500
    na=1.4
    pixel_size = 20

    psf=psfs.paraxial_psf(n, wavelength, na, pixel_size)
    img = forward.forward(ramp, psf,1000,0)

    return ramp, img



