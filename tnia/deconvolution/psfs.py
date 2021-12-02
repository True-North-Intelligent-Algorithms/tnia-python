import microscPSF.microscPSF as msPSF
import numpy as np
from skimage.filters import threshold_otsu
from numpy.fft import ifftn, ifftshift, fftshift
from tnia.segmentation.rendering import draw_centroids 

def gibson_lanni_3D(NA, ni, ns, voxel_size_xy, voxel_size_z, xy_size, z_size, pz, wvl):
    m_params = msPSF.m_params
    m_params['NA']=NA
    m_params['ni']=ni
    m_params['ni0']=ni
    m_params['ns']=ns

    zv = np.arange(-z_size*voxel_size_z/2, z_size*voxel_size_z/2, voxel_size_z)

    psf = msPSF.gLXYZFocalScan(m_params, voxel_size_xy, xy_size, zv, True, pz, wvl)
    psf = psf/psf.sum()

    return psf

def paraxial_otf(n, wavelength, numerical_aperture, pixel_size):
    '''
        Note this function inspired by code from https://github.com/jdmanton/rl_positivity_sim by James Manton
    '''
    nx, ny=(n,n)
    
    resolution  = 0.5 * wavelength / numerical_aperture

    image_centre_x = n / 2 + 1
    image_centre_y = n / 2 + 1

    x=np.linspace(0,nx-1,nx)
    y=np.linspace(0,ny-1,ny)
    x=x-image_centre_x
    y=y-image_centre_y

    X, Y = np.meshgrid(x,y)

    filter_radius = 2 * pixel_size / resolution
    r = np.sqrt(X*X+Y*Y)
    r=r/x.max()
    v=r/filter_radius
    v = v * (r<=filter_radius)
    otf = 2 / np.pi * (np.arccos(v) - v * np.sqrt(1 - v*v))*(r<=filter_radius);
    
    return otf

def paraxial_psf(n, wavelength, numerical_aperture, pixel_size):
    otf = paraxial_otf(n, wavelength, numerical_aperture, pixel_size)
    psf = fftshift(ifftn(ifftshift(otf)).astype(np.float32))
    return psf/psf.sum()

def psf_from_beads(bead_image, background_factor=1.25):
    """ Extracts a PSF from a bead image using reverse deconvolution (a.k.a. PSF Distilling)

    Args:
        bead_image (numpy array): an image of a field of sub-resolution beads
        background_factor (float, optional): Used to modulate background subtraction. Defaults to 1.25.

    Returns:
        [numpy array]: the PSF
    """
    bead_image=bead_image-background_factor*bead_image.mean()
    bead_image[bead_image<=0]=.1

    thresholded = bead_image>threshold_otsu(bead_image)

    centroids = draw_centroids(thresholded)

    from clij2fft.richardson_lucy import richardson_lucy
    #from skimage.restoration import richardson_lucy
    im_32=bead_image.astype('float32')
    centroids_32=centroids.astype('float32')
    print('call skimage rl')
    psf=richardson_lucy(im_32, centroids_32, 200)

    return psf



