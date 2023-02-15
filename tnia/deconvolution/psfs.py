#import microscPSF.microscPSF as msPSF
import numpy as np
from skimage.filters import threshold_otsu
from numpy.fft import ifftn, ifftshift, fftshift
from tnia.segmentation.rendering import draw_centroids 
from skimage.filters import median
from skimage.morphology import cube
import sdeconv
from skimage.measure import label
from skimage.measure import regionprops

def gibson_lanni_3D(NA, ni, ns, voxel_size_xy, voxel_size_z, xy_size, z_size, pz, wvl, confocal = False):
    """
       Generates a 3D PSF using the Gibson-Lanni model.

         Parameters
            ----------
            NA : float
                Numerical aperture of the objective.
            ni : float
                Refractive index of the immersion medium.
            ns : float
                Refractive index of the sample.
            voxel_size_xy : float
                Voxel size in the xy plane in microns.
            voxel_size_z : float    
                Voxel size in the z direction in microns.
            xy_size : int
                Number of voxels in the xy plane.
            z_size : int
                Number of voxels in the z direction.
            pz : float
                Position of the focal plane in microns.
            wvl : float
                Wavelength of the light in microns.
            confocal : bool
                If True, the PSF is convolved with itself to simulate a confocal PSF.
    """ 
    
    version_list=sdeconv.__version__.split('.')
    
    if version_list[0] == '0':
        print('sdeconv 0.x.x detected')
        from sdeconv.deconv import PSFGibsonLanni
        gl = PSFGibsonLanni((z_size, xy_size, xy_size),1000*voxel_size_xy, 1000*voxel_size_z, NA, 1000)
        psf = gl.run()
    elif version_list[0] == '1':
        print('sdeconv 1.x.x detected')
        from sdeconv.psfs import SPSFGibsonLanni

        gl = SPSFGibsonLanni((z_size, xy_size, xy_size), NA=NA, ni=ni, ni0=ni, ns=ns, res_lateral=voxel_size_xy, res_axial=voxel_size_z, wavelength=wvl, pZ=pz)
        psf=gl()
        psf_=psf.cpu()
        psf= psf_.numpy()
    
    psf = psf.astype('float32')
    if (confocal):
        psf=psf*psf
    psf = psf/psf.sum()
    return psf

        #Note this function inspired by code from https://github.com/jdmanton/rl_positivity_sim by James Manton
def paraxial_otf(n, wavelength, numerical_aperture, pixel_size):
    """Generates a paraxial OTF for a given wavelength, numerical aperture, and pixel size

    Args:
        n (int): the size of the OTF
        wavelength (float): the wavelength of the light in microns
        numerical_aperture (float): the numerical aperture of the objective
        pixel_size (float): the pixel size in microns
    """    

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

def psf_from_beads(bead_image, background_factor=1.25, apply_median=False):
    """ Extracts a PSF from a bead image using reverse deconvolution (a.k.a. PSF Distilling)

    Args:
        bead_image (numpy array): an image of a field of sub-resolution beads
        background_factor (float, optional): Used to modulate background subtraction. Defaults to 1.25.
        apply_media (bool, optional): Apply a median filter, use if noise is being segmented as beads
    Returns:
        [numpy array]: the PSF
    """
    bead_image=bead_image-background_factor*bead_image.mean()
    bead_image[bead_image<=0]=.1

    if (apply_median==True):
        bead_image = median(bead_image, cube(3))

    thresholded = bead_image>threshold_otsu(bead_image)

    centroids = draw_centroids(thresholded)

    from clij2fft.richardson_lucy import richardson_lucy
    #from skimage.restoration import richardson_lucy
    im_32=bead_image.astype('float32')
    centroids_32=centroids.astype('float32')
    centroids_32 = centroids_32+0.0000001
    centroids_32 = centroids_32/centroids_32.sum()
    #centroids_32 = centroids_+0.0000001
    print('call skimage rl')
    psf=richardson_lucy(im_32, centroids_32, 200)
    psf=psf/psf.sum()

    return psf, im_32, centroids_32

def gaussian_3d(xy_dim, z_dim, xy_sigma, z_sigma):
    """ Generates a 3D Gaussian PSF

    Args:
        xy_dim (_type_): x and y dimensions of the PSF 
        z_dim (_type_): z dimension of the PSF
        xy_sigma (_type_): x and y sigma of the PSF
        z_sigma (_type_): z sigma of the PSF

    Returns:
        [numpy array]: the PSF
    """
    muu = 0.0
    gauss = np.empty([z_dim,xy_dim,xy_dim])
    x_, y_, z_ = np.meshgrid(np.linspace(-10,10,xy_dim), np.linspace(-10,10,xy_dim), np.linspace(-10,10,z_dim))
    for x in range(xy_dim):
        for y in range(xy_dim):
            for z in range(z_dim):
                tx=x_[x,y,z]
                ty=y_[x,y,z]
                tz=z_[x,y,z]
            
                gauss[z,y,x]=np.exp(-( (tx-muu)**2 / ( 2.0 * xy_sigma**2 ) ) )*np.exp(-( (ty-muu)**2 / ( 2.0 * xy_sigma**2 ) ) )*np.exp(-( (tz-muu)**2 / ( 2.0 * z_sigma**2 ) ) )

    gauss=gauss+0.000000000001
    gauss=gauss/gauss.sum()

    return gauss

def gaussian_2d(xy_dim, xy_sigma):
    muu = 0.0
    gauss = np.empty([xy_dim,xy_dim])
    x_, y_ = np.meshgrid(np.linspace(-10,10,xy_dim), np.linspace(-10,10,xy_dim))
    for x in range(xy_dim):
        for y in range(xy_dim):
            tx=x_[x,y]
            ty=y_[x,y]
            
            gauss[y,x]=np.exp(-( (tx-muu)**2 / ( 2.0 * xy_sigma**2 ) ) )*np.exp(-( (ty-muu)**2 / ( 2.0 * xy_sigma**2 ) ) )

    gauss=gauss+0.000000000001
    gauss=gauss/gauss.sum()

    return gauss

def recenter_psf_axial(psf, newz, return_labels=False):
    """ recenters a PSF.   Useful for recentering a theoretical PSF that was generated at off center z location (often done when modelling spherical aberration) 
    
    Note currently the center is an approximation, in the future this could be improved by using the float center and interpolation 

    Args:
        psf (numpy array): array with off center PSF
        newz (int): desired new z size after centering PSF

    Returns:
        _type_: _description_
    """
    thresholded = psf>threshold_otsu(psf)
    labels=label(thresholded)
    objects = regionprops(labels)
    #cz=int(objects[0].centroid[0])
    cz,cy,cx=np.unravel_index(psf.argmax(), psf.shape)
    psf=psf[int(cz-newz/2):int(cz+newz/2),:,:]

    if return_labels:
        return psf, labels
    else:
        return psf

     
def gibson_lanni_3D_old(NA, ni, ns, voxel_size_xy, voxel_size_z, xy_size, z_size, pz, wvl):
    m_params = msPSF.m_params
    m_params['NA']=NA
    m_params['ni']=ni
    m_params['ni0']=ni
    m_params['ns']=ns

    zv = np.zeros(z_size)

    start = -(z_size-1)*voxel_size_z/2.

    for z in range(z_size):
        zv[z]=start+z*voxel_size_z

    psf = msPSF.gLXYZFocalScan(m_params, voxel_size_xy, xy_size, zv, True, pz, wvl)
    psf = psf/psf.sum()

    return psf, zv

