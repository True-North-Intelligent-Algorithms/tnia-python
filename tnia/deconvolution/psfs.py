#import microscPSF.microscPSF as msPSF
import numpy as np
from skimage.filters import threshold_otsu
from numpy.fft import ifftn, ifftshift, fftshift
from tnia.segmentation.rendering import draw_centroids 
from skimage.filters import median
from skimage.morphology import cube
from skimage.measure import label
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage.io import imread
import json
from tnia.nd.ndutil import centercrop

def gibson_lanni_3D(NA, ni, ns, voxel_size_xy, voxel_size_z, xy_size, z_size, pz, wvl, confocal = False, use_psfm=False, convert_to_32=True):
    """
       Generates a 3D PSF using the Gibson-Lanni model.  If use_psfm is True the psfmodels implementation will be used.  Otherwise, the sdeconv implementation will be used.

       Note this function does NOT center the result from sdeconv, which is not centered by default.

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
            use_psfm : bool
                If True, the PSF is generated using the psfmodels package. Otherwise, the PSF is generated using sdeconv.

            Returns
            -------
            psf : ndarray
    """ 
    
    if use_psfm:
        import psfmodels as psfm
        psf = psfm.make_psf(z_size, xy_size, model='scalar', dxy=voxel_size_xy, dz=voxel_size_z, pz=pz, ni0=ni, ni=ni, ns=ns, NA=NA, wvl=wvl)
        if convert_to_32:
            psf = psf.astype('float32')
        return psf
    else:
        import sdeconv
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

def gibson_lanni_3D_partial_confocal(NA, ni, ns, voxel_size_xy, voxel_size_z, xy_size, z_size, pz, wvl, use_psfm=False, confocal_factor=1):
    """
       Generates a 3D PSF using the Gibson-Lanni model.  If use_psfm is True the psfmodels implementation will be used.  Otherwise, the sdeconv implementation will be used.

       Note this function centers the result from sdeconv, which is not centered by default.

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
            use_psfm : bool
                If True, the PSF is generated using the psfmodels package. Otherwise, the PSF is generated using sdeconv.
            confocal_factor:
                Factor by which the PSF is raised to the power of.  A value of 1 is a widefield PSF (no change)
                A value of 2 is a fully confocal PSF (PSF raised to the power of 2).
                A value between 1 and 2 is an ad-hoc approximation of a partially confocal PSF (this option should be used carefully)
            Returns
            -------
            psf : ndarray
    """ 
    if use_psfm:
        psf = gibson_lanni_3D(NA, ni, ns, voxel_size_xy, voxel_size_z, xy_size, z_size, pz, wvl, False, True)
    else:
        z_compute_psf_dim = z_size+z_size//2
        z_crop_psf_dim = z_size

        psf = gibson_lanni_3D(NA, ni, ns, voxel_size_xy, voxel_size_z, xy_size, z_compute_psf_dim, pz, wvl, False, False)
        psf=recenter_psf_axial(psf, z_crop_psf_dim)
    
    psf = psf.astype('float32')
    psf=psf**confocal_factor
    psf=psf/psf.sum()

    return psf 

#Note this function inspired by code from https://github.com/jdmanton/rl_positivity_sim by James Manton
def paraxial_otf(n, wavelength, numerical_aperture, pixel_size):
    """Generates a paraxial OTF for a given wavelength, numerical aperture, and pixel size

    Parameters:
    ----------
        n (int): the size of the OTF
        wavelength (float): the wavelength of the light in microns
        numerical_aperture (float): the numerical aperture of the objective
        pixel_size (float): the pixel size in microns
    Returns:
    -------
        otf (numpy array): the OTF
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
    """Generates a paraxial PSF for a given wavelength, numerical aperture, and pixel size

    Parameters:
    ----------
        n (int): the size of the PSF
        wavelength (float): the wavelength of the light in microns
        numerical_aperture (float): the numerical aperture of the objective
        pixel_size (float): the pixel size in microns
    Returns:
    -------
        psf (numpy array): the PSF
    """
    otf = paraxial_otf(n, wavelength, numerical_aperture, pixel_size)
    psf = fftshift(ifftn(ifftshift(otf)).astype(np.float32))
    return psf/psf.sum()

def psf_from_beads(bead_image, background_factor=1.25, apply_median=False, peak_method=1, thresh=0):
    """ Extracts a PSF from a bead image using reverse deconvolution (a.k.a. PSF Distilling)

    Parameters:
    ----------
        bead_image (numpy array): an image of a field of sub-resolution beads
        background_factor (float, optional): Used to modulate background subtraction. Defaults to 1.25.
        apply_median (bool, optional): Apply a median filter, use if noise is being segmented as beads
        peak_method (int, optional): 1=centroid of label regions, 2=local max
        thresh (float, optional): threshold for peak detection. Defaults to 0.
    Returns:
    -------
        psf (numpy array): the PSF
        bead_image (numpy array): the bead image (with background subtracted and optional median filter applied)
        centroids (numpy array): the bead centroids
    """
    bead_image=bead_image-background_factor*bead_image.mean()
    bead_image[bead_image<=0]=.1

    if (apply_median==True):
        bead_image = median(bead_image, cube(3))

    thresholded = bead_image>threshold_otsu(bead_image)

    # peak method is centroid
    if peak_method==1:
        centroids = draw_centroids(thresholded,img=bead_image)
    # peak method is local max
    elif peak_method==2:
        # find peaks
        peaks = peak_local_max(bead_image, min_distance=1, threshold_abs=0.1, exclude_border=False)

        centroids = np.zeros_like(bead_image)
        # only keep peaks above threshold
        for peak in peaks:
            idx = tuple(int(p) for p in peak)
            if bead_image[idx]>thresh:
                centroids[idx]=bead_image[idx]
        
    im_32=bead_image.astype('float32')
    centroids_32=centroids.astype('float32')
    centroids_32 = centroids_32+0.0000001
    centroids_32 = centroids_32/centroids_32.sum()
    
    try:
        from tnia.deconvolution.richardson_lucy import richardson_lucy_cp
        rl = richardson_lucy_cp
    except:
        from clij2fft.richardson_lucy import richardson_lucy
        rl = richardson_lucy

    psf = rl(im_32, centroids_32, 200)
    psf=psf/psf.sum()

    return psf, im_32, centroids_32

def recenter_psf_axial(psf, newz, use_centroid=False, return_labels=False):
    """ recenters a PSF.   Useful for recentering a theoretical PSF that was generated at off center z location (often done when modelling spherical aberration) 
    
    Note currently the center is an approximation, in the future this could be improved by using the float center and interpolation 

    Parameters:
    ----------
        psf (numpy array): array with off center PSF
        newz (int): desired new z size after centering PSF
        use_centroid (bool, optional): use the centroid of the PSF to center it. Defaults to False.
        return_labels (bool, optional): return the labels from the segmentation. Defaults to False.

    Returns:
    -------
        [numpy array]: centered PSF
    """
    
    # if using centroid, segment the PSF and find the centroid
    if use_centroid:
        thresholded = psf>threshold_otsu(psf)
        labels=label(thresholded)
        objects = regionprops(labels)
        cz=int(objects[0].centroid[0])
    else:
        cz,cy,cx=np.unravel_index(psf.argmax(), psf.shape)
    
    start=cz-newz//2
    psf=psf[start:start+newz,:,:]

    if use_centroid and return_labels:
        return psf, labels
    else:
        return psf

def load_psf(path_):
    """ loads a PSF

    it is assumed that under the directory path_ there is a file called psf.tif, and a file called psf.json.  The json file contains the metadata for the PSF

    Parameters:
    ----------
        path_ (str): path to the PSF

    Returns:
    -------
        [numpy array]: PSF
        dict: metadata for the PSF
    """
    psf = imread(path_+'//psf.tif')
    json_ = json.load(open(path_+'//psf.json'))
    return psf, json_

def load_and_resize_psf(path_, new_spacing=None, new_size = None):
    """ loads a PSF and resizes it to a new size and spacing

    it is assumed that under the directory path_ there is a file called psf.tif, and a file called psf.json.  The json file contains the metadata for the PSF

    Parameters:
    ----------
        path_ (str): path to the PSF
        new_spacing (float): new spacing of the PSF
        new_size (int): new size of the PSF

    Returns:
    -------
        [numpy array]: resized PSF
    """
    psf, json_ = load_psf(path_)

    if new_spacing is not None:
        xy_spacing = json_.get('xy_spacing', 1)
        z_spacing = json_.get('z_spacing', 1)

        # rescale the PSF
        new_x = int(psf.shape[2]*xy_spacing/new_spacing[2])
        new_y = int(psf.shape[1]*xy_spacing/new_spacing[1])
        new_z = int(psf.shape[0]*z_spacing/new_spacing[0])

        psf = resize(psf, (new_z,new_y,new_x), order=1, mode='constant', cval=0, clip=True, preserve_range=True, anti_aliasing=True)

    # if new size is not none, crop the PSF
    if new_size is not None:
        psf = centercrop(psf, new_size[2]) 
    psf = psf/psf.sum()

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

