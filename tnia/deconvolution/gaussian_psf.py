import numpy as np

def gaussian_3d(xy_dim, z_dim, xy_sigma, z_sigma):
    """ Generates a 3D Gaussian PSF

    Parameters:
    ----------
        xy_dim (int): the size of the PSF in the xy plane
        z_dim (int): the size of the PSF in the z direction
        xy_sigma (float): the sigma of the Gaussian in the xy plane
        z_sigma (float): the sigma of the Gaussian in the z direction
    Returns:
    -------
        psf (numpy array): the PSF
    """
    muu = 0.0
    gauss = np.empty([z_dim,xy_dim,xy_dim])
    x_, y_, z_ = np.meshgrid(np.linspace(-(xy_dim//2),xy_dim//2,xy_dim), np.linspace(-(xy_dim//2),xy_dim//2,xy_dim), np.linspace(-(z_dim//2),z_dim//2,z_dim))
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
    """ Generates a 2D Gaussian PSF
    
    Parameters:
    ----------
        xy_dim (int): the size of the PSF in the xy plane
        xy_sigma (float): the sigma of the Gaussian in the xy plane in pixels
    Returns:
    -------
        psf (numpy array): the PSF
    """
    muu = 0.0
    gauss = np.empty([xy_dim,xy_dim])
    x_, y_ = np.meshgrid(np.linspace(-(xy_dim//2),xy_dim//2,xy_dim), np.linspace(-(xy_dim//2),xy_dim//2,xy_dim))
    for x in range(xy_dim):
        for y in range(xy_dim):
            tx=x_[x,y]
            ty=y_[x,y]
            
            gauss[y,x]=np.exp(-( (tx-muu)**2 / ( 2.0 * xy_sigma**2 ) ) )*np.exp(-( (ty-muu)**2 / ( 2.0 * xy_sigma**2 ) ) )

    #gauss=gauss+0.000000000001
    gauss=gauss/gauss.sum()

    return gauss

