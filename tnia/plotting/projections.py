import matplotlib.pyplot as plt
from matplotlib import gridspec
import pyclesperanto_prototype as cle
import numpy as np
from skimage.transform import resize

def show_xyz_max_clij(image_to_show, labels=False):
    """
    This function generates three projections using clij, in X-, Y- and Z-direction and shows them.
    """
    fig=plt.figure(figsize=(10,10))

    spec=gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[258,100], width_ratios=[258,100])

    ax0=fig.add_subplot(spec[0])
    ax1=fig.add_subplot(spec[1])
    ax2=fig.add_subplot(spec[2])

    projection_x = cle.maximum_x_projection(image_to_show)
    projection_y = cle.maximum_y_projection(image_to_show)
    projection_z = cle.maximum_z_projection(image_to_show)

    ax0.imshow(projection_z)
    ax1.imshow(projection_x)
    ax2.imshow(projection_y)

def show_xy_yz_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,10)):
    """ extracts xy, xz, and zy slices at x, y, z of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
    """
 
    slice_zy = np.flip(np.rot90(image_to_show[:,:,x],1),0)
    slice_xy = image_to_show[z,:,:]

    return show_xy_zy(slice_xy, slice_zy, sxy, sz,figsize)

def show_xyz_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,10)):
    """ extracts xy, xz, and zy slices at x, y, z of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
    """
 
    slice_zy = np.flip(np.rot90(image_to_show[:,:,x],1),0)
    slice_xz = image_to_show[:,y,:]
    slice_xy = image_to_show[z,:,:]

    return show_xyz(slice_xy, slice_xz, slice_zy, sxy, sz, figsize)


def show_xyz_max(image_to_show, sxy=1, sz=1,figsize=(10,10)):
    return show_xyz_projection(image_to_show, sxy, sz, figsize, np.max)
 
def show_xyz_sum(image_to_show, sxy=1, sz=1,figsize=(10,10)):
    show_xyz_projection(image_to_show, sxy, sz, figsize, np.sum)
    
def show_xyz_projection(image_to_show, sxy=1, sz=1,figsize=(10,10), projector=np.max):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple): size of figure to
        projector: function to project with
    """
    projection_y = projector(image_to_show,1)
    projection_x = np.flip(np.rot90(projector(image_to_show,2),1),0)
    projection_z = projector(image_to_show,0)

    return show_xyz(projection_z, projection_y, projection_x, sxy, sz, figsize)

def show_xyz(xy, xz, zy, sxy=1, sz=1,figsize=(10,10)):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        zy (2d numpy array): zy projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.

    Returns:
        [type]: [description]
    """
    
    fig=plt.figure(figsize=figsize)
    
    xdim = xy.shape[1]
    ydim = xy.shape[0]
    zdim = xz.shape[0]

    z_xy_ratio=1

    if sxy!=sz:
        z_xy_ratio=sz/sxy

    spec=gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[ydim,zdim*z_xy_ratio], width_ratios=[xdim,zdim*z_xy_ratio],hspace=.01)

    ax0=fig.add_subplot(spec[0])
    ax1=fig.add_subplot(spec[1])
    ax2=fig.add_subplot(spec[2])
   
    if z_xy_ratio!=1:
        xz=resize(xz, (int(xz.shape[0]*z_xy_ratio), xz.shape[1]))
        zy=resize(zy, (zy.shape[0], int(zy.shape[1]*z_xy_ratio)))


    ax0.imshow(xy)
    ax0.set_title('xy')
    ax1.imshow(zy)
    ax1.set_title('zy')
    ax2.imshow(xz)
    ax2.set_title('xz')

    return fig


# Copyright tnia 2021 - BSD License
def show_xy_zy_max(image_to_show, sxy=1, sz=1,figsize=(10,3)):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
    """
    projection_x = np.flip(np.rot90(np.max(image_to_show,2),1),0)
    projection_z = np.max(image_to_show,0)

    return show_xy_zy(projection_z, projection_x, sxy, sz, figsize)


# Copyright tnia 2021 - BSD License
def show_xy_zy(xy, zy, sxy=1, sz=1,figsize=(10,3)):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        zy (2d numpy array): zy projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.

    Returns:
        [type]: [description]
    """
    
    fig=plt.figure(figsize=figsize)
    
    xdim = xy.shape[1]
    ydim = xy.shape[0]
    zdim = zy.shape[1]

    z_xy_ratio=1

    if sxy!=sz:
        z_xy_ratio=sz/sxy

    spec=gridspec.GridSpec(ncols=2, nrows=1, height_ratios=[ydim], width_ratios=[xdim,zdim*z_xy_ratio],hspace=.01)

    ax0=fig.add_subplot(spec[0])
    ax1=fig.add_subplot(spec[1])
   
    if z_xy_ratio!=1:
        xz=resize(xz, (int(xz.shape[0]*z_xy_ratio), xz.shape[1]))
        zy=resize(zy, (zy.shape[0], int(zy.shape[1]*z_xy_ratio)))


    ax0.imshow(xy)
    ax0.set_title('xy')
    ax1.imshow(zy)
    ax1.set_title('zy')

    return fig
