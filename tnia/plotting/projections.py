import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from skimage.transform import resize
from matplotlib.colors import PowerNorm

def show_xy_zy_slice_center(im,  sxy=1, sz=1,figsize=(10,3), colormap=None, vmax=None, gamma=None):
    """ extracts xy, and zy center slices of a 3D image and plots them

    Args:
        image_to_show (_type_): _description_
        sxy (int, optional): _description_. Defaults to 1.
        sz (int, optional): _description_. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """

    return show_xy_zy_slice(im, int(im.shape[2]/2), int(im.shape[1]/2), int(im.shape[0]/2), sxy, sz, figsize, colormap, vmax, gamma)

def show_xy_zy_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,3), colormap=None, vmax=None, gamma=None):
    """ extracts xy, and zy slices at x, y, z of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        x (int): x position of slice
        y (int): y position of slice
        z (int): z position of slice
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (colormap, optional): pyplot colormap to use . Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
 
    slice_zy = np.flip(np.rot90(image_to_show[:,:,x],1),0)
    slice_xy = image_to_show[z,:,:]

    return show_xy_zy(slice_xy, slice_zy, sxy, sz,figsize, colormap, vmax, gamma)

def show_xy_xz_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,3), colormap=None, vmax=None, gamma=None):
    """ extracts xy, and xz slices at x, y, z of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        x (int): x position of slice
        y (int): y position of slice
        z (int): z position of slice
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (colormap, optional): pyplot colormap to use . Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
 
    slice_xz = image_to_show[:,y,:]
    slice_xy = image_to_show[z,:,:]

    return show_xy_xz(slice_xy, slice_xz, sxy, sz,figsize, colormap, vmax, gamma)


def show_xyz_slice_center(image_to_show, sxy=1, sz=1, figsize=(10,10), colormap=None, vmax=None, gamma=None):
    """ extracts xy, xz, and zy slices at center of a 3D image and plots them

    Args:
        image_to_show (_type_): _description_
        sxy (int, optional): _description_. Defaults to 1.
        sz (int, optional): _description_. Defaults to 1.
        figsize (tuple, optional): _description_. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    Returns:
        _type_: _description_
    """
    xc=int(image_to_show.shape[2]/2)
    yc=int(image_to_show.shape[1]/2)
    zc=int(image_to_show.shape[0]/2)

    return show_xyz_slice(image_to_show, xc, yc, zc, sxy, sz,figsize, colormap, vmax, gamma)

def show_xyz_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,10), colormap=None, vmax=None, gamma=None):
    """ extracts xy, xz, and zy slices at x, y, z of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        x (int): x position of slice
        y (int): y position of slice
        z (int): z position of slice
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
 
    slice_zy = np.flip(np.rot90(image_to_show[:,:,x],1),0)
    slice_xz = image_to_show[:,y,:]
    slice_xy = image_to_show[z,:,:]

    return show_xyz(slice_xy, slice_xz, slice_zy, sxy, sz, figsize, colormap, vmax, gamma)

def show_xyz_max(image_to_show, sxy=1, sz=1,figsize=(10,10), colormap=None, vmax=None, gamma=None):
    """ plots max xy, xz, and zy projections of a 3D image

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
 
    return show_xyz_projection(image_to_show, sxy, sz, figsize, np.max, colormap, vmax)
 
def show_xyz_sum(image_to_show, sxy=1, sz=1,figsize=(10,10), colormap=None, vmax=None, gamma=None):
    show_xyz_projection(image_to_show, sxy, sz, figsize, np.sum, colormap, vmax, gamma)
    
def show_xyz_projection(image_to_show, sxy=1, sz=1,figsize=(10,10), projector=np.max, colormap=None, vmax=None, gamma=None):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple): size of figure to
        projector: function to project with
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
    projection_y = projector(image_to_show,1)
    projection_x = np.flip(np.rot90(projector(image_to_show,2),1),0)
    projection_z = projector(image_to_show,0)

    return show_xyz(projection_z, projection_y, projection_x, sxy, sz, figsize, colormap, vmax, gamma)

def show_xyz(xy, xz, zy, sxy=1, sz=1,figsize=(10,10), colormap=None, vmax=None, gamma=None):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        zy (2d numpy array): zy projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
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

    if gamma is None:
        ax0.imshow(xy, colormap, vmax=vmax, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, vmax=vmax, extent=[0,zdim*sz,ydim*sxy,0])
        ax1.set_title('zy')
        ax2.imshow(xz, colormap, vmax=vmax, extent=[0,xdim*sxy,zdim*sz,0])
        ax2.set_title('xz')
    else:
        norm=PowerNorm(gamma=gamma, vmax=vmax)
        ax0.imshow(xy, colormap, norm=norm, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, norm=norm, extent=[0,zdim*sz,ydim*sxy,0])
        ax1.set_title('zy')
        ax2.imshow(xz, colormap, norm=norm, extent=[0,xdim*sxy,zdim*sz,0])
        ax2.set_title('xz')

    return fig


# Copyright tnia 2021 - BSD License
def show_xy_zy_max(image_to_show, sxy=1, sz=1,figsize=(10,3), colormap=None, vmax=None, gamma=None ):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
    projection_x = np.flip(np.rot90(np.max(image_to_show,2),1),0)
    projection_z = np.max(image_to_show,0)

    return show_xy_zy(projection_z, projection_x, sxy, sz, figsize, colormap, vmax, gamma)


# Copyright tnia 2021 - BSD License
def show_xy_zy(xy, zy, sxy=1, sz=1,figsize=(10,3), colormap=None, vmax=None, gamma=None):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        zy (2d numpy array): zy projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.

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
        #xz=resize(xz, (int(xz.shape[0]*z_xy_ratio), xz.shape[1]))
        zy=resize(zy, (zy.shape[0], int(zy.shape[1]*z_xy_ratio)))

    if gamma is None:
        ax0.imshow(xy, colormap, vmax=vmax, extent = [0, xdim*sxy, ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, vmax=vmax, extent = [0, zdim*sz, ydim*sxy,0])
        ax1.set_title('zy')
    else:
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
        ax0.imshow(xy, colormap, norm=norm, extent = [0, xdim*sxy, ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, norm=norm, extent = [0, zdim*sz, ydim*sxy,0])
        ax1.set_title('zy')
    
    return fig


def show_xy_xz(xy, xz, sxy=1, sz=1,figsize=(10,3), colormap=None, vmax=None, gamma=None):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.

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

    spec=gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[ydim,zdim*z_xy_ratio], width_ratios=[xdim],hspace=.01)

    ax0=fig.add_subplot(spec[0])
    ax1=fig.add_subplot(spec[1])
   
    if z_xy_ratio!=1:
        xz=resize(xz, (int(xz.shape[0]*z_xy_ratio), xz.shape[1]))

    if gamma is None:
        ax0.imshow(xy, colormap, vmax=vmax, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(xz, colormap, vmax=vmax, extent=[0,xdim*sxy,zdim*sz,0])
        ax1.set_title('xz')
    else:
        norm=PowerNorm(gamma=gamma, vmax=vmax)
        ax0.imshow(xy, colormap, norm=norm, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(xz, colormap, norm=norm, extent=[0,xdim*sxy,zdim*sz,0])
        ax1.set_title('xz')
      
    return fig

