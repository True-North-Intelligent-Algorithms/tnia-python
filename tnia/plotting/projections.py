import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import gridspec
import numpy as np
from skimage.transform import resize
from matplotlib.colors import PowerNorm

def show_xy_zy_slice_center(im,  sxy=1, sz=1,figsize=(10,3), colormap=None, vmin=None, vmax=None, gamma=None):
    """ extracts xy, and zy center slices of a 3D image and plots them

    Args:
        image_to_show (_type_): _description_
        sxy (int, optional): _description_. Defaults to 1.
        sz (int, optional): _description_. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.

    Returns:
        fig: matplotlib figure

    """

    return show_xy_zy_slice(im, int(im.shape[2]/2), int(im.shape[1]/2), int(im.shape[0]/2), sxy, sz, figsize, colormap, vmin, vmax, gamma)

def show_xy_zy_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,3), colormap=None, vmin=None, vmax=None, gamma=None, show_cross_hairs=True):
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
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.
        show_cross_hairs (bool, optional): show cross hairs at x, y, z. Defaults to True.

    Returns:
        fig: matplotlib figure
    """
 
    slice_zy = np.flip(np.rot90(image_to_show[:,:,x],1),0)
    slice_xy = image_to_show[z,:,:]

    fig =  show_xy_zy(slice_xy, slice_zy, sxy, sz,figsize, colormap, vmin, vmax, gamma)

    if show_cross_hairs:
        fig.axes[0].axvline(x*sxy+0.5, color='r')
        fig.axes[0].axhline(y*sxy+0.5, color='r')
        fig.axes[1].axvline(z*sz+0.5*sz, color='r')
        fig.axes[1].axhline(y*sxy+0.5, color='r')    

    return fig


def show_xy_xz_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,3), colormap=None, vmin=None, vmax=None, gamma=None, show_cross_hairs=True):
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
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.
        show_cross_hairs (bool, optional): show cross hairs at x, y, z. Defaults to True.

    Returns:
        fig: matplotlib figure
    """
 
    slice_xz = image_to_show[:,y,:]
    slice_xy = image_to_show[z,:,:]

    fig = show_xy_xz(slice_xy, slice_xz, sxy, sz,figsize, colormap, vmin, vmax, gamma)

    if show_cross_hairs:
        fig.axes[0].axvline(x*sxy+0.5, color='r')
        fig.axes[0].axhline(y*sxy+0.5, color='r')
        fig.axes[1].axvline(x*sxy+0.5, color='r')
        fig.axes[1].axhline(z*sz+0.5*sz, color='r')

    return fig

def show_xyz_slice_center(image_to_show, sxy=1, sz=1, figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None, show_cross_hairs=False):
    """ extracts xy, xz, and zy slices at center of a 3D image and plots them

    Args:
        image_to_show (_type_): _description_
        sxy (int, optional): _description_. Defaults to 1.
        sz (int, optional): _description_. Defaults to 1.
        figsize (tuple, optional): _description_. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.
        show_cross_hairs (bool, optional): show cross hairs at center. Defaults to False.

    Returns:
        fig: matplotlib figure
    """
    xc=int(image_to_show.shape[2]/2)
    yc=int(image_to_show.shape[1]/2)
    zc=int(image_to_show.shape[0]/2)

    return show_xyz_slice(image_to_show, xc, yc, zc, sxy, sz,figsize, colormap, vmin, vmax, gamma, show_cross_hairs=show_cross_hairs)

def show_xyz_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None, use_plt=True, show_cross_hairs=True):
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
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.
        use_plt (bool, optional): use matplotlib.pyplot. Defaults to True.
        show_cross_hairs (bool, optional): show cross hairs at x, y, z. Defaults to True.

    Returns:
        fig: matplotlib figure
    """
 
    slice_zy = np.flip(np.rot90(image_to_show[:,:,x],1),0)
    slice_xz = image_to_show[:,y,:]
    slice_xy = image_to_show[z,:,:]

    fig = show_xyz(slice_xy, slice_xz, slice_zy, sxy, sz, figsize, colormap, vmin, vmax, gamma, use_plt)
    
    if show_cross_hairs:
        fig.axes[0].axvline(x*sxy+0.5, color='r')
        fig.axes[0].axhline(y*sxy+0.5, color='r')
        fig.axes[1].axvline(z*sz+0.5*sz, color='r')
        fig.axes[1].axhline(y*sxy+0.5, color='r')
        fig.axes[2].axvline(x*sxy+0.5, color='r')
        fig.axes[2].axhline(z*sz+0.5*sz, color='r')

    return fig

def show_xyz_max(image_to_show, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None):
    """ plots max xy, xz, and zy projections of a 3D image

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size. Defaults to 1.
        sz (float, optional): z pixel size. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.

    Returns:
        fig: matplotlib figure
    """
 
    return show_xyz_projection(image_to_show, sxy, sz, figsize, np.max, colormap, vmin, vmax)
 
def show_xyz_sum(image_to_show, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None):
    show_xyz_projection(image_to_show, sxy, sz, figsize, np.sum, colormap, vmin, vmax, gamma)
    
def show_xyz_projection(image_to_show, sxy=1, sz=1,figsize=(10,10), projector=np.max, colormap=None, vmin=None, vmax=None, gamma=None):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple): size of figure to
        projector: function to project with
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.

    Returns:
        fig: matplotlib figure
    """
    projection_y = projector(image_to_show,1)
    projection_x = np.flip(np.rot90(projector(image_to_show,2),1),0)
    projection_z = projector(image_to_show,0)

    return show_xyz(projection_z, projection_y, projection_x, sxy, sz, figsize, colormap, vmin, vmax, gamma)

def show_xyz(xy, xz, zy, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None, use_plt=True):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        zy (2d numpy array): zy projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.
        use_plt (bool, optional): use matplotlib.pyplot. Defaults to True.
    Returns:
        fig: matplotlib figure
    """
    
    if use_plt:
        fig=plt.figure(figsize=figsize, layout='constrained')
    else:
        fig = Figure(figsize=figsize, constrained_layout=True)
    
    
    xdim = xy.shape[1]
    ydim = xy.shape[0]
    zdim = xz.shape[0]

    z_xy_ratio=1

    if sxy!=sz:
        z_xy_ratio=sz/sxy

    spec=gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[ydim,zdim*z_xy_ratio], width_ratios=[xdim,zdim*z_xy_ratio],hspace=.01, figure = fig)

    ax0=fig.add_subplot(spec[0])
    ax1=fig.add_subplot(spec[1])
    ax2=fig.add_subplot(spec[2])
   
    if z_xy_ratio!=1:
        xz=resize(xz, (int(xz.shape[0]*z_xy_ratio), xz.shape[1]))
        zy=resize(zy, (zy.shape[0], int(zy.shape[1]*z_xy_ratio)))

    if len(xy.shape)==3 and vmax is not None:
        if vmin is None:
            vmin = 0
        xy = np.clip((xy-vmin)/(vmax-vmin), 0, 1)
        xz = np.clip((xz-vmin)/(vmax-vmin), 0, 1)
        zy = np.clip((zy-vmin)/(vmax-vmin), 0, 1)

    if gamma is None:
        ax0.imshow(xy, colormap, vmin=vmin, vmax=vmax, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, vmin=vmin, vmax=vmax, extent=[0,zdim*sz,ydim*sxy,0])
        ax1.set_title('zy')
        ax2.imshow(xz, colormap, vmin=vmin, vmax=vmax, extent=[0,xdim*sxy,zdim*sz,0])
        ax2.set_title('xz')
    else:
        norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        ax0.imshow(xy, colormap, norm=norm, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, norm=norm, extent=[0,zdim*sz,ydim*sxy,0])
        ax1.set_title('zy')
        ax2.imshow(xz, colormap, norm=norm, extent=[0,xdim*sxy,zdim*sz,0])
        ax2.set_title('xz')

    return fig

def show_xy_zy_max(image_to_show, sxy=1, sz=1,figsize=(10,3), colormap=None, vmin=None, vmax=None, gamma=None ):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.

    Returns:
        fig: matplotlib figure
    """
    projection_x = np.flip(np.rot90(np.max(image_to_show,2),1),0)
    projection_z = np.max(image_to_show,0)

    return show_xy_zy(projection_z, projection_x, sxy, sz, figsize, colormap, vmin, vmax, gamma)

def show_xy_zy(xy, zy, sxy=1, sz=1,figsize=(10,3), colormap=None, vmin=None, vmax=None, gamma=None):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        zy (2d numpy array): zy projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.

    Returns:
        fig: matplotlib figure
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

    # if rgb
    if len(xy.shape)==3 and vmax is not None:
        if vmin is None:
            vmin = 0
        xy = np.clip((xy-vmin)/(vmax-vmin), 0, 1)
        zy = np.clip((zy-vmin)/(vmax-vmin), 0, 1)
    
    if gamma is None:
        ax0.imshow(xy, colormap, vmin=vmin, vmax=vmax, extent = [0, xdim*sxy, ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, vmin=vmin, vmax=vmax, extent = [0, zdim*sz, ydim*sxy,0])
        ax1.set_title('zy')
    else:
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
        ax0.imshow(xy, colormap, norm=norm, extent = [0, xdim*sxy, ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, norm=norm, extent = [0, zdim*sz, ydim*sxy,0])
        ax1.set_title('zy')
    
    return fig


def show_xy_xz(xy, xz, sxy=1, sz=1,figsize=(10,3), colormap=None, vmin=None, vmax=None, gamma=None):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.

    Returns:
        fig: matplotlib figure
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

    if len(xy.shape)==3 and vmax is not None:
        if vmin is None:
            vmin = 0
        xy = np.clip((xy-vmin)/(vmax-vmin), 0, 1)
        xz = np.clip((xz-vmin)/(vmax-vmin), 0, 1)

    if gamma is None:
        ax0.imshow(xy, colormap, vmin=vmin, vmax=vmax, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(xz, colormap, vmin=vmin, vmax=vmax, extent=[0,xdim*sxy,zdim*sz,0])
        ax1.set_title('xz')
    else:
        norm=PowerNorm(gamma=gamma, vmax=vmax)
        ax0.imshow(xy, colormap, norm=norm, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        ax1.imshow(xz, colormap, norm=norm, extent=[0,xdim*sxy,zdim*sz,0])
        ax1.set_title('xz')
      
    return fig

def show_xyz_max_slabs(image_to_show, x = [0,1], y = [0,1], z = [0,1], sxy=1, sz=1,figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None):
    """ plots max xy, xz, and zy projections of a 3D image SLABS (slice intervals)

    Author: PanosOik https://github.com/PanosOik

    Args:
        image_to_show (3d numpy array): image to plot
        x: slices for x in format [x_1, x_2] where values are integers, to be passed as slice(x_1, x_2, None)
        y: slices for y in format [y_1, y_2] where values are integers
        z: slices for z in format [z_1, z_2] where values are integers
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.

    Returns:
        fig: matplotlib figure
    """
    ### Coerce into integers for slices
    x_ = [int(i) for i in x]
    y_ = [int(i) for i in y]
    z_ = [int(i) for i in z]

    x_slices = slice(*x)
    y_slices = slice(*y)
    z_slices = slice(*z)

    return show_xyz_projection_slabs(image_to_show, x_slices, y_slices, z_slices, sxy, sz, figsize, np.max, colormap, vmin, vmax)

def show_xyz_projection_slabs(image_to_show, x_slices, y_slices, z_slices, sxy=1, sz=1,figsize=(10,10), projector=np.max, colormap=None, vmin=None, vmax=None, gamma=None):
    """ generates xy, xz, and zy max projections of a 3D image and plots them
    
    Author: PanosOik https://github.com/PanosOik

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple): size of figure to
        projector: function to project with
        colormap (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): minimum value for display range. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
        gamma (float, optional): gamma value for display range. Defaults to None.

    Returns:
        fig: matplotlib figure
    """
    projection_y = projector(image_to_show[:,y_slices,:],1)
    projection_x = np.flip(np.rot90(projector(image_to_show[:,:,x_slices],2),1),0)
    projection_z = projector(image_to_show[z_slices,:,:],0)

    return show_xyz(projection_z, projection_y, projection_x, sxy, sz, figsize, colormap, vmin, vmax, gamma)
