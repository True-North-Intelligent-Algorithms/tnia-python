import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import gridspec
import numpy as np
from skimage.transform import resize
from matplotlib.colors import PowerNorm

from matplotlib import rcParams
# Set the global interpolation to 'none'
rcParams['image.interpolation'] = 'none'

def show_xy_zy_slice_center(im,  sxy=1, sz=1,figsize=(10,3), colormap=None, vmin=None, vmax=None, gamma=None, show_crosshairs=False):
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

    return show_xy_zy_slice(im, int(im.shape[2]/2), int(im.shape[1]/2), int(im.shape[0]/2), sxy, sz, figsize, colormap, vmin, vmax, gamma, show_cross_hairs=show_crosshairs)

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
        fig.axes[0].axvline(x*sxy+0.5*sxy, color='r')
        fig.axes[0].axhline(y*sxy+0.5*sxy, color='r')
        fig.axes[1].axvline(z*sz+0.5*sz, color='r')
        fig.axes[1].axhline(y*sxy+0.5*sxy, color='r')    

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

def show_xyz_max(image_to_show, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None, extentxy=None, extentzy=None, extentxz=None):
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
 
    return show_xyz_projection(image_to_show, sxy, sz, figsize, np.max, colormap, vmin, vmax, gamma, extentxy=extentxy, extentzy=extentzy, extentxz=extentxz)
 
def show_xyz_sum(image_to_show, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None, extentxy=None, extentzy=None, extentxz=None):
    show_xyz_projection(image_to_show, sxy, sz, figsize, np.sum, colormap, vmin, vmax, gamma, extentxy=extentxy, extentzy=extentzy, extentxz=extentxz)
    
def show_xyz_projection(image_to_show, sxy=1, sz=1,figsize=(10,10), projector=np.max, colormap=None, vmin=None, vmax=None, gamma=None, extentxy=None, extentzy=None, extentxz=None):
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

    return show_xyz(projection_z, projection_y, projection_x, sxy, sz, figsize, colormap, vmin, vmax, gamma, use_plt=True, extentxy=extentxy, extentzy=extentzy, extentxz=extentxz)

def show_xyz(xy, xz, zy, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin=None, vmax=None, gamma=None, use_plt=True, extentxy=None, extentzy=None, extentxz=None):
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

    if extentxy is None:
    # if extentxy is not provided, calculate it based on sxy and dimensions
        extentxy = [0, xdim*sxy, ydim*sxy, 0]
    if extentzy is None:
    # if extentzy is not provided, calculate it based on sxy and dimensions
        extentzy = [0, zdim*sz, ydim*sxy, 0]
    if extentxz is None:
        extentxz = [0, xdim*sxy, zdim*sz, 0]

    if gamma is None:
        ax0.imshow(xy, colormap, vmin=vmin, vmax=vmax, extent=extentxy)
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, vmin=vmin, vmax=vmax, extent=extentzy, aspect='auto')
        ax1.set_title('zy')
        ax2.imshow(xz, colormap, vmin=vmin, vmax=vmax, extent=extentxz, aspect='auto')
        ax2.set_title('xz')
    else:
        norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        ax0.imshow(xy, colormap, norm=norm, extent=extentxy)
        ax0.set_title('xy')
        ax1.imshow(zy, colormap, norm=norm, extent=extentzy, aspect='auto')
        ax1.set_title('zy')
        ax2.imshow(xz, colormap, norm=norm, extent=extentxz, aspect='auto')
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

def show_xyz_projection_peaks(img, x_peaks, y_peaks, xy_axis, zy_axis, xz_axis, color='red', marker='ro', label=None):
    """ shows peaks in pre-computed xy, xz and zy of a 3D image in a plot
    Args:
        x_peaks (list): list of x peaks
        y_peaks (list): list of y peaks
        xy_axis (matplotlib axis): axis for xy projection
        zy_axis (matplotlib axis): axis for zy projection
        xz_axis (matplotlib axis): axis for xz projection
        label (str, optional): legend label for these peaks (on the xy axis; call ``xy_axis.legend()``)
    """
    margin = 5

    for i, (x, y) in enumerate(zip(x_peaks, y_peaks)):
        #print(f"Peak at x: {x}, y: {y}")
        xy_axis.plot(x, y, marker,  markerfacecolor='none', markeredgecolor=color, label=label if i == 0 else None)
        
        x_dim = img.shape[2]
        y_dim = img.shape[1]

        # Define bounds, with clipping to stay inside array dimensions
        x_min = max(0, int(x) - margin)
        x_max = min(x_dim, int(x) + margin + 1)
        y_min = max(0, int(y) - margin)
        y_max = min(y_dim, int(y) + margin + 1)

        # Extract the sub-volume around (x, y)
        sub_volume = img[:, y_min:y_max, x_min:x_max]

        # Get the flat index of the maximum value
        flat_index = np.argmax(sub_volume)

        # Convert flat index to (z, y, x) within the sub-volume
        z_rel, y_rel, x_rel = np.unravel_index(flat_index, sub_volume.shape)

        # Map back to original array coordinates
        z_abs = z_rel #*0.5
        y_abs = y_min + y_rel
        x_abs = x_min + x_rel

        half_width = 5

        zy_axis.plot( [z_abs, z_abs], [max(y_abs - half_width,0), min(y_abs + half_width,img.shape[1])],color=color, linewidth=1)

        xz_axis.plot([max(x_abs - half_width,0), min(x_abs + half_width, img.shape[2])], [z_abs, z_abs], color=color, linewidth=1)


def show_projection_grid(volumes, rows, voxel=1.0, bounds=None, projector=np.sum, transform=None,
                         cmap='magma', gamma=None, share='row', clip_percentile=None, origin='lower',
                         aspect='equal', height_ratios=None, figsize=None, col_width=6.0,
                         xy_scale=1.0, z_scale=None, z_stretch=1.0, ylabels=None, xlabels=None, hlines=None,
                         vlines=None, line_color='cyan', vline_color=None, hide=(), transpose=False, ticks=True,
                         title_format='{name}: {row}', titles=None, title_fontsize=10, tick_fontsize=7,
                         suptitle=None, suptitle_fontsize=12, tight_rect=None, h_pad=None):
    """Projections of several 3D volumes side by side: one column per volume, one row per view.

    Made for comparing volumes of the same region (for example two localization programs rendered with
    ``tnia.plotting.points.points_to_volume``): every row shows the same view of every volume, on one
    colour scale, with axes in real-world units.

    Args:
        volumes (dict): ``{name: volume}``, each indexed ``[z, y, x]``, all the same shape.
        rows (list): one entry per row, ``(view, slabs, title)``; ``view`` is ``'xy'``, ``'xz'`` or ``'yz'``
            (the first letter is the horizontal axis), or ``'zx'`` / ``'zy'`` for depth running across;
            ``slabs`` (optional) maps ``'x'`` / ``'y'`` / ``'z'`` to a slice of voxel indices: only those
            voxels count (the rest are zeroed; the view still shows the whole volume). Use
            ``points.range_to_slice`` to make the slices. ``title`` (optional) defaults to the view.
        voxel (float or (xy, z)): voxel size, in the units of ``bounds``.
        bounds (sequence of 3 (lo, hi), optional): the volume's ``(x, y, z)`` origin, as returned by
            ``points_to_volume``; the extent is ``lo + n * voxel``. Default: from 0.
        projector (callable): ``np.sum`` (counts) or ``np.max``.
        transform (callable, optional): applied to each projection before display, for example ``np.log1p``.
        cmap (str), gamma (float, optional): colour map, and a power-law display norm if given.
        share (str): ``'row'`` one colour scale per row (vmin 0), ``'all'`` one for the whole grid,
            ``'none'`` each panel its own (matplotlib's default range unless ``clip_percentile`` is set).
        clip_percentile (float, optional): vmax at this percentile of the non-zero pixels instead of the max.
        origin (str): ``'lower'`` puts y and z upward; ``'upper'`` is image convention.
        aspect (str or list): ``'equal'`` (real proportions) or ``'auto'``, for all rows or one per row.
        height_ratios (list, optional): row heights; default from each row's real height / width.
        figsize (tuple, optional): default from ``col_width`` and the row heights.
        xy_scale, z_scale (float): factor from volume units to axis units (for example 1e-3 for nm -> um);
            ``z_scale`` defaults to ``xy_scale``.
        z_stretch (float): extra display-only stretch of z.
        ylabels (list, optional): a y-axis label per row (``None`` to leave one out).
        xlabels (list, optional): an x-axis label per row (view), likewise; with ``transpose`` only on the bottom
            panel of each column.
        hlines, vlines (dict, optional): ``{row index: [values]}``, dotted lines at those positions, in
            volume units (scaled like the axis they sit on).
        line_color, vline_color (str): colour of the hlines, and of the vlines (default: the same).
        hide (iterable): ``(name, row index)`` panels to leave empty.
        transpose (bool): one row per volume, one column per view instead.
        ticks (bool): show axis ticks.
        title_format (str): panel title, from ``{name}`` and ``{row}``.
        titles (dict, optional): ``{(name, row index): title}`` for panels that need their own title.
        suptitle (str, optional), suptitle_fontsize (float): figure title.
        tight_rect, h_pad: passed to ``fig.tight_layout``.

    Returns:
        fig, axes (2D array in the displayed layout), images (``images[row][col] = (projection, extent)`` in
        the ``rows`` x ``volumes`` order, before ``transform``), for adding overlays.
    """
    names = list(volumes)
    shape = volumes[names[0]].shape
    sxy, sz = (voxel, voxel) if np.isscalar(voxel) else voxel
    zsc = xy_scale if z_scale is None else z_scale
    lo = [0., 0., 0.] if bounds is None else [b[0] for b in bounds]
    span = {'x': (lo[0], lo[0] + shape[2] * sxy, xy_scale), 'y': (lo[1], lo[1] + shape[1] * sxy, xy_scale),
            'z': (lo[2], lo[2] + shape[0] * sz, zsc * z_stretch)}
    # view -> (horizontal axis, vertical axis, axis projected over); 'zx' / 'zy' are 'xz' / 'yz' turned on their side
    axes_of = {'xy': ('x', 'y', 0), 'xz': ('x', 'z', 1), 'yz': ('y', 'z', 2), 'zx': ('z', 'x', 1), 'zy': ('z', 'y', 2)}
    rows = [(r[0], (r[1] if len(r) > 1 and r[1] else {}), (r[2] if len(r) > 2 else r[0])) for r in rows]
    aspects = [aspect] * len(rows) if isinstance(aspect, str) else list(aspect)

    images, displays = [], []
    for view, slabs, _ in rows:
        h_ax, v_ax, proj_axis = axes_of[view]
        ext = [span[h_ax][0] * span[h_ax][2], span[h_ax][1] * span[h_ax][2],
               span[v_ax][0] * span[v_ax][2], span[v_ax][1] * span[v_ax][2]]
        row_imgs, row_disp = [], []
        for name in names:
            vol = volumes[name]
            if slabs:
                keep = np.zeros(vol.shape, bool)
                keep[slabs.get('z', slice(None)), slabs.get('y', slice(None)), slabs.get('x', slice(None))] = True
                vol = np.where(keep, vol, 0)
            img = projector(vol, axis=proj_axis)               # xy -> [y, x], xz -> [z, x], yz -> [z, y]
            if view in ('zx', 'zy'):
                img = img.T                                    # zx -> [x, z], zy -> [y, z]
            row_imgs.append((img, ext))
            row_disp.append(transform(img) if transform is not None else img)
        images.append(row_imgs)
        displays.append(row_disp)

    def top(d):
        nz = d[d > 0]
        if nz.size == 0:
            return 1.
        return float(np.percentile(nz, clip_percentile)) if clip_percentile is not None else float(d.max())

    if share == 'all':
        vmax_all = max(top(d) for row in displays for d in row)
    ratio = [(e[3] - e[2]) / (e[1] - e[0]) for e in (images[r][0][1] for r in range(len(rows)))]
    if transpose:
        n_r, n_c = len(names), len(rows)
        hr = [1.] * n_r if height_ratios is None else height_ratios
        fs = figsize or (col_width * n_c, col_width * max(ratio) * n_r + 0.8 * n_r)
    else:
        n_r, n_c = len(rows), len(names)
        hr = ratio if height_ratios is None else height_ratios
        fs = figsize or (col_width * n_c, col_width * sum(hr) / max(hr[0], 1e-9) * hr[0] + 0.8 * n_r)
    fig, axes = plt.subplots(n_r, n_c, figsize=fs, squeeze=False, gridspec_kw={'height_ratios': hr})
    hide = set(hide)
    for r, (view, slabs, row_title) in enumerate(rows):
        if share == 'row':
            vmax_row = max(top(d) for d in displays[r])
        for c, name in enumerate(names):
            ax = axes[c, r] if transpose else axes[r, c]
            if (name, r) in hide:
                ax.axis('off')
                continue
            disp, (img, ext) = displays[r][c], images[r][c]
            if share == 'row':
                vmin, vmax = 0., vmax_row
            elif share == 'all':
                vmin, vmax = 0., vmax_all
            else:
                vmin, vmax = (0., top(disp)) if clip_percentile is not None else (None, None)
            norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax) if gamma is not None else None
            if origin == 'upper':                              # row 0 is drawn at the top: keep the axis labels right
                ext = [ext[0], ext[1], ext[3], ext[2]]
            ax.imshow(disp, origin=origin, cmap=cmap, extent=ext, aspect=aspects[r],
                      **({'norm': norm} if norm is not None else {'vmin': vmin, 'vmax': vmax}))
            title = (titles or {}).get((name, r), title_format.format(name=name, row=row_title))
            ax.set_title(title, fontsize=title_fontsize)
            ax.tick_params(labelsize=tick_fontsize)
            if not ticks:
                ax.set_xticks([]); ax.set_yticks([])
            if ylabels is not None and ylabels[r] is not None:
                ax.set_ylabel(ylabels[r], fontsize=tick_fontsize + 1)
            if xlabels is not None and xlabels[r] is not None and (not transpose or c == len(names) - 1):
                ax.set_xlabel(xlabels[r], fontsize=tick_fontsize + 1)
            h_ax, v_ax, _ = axes_of[view]
            for v in (hlines or {}).get(r, []):
                ax.axhline(v * span[v_ax][2], color=line_color, lw=0.6, ls=':')
            for v in (vlines or {}).get(r, []):
                ax.axvline(v * span[h_ax][2], color=vline_color or line_color, lw=0.6, ls=':')
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    layout = {}
    if tight_rect is not None or suptitle is not None:
        layout['rect'] = tight_rect or (0, 0, 1, 0.97)
    if h_pad is not None:
        layout['h_pad'] = h_pad
    fig.tight_layout(**layout)
    return fig, axes, images
