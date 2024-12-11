from ipywidgets import interact, IntSlider, FloatRangeSlider
from tnia.plotting.projections import show_xyz_slice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage.transform import resize
from matplotlib.colors import PowerNorm

def show_xyz_slice_interactive(im, sxy=1, sz=1,figsize=(10,10), colormap=None, vmax=None, gamma=None):
    """
    Display an interactive widget to explore a 3D image by showing a slice in the x, y, and z directions.

    Requires ipywidgets to be installed.
    
    Parameters
    ----------
    im : array
        3D image to display
    sxy : float
        scaling factor for x and y dimensions
    sz : float
        scaling factor for z dimension
    figsize : tuple
        size of the figure
    colormap : str
        colormap to use
    vmax : float
        maximum value to use for the colormap
    """
    def display(x,y,z):
        fig = show_xyz_slice(im, x, y, z, sxy, sz,figsize, colormap, vmax, gamma)
        fig.axes[0].axvline(x*sxy+0.5, color='r')
        fig.axes[0].axhline(y*sxy+0.5, color='r')
        fig.axes[1].axvline(z*sz+0.5*sz, color='r')
        fig.axes[1].axhline(y*sxy+0.5, color='r')
        fig.axes[2].axvline(x*sxy+0.5, color='r')
        fig.axes[2].axhline(z*sz+0.5*sz, color='r')
    x_slider = IntSlider(min=0, max=im.shape[2]-1, step=1, value=im.shape[2]//2)
    y_slider = IntSlider(min=0, max=im.shape[1]-1, step=1, value=im.shape[1]//2)
    z_slider = IntSlider(min=0, max=im.shape[0]-1, step=1, value=im.shape[0]//2)

    interact(display, x=x_slider, y=y_slider, z=z_slider)


def show_xyz_slice_interactive_2(im, sxy=1, sz=1, figsize=(10,10), colormap=None, vmax=None, gamma=None):
    """
    Display an interactive widget to explore a 3D image by showing a slice in the x, y, and z directions.
    
    This version of 'show_xyz_slice_interactive' keeps track of the im objects returned form imshow and sets data on them directory. 
     
    It may be a bit faster than show_xyz_slice_interactive but requires '%matplotlib widget' to work. 
    
    Requires ipywidgets to be installed.
    
    Parameters
    ----------
    im : array 3D image to display
    sxy : float scaling factor for x and y dimensions
    sz : float scaling factor for z dimension
    figsize : tuple size of the figure
    colormap : str colormap to use
    vmax : float maximum value to use for the colormap
    gamma : float gamma value for colormap

    Returns
    -------
    fig : matplotlib figure
    """

    def make_slices(x,y,z):
        slice_zy = np.flip(np.rot90(im[:,:,x],1),0)
        slice_xz = im[:,y,:]
        slice_xy = im[z,:,:]

        return slice_xy, slice_zy, slice_xz

    fig=plt.figure(figsize=figsize, layout='constrained')
    #fig.suptitle('Interactive 3D Image Viewer')
    
    xdim = im.shape[2]
    ydim = im.shape[1]
    zdim = im.shape[0]
    
    z_xy_ratio=1

    if sxy!=sz:
        z_xy_ratio=sz/sxy

    spec=gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[ydim,zdim*z_xy_ratio], width_ratios=[xdim,zdim*z_xy_ratio],hspace=.01, figure = fig)

    ax0=fig.add_subplot(spec[0])
    ax1=fig.add_subplot(spec[1])
    ax2=fig.add_subplot(spec[2])
        
    vline_xy = fig.axes[0].axvline(xdim//2*sxy+0.5, color='r')
    hline_xy = fig.axes[0].axhline(ydim//2*sxy+0.5, color='r')
    vline_zy = fig.axes[1].axvline(zdim//2*sz+0.5*sz, color='r')
    hline_zy = fig.axes[1].axhline(ydim//2*sxy+0.5, color='r')
    vline_xz = fig.axes[2].axvline(xdim//2*sxy+0.5, color='r')
    hline_xz = fig.axes[2].axhline(zdim//2*sz+0.5*sz, color='r')

    xy, zy, xz = make_slices(xdim//2, ydim//2, zdim//2)

    if z_xy_ratio!=1:
        xz=resize(xz, (int(xz.shape[0]*z_xy_ratio), xz.shape[1]))
        zy=resize(zy, (zy.shape[0], int(zy.shape[1]*z_xy_ratio)))

    if gamma is None:
        im0 = ax0.imshow(xy, colormap, vmax=vmax, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        im1 = ax1.imshow(zy, colormap, vmax=vmax, extent=[0,zdim*sz,ydim*sxy,0])
        ax1.set_title('zy')
        im2 = ax2.imshow(xz, colormap, vmax=vmax, extent=[0,xdim*sxy,zdim*sz,0])
        ax2.set_title('xz')
    else:
        norm=PowerNorm(gamma=gamma, vmax=vmax)
        im0 = ax0.imshow(xy, colormap, norm=norm, extent=[0,xdim*sxy,ydim*sxy,0])
        ax0.set_title('xy')
        im1 = ax1.imshow(zy, colormap, norm=norm, extent=[0,zdim*sz,ydim*sxy,0])
        ax1.set_title('zy')
        im2 = ax2.imshow(xz, colormap, norm=norm, extent=[0,xdim*sxy,zdim*sz,0])
        ax2.set_title('xz')
    
    x_slider = IntSlider(min=0, max=im.shape[2]-1, step=1, value=im.shape[2]//2, layout={'width': '800px'})
    y_slider = IntSlider(min=0, max=im.shape[1]-1, step=1, value=im.shape[1]//2, layout={'width': '800px'})
    z_slider = IntSlider(min=0, max=im.shape[0]-1, step=1, value=im.shape[0]//2, layout={'width': '800px'})
    
    def display(x,y,z):
        xy, zy, xz = make_slices(x,y,z)
        im0.set_data(xy)
        im1.set_data(zy)
        im2.set_data(xz)
        vline_xy.set_xdata(x*sxy+0.5)
        hline_xy.set_ydata(y*sxy+0.5)
        vline_zy.set_xdata(z*sz+0.5*sz)
        hline_zy.set_ydata(y*sxy+0.5)
        vline_xz.set_xdata(x*sxy+0.5)
        hline_xz.set_ydata(z*sz+0.5*sz)

    interact(display, x=x_slider, y=y_slider, z=z_slider)

    return fig




