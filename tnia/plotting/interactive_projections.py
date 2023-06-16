
from ipywidgets import interact, IntSlider, FloatRangeSlider
from tnia.plotting.projections import show_xyz_slice

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

