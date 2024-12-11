from operator import is_
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm, to_rgb, CSS4_COLORS
import colorsys
from skimage import color



color_dictionary = {
    "Cy5": [1, 0, 0],        # Red
    "DAPI": [0, 0, 1],       # Blue
    "FITC": [0, 1, 0],       # Green
    "Texa": [1, 0.5, 0],     # Orange
    "AF594":[1,0.2,0],       # Red
    "Cy2": [0, 1, 0],
}

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):
    """
    Create a random colormap for labels
    
    Args:
        n (int, optional): number of labels. Defaults to 2**16.
        h (tuple, optional): hue range. Defaults to (0,1).
        l (tuple, optional): lightness range. Defaults to (.4,1).
        s (tuple, optional): saturation range. Defaults to (.2,.8).
        
        Returns:
        matplotlib.colors.ListedColormap: colormap for labels
    """
    import matplotlib
    import colorsys
    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

def imshow2d(im, width=8, height=6, colormap=None, vmin=None, vmax=None):
    """ a little helper to show images of differnt sizes using pyplot.  Just makes calls to imshow more readable. 

    Args:
        im ([type]): input image
        width (int, optional): Width of plot. Defaults to 8.
        height (int, optional): Height of plot. Defaults to 6.
        colormap ([type], optional): colormap for image. Defaults to None.
        vmin ([type], optional): min value for colormap. Defaults to None.
        vmax ([type], optional): max value for colormap. Defaults to None.

    Returns:
        [type]: returns the figure
    """
    fig, ax = plt.subplots(figsize=(width,height))
    ax.imshow(im, colormap, vmin=vmin, vmax=vmax)
    return fig

def imshow_multi2d(ims, titles, rows, cols, width=10, height=4, colormaps=None, vmin=None, vmax=None, gamma=None, plottypes=None, xlabels = None, ylabels = None):
    """ a little helper to show a grid of images of differnt sizes using pyplot.  Just makes calls to imshow more readable. 

    Args:
        ims (list of np arrays): input images
        titles(list of strings): titles of images
        rows: number of rows of images
        cols: number of collumns of images (not rows*cols needs to equal number of images) 
        width (int, optional): Width of plot. Defaults to 8.
        height (int, optional): Height of plot. Defaults to 6.
        colormaps (list of colormaps, optional): list of colormaps for each image. Defaults to None.
        vmin (float, optional): min value for colormap. Defaults to None.
        vmax (float, optional): max value for colormap. Defaults to None.
        gamma (float, optional): gamma value for colormap. Defaults to None.
        plottypes (list of strings, optional): list of plot types. Defaults to None.
        xlabels (list of strings, optional): list of xlabels. Defaults to None.
        ylabels (list of strings, optional): list of ylabels. Defaults to None.

    Returns:
        [type]: returns the figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=(width,height))

    if colormaps is None:
        colormaps = [None]*len(ims)

    if plottypes is None:
        plottypes = ['imshow']*len(ims)

    if xlabels is None:
        xlabels = [None]*len(ims)
    
    if ylabels is None:
        ylabels = [None]*len(ims)

    for im,ax,title,colormap,plottype,xlabel,ylabel in zip(ims,np.ndarray.flatten(axes),titles,colormaps,plottypes,xlabels,ylabels):
        #print('types')
        #print(type(im), type(ax), type(title))
        #print()

        if plottype == 'imshow':    
            if gamma is not None:
                norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
                ax.imshow(im.squeeze(), colormap,norm=norm)
            else:
                ax.imshow(im.squeeze(), colormap,vmin=vmin, vmax=vmax)

            if xlabel is not None:
                ax.set_xlabel(xlabel)

            if ylabel is not None:
                ax.set_ylabel(ylabel)
        elif plottype == 'hist':
            ax.hist(im.flatten(), bins=100)
        
        ax.set_title(title)

    return fig 

def plot_line(line, ax, coords=None, linestyle='-',linewidth=1, color='k', bars=False, label=None):
    if coords is None:
        coords = np.arange(len(line))
    
    if bars:
        ax.bar(coords, line, width=linewidth, color=color, label=label)
    else:
        ax.plot(coords, line, linestyle=linestyle, linewidth=linewidth, color=color, label=label)

def plot_lines(lines, ax=None, coords=None, linestyles=None, linewidths=None, linecolors=None, bars=None, labels=None):
    
    if (ax==None):
        fig, ax = plt.subplots()
    else:
        fig=None
    
    if linestyles is None:
        linestyles = ['-']*len(lines)
    if linewidths is None:
        linewidths = [1]*len(lines)
    if linecolors is None:
        linecolors = ['k']*len(lines)
    if bars is None:
        bars = [False]*len(lines)
    if coords is None:
        coords = [np.arange(len(line))]*len(lines)
    if labels is None:
        labels = [None]*len(lines)

    for coord, line, linestyle, linewidth, linecolor, bar, label in zip(coords, lines, linestyles, linewidths, linecolors, bars, labels):
        plot_line(line, ax, coord, linestyle=linestyle, linewidth=linewidth, color=linecolor, bars=bar, label=label)

    ax.legend()

    return fig

def plot_color_space(im, rgb_func, width=10, height=10, ax=None, titles=None):
    """
    plot a color space image with 3 channels
    
    
    Args:
        im (np.array): input image
        rgb_func (function): function to convert from rgb to desired color space
        width (int, optional): width of plot. Defaults to 10.
        height (int, optional): height of plot. Defaults to 10.
        ax (matplotlib axis, optional): axis to plot on. Defaults to None.
        titles (list of strings, optional): list of titles for each channel. Defaults to None.
        """
    if (ax==None):
        fig, ax = plt.subplots(1, 3, figsize=(width,height))
    else:
        fig=None

    null = np.zeros_like(im[:, :, 0])
    im_a = rgb_func(np.stack((im[:,:,0], null, null), axis=-1))
    im_b = rgb_func(np.stack((null, im[:,:,1], null), axis=-1))
    im_c = rgb_func(np.stack((null, null, im[:,:,2]), axis=-1))

    ax[0].imshow(im_a)
    if titles is not None:
        ax[0].set_title(titles[0])
    
    ax[1].imshow(im_b)
    if titles is not None:
        ax[1].set_title(titles[1])

    ax[2].imshow(im_c)
    if titles is not None:
        ax[2].set_title(titles[2])
        
    return fig


def hsv_to_rgb(arr):
    """Convert an array of HSV values to RGB values.

     Args:
        arr (np.array): Array of HSV values.

    Returns:
        np.array: Array of RGB values.
    """
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r, g, b), axis=-1)
    return rgb


def mask_overlay(img, masks, colors=None):
    """Overlay masks on image (set image to grayscale).

    (Vendored from Cellpose codebase)

    Args:
        img (int or float, 2D or 3D array): Image of size [Ly x Lx (x nchan)].
        masks (int, 2D array): Masks where 0=NO masks; 1,2,...=mask labels.
        colors (int, 2D array, optional): Size [nmasks x 3], each entry is a color in 0-255 range.

    Returns:
        RGB (uint8, 3D array): Array of masks overlaid on grayscale image.
    """
    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim > 2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)

    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:, :, 2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max() + 1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        if colors is None:
            HSV[ipix[0], ipix[1], 0] = hues[n]
        else:
            HSV[ipix[0], ipix[1], 0] = colors[n, 0]
        HSV[ipix[0], ipix[1], 1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def create_rgb(ims, color_names=None, color_dictionary_=None, channel_pos=-1, vmin=None, vmax=None, gamma=1):
    """create an RGB color image by mixing multiple images, a list of wavelengths and a color dictionary

    Author: Panos Oikonomou https://github.com/eigenP and Brian Northan
    
    Args:
        ims (either list of np arrays each representing a channel or a single np array): 
            if a list of images each entry in the list represents a channel
            if a nd numpy array the channel_pos parameter is used to determine the channel
            if the channel pos is -1 the smallest dimension is assumed to be the channel
        color_names (list of strings): list of color names
            if None the color dictionary is used in order
            if the color name strings are stain names (ie DAPI, FITC, Cy5) the color dictionary is used
            if the color name names are CSS4 color names (ie 'red', 'blue', 'green') the to_rgb function is used to convert to rgb
        color_dictionary_ (dict, optional): dictionary of colors. Defaults to None.
            Color dictionary is used to convert stain names to rgb values.

            It is only used if the waves parameter represent stain names

            If None the global color dictionary is used
            
            color_dictionary_ should be in format
            {
                'wave1': [r,g,b],
                'wave2': [r,g,b],
                ...
            }
            ie 
            color_dictionary = {
                "Cy5": [1, 0, 0],        # Red
                "DAPI": [0, 0, 1],       # Blue
                "FITC": [0, 1, 0],       # Green
                "Texa": [1, 0.5, 0],     # Orange
                "AF594":[1,0.2,0],       # Red
                "Cy2": [0, 1, 0],
            }
        channel_pos (int, optional): channel position. Defaults to -1. Only needed if ims is a numpy array (not a list of numpy arrays)
        vmim (list of floats, optional): min values for each channel. Defaults to None.
        vmax (list of floats, optional): max values for each channel. Defaults to None.
        gamma (list of floats, optional): gamma values for each channel. Defaults to 1.

    Returns:
        np.array: rgb image

    """
    # if no color dictionary is provided use the global scope one    
    if color_dictionary_ is None:
        color_dictionary_ =  color_dictionary

    # check if ims is list, if not convert to list
    if not isinstance(ims, list):

        if channel_pos == -1:
            # guess channel index as smallest dim
            channel_pos = np.argmin(ims.shape)

        ims_temp = []
        for i in range(ims.shape[channel_pos]):
            slice_at_pos = np.take(ims, i, axis=channel_pos)
            ims_temp.append(slice_at_pos)
        ims = ims_temp

    # create empty rgb image 
    rgb_im = np.zeros((np.append(ims[0].shape, 3)), dtype=np.float32)

    color_list = []

    # if no color names are provided use the color dictionary in order 
    if color_names is None:
        for im_, color_values in zip(ims, color_dictionary_.values()):
            color_list.append(color_values)
    # else if color names are named colors ('red', 'blue', 'green', etc) convert the name to rgb
    elif is_named_color_list(color_names):
        for color_str in color_names:
            color_list.append(to_rgb(color_str))    
    # else if color names are stain names (DAPI, FITC, Cy5, etc) use the color dictionary to look up the color
    else:    
        # loop over images and maps and add to color image
        for wave, im_ in zip(color_names, ims ):
            color_values = np.array(color_dictionary_[wave])
            color_list.append(color_values)

    # loop over images and colors and add to color image
    for idx_i, (im_, color_values) in enumerate(zip(ims, color_list)): 
            temp = color.gray2rgb(im_) * color_values
            temp = temp/temp.max()

            if any([vmin, vmax]):
                norm = PowerNorm(gamma=gamma[idx_i], vmin=vmin[idx_i], vmax=vmax[idx_i])
                temp = norm(temp)

            rgb_im+=temp
    return rgb_im

def is_named_color_list(color_list):
    """
    Check if all elements in the list are valid Matplotlib named colors.
    
    Parameters:
        color_list (list of str): List of color names to check.
    
    Returns:
        bool: True if all elements are valid named colors, False otherwise.
    """
    return all(color in CSS4_COLORS for color in color_list)

def get_color(i):
    """
    Get color from a list of colors
    
    Args:
        i (int): index of color to get
    """
    if i == 0:
        return [1, 0, 0]
    elif i == 1:
        return [0, 0, 1]
    elif i == 2:
        return [0, 1, 0]
    elif i == 3:
        return [1, 1, 0]
    elif i == 4:
        return [1, 0, 1]
    elif i == 5:
        return [0, 1, 1]
    else:
        return [.5, 1, 1]

def create_linear_napari_color_map(name, color_values):
    """
    Create a napari color map as a linear interpolation of a color (ie a list of rgb values)
    
    Args:
        name (str): name of color map
        color_values (list): list of rgb values, [r,g,b] for example [1,0,0] for red

    Returns:
        napari format color map (dict): dictionary with keys 'colors', 'name', 'interpolation' that can be used in napari
    """
    map_end = np.array(color_values)
    map_end = np.append(map_end, 1)

    map_start = np.zeros_like(map_end)
    map_start[3]=1

    color = np.linspace(
        start=map_start,
        stop=map_end,
        num=10,
        endpoint=True
    )
    
    return {
        'colors': color,
        'name': name,
        'interpolation': 'linear'
    }
