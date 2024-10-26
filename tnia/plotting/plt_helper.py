import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
import colorsys

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):
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

def plot_color_space(im, rgb_func, width=10, height=10, ax=None, title='color space', titles=None):
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
'''
def draw_bimodal_hist(data, thresh, params, ax=None, title='bimodal hist'):
    if (ax==None):
        fig, ax = plt.subplots()
    else:
        fig=None
'''


def hsv_to_rgb(arr):
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