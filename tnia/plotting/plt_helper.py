import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

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

def imshow_multi2d(ims, titles, rows, cols, width=10, height=4, colormap=None, vmin=None, vmax=None, gamma=None):
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

    for im,ax,title in zip(ims,np.ndarray.flatten(axes),titles):
        #print('types')
        #print(type(im), type(ax), type(title))
        #print()

        if gamma is not None:
            norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
            ax.imshow(im, colormap,norm=norm)
        else:
            ax.imshow(im, colormap,vmin=vmin, vmax=vmax)

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
'''
def draw_bimodal_hist(data, thresh, params, ax=None, title='bimodal hist'):
    if (ax==None):
        fig, ax = plt.subplots()
    else:
        fig=None
'''