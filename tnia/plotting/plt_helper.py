import matplotlib.pyplot as plt
import numpy as np

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):
    import matplotlib
    import colorsys
    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

def imshow2d(im, width=8, height=6, colormap=None):
    """ a little helper to show images of differnt sizes using pyplot.  Just makes calls to imshow more readable. 

    Args:
        im ([type]): input image
        width (int, optional): Width of plot. Defaults to 8.
        height (int, optional): Height of plot. Defaults to 6.

    Returns:
        [type]: returns the figure
    """
    fig, ax = plt.subplots(figsize=(width,height))
    ax.imshow(im, colormap)
    return fig

def imshow_multi2d(ims, titles, rows, cols, width=10, height=4, colormap=None):
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
        ax.imshow(im, colormap)
        ax.set_title(title)

    return fig 

#def plot_line(line, tittle)