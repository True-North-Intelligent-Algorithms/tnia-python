import matplotlib.pyplot as plt
from matplotlib import gridspec
import pyclesperanto_prototype as cle
import numpy as np

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


def show_xyz_max(image_to_show, labels=False):
    """
    This function generates three projections in X-, Y- and Z-direction and shows them.
    """
    fig=plt.figure(figsize=(10,10))
    
    xdim = image_to_show.shape[2]
    ydim = image_to_show.shape[1]
    zdim = image_to_show.shape[0]

    spec=gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[xdim,zdim], width_ratios=[ydim,zdim])

    ax0=fig.add_subplot(spec[0])
    ax1=fig.add_subplot(spec[1])
    ax2=fig.add_subplot(spec[2])

    projection_x = np.max(image_to_show,2)
    projection_y = np.rot90(np.max(image_to_show,1))
    projection_z = np.max(image_to_show,0)

    ax0.imshow(projection_z)
    ax1.imshow(projection_y)
    ax2.imshow(projection_x)