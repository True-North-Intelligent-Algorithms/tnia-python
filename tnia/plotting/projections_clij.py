import pyclesperanto_prototype as cle
from matplotlib import gridspec
import matplotlib.pyplot as plt

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

