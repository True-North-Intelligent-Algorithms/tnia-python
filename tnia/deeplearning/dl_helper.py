from glob import glob
from tqdm import tqdm
from tifffile import imread
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt

""" Note:
Original source for much of this code is here https://github.com/stardist/stardist/tree/master/examples
"""

def get_training_data(labeled_dir):
    """ loads images and masks that can be used for training data for an air

    Original source for much of this code is here https://github.com/stardist/stardist/tree/master/examples

    The data must exist under the directory 'labeled_dir' and be organized as follows to
        - images in a sub-directory called 'images'
        - labeled masks in a sub-directory called 'masks' with each mask having the same name as the corresponding image.

    Args:
        labeled_dir (_type_): loacation of the image and mask data

    Returns:
        likst(numpy array), list(numpy array): Two lists of numpy arrays containing the images and masks 
    """

    X_ = sorted(glob(labeled_dir+'images/*.tif'))
    Y_ = sorted(glob(labeled_dir+'masks/*.tif'))

    X = list(map(imread,X_))
    Y = list(map(imread,Y_))

    # relabel the data just to make sure the objects are indexed
    labeled=[]
    for y in Y:
        labeled.append(label(y))
    
    Y=labeled

    return X, Y

def split_training_data(X, Y):
    """ splits data into a training and validation set

    Original source for much of this code is here https://github.com/stardist/stardist/tree/master/examples

    Args:
        X: set of images
        Y: set of ground truth masks

    Returns:
        X_val, Y_val, X_trn, Y_trn : images and masks split into validation and training partitions
    """

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    return X_val, Y_val, X_trn, Y_trn

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):
    import matplotlib
    import colorsys
    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    """ plots an image and label

    Args:
        img (numpy array): the image 
        lbl (numpy array): the label 
        img_title (str, optional): _description_. Defaults to "image".
        lbl_title (str, optional): _description_. Defaults to "label".

    Returns:
        figure: the figure
    """
    
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, random_label_cmap())
    al.set_title(lbl_title)
    plt.tight_layout()

    return fig

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y