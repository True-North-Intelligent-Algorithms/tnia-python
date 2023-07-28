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
    im = ai.imshow(img, cmap='gray')
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

import os
import json

def make_patch_directory(num_inputs, num_truths, parent_dir, sub_sample=1):
    """ makes a directory of patches from an image and its corresponding ground truth

    Note: the input crop_size is with respect to the image, the final size of the patch will be crop_size/sub_sample

    Args:
        img (numpy array): the image
        truth (numpy array): the ground truth
        parent_dir (str): the location of the directory to save the patches
        crop_size (int): the size of the crop with respect to the image
        sub_sample (int): the subsampling rate to apply to the patch in lateral xy direction
    """

    input_paths = []
    truth_paths = []

    for i in range(num_inputs):
        input_path = os.path.join(parent_dir, "input" + str(i))
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        input_paths.append(input_path)

    for i in range(num_truths):
        truth_path = os.path.join(parent_dir, "ground truth" + str(i))
        if not os.path.exists(truth_path):
            os.makedirs(truth_path)
        truth_paths.append(truth_path)

    json_ = {
        "num_inputs": num_inputs,
        "num_truths": num_truths
    }

    json_file = os.path.join(parent_dir, "info.json")

    with open(json_file, 'w') as outfile:
        json.dump(json_, outfile)

    return input_paths, truth_paths


def make_random_patch(img, truth, patch_size, ind=None, sub_sample_xy=1):
    """ makes a random patch from an image and its corresponding ground truth

    Args:
        img (numpy array): the image
        truth (numpy array): the ground truth
        patch_size (int): the size of the patch
        ind (index from np.s_): (optional) the index of the patch to crop defaults to None, if None a random index will be generated
        sub_sample_xy (int): the subsampling rate to apply to the image and truth in xy direction before cropping the patch

    Returns:
        numpy array, numpy array: the cropped image and ground truth of size crop_size/sub_sample
    
    """


    if (len(img.shape)==2):

        if (sub_sample_xy>1):
            img = img[::sub_sample_xy, ::sub_sample_xy]
            truth = truth[::sub_sample_xy, ::sub_sample_xy]

        # get the size of the image
        img_size = img.shape
        
        # get the random location of the patch
        if ind is None:
            y = np.random.randint(0, img_size[0] - patch_size[0]+1)
            x = np.random.randint(0, img_size[1] - patch_size[1]+1)
            ind = np.s_[y:y+patch_size[0], x:x+patch_size[1]]

        # crop the image and truth
        img_crop = img[ind]
        truth_crop = truth[ind]

    elif len(img.shape)==3:

        if (sub_sample_xy>1):
            img = img[:,::sub_sample_xy, ::sub_sample_xy]
            truth = truth[:,::sub_sample_xy, ::sub_sample_xy]

        # get the size of the image
        img_size = img.shape
 
        if ind is None:
            # get the random location of the patch
            if patch_size[0]>img.shape[0]:
                z=0
            else:
                z = np.random.randint(0, img_size[0] - patch_size[0]+1)
            y = np.random.randint(0, img_size[1] - patch_size[1]+1)
            x = np.random.randint(0, img_size[2] - patch_size[2]+1)
            ind = np.s_[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]

        if patch_size[0]>img.shape[0]:
            # pad the image in z
            pad = patch_size[0]-img.shape[0]
            img = np.pad(img, ((pad//2, pad-pad//2), (0,0), (0,0)), 'constant', constant_values=np.percentile(img, 0.15))
            truth = np.pad(truth, ((pad//2, pad-pad//2), (0,0), (0,0)), 'constant', constant_values=0)
        
        # crop the image and truth
        img_crop = img[ind]
        truth_crop = truth[ind]

    return img_crop, truth_crop, ind

from csbdeep.utils import normalize
from skimage.transform import resize
import json
import skimage.io as io

def collect_training_data(data_path, sub_sample=1, downsample=False):
    # open info.json
    with open(os.path.join(data_path, "info.json")) as json_file:
        info = json.load(json_file)

    num_inputs = info["num_inputs"]
    num_truths = info["num_truths"]

    X=[]
    Y=[]

    # todo handle different number of inputs and truths
    i=0
    input_path = os.path.join(data_path, "input" + str(i))
    truth_path = os.path.join(data_path, "ground truth" + str(i))

    input_files = os.listdir(input_path)[0::sub_sample]
    truth_files = os.listdir(truth_path)[0::sub_sample]

    for i in range(len(input_files)):
        # Load the corrupted image and ground truth image
        input_img = io.imread(os.path.join(input_path, input_files[i]), plugin='tifffile')
        ground_truth_img = io.imread(os.path.join(truth_path, truth_files[i]), plugin='tifffile')

        if downsample:
            input_img = input_img[:,::2,::2]
            ground_truth_img = ground_truth_img[:,::2,::2]

        # Add a trivial channel dimension using np.newaxis (CARE/Stardist seem to expect his) 
        input_img = input_img[..., np.newaxis]

        min_ = input_img.min()
        max_ = input_img.max()

        #print('min/max', min_, max_)

        # Normalize the pixel values to [0, 1]
        input_img = (input_img.astype('float32')-input_img.min()) / (input_img.max() - input_img.min())
        
        # Append the preprocessed images to the training set
        X.append(input_img)
        Y.append(ground_truth_img)
    
    return X,Y



        

def apply_stardist(img, model, prob_thresh=0.5, nms_thresh=0.3, down_sample=1, pmin=1, pmax=99.8):
    """ applies stardist to an image

    Args:
        img (numpy array): the image
        model (stardist model): the stardist model
        prob_thresh (float, optional): probability threshold. Defaults to 0.5.
        nms_thresh (float, optional): nms threshold. Defaults to 0.3.
        down_sample (int, optional): downsampling factor. Defaults to 1.

    Returns:
        numpy array: the predicted labels
    """

    if (down_sample>1):
        old_size = img.shape
        if (len(old_size)==2):
            new_size=[old_size[0]//down_sample, old_size[1]//down_sample]
        elif (len(old_size)==3):
            new_size = [old_size[0], old_size[1]//down_sample, old_size[2]//down_sample]
        
        # downsample the image
        img = resize(img, new_size, anti_aliasing=True)
    
    # normalize the image
    img = normalize(img, pmin, pmax)
    #img = img.astype('float32')
    #img = (img-img.min()) / (img.max() - img.min())
    
    # apply the model
    labels, details = model.predict_instances(img, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    
    if (down_sample>1):
        # upsample the labels
        labels = resize(labels, old_size, order=0, preserve_range=True, anti_aliasing=False) 
        
    return labels, details