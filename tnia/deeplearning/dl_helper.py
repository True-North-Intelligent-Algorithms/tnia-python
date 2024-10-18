from glob import glob
from tqdm import tqdm
from tifffile import imread, imsave
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt
try:
    from csbdeep.utils import normalize
except ImportError:
    "csbdeep normalize did not import.  We will continue without it."
from skimage.transform import resize
import json
import skimage.io as io
import os
import math
try:
    import raster_geometry as rg
    from tnia.simulation.phantoms import add_small_to_large, add_small_to_large_2d
except ImportError:
    print("raster_geometry not imported.  This is only needed for the ellipsoid rendering in apply_stardist")
import random
from tqdm import tqdm

""" Note:
Original source for some of this code is here https://github.com/stardist/stardist/tree/master/examples
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

def get_patch_directory(num_inputs, num_truths, parent_dir):
    """ gets the directory to put patches from an image and its corresponding ground truth

    Args:
        img (numpy array): the image
        truth (numpy array): the ground truth
        parent_dir (str): the location of the directory to save the patches
    """
    
    input_paths = []
    truth_paths = []

    for i in range(num_inputs):
        input_path = os.path.join(parent_dir, "input" + str(i))
        input_paths.append(input_path)

    for i in range(num_truths):
        truth_path = os.path.join(parent_dir, "ground truth" + str(i))
        truth_paths.append(truth_path)

    return input_paths, truth_paths

def get_label_directory(num_inputs, num_truths, parent_dir):
    """ gets the directory to put labeks from an image and its corresponding ground truth

    Note1: Labels are subtly different than patches.  Labels can be the entire image, and labels in the same directory can be different sizes.
    Patches are always the same size and are always cropped from the image.

    Note2:  Right now naming scheme for labels and patches are the same so this function is not really necessary.  However, it is included for completeness.
    
    Args:
        img (numpy array): the image
        truth (numpy array): the ground truth
        parent_dir (str): the location of the directory to save the patches
    """
    return get_patch_directory(num_inputs, num_truths, parent_dir)

from pathlib import Path

def get_label_paths(num_inputs, num_truths, parent_dir):
    """ gets the paths to put labeks from an image and its corresponding ground truth

    Note1: Labels are subtly different than patches.  Labels can be the entire image, and labels in the same directory can be different sizes.
    Patches are always the same size and are always cropped from the image.

    Note2:  Right now naming scheme for labels and patches are the same so this function is not really necessary.  However, it is included for completeness.
    
    Args:
        img (numpy array): the image
        truth (numpy array): the ground truth
        parent_dir (str): the location of the directory to save the patches
    """
    image_dirs, label_dirs = get_patch_directory(num_inputs, num_truths, parent_dir)

    image_paths = []
    for p in image_dirs:
        as_path = Path(p)
        image_paths.append(as_path)

    label_paths=[]
    for p in label_dirs:
        as_path = Path(p)
        label_paths.append(as_path)
    
    return image_paths, label_paths

def make_patch_directory(num_inputs, num_truths, parent_dir):
    """ makes a directory to put patches from an image and its corresponding ground truth

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

def make_label_directory(num_inputs, num_truths, parent_dir):
    """ makes a directory to put labeks from an image and its corresponding ground truth

    Note1: Labels are subtly different than patches.  Labels can be the entire image, and labels in the same directory can be different sizes.
    Patches are always the same size and are always cropped from the image.

    Note2:  Right now naming scheme for labels and patches are the same so this function is not really necessary.  However, it is included for completeness.
    
    Args:
        img (numpy array): the image
        truth (numpy array): the ground truth
        parent_dir (str): the location of the directory to save the patches
        crop_size (int): the size of the crop with respect to the image
        sub_sample (int): the subsampling rate to apply to the patch in lateral xy direction
    """
    return make_patch_directory(num_inputs, num_truths, parent_dir)


def make_patch_source_divide2(source, axes, label_dir):
    """ 
    This helper was motivated by a fairly common scenario when quickly evaluating DL on an image type. 
    It is common that a single large will be provided, so this utility divides the image in 2 such that
    some of the data remains 'unseen'.  The intention is that patches will be generated form one half 
    (the train half) and testing can be done on the other half. 
    
    Args:
        source (numpy array): the image
        axes (str): the axes to divide the image in 2
        label_dir (str): the location of the directory to save the source
    
    """
    # divide image in 2 in each dimensions
    if axes == 'YX':
        # Divide image in 2 dimensions
        train = source[:, :source.shape[1]//2]
        test = source[:, source.shape[1]//2:]
    elif axes == 'ZYX':
        # Divide volume in 3 dimensions
        train = source[:, :, :source.shape[2]//2]
        test = source[:, :, source.shape[2]//2:]
    elif axes == 'YXC':
        train = source[:, :source.shape[1]//2, :]
        test = source[:, source.shape[1]//2:, :]
    else:
        raise ValueError("Invalid axes. Only 'YX', 'ZYX' and 'YXC are supported.")
    
    imsave(os.path.join(label_dir,'train.tif'), train)
    imsave(os.path.join(label_dir, 'test.tif'), test)
    
    return train, test


def make_random_patch(img, truth, patch_size, axes, ind=None, sub_sample_xy=1):
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
    if axes == 'YX':

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
            truth_ind = ind

        # crop the image and truth
        img_crop = img[ind]
        truth_crop = truth[ind]
    elif axes == 'YXC':
        if (sub_sample_xy>1):
            img = img[::sub_sample_xy, ::sub_sample_xy, :]
            truth = truth[::sub_sample_xy, ::sub_sample_xy]

        # get the size of the image
        img_size = img.shape
        
        # get the random location of the patch
        if ind is None:
            y = np.random.randint(0, img_size[0] - patch_size[0]+1)
            x = np.random.randint(0, img_size[1] - patch_size[1]+1)
            ind = np.s_[y:y+patch_size[0], x:x+patch_size[1], :]
            truth_ind = np.s_[y:y+patch_size[0], x:x+patch_size[1]]

        # crop the image and truth
        img_crop = img[ind]
        truth_crop = truth[truth_ind]
    elif axes == 'ZYX':

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
            truth_ind = ind
        if patch_size[0]>img.shape[0]:
            # pad the image in z
            pad = patch_size[0]-img.shape[0]
            img = np.pad(img, ((pad//2, pad-pad//2), (0,0), (0,0)), 'constant', constant_values=np.percentile(img, 0.15))
            truth = np.pad(truth, ((pad//2, pad-pad//2), (0,0), (0,0)), 'constant', constant_values=0)
        
        # crop the image and truth
        img_crop = img[ind]
        truth_crop = truth[ind]

    return img_crop, truth_crop, ind, truth_ind

def collect_training_data(data_path, sub_sample=1, downsample=False,pmin=3, pmax=99.8, normalize_input=True, normalize_truth=False, training_multiple=1, patch_size=None, add_trivial_channel=True, relabel=False):
    """
    Collect training data for image processing models.

    Parameters:
    data_path (str): Path to the directory containing input and ground truth data.
    sub_sample (int, optional): Wether to subsample the images in the collection. Defaults to 1.  
        for example if sumbsample is 2, every other image will be used. 
    downsample (bool, optional): If True, downsample the images by a factor of 2. Defaults to False.
    pmin (float, optional): Minimum percentile for normalization. Defaults to 3.
    pmax (float, optional): Maximum percentile for normalization. Defaults to 99.8.
    normalize_input (bool, optional): If True, normalize the input images. Defaults to True.
    normalize_truth (bool, optional): If True, normalize the ground truth images. Defaults to False.
    training_multiple (int, optional): Ensures that the dimensions of the training data are multiples of this value. Defaults to 1.
        This is needed because some networks require that the dimensions of the training data are multiples of a certain value 
        (for example 16 or 32 works well because we can downsample the image by a factor of 2 multiple times).
    patch_size (tuple, optional): If provided, the function will extract random patches of this size from the images. Defaults to None.
    add_trivial_channel (bool, optional): If True, add a trivial channel dimension to the input images. Defaults to True.
        This is needed because some networks expect the input images to have a channel dimension.

    Returns:
    tuple: A tuple containing two lists:
        X (list): List of input images.
        Y (list): List of ground truth images.

    Example:
    >>> X, Y = collect_training_data('/path/to/data', sub_sample=2, downsample=True)
    """ 

    X=[]
    Y=[]

    # todo handle different number of inputs and truths
    i=0
    input_path = os.path.join(data_path, "input" + str(i))
    truth_path = os.path.join(data_path, "ground truth" + str(i))

    input_files = [f for f in os.listdir(input_path)[0::sub_sample] if f.endswith('.tif')]
    truth_files = [f for f in os.listdir(truth_path)[0::sub_sample] if f.endswith('.tif')]

    for i in range(len(input_files)):
        # Load the corrupted image and ground truth image
        input_img = io.imread(os.path.join(input_path, input_files[i]), plugin='tifffile')
        ground_truth_img = io.imread(os.path.join(truth_path, truth_files[i]), plugin='tifffile')

        if patch_size is not None:
            input_img, ground_truth_img, ind = make_random_patch(input_img, ground_truth_img, patch_size, sub_sample_xy=1)
        
        if downsample:
            input_img = input_img[:,::2,::2]
            ground_truth_img = ground_truth_img[:,::2,::2]

        # Add a trivial channel dimension using np.newaxis (CARE/Stardist seem to expect his)  
        if add_trivial_channel:
            input_img = input_img[..., np.newaxis]

        if normalize_input:
            # Normalize the pixel values to [0, 1]
            input_img =  quantile_normalization(input_img, pmin/100., pmax/100.)

        if (normalize_truth):
            ground_truth_img = quantile_normalization(ground_truth_img, pmin/100., pmax/100.)

        if (training_multiple>1):
            new_z = training_multiple*(input_img.shape[0]//training_multiple)
            new_y = training_multiple*(input_img.shape[1]//training_multiple)
            new_x = training_multiple*(input_img.shape[2]//training_multiple)

            input_img = input_img[:new_z, :new_y, :new_x]
            ground_truth_img = ground_truth_img[:new_z, :new_y, :new_x]
        
        # Append the preprocessed images to the training set
        X.append(input_img)

        if relabel:
            ground_truth_img = label(ground_truth_img)
            
        Y.append(ground_truth_img)
    
    return X,Y

def shuffle(X, Y):
    """
    Shuffles two lists (X and Y) together. This is useful when the two lists are related to each other 
    (e.g., X is a list of inputs and Y is a list of corresponding outputs).

    Args:
        X (list): The first list to be shuffled.
        Y (list): The second list to be shuffled. Must be the same length as X.

    Returns:
        tuple: A tuple containing two lists (X, Y), shuffled in unison.
    """

    combined = list(zip(X, Y))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip the shuffled list back into X and Y
    X, Y = zip(*combined)
    X = list(X)
    Y = list(Y)

    return X, Y

def divide_training_data(X, Y, val_size=3, shuffle_data=True, to_numpy=True):
    """
    Divides a dataset into training and validation sets, with the option to shuffle the data and convert it to numpy arrays.

    Args:
        X (list): The list of inputs.
        Y (list): The list of corresponding outputs. Must be the same length as X.
        val_size (int, optional): The size of the validation set. Default is 3.
        shuffle_data (bool, optional): Whether to shuffle the data before dividing it. Default is True.
        to_numpy (bool, optional): Whether to convert the lists to numpy arrays. Default is True.

    Returns:
        tuple: A tuple containing four lists (or numpy arrays if to_numpy is True): 
               the training inputs, the training outputs, the validation inputs, and the validation outputs.
    """
    if shuffle_data:
        X, Y = shuffle(X, Y)

    if to_numpy:
        X = np.array(X)
        Y = np.array(Y)

        X = np.nan_to_num(X, nan=0)

    # divide the training set into training and validation sets
    X_train=X[val_size:]
    Y_train=Y[val_size:]
    X_val=X[:val_size]
    Y_val=Y[:val_size]

    return X_train, Y_train, X_val, Y_val

def apply_stardist(img, model, prob_thresh=0.5, nms_thresh=0.3, down_sample=1, pmin=1, pmax=99.8, render_mode="default"):
    """ applies stardist to an image with an option to downsample

    Args:
        img (numpy array): the image
        model (stardist model): the stardist model
        prob_thresh (float, optional): probability threshold. Defaults to 0.5.
        nms_thresh (float, optional): nms threshold. Defaults to 0.3.
        down_sample (int, optional): downsampling factor. Defaults to 1.
        pmin (float, optional): min percentile for normalization. Defaults to 1.
        pmax (float, optional): max percentile for normalization. Defaults to 99.8.
        render_mode (str, optional): render mode. If 'ellipsoid' ellipsoids will be rendered.  Defaults to default.  
            If default model type will be checked and mode will be set to ellipsoid of model is Octohedron. 

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
    
    # apply the model
    labels, details = model.predict_instances(img, prob_thresh=prob_thresh, nms_thresh=nms_thresh)

    if render_mode=="ellipsoid":
        labels = octo_to_ellipsoid_labels(details['points'], details['dist'], shape=img.shape)
    
    if (down_sample>1):
        # upsample the labels
        labels = resize(labels, old_size, order=0, preserve_range=True, anti_aliasing=False) 

    return labels, details

def generate_patch_names(image_path, mask_path, data_name):
    
    index=0
    image_name=image_path+'/'+data_name+'_'+str(index)+'.tif'
    mask_name=mask_path+'/'+data_name+'_'+str(index)+'.tif'

    while (os.path.exists(image_name)==True):
        index=index+1
        image_name=image_path+'/'+data_name+'_'+str(index)+'.tif'
        mask_name=mask_path+'/'+data_name+'_'+str(index)+'.tif'

    return image_name, mask_name

def generate_label_names(image_path, mask_path, data_name):
    return generate_patch_names(image_path, mask_path, data_name)

def generate_next_patch_name(image_path, name):
    
    index=0
    image_name=image_path+'/'+name+'_'+str(index)+'.tif'

    while (os.path.exists(image_name)==True):
        index=index+1
        image_name=image_path+'/'+name+'_'+str(index)+'.tif'

    base_name = os.path.basename(image_name)
    base_name = os.path.splitext(base_name)[0]
    
    return base_name

def generate_next_label_name(image_path, name):
    return generate_next_patch_name(image_path, name)

def compute_centroid(vertices):
    """
    This function calculates the centroid (geometric center) of a set of vertices in 2D or 3D space.

    Parameters:
    vertices (list): A list of lists or tuples, where each inner list or tuple contains two or three numbers 
                     representing the x, y, (and z) coordinates of a vertex.

    Returns:
    list: A list containing the x, y, (and z) coordinates of the centroid.
    """
    
    sum_x = sum_y = sum_z = 0
    for vertex in vertices:
        sum_x += vertex[0]
        sum_y += vertex[1]
        if len(vertex) == 3:
            sum_z += vertex[2]
    num_vertices = len(vertices)
    
    if len(vertices[0]) == 3:
        return [sum_x / num_vertices, sum_y / num_vertices, sum_z / num_vertices]
    else:
        return [sum_x / num_vertices, sum_y / num_vertices]
    
def octo_to_ellipsoid_labels(points, distances, shape, up_sample=1, scale=[1,1,1]):
    """ Converts octohedron labels to ellipsoid labels

    Note:  Only considers the first 6 Rays which are assumed to be parallel to the x, y, and z axes
    in the order: -x, -y, x, y, -z, z

    Args:
        points (list): a list of centers of the octohedrons
        distances (list): a list of distances from the center of the octohedron to the vertices
        shape (int array): size of the image
        up_sample (int, optional): Whether to upsample after creating the labels (this is useful if the octohedrons were derived in a downsampled space). Defaults to 1.
        scale (list, optional): Used for similar purpose as up_sample, but in this case we scale the rays instead of up_sampling the labels at the end. Defaults to [1,1,1].

        Todo:  Could probably eliminate the need for both up_sample and scale and reconcile to one consistent approach

    Returns:
        numpy array (int32): A new label image with ellipsoids instead of octohedrons

    Limitations:
        - Does not consider that the ellipsoid may not be symmetrical.
    """
    labels = np.zeros(shape, dtype=np.float32)
    
    label_num=1
    for point in points:

        dx1 = scale[2]*distances[label_num-1, 0]
        dy1 = scale[1]*distances[label_num-1, 1]
        dx2 = scale[2]*distances[label_num-1, 2]
        dy2 = scale[1]*distances[label_num-1, 3]
        dz1 = scale[0]*distances[label_num-1, 4]
        dz2 = scale[0]*distances[label_num-1, 5]

        dx = dx1+dx2
        dy = dy1+dy2
        dz = dz1+dz2

        # compute the centroid of the octohedron
        centroid = [0,0,0]
        centroid[2]=(point[2]-dx2+point[2]+dx1)/2
        centroid[1]=(point[1]-dy2+point[1]+dy1)/2
        centroid[0]=(point[0]-dz2+point[0]+dz1)/2

        # size of bounding box surrounding the ellipsoid
        size = [math.ceil(dz),math.ceil(dy),math.ceil(dx)]

        # when we add the ellipsoid to the label image, we need an integer location for the centroid
        ceil_centroid = [math.ceil(centroid[0]), math.ceil(centroid[1]), math.ceil(centroid[2])]
        
        # compute the percentage offset of the centroid from the integer centroid
        px = centroid[2]-ceil_centroid[2]
        py = centroid[1]-ceil_centroid[1]
        pz = centroid[0]-ceil_centroid[0]

        # the centroid that is passed to the ellipsoid function should be in the range 0 to 1
        # in this case we adjust it very slightly to compensate for the difference between the centroid and the integer centroid
        px = 0.5 + px/size[2]
        py = 0.5 + py/size[1]
        pz = 0.5 + pz/size[0]

        # draw ellipsoid in bounding box 'size', with radius [dz/2, dy/2, dx/2], add percentage offset [pz, py, px]
        ellipsoid_ = rg.ellipsoid(size, [dz/2, dy/2, dx/2], [pz, py, px]).astype(np.float32)
        
        # add the ellipsoid to the label image 
        add_small_to_large(labels, label_num*ellipsoid_, ceil_centroid[2], ceil_centroid[1], ceil_centroid[0], mode = 'replace_non_zero')
        
        label_num += 1

    if (up_sample>1):

        new_size = [labels.shape[0], labels.shape[1]*up_sample, labels.shape[2]*up_sample]
        # upsample the labels
        labels = resize(labels, new_size, order=0, preserve_range=True, anti_aliasing=False) 
    

    return labels.astype(np.int32)

def ray4_to_ellipsoid2d_labels(coords, shape):
    """ Converts ray4 labels to 2d ellipsoid labels

    Args:
        coords (list): The coordinates of the rays
        shape (int array): size of the image

    Returns:
        numpy array (int32): A new label image with ellipsoids instead of octohedrons
    
    Limitations:
        - Does not consider that the ellipsoid may not be symmetrical.
    """
    labels = np.zeros(shape, dtype=np.float32)
    
    label_num=1
    for coord in coords:
        # collect the vertices of the polyhedron
        v = []
        v.append([coord[0][0], coord[1][0]])
        v.append([coord[0][1], coord[1][1]])
        v.append([coord[0][2], coord[1][2]])
        v.append([coord[0][3], coord[1][3]])

        # compute the centroid
        centroid = compute_centroid(v)

        # compute the size of the bounding box
        dx = max(coord[0])-min(coord[0])
        dy = max(coord[1])-min(coord[1])
        size = [math.ceil(dy), math.ceil(dx)]

        # draw ellipsoid in bounding box 'size', with radius [dz/2, dy/2, dx/2], add percentage offset [pz, py, px]
        ellipsoid_ = rg.ellipse(size, [dy/2, dx/2]).astype(np.float32)
        if centroid[0]<labels.shape[0] and centroid[1]<labels.shape[1]:
            add_small_to_large_2d(labels, label_num*ellipsoid_, int(centroid[1]), int(centroid[0]), 0, mode = 'replace_non_zero')
        
        label_num += 1

    return labels.astype(np.int32)


def stardist_2d_slicewise(im, model, nmin=2, nmax=98, prob_thresh=0.3, nms_thresh=0.3, perform_normalization=True, use_tqdm=False):
    label_list=[]

    range_func = tqdm(range(im.shape[0])) if use_tqdm else range(im.shape[0])
    for i in range_func:
        if perform_normalization:    
            im_n = normalize(im[i],nmin,nmax)
        else:
            im_n = im[i]
        
        labels, details = model.predict_instances(im_n, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        label_list.append(labels)

    return np.stack(label_list,0)

def normalize_(img, low, high, eps=1.e-20, clip=True):
    # we have to add a small eps to handle the case where both quantiles are equal
    # to avoid dividing by zero
    scaled = (img - low) / (high - low + eps)

    if clip:
        scaled = np.clip(scaled, 0, 1)

    return scaled


def quantile_normalization(img, quantile_low=0.01, quantile_high=0.998, eps=1.e-20, clip=True, channels = False):
    """
    Copying this from PolBias GPU course....  it is an easy piece of code that is also in stardist.  

    But... sometimes Stardist isn't installed, sometimes it is, sometimes PyTorch is there sometimes it isn't.  So I'm copying it here.

    First scales the data so that values below quantile_low are smaller
    than 0 and values larger than quantile_high are larger than one.
    Then optionally clips to (0, 1) range.
    """

    if channels == False:
        qlow = np.quantile(img, quantile_low)
        qhigh = np.quantile(img, quantile_high)

        scaled = normalize_(img, low=qlow, high=qhigh, eps=eps, clip=clip)
        return scaled
    else:
        num_channels = img.shape[-1]
        scaled = np.zeros(img.shape, dtype=np.float32)
        for i in range(num_channels):
            qlow = np.quantile(img[...,i], quantile_low)
            qhigh = np.quantile(img[...,i], quantile_high)
            scaled[...,i] = normalize_(img[...,i], low=qlow, high=qhigh, eps=eps, clip=clip)
        return scaled
