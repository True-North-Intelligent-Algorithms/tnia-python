import numpy as np
from skimage.feature import multiscale_basic_features
from functools import partial

default_feature_params = {
        "sigma_min": 1,
        "sigma_max": 15,
        "num_sigma": 3,
        "intensity": True,
        "edges": True,
        "texture": True,
    }

def extract_features(image, feature_params=default_feature_params):
    """
    Extract features from an image

    Args:
        image (numpy.ndarray): image to extract features from
        feature_params (dict): dictionary of feature parameters

    Returns:
        numpy.ndarray: extracted features
    """
    features_func = partial(
        multiscale_basic_features,
        intensity=feature_params["intensity"],
        edges=feature_params["edges"],
        texture=feature_params["texture"],
        sigma_min=feature_params["sigma_min"],
        sigma_max=feature_params["sigma_max"],
        num_sigma = feature_params["num_sigma"],
        channel_axis=None,
    )
    # print(f"image shape {image.shape} feature params {feature_params}")

    if len(image.shape) == 2:
        features = features_func(image)
    else:
    
        for c in range(image.shape[-1]):
            
            features_temp = features_func(np.squeeze(image[..., c]))
            if c == 0:
                features = features_temp
            else:
                features = np.concatenate((features, features_temp), axis=2)
        
    return features

def extract_features_sequence(images, labels, features):
    """
    Extract features from a sequence of images and labels

    We check to see if the labels have any data in them, and if they do, we extract features from the corresponding image.

    We then concatenate the features and labels into a single label and feature vector respectively.

    In the future we could potentially also check the size of the region that is labeled then only compute features for that region... though
    we don't yet do that.. 

    Args:
        images (numpy.ndarray): sequence of images
        labels (numpy.ndarray): sequence of labels
        features (numpy.ndarray): sequence of features

    Returns:
        tuple: label_vector, feature_vector

    """

    num_features = features.shape[-1]

    label_vector = np.empty((0,))
    feature_vector = np.empty((0, num_features))

    # loop through all images
    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]

        # if the label has any data in it, extract features
        if label.sum() > 0:
            print(f"image {i} has shape {image.shape}")
            print(f"labels {i} has sum {label.sum()}")

            # use sum of first feature to check if features have been extracted 
            if True:#features[i,:,:,0].sum() == 0:
                print(f"extracting features for image {i}")
                features[i,:,:,:] = extract_features(image, default_feature_params)
            else:
                print(f"features {i} already exist")

            label_vector_temp = label[label>0]

            temp = features[i, :, :]
            feature_vector_temp = temp[label>0, :]

            label_vector = np.concatenate((label_vector, label_vector_temp), axis=0)
            feature_vector = np.concatenate((feature_vector, feature_vector_temp), axis=0)
    
    return label_vector, feature_vector