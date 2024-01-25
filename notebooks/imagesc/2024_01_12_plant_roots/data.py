"""Note: The dataset and preprocessing code are largely taken from the StarDist github repository available at https://github.com/stardist"""

import numpy as np
import torch

from pathlib import Path
from scipy.ndimage import binary_fill_holes
from tifffile import imread
from torchvision import transforms
from tqdm import tqdm

SRC_DIR = Path("./data/dsb2018")

def get_dsb2018_files(subset, rootdir=SRC_DIR):
    assert subset in ["train", "validation", "test"]
    src_dir = rootdir / subset
    
    assert src_dir.exists(), f"root directory with images and masks {src_dir} does not exist"
    
    X = sorted(src_dir.rglob('**/images/*.tif'))
    Y = sorted(src_dir.rglob('**/masks/*.tif'))
    assert len(X) > 0, f"error finding the right structure in {src_dir}\n{list(src_dir.glob('*'))}"
    assert len(X) == len(Y), print(f"X has length {len(X)} and Y has length {len(Y)}")
    assert all(x.name==y.name for x,y in zip(X,Y))

    return X, Y


def get_dsb2018_train_files():
    return get_dsb2018_files(subset="train")


def get_dsb2018_validation_files():
    return get_dsb2018_files(subset="validation")


def get_dsb2018_test_files():
    return get_dsb2018_files(subset="test")


def fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def normalize(img, low, high, eps=1.e-20, clip=True):
    # we have to add a small eps to handle the case where both quantiles are equal
    # to avoid dividing by zero
    scaled = (img - low) / (high - low + eps)

    if clip:
        scaled = np.clip(scaled, 0, 1)

    return scaled


def quantile_normalization(img, quantile_low=0.01, quantile_high=0.998, eps=1.e-20, clip=True):
    """
    First scales the data so that values below quantile_low are smaller
    than 0 and values larger than quantile_high are larger than one.
    Then optionally clips to (0, 1) range.
    """

    qlow = np.quantile(img, quantile_low)
    qhigh = np.quantile(img, quantile_high)

    scaled = normalize(img, low=qlow, high=qhigh, eps=eps, clip=clip)
    return scaled, qlow, qhigh


class DSBData():
    def __init__(self, image_files, label_files, target_shape=(256, 256)):
        """
        Parameters
        ----------
        image_files: list of pathlib.Path objects pointing to the *.tif images
        label_files: list of pathlib.Path objects pointing to the *.tif segmentation masks
        target_shape: tuple of length 2 specifying the sample resolutions of files that
                      will be kept. All other files will NOT be used.
        """
        assert len(image_files) == len(label_files)
        assert all(x.name==y.name for x,y in zip(image_files, label_files))

        self.images = []
        self.labels = []

        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        for idx in tqdm(range(len(image_files))):
            # we use the same data reading approach as in the previous notebook
            image = imread(image_files[idx])
            label = imread(label_files[idx])

            if image.shape != target_shape:
                continue

            # do the normalizations
            image = quantile_normalization(
                image,
                quantile_low=0.01,
                quantile_high=0.998,
                clip=True)[0].astype(np.float32)

            # NOTE: we convert the label to dtype float32 and not uint8 because
            # the tensor transformation does a normalization if the input is of
            # dtype uint8, destroying the 0/1 labelling which we want to avoid.
            label = fill_label_holes(label)
            label_binary = np.zeros_like(label).astype(np.float32)
            label_binary[label != 0] = 1.

            # convert to torch tensor: adds an artificial color channel in the front
            # and scales inputs to have same size as samples tend to differ in image
            # resolutions
            image = tensor_transform(image)
            label = tensor_transform(label_binary)

            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images)
        self.labels = torch.stack(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)
