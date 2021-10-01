import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float

def open_and_subsample(im_name, to_float):
    image = imread(im_name)
    image=image[:,:,1]
    image=image[0:-1:4,0:-1:4]

    if to_float:
        image = img_as_float(image)

    return image 

def plot_result(image, background):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5),
                                    sharex=True,
                                    sharey=True)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(background, cmap='gray')
    ax[1].set_title('Background')
    ax[1].axis('off')

    ax[2].imshow(image - background, cmap='gray')
    ax[2].set_title('Result')
    ax[2].axis('off')

    fig.tight_layout()
    plt.show()

