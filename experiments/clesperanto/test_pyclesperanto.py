import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt

from skimage.io import imread, imsave

# initialize GPU
cle.select_device("RTX")
print("Used GPU: " + cle.get_device().name)

# load data
image = imread('https://imagej.nih.gov/ij/images/blobs.gif')
print("Loaded image size: " + str(image.shape))


# push image to GPU memory
input = cle.push(image)
print("Image size in GPU: " + str(input.shape))

# process the image
inverted = cle.subtract_image_from_scalar(image, scalar=255)
blurred = cle.gaussian_blur(inverted, sigma_x=1, sigma_y=1)
binary = cle.threshold_otsu(blurred)
labeled = cle.connected_components_labeling_box(binary)

binary_cpu = cle.pull(binary)
labeled_cpu = cle.pull(labeled)

plt.imshow(image)
plt.figure() 
plt.imshow(labeled_cpu)

plt.show()