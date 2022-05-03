from skimage.morphology import remove_small_holes
import numpy as np

def fill_holes_3d_slicer_labels(labels, area_threshold=1000, num_iterations=1):
    """ calls fill_holes_3d_slicer on each label separately

    Args:
        labels (): 3D labeled image
        area_threshold (int, optional): _description_. Defaults to 1000.
        num_iterations (int, optional): _description_. Defaults to 1.
    """
    max_label=labels.max()

    for i in range(1,max_label):
        binary = np.zeros_like(labels)
        binary[labels==i] = 1
        binary = binary>0
        fill_holes_3d_slicer(binary, area_threshold, num_iterations)
        labels[binary==True]=i
        

def fill_holes_3d_slicer(binary, area_threshold=1000, num_iterations=1):
    """ fills holes in a 3D image using a 2D 'slicer' strategy.
        Useful for filling 'tunnels'

    Args:
        binary (binary numpy array): binary image to fill holes (or tunnels) in
        area_threshold (int, optional): maximum 2d hole size
    """
    for i in range(num_iterations):
        for i in range(binary.shape[0]):
            binary[i,:,:] = remove_small_holes(binary[i,:,:],area_threshold)

        for i in range(binary.shape[1]):
            binary[:,i,:] = remove_small_holes(binary[:,i,:],area_threshold)

        for i in range(binary.shape[2]):
            binary[:,:,i] = remove_small_holes(binary[:,:,i],area_threshold)

def fill_holes_slicer(binary, area_threshold=1000):
    """ fills holes in a 3D image using a 2D 'slicer' strategy only in xy direction.

    Args:
        binary (binary numpy array): binary image to fill holes (or tunnels) in
        area_threshold (int, optional): maximum 2d hole size
    """
    for i in range(binary.shape[0]):
        binary[i,:,:] = remove_small_holes(binary[i,:,:],area_threshold)

