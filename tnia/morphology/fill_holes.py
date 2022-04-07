from skimage.morphology import remove_small_holes

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


