
def calculate_array_size_gb(arr):
    """ Calculates the size of an array in GB

    Args:

        arr (numpy array): the array to calculate the size of

    Returns:
        the size of the array in GB
    """
    return arr.size*arr.itemsize/1e9

def print_mem_info(arr):
    """ Prints information about an array

    Args:

        arr (numpy array): the array to print information about
    """
    print()
    print(f"Array shape: {arr.shape}")
    print(f"Array size: {arr.size}")
    print(f"Array item size: {arr.itemsize}")
    print(f"Array size in GB: {calculate_array_size_gb(arr)}")
