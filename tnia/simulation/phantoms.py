import numpy as np
import raster_geometry as rg
from random import uniform, seed
import math
import random

def sphere3d(size, radius, intensity=1, z_down_sample=1):
    """ Creates a 3D sphere.

    Args:
        size (list or tuple): The size of the volume to create.
        radius (float): The radius of the sphere.
        intensity (float, optional): The intensity of the sphere. Default is 1.
        z_down_sample (int, optional): The downsampling factor in the z direction. Default is 1.
    Returns:
        numpy.ndarray: A 3D volume of size 'size' containing a sphere of radius 'radius'.
    """
    sphere=intensity*rg.sphere(size, radius).astype(np.float32)

    return sphere[::z_down_sample,:,:]

def sphere_fits(large_arr, small_arr,x,y,z):
    """ checks if a sphere fits in a larger array

    Args:
        large_arr (numpy.ndarray): The larger array.
        small_arr (numpy.ndarray): The smaller array which contains the sphere.
        x (int): The x coordinate of the center of the sphere.
        y (int): The y coordinate of the center of the sphere.
        z (int): The z coordinate of the center of the sphere.
    Returns:
        bool: True if the sphere fits in the larger array, False otherwise.
    """
    # Get the dimensions of the arrays
    l, w, h = large_arr.shape
    sl, sw, sh = small_arr.shape

    # Calculate the start and end indices for the small array in each dimension
    start_x = x - sw//2
    end_x = start_x + sw
    start_y = y - sh//2
    end_y = start_y + sh
    start_z = z - sl//2
    end_z = start_z + sl

    # Calculate the overlapping slice of the small array
    slice_z = slice(max(start_z, 0), min(end_z, l))
    slice_y = slice(max(start_y, 0), min(end_y, h))
    slice_x = slice(max(start_x, 0), min(end_x, w))
    small_slice_z = slice(max(-start_z, 0), min(sl - (end_z - l), sl))
    small_slice_y = slice(max(-start_y, 0), min(sh - (end_y - h), sh))
    small_slice_x = slice(max(-start_x, 0), min(sw - (end_x - w), sw))
    small_slice = small_arr[small_slice_z, small_slice_y, small_slice_x]

    # Check if the slice is empty
    if np.any(np.logical_and(large_arr[slice_z, slice_y, slice_x]>0,small_slice>0)):
        return False
    
    return True

def add_small_to_large(large_arr, small_arr,x,y,z, check_empty=False, mode='add'):
    """ Adds a small array to a larger array.

    Args:
        large_arr (numpy.ndarray): The larger array.
        small_arr (numpy.ndarray): The smaller array to add to the larger array.
        x (int): The x coordinate of the center of the smaller array.
        y (int): The y coordinate of the center of the smaller array.
        z (int): The z coordinate of the center of the smaller array.
        check_empty (bool, optional): If True, the function will check if the overlapping slice is empty before adding the small array. Default is False.
    Returns:
        bool: True if the small array was added to the larger array, False otherwise.
    """
    # Get the dimensions of the arrays
    l, h, w = large_arr.shape
    sl, sh, sw = small_arr.shape

    # Calculate the start and end indices for the small array in each dimension
    start_x = x - sw//2
    end_x = start_x + sw
    start_y = y - sh//2
    end_y = start_y + sh
    start_z = z - sl//2
    end_z = start_z + sl

    # Calculate the overlapping slice of the small array
    slice_z = slice(max(start_z, 0), min(end_z, l))
    slice_y = slice(max(start_y, 0), min(end_y, h))
    slice_x = slice(max(start_x, 0), min(end_x, w))
    small_slice_z = slice(max(-start_z, 0), min(sl - (end_z - l), sl))
    small_slice_y = slice(max(-start_y, 0), min(sh - (end_y - h), sh))
    small_slice_x = slice(max(-start_x, 0), min(sw - (end_x - w), sw))
    small_slice = small_arr[small_slice_z, small_slice_y, small_slice_x]

    if check_empty:
        # Check if the slice is empty
        if np.any(np.logical_and(large_arr[slice_z, slice_y, slice_x]>0,small_slice>0)):
            return False

    # Add the small slice to the large array
    if mode=='add':
        large_arr[slice_z, slice_y, slice_x] += small_slice
    elif mode=='replace':
        large_arr[slice_z, slice_y, slice_x] = small_slice
    elif mode=='replace_non_zero':
        indices = np.where(small_slice>0)
        large_arr[slice_z, slice_y, slice_x][indices] = small_slice[indices]
        #large_arr[slice_z, slice_y, slice_x][large_arr[slize_z, slize_y, slize_x]>0] = small_slice[small_slice>0]
    return True

def add_small_to_large_2d(large_arr, small_arr, x, y, check_empty=False, mode='add', center = True):
    """ Adds a small array 2d to a larger array.

    Notes:  This function is essentially the same as add_small_to_large but for 2D.  The code could be made more 'DRY' in the future.

    Args:
        large_arr (numpy.ndarray): The larger array.
        small_arr (numpy.ndarray): The smaller array to add to the larger array.
        x (int): The x coordinate of the center of the smaller array.
        y (int): The y coordinate of the center of the smaller array.
        check_empty (bool, optional): If True, the function will check if the overlapping slice is empty before adding the small array. Default is False.
    Returns:
        bool: True if the small array was added to the larger array, False otherwise.
    """
    # Get the dimensions of the arrays
    h, w = large_arr.shape
    sh, sw = small_arr.shape

    if center:
        # Calculate the start and end indices for the small array in each dimension
        start_x = x - sw//2
        end_x = start_x + sw
        start_y = y - sh//2
        end_y = start_y + sh
    else:
        start_x = x
        end_x = x + sw
        start_y = y
        end_y = y + sh

    # Calculate the overlapping slice of the small array
    slice_y = slice(max(start_y, 0), min(end_y, h))
    slice_x = slice(max(start_x, 0), min(end_x, w))
    small_slice_y = slice(max(-start_y, 0), min(sh - (end_y - h), sh))
    small_slice_x = slice(max(-start_x, 0), min(sw - (end_x - w), sw))
    small_slice = small_arr[small_slice_y, small_slice_x]

    if check_empty:
        # Check if the slice is empty
        if np.any(np.logical_and(large_arr[slice_y, slice_x]>0,small_slice>0)):
            return False

    # Add the small slice to the large array
    if mode=='add':
        large_arr[slice_y, slice_x] += small_slice
    elif mode=='replace':
        large_arr[slice_y, slice_x] = small_slice
    elif mode=='replace_non_zero':
        indices = np.where(small_slice>0)
        large_arr[slice_y, slice_x][indices] = small_slice[indices]
    return True



def mask_small_to_large(large_arr, small_arr,x,y,z):
    """
    Masks a small array to a larger array.
    
    Args:
        large_arr (numpy.ndarray): The larger array.
        small_arr (numpy.ndarray): The smaller array to add to the larger array.
        x (int): The x coordinate of the center of the smaller array.
        y (int): The y coordinate of the center of the smaller array.
        z (int): The z coordinate of the center of the smaller array.
    """
    # Get the dimensions of the arrays
    l, w, h = large_arr.shape
    sl, sw, sh = small_arr.shape

    # Calculate the start and end indices for the small array in each dimension
    start_x = x - sw//2
    end_x = start_x + sw
    start_y = y - sh//2
    end_y = start_y + sh
    start_z = z - sl//2
    end_z = start_z + sl

    # Calculate the overlapping slice of the small array
    slice_z = slice(max(start_z, 0), min(end_z, l))
    slice_y = slice(max(start_y, 0), min(end_y, h))
    slice_x = slice(max(start_x, 0), min(end_x, w))
    small_slice_z = slice(max(-start_z, 0), min(sl - (end_z - l), sl))
    small_slice_y = slice(max(-start_y, 0), min(sh - (end_y - h), sh))
    small_slice_x = slice(max(-start_x, 0), min(sw - (end_x - w), sw))
    small_slice = small_arr[small_slice_z, small_slice_y, small_slice_x]
    
    # mask out the area of the smaller array within the larger array
    large_arr[slice_z, slice_y, slice_x] = 0 


def add_sphere3d(img, radius, x_center, y_center, z_center, intensity=1, z_down_sample=1):
    """Adds a sphere to a 3D numpy array.

    Args:
        im (numpy.ndarray): The 3D numpy array to add the sphere to.
        radius (int): The radius of the sphere.
        x_center (int): The x-coordinate of the center of the sphere.
        y_center (int): The y-coordinate of the center of the sphere.
        z_center (int): The z-coordinate of the center of the sphere.
        intensity (float, optional): The intensity of the sphere. Default is 1.
        z_down_sample (int, optional): The downsampling factor in the z direction. Default is 1.

    Returns:
        None
    """

    size = [2*radius, 2*radius, 2*radius]
    sphere = rg.sphere(size, radius).astype(np.float32)
    add_small_to_large(img, intensity*sphere, x_center, y_center, z_center, True)
    
def ramp2d(shape,min,max):
    """ create an image of a 2D ramp to simulate uneven background

    Args:
        shape (list of 2 integers): Size of ramp image
        min (number): min of ramp
        max (number): max of ramp

    Returns:
        [2d numpy array]: ramp image 
    """

    ramp=np.empty(shape)

    for y in range(shape[0]):
        for x in range(shape[1]):
            ramp[y,x]=(max-min)*(y+x)/(shape[0]+shape[1])+min
    return ramp

def random_circles(im, num, min_r, max_r, min_intensity, max_intensity, seed_val=-1):
    """ draws random circles in an image useful for testing segmentation, background subtraction etc.

    Args:
        im (2d numpy array): input image, will be changed
        num (number): number of circles to draw
        min_r (number): min radius of circles
        max_r (number): max radius of circles
        min_intensity (number): max intensity of circles
        max_intensity (number): min intensity of circles
        seed_val (int, optional): use seed if you need to replicate the same random image defalts to -1 (no seed used).
    """
    if seed_val!=-1:
        seed(seed_val)

    for i in range(num):
        r=uniform(min_r, max_r)
        cx=round(uniform(r,im.shape[1]-r))
        cy=round(uniform(r,im.shape[0]-r))
        intensity=round(uniform(min_intensity, max_intensity))
        #print(r,cx,cy,intensity)
        
        temp1=rg.circle([r*2,r*2],r)
        temp2=np.zeros_like(im)
        temp2[cy-r:cy+r,cx-r:cx+r]=temp1
        im[temp2>0]=im[temp2>0]+intensity

def grid_of_circles(im, radius, space,border,intensity):
    """ draws a grid of circles on a background image

    Args:
        im ([type]): background image
        radius ([type]): radius of circles 
        space ([type]): space between circles 
        border (): border around circles
        intensity ([type]): intensity of circles
    """
    num_circles_x = math.floor(im.shape[1]/(space))
    num_circles_y = math.floor(im.shape[0]/(space))

    for x in range(num_circles_x):
        for y in range(num_circles_y):
            cx=(x)*space+border
            cy=(y)*space+border
            if cx < (im.shape[1]-radius/2-1) and cy < (im.shape[0]-radius/2-1):
                temp1=rg.circle([radius*2,radius*2],radius)
                temp2=np.zeros_like(im)
                #print(cx,cy)
                temp2[cy-radius:cy+radius,cx-radius:cx+radius]=temp1
                im[temp2>0]=intensity

def random_sphere_phantom(xdim, ydim, zdim, number_spheres, min_radius, max_radius, phantom=None, min_intensity=50, max_intensity=800):

    if phantom is None:
        phantom = np.zeros((zdim, ydim, xdim), dtype=np.float32)
        
    number_spheres_added = 0

    for i in range(0,number_spheres):
        x=random.randint(0,xdim)
        y=random.randint(0,ydim)
        z=random.randint(0,zdim)
        intensity = random.randint(min_intensity, max_intensity) 

        #print(i,x,y,z)

        r=random.randint(min_radius,max_radius)

        size = [2*r, 2*r, 2*r]
        sphere = rg.sphere(size, r).astype(np.float32)
        sphere = sphere*intensity
        #add_sphere3d(phantom, 20, x, y, z, intensity, 2)
        if add_small_to_large(phantom, sphere, x, y, z, True) == True:
            number_spheres_added += 1

    return phantom

