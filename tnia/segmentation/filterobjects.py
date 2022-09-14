from skimage.measure import regionprops
import math
import numpy as np

def find_circular_objects(image, label_image, min_area, min_intensity, min_circularity):
    """ filters dim, small and non-circular objects and keeps bright, large and circular objectgs 

    Args:
        image (np array): original intensity image which objects were segmented from
        label_image (np array):  segmented objects labeled with integer indexes
        max_area (number): min area to keep 
        max_intensity (number): min intesnity to keep
        max_circularity (number): min circularity to keep (circularity goes from 0 to 1, closer to 1 is more circular)

    Returns:
        [np array]: label image with filtered labels set to zero
    """
    object_list=regionprops(label_image,image)
    label_image_filtered=label_image.copy()

    for obj in object_list:

        circularity = (4*math.pi*obj.area)/(obj.perimeter**2)
        if circularity < min_circularity or obj.area < min_area or obj.mean_intensity<min_intensity:
            label_image_filtered[label_image_filtered==obj.label]=0

    return label_image_filtered

def find_solid_objects(image, label_image, min_area, min_intensity, min_solidity):
    """ filters dim, small and non-solid objects and keeps bright, large and solid objects 

    Args:
        image (np array): original intensity image which objects were segmented from
        label_image (np array):  segmented objects labeled with integer indexes
        max_area (number): min area to keep 
        max_intensity (number): min intesnity to keep
        max_solidity (number): min solidity to keep (solidity goes from 0 to 1, closer to 1 is more solid)

    Returns:
        [np array]: label image with filtered labels set to zero
    """
    object_list=regionprops(label_image,image)
    label_image_filtered=label_image.copy()

    for obj in object_list:
        
        if obj.solidity < min_solidity or obj.area < min_area or obj.mean_intensity<min_intensity:
            label_image_filtered[label_image_filtered==obj.label]=0

    return label_image_filtered

def find_large_objects(image, label_image, min_area):
    """ filters small objects and keeps large objects 

    Args:
        image (np array): original intensity image which objects were segmented from
        label_image (np array):  segmented objects labeled with integer indexes
        max_area (number): min area to keep 

    Returns:
        [np array]: label image with filtered labels set to zero
    """

    object_list=regionprops(label_image)
    import numpy as np
    label_image_filtered=np.zeros_like(label_image)

    for obj in object_list:
        
        if obj.area > min_area:
            label_image_filtered[label_image==obj.label]=1

    return label_image_filtered

def filter_flat_objects(label_image, min_depth):
    object_list=regionprops(label_image)
    label_image_filtered=np.zeros_like(label_image)

    for obj in object_list:
        if obj.bbox[3]-obj.bbox[0]>min_depth:
            for c in obj.coords:
                label_image_filtered[c[0],c[1],c[2]]=obj.label


    return label_image_filtered

def filter_objects_zlocation(label_image, start_z, end_z):
    object_list=regionprops(label_image)
    label_image_filtered=np.zeros_like(label_image)

    for obj in object_list:
        if obj.centroid[0]>=start_z and obj.centroid[0]<=end_z:
            for c in obj.coords:
                label_image_filtered[c[0],c[1],c[2]]=obj.label


    return label_image_filtered

