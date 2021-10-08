from skimage.measure import regionprops
import math

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
        circularity = 4*math.pi*(obj.area/obj.perimeter**2)    
        if circularity < min_circularity or obj.area < min_area or obj.mean_intensity<min_intensity:
            label_image_filtered[label_image_filtered==obj.label]=0

    return label_image_filtered