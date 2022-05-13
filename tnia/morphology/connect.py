from skimage.segmentation import relabel_sequential
import numpy as np
from skimage.util._map_array import ArrayMap
from skimage import measure
from scipy import spatial
from skimage.segmentation import relabel_sequential

def connect_2d_in_3d(labeled, threshold):
    """ Given a stack of 2D labels connect nearest neighbors in adjacent planes 
    Args:
        labeled (3d integer np array): the labels
        threshold (float): max 2d distance to consider objects connected
    """
    relabeled=[labeled[0,:,:]]
    for i in range(1,labeled.shape[0]):
        print(i,end=' ')
        previous=relabeled[i-1]
        current=labeled[i,:,:]
        relabeled.append(connect_labels(previous, current, threshold))

    return np.stack(relabeled)

def connect_labels(previousImage, currentImage, threshold):
    """_summary_

    A algorithm that connect 2D labels in 3D.  Code mostly written by 
    Varun Kapoor and posted on the SC forum here https://forum.image.sc/t/2d-3d-integer-labels/38732/10

    Some improvements were made by Volker Hilsenstein 

    Given image n and n-1 of the stack connect nearest neigbors between n and n-1
    Args:
        previousImage (2d integer labeled np array): image n-1, presumably from a stack 
        currentImage (2d integer labeled np array):  image n of the stack
        threshold (float): max 2d distance to consider objects connected

    Returns:
        relabeled image
    """
    # This line ensures non-intersecting label sets
    currentImage = relabel_sequential(currentImage,offset=previousImage.max()+1)[0]
    # I also don't like modifying the input image, so we take a copy
    relabelimage = currentImage.copy()
    waterproperties = measure.regionprops(previousImage, previousImage)
    indices = [prop.centroid for prop in waterproperties] 
    labels = [prop.label for prop in waterproperties]
    if len(indices) > 2:
       tree = spatial.cKDTree(indices)
       currentwaterproperties = measure.regionprops(currentImage, currentImage)
       currentindices = [prop.centroid for prop in currentwaterproperties] 
       currentlabels = [prop.label for prop in currentwaterproperties] 
       if len(currentindices) > 2: #why only > : ?
           for i in range(0,len(currentindices)):
               index = currentindices[i]
               #print(f"index {index}")
               currentlabel = currentlabels[i] 
               #print(f"currentlabel {currentlabel}")
               if currentlabel > 0:
                      previouspoint = tree.query(index)
                      #print(f"prviouspoint {previouspoint}")
                      previouslabel = previousImage[int(indices[previouspoint[1]][0]), int(indices[previouspoint[1]][1])]
                      #print()
                      #print(f"previouslabels {previouslabel}")
                      #hjkhkj
                      if previouspoint[0] > threshold:
                             relabelimage[np.where(currentImage == currentlabel)] = currentlabel
                      else:
                             #print('match', previouspoint, 'threshold', threshold)
                             relabelimage[np.where(currentImage == currentlabel)] = previouslabel
    return relabelimage 