import SimpleITK as sitk
import numpy as np

def rigid(fixed, mov):
    """Registers moving image to fixed using bspline approach

    Args:
        fixed (2d numpy array): fixed image
        mov (2d numpy array): moving image

    Returns:
        (2d numpy array, transformparametervector): returns the registered image and the parameter map 
    """
     # convert arrays to simple ITK images
    fixed = sitk.GetImageFromArray(fixed)
    mov= sitk.GetImageFromArray(mov)

    # set up the filter
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(mov)
    
    # TODO consider masks
    #elastixImageFilter.SetFixedMask(mask)
    #elastixImageFilter.SetMovingMask(mask)

    # set up parameter map for affine followed by bspline registration
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
 
    elastixImageFilter.Execute()

    # convert simple itk to numpy array, and return array and parameter map 
    return sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()), elastixImageFilter.GetTransformParameterMap()

def bspline(fixed, mov, grid_spacing, order):
    """Registers moving image to fixed using bspline approach

    Args:
        fixed (2d numpy array): fixed image
        mov (2d numpy array): moving image
        grid_spacing: Final grid spacing in physical units.
        order: order of bspline

    Returns:
        (2d numpy array, transformparametervector): returns the registered image and the parameter map 
    """
    # convert arrays to simple ITK images
    fixed = sitk.GetImageFromArray(fixed)
    mov= sitk.GetImageFromArray(mov)

    # set up the filter
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(mov)
    
    # TODO consider masks
    #elastixImageFilter.SetFixedMask(mask)
    #elastixImageFilter.SetMovingMask(mask)

    # set up parameter map for affine followed by bspline registration
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    bspline=sitk.GetDefaultParameterMap("bspline")

    # defaults below work OK in general, but in the future maybe make these parameters
    bspline['NumberOfResolutions']=['2']
    bspline['FinalGridSpacingInPhysicalUnits']=[grid_spacing]
    bspline['BSplineTransformSplineOrder']=[order]

    # set parameter map and execute
    parameterMapVector.append(bspline)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    # convert simple itk to numpy array, and return array and parameter map 
    return sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()), elastixImageFilter.GetTransformParameterMap()

def applymap(img, transform):
    """Applies a pre-compated simple elastics transform to img

    Args:
        img (2d numpy array): image to transform
        transform ([type]): simple elastic vector of ParameterMap, that contains pre-computed transform 

    Returns:
        2d numpy array: transformed image
    """
    img = sitk.GetImageFromArray(img)
    return sitk.GetArrayFromImage(sitk.Transformix(img, transform))

def applymap_zcyx(img, transform, transform_channel):

    aligned = np.zeros(img.shape, 'float32')
    
    for i in range(0,img.shape[0]):
        print('registering frame',i)
        for c in range(0,img.shape[1]):
        
            if transform_channel[c]==0:
                aligned[i,c,:,:]=img[i,c,:,:]
            else:
                aligned[i,c,:,:]=applymap(img[i,c,:,:],transform)

            aligned[aligned<0]=0;
        
    return aligned        