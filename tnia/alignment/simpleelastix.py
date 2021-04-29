import SimpleITK as sitk

def bspline(fixed, mov):
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
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    bspline=sitk.GetDefaultParameterMap("bspline")

    # defaults below work OK in general, but in the future maybe make these parameters
    bspline['NumberOfResolutions']=['2']
    bspline['FinalGridSpacingInPhysicalUnits']=['32']

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
