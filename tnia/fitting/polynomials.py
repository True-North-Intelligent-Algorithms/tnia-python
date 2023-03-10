import numpy as np
from scipy.optimize import curve_fit

def paraboloid(data, a, b, c, d, e, f, g, h, i, shiftx, shifty):
    """
    Returns the value of a paraboloid function at a given point.

    Parameters:
        - data (list-like): The point at which to evaluate the paraboloid function. Should be a list-like object containing two elements: x and y.
        - a (float): The constant offset of the paraboloid.
        - b (float): The coefficient of x in the linear term of the paraboloid.
        - c (float): The coefficient of y in the linear term of the paraboloid.
        - d (float): The coefficient of x^2 in the quadratic term of the paraboloid.
        - e (float): The coefficient of x^2*y in the quadratic term of the paraboloid.
        - f (float): The coefficient of x^2*y^2 in the quadratic term of the paraboloid.
        - g (float): The coefficient of y^2 in the quadratic term of the paraboloid.
        - h (float): The coefficient of x*y^2 in the quadratic term of the paraboloid.
        - i (float): The coefficient of x*y in the quadratic term of the paraboloid.
        - shiftx (float): A value to shift the x-axis by.
        - shifty (float): A value to shift the y-axis by.

    Returns:
        The value of the paraboloid function at the point specified by data.

    Example:
    >>> data = [1, 2]
    >>> a = 1
    >>> b = 2
    >>> c = 3
    >>> d = 4
    >>> e = 5
    >>> f = 6
    >>> g = 7
    >>> h = 8
    >>> i = 9
    >>> shiftx = 0
    >>> shifty = 0
    >>> paraboloid(data, a, b, c, d, e, f, g, h, i, shiftx, shifty)
    93
    """
    x = data[0] - shiftx
    y = data[1] - shifty
    return a + x * b + y * c + x * x * d + x * x * y * e + x * x * y * y * f + y * y * g + x * y * y * h + x * y * i

def fit_paraboloid_to_image(im):
    """
    Fits a paraboloid to the given 2D array of image data using least squares curve fitting.

    Parameters:
        - im (numpy.ndarray): A 2D array of image data.

    Returns:
        A 2D numpy.ndarray representing the fitted paraboloid surface.

    Example:
    >>> im = np.random.rand(100, 100)
    >>> fit_surf = fit_paraboloid_to_image(im)
    """
    
    x_values = np.linspace(0, im.shape[1], im.shape[1])
    y_values = np.linspace(0, im.shape[0], im.shape[0])

    X, Y = np.meshgrid(x_values, y_values)

    X = X.flatten()
    Y = Y.flatten()
    Z = im.flatten()

    # fit the curve using least squares
    parameters, covariance = curve_fit(paraboloid, np.array([X, Y]), Z)
    
    # calculate Z coordinate array
    fit_surf = paraboloid(np.array([X, Y]), *parameters)
    
    # reshape to 2D array
    fit_surf = fit_surf.reshape(im.shape[0], im.shape[1])
    
    return fit_surf

def fit_2d_image(im):
    """ fits 2d polynomial to im

    Args:
        im (np array): input array

    Returns:
        np array: array containing 2d polynomial fit of input 
    """
    X=[]
    Y=[]
    Z=[]

    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            if im[y,x]>0:
                X.append(x)
                Y.append(y)
                Z.append(im[y,x])

    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)

    # we want to fit a 3D (position = f(X,Y)) to the data, so form an array filled with
    # the X, Y, XY, X^2, XY^2, X^2Y^2, Y^2, XY^2 and XY values 
    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B=Z

    # fit the curve using least squares
    coeff, r, rank, s = np.linalg.lstsq(A, B)

    fitted=np.zeros_like(im)
    
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            fitted[y,x]=coeff[0]+x*coeff[1]+y*coeff[2]+x*x*coeff[3]+x*x*y*coeff[4]+x*x*y*y*coeff[5]+y*y*coeff[6]+x*y*y*coeff[7]+x*y*coeff[8]

    return fitted,coeff
