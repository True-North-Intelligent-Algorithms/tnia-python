import numpy as np

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
 
    