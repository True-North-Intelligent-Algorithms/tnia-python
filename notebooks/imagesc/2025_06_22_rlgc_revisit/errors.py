import numpy as np

def MSE(a, b, mask=None, backend=np):
    """
    Mean Squared Error (MSE) for NumPy or CuPy arrays.
    
    Parameters:
        a (array): First array (NumPy or CuPy).
        b (array): Second array (NumPy or CuPy).
        mask (array, optional): Mask to apply. Defaults to None.
        backend (module, optional): NumPy or CuPy module. Defaults to NumPy.
        
    Returns:
        float: MSE value.
    """
    if mask is not None:
        a = a[mask!=0]
        b = b[mask!=0]

    print(a.min(), a.max())

    diff = backend.subtract(a, b)
    squared_diff = backend.square(diff)
    return squared_diff.mean()

def RMSE(a, b, mask=None, backend=np):
    """
    Root Mean Squared Error (RMSE) for NumPy or CuPy arrays.
    
    Parameters:
        a (array): First array (NumPy or CuPy).
        b (array): Second array (NumPy or CuPy).
        mask (array, optional): Mask to apply. Defaults to None.
        backend (module, optional): NumPy or CuPy module. Defaults to NumPy.
        
    Returns:
        float: RMSE value.
    """
    #print(squared_diff.shape, squared_diff.min(), squared_diff.max())
    #print(a.mean(), b.mean(), a.min(), a.max(), b.min(), b.max())
    
    return backend.sqrt(MSE(a, b, mask, backend=backend))
