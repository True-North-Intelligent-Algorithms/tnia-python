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
    
    if mask is None:
        diff = backend.subtract(a, b)
        squared_diff = backend.square(diff)
        return squared_diff.mean()
    else:
        diff = backend.subtract(a*mask, b*mask)
        squared_diff = backend.square(diff)
        return squared_diff.sum() / backend.sum(mask)

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
    return backend.sqrt(MSE(a, b, mask=mask, backend=backend))

