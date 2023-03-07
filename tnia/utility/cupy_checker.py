import numpy as np
import importlib

try:
    cupy = importlib.import_module('cupy')
    has_cupy = True
except ImportError:
    has_cupy = False

def is_cupy_installed():
    """
    Returns True if Cupy is installed, False otherwise.
    """
    return has_cupy

def get_platform(x):
    """
    Returns the appropriate package (NumPy or Cupy) depending on whether Cupy is installed and `x` is a
    Cupy array.

    Args:
        x (ndarray): Input array

    Returns:
        Module: The appropriate package (either NumPy or Cupy)
    """
    if has_cupy:
        if hasattr(cupy, 'ndarray') and isinstance(x, cupy.ndarray):
            return cupy
    return np
