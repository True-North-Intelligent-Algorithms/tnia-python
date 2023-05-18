import pywt
import pywt.data

def dwt2d(img, wavelet='db4', level=1):
    """ convenience helper to perform a 2d discrete wavelet transform on an image and return the wavelet coefficient images as a list
    Args:
        img ([nd array]): image to transform 
        wavelet ([str], optional): wavelet to use. Defaults to 'db4'.
        level ([int], optional): number of levels to use. Defaults to 1.
    Returns:
        [list]: list of wavelet coefficients
    """
    # Transform
    coeffs = pywt.dwt2(img, wavelet, mode='symmetric')
    LL, (LH, HL, HH) = coeffs
    return [LL, LH, HL, HH]

def idwt2d(coeffs, wavelet='db4'):
    """ convenience helper to perform a 2d inverse discrete wavelet transform on a list of wavelet coefficients
    Args:
        coeffs ([list]): list of wavelet coefficients
        wavelet ([str], optional): wavelet to use. Defaults to 'db4'.
    Returns:
        [nd array]: reconstructed image
    """
    LL, LH, HL, HH = coeffs
    coeffs2 = LL, (LH, HL, HH)
    # Reconstruction
    return pywt.idwt2(coeffs2, wavelet, mode='symmetric')
