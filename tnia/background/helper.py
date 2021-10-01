import numpy as np

def subtract_clip_negatives(a, b):
    c = np.zeros(a.shape)
    np.putmask(c,b>=a,0)
    np.putmask(c,b<a,a-b)
    return c
