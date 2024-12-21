import raster_geometry as rg
import numpy as np
from tnia.simulation.phantoms import add_small_to_large_2d, add_small_to_large
from random import seed, uniform
import matplotlib.pyplot as plt
import math

def make_random_spheres():
    width = 512
    height = 512

    random_spheres = np.zeros([height, width], dtype=np.float32)

    seed(354)

    for i in range(10):
        x = int(uniform(0, width))
        y = int(uniform(0, height))
        r = int(uniform(20, 30))
        size = [math.ceil(r*2), math.ceil(r*2)]
        temp=rg.circle(size, r)

        add_small_to_large_2d(random_spheres, temp, x, y, mode='replace_non_zero')

    return random_spheres

def make_3D_random_spheres():
    width = 64
    height = 64
    depth = 16

    random_spheres = np.zeros([depth, height, width], dtype=np.float32)

    seed(354)

    for i in range(10):
        x = int(uniform(0, width))
        y = int(uniform(0, height))
        z = int(uniform(0, depth))
        r = int(uniform(2, 10))
        size = [math.ceil(r*2), math.ceil(r*2), math.ceil(r*2)]
        temp=rg.sphere(size, r)

        add_small_to_large(random_spheres, temp, x, y, z, mode='replace_non_zero')

    return random_spheres
