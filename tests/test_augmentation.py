import raster_geometry as rg
import numpy as np
from tnia.simulation.phantoms import add_small_to_large_2d
from random import seed, uniform
import matplotlib.pyplot as plt
import math

def test_augmentation():
    width = 512
    height = 512

    truth = np.zeros([height, width], dtype=np.float32)

    seed(354)

    for i in range(10):
        x = int(uniform(0, width))
        y = int(uniform(0, height))
        r = int(uniform(20, 30))
        size = [math.ceil(r*2), math.ceil(r*2)]
        temp=rg.circle(size, r)

        add_small_to_large_2d(truth, temp, x, y, mode='replace_non_zero')

    assert truth.sum() == 16355.0

    print(truth.sum())
    #plt.imshow(truth)
    #plt.show()
    #stop = 5

# make main
if __name__ == "__main__":
    test_augmentation()
    



