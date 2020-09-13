import numpy as np
from random import gauss


def create_2d_dots_with_gaussian(mean, variance, count):
    mean_for_x = mean[0]
    mean_for_y = mean[1]

    variance_for_x = variance[0]
    variance_for_y = variance[1]

    dots = np.empty((count, 2))
    for i in range(count):
        dots[i, 0] = gauss(mean_for_x, variance_for_x)
        dots[i, 1] = gauss(mean_for_y, variance_for_y)

    return dots
