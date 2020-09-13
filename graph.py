import numpy as np
import matplotlib.pyplot as plt
import random

from solution import Solution

test_solution = Solution.load("best_solution")
dots = np.load('data/dots.npy')


plot_true = plt.figure(1)
ax_true=plot_true.add_axes([0,0,1,1])

plot_test = plt.figure(2)
ax_test=plot_test.add_axes([0,0,1,1])


ax_true.scatter(dots[:, 0], dots[:, 1], s=4)

dot_to_cluster_matrix = test_solution.dot_to_cluster_matrix()

for column in dot_to_cluster_matrix.T:
    rgb = (random.random(), random.random(), random.random())

    cluster_indexes = (column == 1).nonzero()[0]
    cluster_dots = dots[cluster_indexes]
    ax_test.scatter(cluster_dots[:, 0], cluster_dots[:, 1],s=4, c=[rgb])

plt.show()