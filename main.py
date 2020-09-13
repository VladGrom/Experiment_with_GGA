from synthetic_data import create_2d_dots_with_gaussian
import matplotlib.pyplot as plt
import numpy as np
from ga import get_davis_bouldin_index

dots_count = 320
dots_count_in_cluster = 40

cluster_centers = np.array([
    [-100, 0],
    [-1, -10],
    [-40, -30],
    [300, -1],
    [-10, 10],
    [250, -210],
    [100, 23],
    [356, 111],
])

variance = np.array([0.35**2, 0.35**2])

dots = np.empty((dots_count, 2))
dot_to_cluster_matrix = np.empty((dots_count, 8))
for cluster_center_number, cluster_center in enumerate(cluster_centers):
    dot_index_from = cluster_center_number * dots_count_in_cluster
    dot_index_to = (cluster_center_number + 1) * dots_count_in_cluster

    dots[dot_index_from:dot_index_to] = create_2d_dots_with_gaussian(
        cluster_center, variance, dots_count_in_cluster)
    dot_to_cluster_matrix[dot_index_from:dot_index_to, cluster_center_number] = 1

plt.scatter(dots[:, 0], dots[:, 1], s=4)

for cluster_center in cluster_centers:
    plt.plot(cluster_center[0], cluster_center[1], 'ro', markersize=5)

np.save('data/cluster_centers.npy', cluster_centers)
np.save('data/dots.npy', dots)
np.save('data/dot_to_cluster_matrix.npy', dot_to_cluster_matrix)



plt.show()
