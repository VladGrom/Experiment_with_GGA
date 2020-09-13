
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from solution import Solution
from helpers import calculate_centers_for_clusters, get_all_cluster_to_dots_list

import numpy as np

from helpers import get_labels_for_dots

if __name__ == '__main__':
    # get prepared data from main.py
    cluster_centers = np.load('data/cluster_centers.npy')
    dots = np.load('data/dots.npy')
    dot_to_cluster_matrix = np.load('data/dot_to_cluster_matrix.npy')

    test_solution = Solution.load("best_solution")
    dot_to_cluster_matrix_for_solution = test_solution.dot_to_cluster_matrix()
    centers_for_solution = calculate_centers_for_clusters(dots, dot_to_cluster_matrix_for_solution)

    labels = get_labels_for_dots(dots, cluster_centers, dot_to_cluster_matrix)
    labels_for_solution = get_labels_for_dots(dots, centers_for_solution, dot_to_cluster_matrix_for_solution)
    print(get_all_cluster_to_dots_list(dots, dot_to_cluster_matrix_for_solution))

    print(davies_bouldin_score(dots, labels))
    print(davies_bouldin_score(dots, labels_for_solution))