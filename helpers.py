import numpy as np
import random


def get_all_cluster_to_dots_list(dots, dot_to_cluster_matrix):
    cluser_to_dots = [None] * len(dot_to_cluster_matrix[1])
    for cluster_index in range(len(dot_to_cluster_matrix[1])):
        cluser_to_dots[cluster_index] = np.take(dots[:], np.where(
            dot_to_cluster_matrix[:, cluster_index] == 1), axis=0)[0]

    return cluser_to_dots


def get_labels_for_dots(dots, clusters, dot_to_cluster_matrix):
    labels = np.empty(len(dots))
    for dot_index, dot in enumerate(dots):
        for cluster_index in range(len(clusters)):
            if(dot_to_cluster_matrix[dot_index, cluster_index] == 1):
                labels[dot_index] = cluster_index
                break

    return labels


def calculate_centers_for_clusters(dots, dot_to_cluster_matrix):
    cluster_to_dots = get_all_cluster_to_dots_list(dots, dot_to_cluster_matrix)

    centers = np.zeros((len(cluster_to_dots), 2))

    for cluster_index, dots in enumerate(cluster_to_dots):
        centers[cluster_index] = np.mean(dots, axis=0)

    return centers


def check_event_success(probability):

    if probability < 0:
        raise Exception("probability cant be fewer than 0")

    if probability > 1:
        raise Exception("probability cant be greater than 1")

    probability_100 = probability * 100
    number = random.randint(1, 100)

    return number <= probability_100


def get_event_for_discrete_probability(discrete_distribution):
    if isinstance(discrete_distribution, list):
        discrete_distribution_100 = {
            key: value * 100 for key, value in enumerate(discrete_distribution)}
    elif isinstance(discrete_distribution, dict):
        discrete_distribution_100 = {
            key: value * 100 for key, value in discrete_distribution.items()}

    number = random.randint(1, 100)

    needed_key = None
    prob_sum = 0
    for key in discrete_distribution_100:
        prob_sum += discrete_distribution_100[key]
        if number <= prob_sum:
            needed_key = key
            break

    if needed_key == None:
        return list(discrete_distribution_100.keys())[len(discrete_distribution_100) - 1]

    return needed_key


if __name__ == '__main__':
    # get prepared data from main.py
    #cluster_centers = np.load('data/cluster_centers.npy')
    #dots = np.load('data/dots.npy')
    #dot_to_cluster_matrix = np.load('data/dot_to_cluster_matrix.npy')

    #print(get_labels_for_dots(dots, cluster_centers, dot_to_cluster_matrix))

    print(get_event_for_discrete_probability({1: 0.1, 2: 0.3, 3: 0.2, 4: 0.4}))
