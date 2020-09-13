import numpy as np
import math

from helpers import get_all_cluster_to_dots_list, calculate_centers_for_clusters, get_event_for_discrete_probability
from solution import Solution


def get_euclidean_distance_squered(vector_x, vector_y):
    distance_squared = np.power(vector_x - vector_y, 2)

    return math.sqrt(np.sum(distance_squared))


def get_sum_of_quadratic_errors_for_clusters(dots, centers, dot_to_cluster_matrix, distance_func=get_euclidean_distance_squered):
    SSE = 0
    for center_index, center in enumerate(centers):
        for dot_index, dot in enumerate(dots):
            SSE += dot_to_cluster_matrix[dot_index,
                                         center_index] * distance_func(dot, center)

    return SSE


def get_davis_bouldin_index(dots, centers, dot_to_cluster_matrix, distance_func=get_euclidean_distance_squered):
    cluster_to_dots = get_all_cluster_to_dots_list(dots, dot_to_cluster_matrix)

    inter_cluster_distance = np.zeros(len(cluster_to_dots))
    for cluster_index, dots in enumerate(cluster_to_dots):
        inter_cluster_distance[cluster_index] = np.average(
            np.array(list(map(lambda dot: distance_func(dot, centers[cluster_index]), dots))))

    davis_bouldin_index = 0
    for center_index_1 in range(len(centers)):
        distances_for_other_clusters = np.empty(0)
        for center_index_2 in range(len(centers)):
            if(center_index_1 == center_index_2):
                continue

            dist = (inter_cluster_distance[center_index_1] +
                    inter_cluster_distance[center_index_2])

            dist /= distance_func(centers[center_index_1],
                                  centers[center_index_2])

            distances_for_other_clusters = np.append(
                distances_for_other_clusters, dist)

        davis_bouldin_index += max(distances_for_other_clusters)

    return davis_bouldin_index/len(centers)


def get_davis_boulding_index_for_solution(dots, solution, distance_func=get_euclidean_distance_squered):
    dot_to_cluster_matrix = solution.dot_to_cluster_matrix()
    clusters_centers = calculate_centers_for_clusters(
        dots, dot_to_cluster_matrix)

    return get_davis_bouldin_index(dots, clusters_centers, dot_to_cluster_matrix)


def create_init_population(elements_count,population_size=50, cluster_count_from=2, cluster_count_to=5):
    solutions = [None for i in range(population_size)]

    for i in range(population_size):
        solutions[i] = Solution(elements_count)
        solutions[i].randomize_params(cluster_count_from, cluster_count_to)

    return solutions


def __calculate_probability(
    probability_start,
    probability_end,
    generation_number,
    generation_count
):
    return probability_start + (generation_number/generation_count) * (probability_end - probability_start)


def __get_probability_for_crossover_for_solutions_list(dots, solutions_sorted, measure=get_davis_boulding_index_for_solution):

    max_rank = len(solutions_sorted)
    probability_for_crossover_for_solution = [
        0 for i in range(len(solutions_sorted))]
    for solution_index, solution in enumerate(solutions_sorted):
        probability_for_crossover_for_solution[solution_index] = 2 * (
            len(solutions_sorted) - solution_index) / ((max_rank) * (max_rank + 1))

    return probability_for_crossover_for_solution


def run(
    dots,
    mutation_probability_start,
    mutation_probability_end,
    population_size=50,
    iteration_count=1000,
    cluster_count_from=7,
    cluster_count_to=10,
    elitism=False,
    island_model=False
):
    # Init Block
    solutions = create_init_population(len(dots),
                                       population_size, cluster_count_from=cluster_count_from, cluster_count_to=cluster_count_to)

    best_solution = None
    for iteration_index in range(iteration_count):
        # sorted solution by davis boulding index
        solutions_sorted = sorted(
            solutions, key=lambda solution: get_davis_boulding_index_for_solution(dots, solution))

        if best_solution == None or get_davis_boulding_index_for_solution(dots, solutions_sorted[0]) < get_davis_boulding_index_for_solution(dots, best_solution):
            best_solution = solutions_sorted[0]

        print("Itteration: {} \r\n Solution: {} \r\n Index: {}".format(iteration_index, str(
            best_solution), get_davis_boulding_index_for_solution(dots, best_solution)))

        mutation_probability = __calculate_probability(
            mutation_probability_start, mutation_probability_end, iteration_index, iteration_count)

        new_population = [None for i in range(population_size)]
        # указатель на место куда нужно вставить новое решение в поплуяции
        new_population_index = 0

        if elitism:
            new_population_index = int(population_size/10) - 1
            for solution_index in range(int(population_size/10)):
                new_population[solution_index] = solutions_sorted[solution_index]

        # Crossover block
        probability_for_crossover_for_solution = __get_probability_for_crossover_for_solutions_list(
            dots, solutions_sorted)

        while new_population_index != population_size:
            parent_1_key = get_event_for_discrete_probability(
                probability_for_crossover_for_solution)

            parent_1 = solutions_sorted[parent_1_key]

            parent_2_key = parent_1_key
            while parent_2_key != parent_1_key:
                parent_2_key = get_event_for_discrete_probability(
                    probability_for_crossover_for_solution)

            parent_2 = solutions_sorted[parent_2_key]

            new_population[new_population_index] = parent_1.crossover(parent_2)
            # Mutation Block
            new_population[new_population_index].mutate(mutation_probability)

            new_population_index += 1

        solutions = new_population
       # print(best_solution, iteration_index)

    return best_solution


if __name__ == '__main__':
    # get prepared data from main.py
    # cluster_centers = np.load('data/cluster_centers.npy')
    dots = np.load('data/dots.npy')
    dot_to_cluster_matrix = np.load('data/dot_to_cluster_matrix.npy')

    best_solution = run(dots, 0.5, 0.2,
                        iteration_count=100, elitism=True, population_size=100)
    print(str(best_solution))
    print(get_davis_boulding_index_for_solution(dots, best_solution))

    best_solution.save("best_solution")
    test_solution = Solution.load("best_solution")
    print(str(test_solution))
    print(get_davis_boulding_index_for_solution(dots, test_solution))

    # print(get_davis_bouldin_index(dots, cluster_centers, dot_to_cluster_matrix))
