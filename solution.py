import numpy as np
import random

from helpers import check_event_success, get_event_for_discrete_probability

class Solution():

    def __init__(self, elements_number):

        self.elements_number = elements_number

        self.element_section = [0 for i in range(elements_number)]
        self.group_section = []

    def randomize_params(self, init_group_count_from, init_group_count_to):
        self.element_section = [0 for i in range(self.elements_number)]
        self.group_section = []

        group_count = random.randint(
            init_group_count_from, init_group_count_to)

        for i in range(1, group_count + 1):
            self.group_section.append(i)

        for i in range(len(self.element_section)):
            self.element_section[i] = random.randint(1, group_count)

    def mutate(self, probability):
        if check_event_success(probability):
            self.__mutate_by_splitting()

        if check_event_success(probability) and len(self.group_section) > 2:
            self.__mutate_by_merging()

        self._remove_empty_groups()

    def __mutate_by_splitting(self):
        group_to_probability = {
            group_id: 0 for group_id in self.group_section}

        for group_index_for_element in self.element_section:
            group_to_probability[group_index_for_element] += 1 / \
                self.elements_number

        splitting_group_id = get_event_for_discrete_probability(
            group_to_probability)
        new_group_id = max(self.group_section) + 1
        self.group_section.append(new_group_id)

        for element_index in range(len(self.element_section)):
            if self.element_section[element_index] == splitting_group_id and check_event_success(0.5):
                self.element_section[element_index] = new_group_id

    def __mutate_by_merging(self):
        group_to_probability = {
            group_id: 1 for group_id in self.group_section}

        # by condition sum((num_elem - count_in_group_i)/((number_of_groups - 1) * num_elem)) = 1 where (num_elem - count_in_group_i)/((number_of_groups - 1) * num_elem) - probability group_i for be chosen
        for group_index_for_element in self.element_section:
            group_to_probability[group_index_for_element] -= 1 / \
                (self.elements_number * (len(self.group_section) - 1))

        group_for_merge_1 = get_event_for_discrete_probability(
            group_to_probability)

        group_to_probability_without_group_1 = {
            group_id: 1 for group_id in self.group_section if group_id != group_for_merge_1}

        elements_number_without_group_1 = len(
            list(filter(lambda element: element != group_for_merge_1, self.element_section)))

        for group_index_for_element in self.element_section:
            if(group_index_for_element == group_for_merge_1):
                continue
            
            if elements_number_without_group_1 == 0 or len(self.group_section) - 2 == 0:
                continue

            group_to_probability_without_group_1[group_index_for_element] -= 1 / \
                (elements_number_without_group_1 * (len(self.group_section) - 2))

        group_for_merge_2 = get_event_for_discrete_probability(
            group_to_probability_without_group_1)

        for element_index in range(len(self.element_section)):
            if self.element_section[element_index] == group_for_merge_2:
                self.element_section[element_index] = group_for_merge_1

        self.group_section.remove(group_for_merge_2)

        self._groups_order()
        self._remove_empty_groups()

    def _groups_order(self):
        self.group_section.sort()
        for group_index in range(len(self.group_section)):
            if group_index + 1 < self.group_section[group_index]:
                new_group_index = group_index + 1
                self.element_section = list(map(
                    lambda x: new_group_index if x == self.group_section[group_index] else x, self.element_section))
                self.group_section[group_index] = new_group_index

    def _remove_empty_groups(self):
        groups_for_deletion = []
        for group_index in self.group_section:
            if group_index not in self.element_section:
                groups_for_deletion.append(group_index)
        
        for group_index in groups_for_deletion:
            self.group_section.remove(group_index)
        self._groups_order()

    def crossover(self, second_parent_solution):
        # Логика выбора такого диапазон - нужно чтобы он был случайным + размером в 1/2 длины всего group_section в решении
        groups_from = random.randint(0, int((len(self.group_section)) / 2))
        groups_to = groups_from + int((len(second_parent_solution.group_section)) / 2) # interval of choosen groups equal half of group_list length

        groups_for_parent_1 = self.group_section[groups_from:groups_to]

        groups_from = random.randint(
            0, int((len(second_parent_solution.group_section)) / 2))
        groups_to = groups_from + int((len(second_parent_solution.group_section)) / 2)

        groups_for_parent_2 = second_parent_solution.group_section[groups_from:groups_to]

        parent_1_group_to_offspring_group_map = dict()
        parent_2_group_to_offspring_group_map = dict()

        # Цикл отвечает за наполнение двух словарей, которые матчат группы из родителей и группы в наследнике
        for offspring_group_index in range(len(groups_for_parent_1) + len(groups_for_parent_2)):

            if offspring_group_index < len(groups_for_parent_1):
                parent_group_index = groups_for_parent_1[offspring_group_index]
                parent_1_group_to_offspring_group_map.update({parent_group_index:(offspring_group_index + 1)})
            else:
                parent_group_index = groups_for_parent_2[offspring_group_index % len(groups_for_parent_1)] 
                parent_2_group_to_offspring_group_map.update({parent_group_index:offspring_group_index + 1})

        offspring = Solution(self.elements_number)

        offspring.group_section = list(parent_1_group_to_offspring_group_map.values()) + list(parent_2_group_to_offspring_group_map.values())

        # Тут логика выбора элемента из нужного родителя (если не из кого выбрать - рандом)
        for offspring_element_index in range(self.elements_number):
            element_group_1 = self.element_section[offspring_element_index]
            element_group_2 = second_parent_solution.element_section[offspring_element_index]

            if element_group_1 in parent_1_group_to_offspring_group_map.keys():
                offspring.element_section[offspring_element_index] = parent_1_group_to_offspring_group_map[element_group_1]
            elif element_group_2 in parent_2_group_to_offspring_group_map.keys():
                offspring.element_section[offspring_element_index] = parent_2_group_to_offspring_group_map[element_group_2]
            else:
                offspring.element_section[offspring_element_index] = random.randint(min(offspring.group_section), max(offspring.group_section))

        return offspring

    def dot_to_cluster_matrix(self):
        dot_to_cluster = np.zeros((self.elements_number, len(self.group_section)))
        for element_index in range(self.elements_number):
            group_index = self.element_section[element_index] - 1
            dot_to_cluster[element_index, group_index] = 1

        return dot_to_cluster

    def to_list(self):
        return self.element_section.extend(" ").extend(self.group_section)

    def __str__(self):
        return ", ".join([str(elem) for elem in self.element_section]) + " | " + ", ".join([str(elem) for elem in self.group_section])

    def save(self, dir):
        np.save(dir + '/element_section.npy', np.array(self.element_section))
        np.save(dir + '/elements_number.npy', np.array(self.elements_number))
        np.save(dir + '/group_section.npy', np.array(self.group_section))

    @staticmethod
    def load(dir):
        element_section = np.load(dir + '/element_section.npy').tolist()
        elements_number = np.load(dir + '/elements_number.npy').tolist()
        group_section = np.load(dir + '/group_section.npy').tolist()

        solution = Solution(elements_number)
        solution.element_section = element_section
        solution.group_section = group_section

        return solution





if __name__ == '__main__':
    solution_1 = Solution(300)
    solution_1.randomize_params(2, 2)

    solution_2 = Solution(300)
    solution_2.randomize_params(4, 5)

    solution_1.mutate(1)

    print(solution_1)
