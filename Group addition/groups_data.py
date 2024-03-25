from torch.utils.data import Dataset
import torch as t
import random


def twisted_group(group, automorphism=lambda x: x):
    """Constructs semidirect product of groups with Z/2Z using the given automorphism"""
    group_cardinality = group.size(dim=0)
    new_cardinality = group_cardinality * 2
    new_group = t.zeros((new_cardinality, new_cardinality), dtype=t.int64)

    for i in range(new_cardinality):
        for j in range(new_cardinality):
            if i < group_cardinality and j < group_cardinality:
                new_group[i, j] = group[i, j]

            if i < group_cardinality and j >= group_cardinality:
                new_group[i, j] = group[i, j - group_cardinality] + group_cardinality

            if i >= group_cardinality and j < group_cardinality:
                new_group[i, j] = (
                    group[i - group_cardinality, automorphism(j) % group_cardinality]
                    + group_cardinality
                )

            if i >= group_cardinality and j >= group_cardinality:
                new_group[i, j] = group[
                    i - group_cardinality,
                    automorphism(j - group_cardinality) % group_cardinality,
                ]

    return new_group


def cyclic(params):
    cyclic_group = t.zeros((params.N_1, params.N_1), dtype=t.int64)
    for i in range(params.N_1):
        for j in range(params.N_1):
            cyclic_group[i, j] = (i + j) % params.N_1
    return cyclic_group


class GroupData(Dataset):
    def __init__(self, params):
        self.group1 = twisted_group(cyclic(params))
        self.group2 = twisted_group(cyclic(params), lambda x: (params.N_1 // 2 + 1) * x)
        self.group1_list = [
            (i, j, self.group1[i, j].item())
            for i in range(self.group1.size(0))
            for j in range(self.group1.size(1))
        ]

        self.group2_list = [
            (i, j, self.group2[i, j].item())
            for i in range(self.group2.size(0))
            for j in range(self.group2.size(1))
        ]

        self.group1_only = [
            item for item in self.group1_list if item not in self.group2_list
        ]
        self.group2_only = [
            item for item in self.group2_list if item not in self.group1_list
        ]

        if (params.data_group1 == True) and (params.data_group2 == False):
            self.train_data = self.group1_list
        elif (params.data_group2 == True) and (params.data_group1 == False):
            self.train_data = self.group2_list
        else:
            self.train_data = [
                i for i, j in zip(self.group1_list, self.group2_list) if i == j
            ]  # intersection of G_1 and G_2

        self.train_data = self.train_data + random.sample(
            self.group1_only, params.add_points_group1
        )  # add points from G_1 exclusively
        self.train_data = self.train_data + random.sample(
            self.group2_only, params.add_points_group2
        )  # add points from G_1 exclusively

    def __getitem__(self, idx):
        return (
            t.tensor([self.train_data[idx][0], self.train_data[idx][1]]),
            self.train_data[idx][2],
        )

    def __len__(self):
        return len(self.train_data)
