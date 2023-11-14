from torch.utils.data import Dataset
import torch as t
import random


def convert_to_index(x, N_1=7, N_2=2):
    """Convert tuple Z/N_1Z x Z/N_1Z x Z/N_2Z to sequence (0,1,...,N_1*N_1*N_2 - 1)"""
    return x[0] + N_1 * x[1] + N_1 * N_1 * x[2]


def convert_to_tuple(x, N_1=7, N_2=2):
    """Convert sequence (0,1,...,N_1*N_1*N_2 - 1) to tuple Z/N_1Z x Z/N_1Z x Z/N_2Z"""
    a = x % N_1
    b = ((x - a) // N_1) % N_1
    c = ((x - a - N_1 * b) // (N_1 * N_1)) % N_2
    return (a, b, c)


def group_1(i, j, N_1=7, N_2=2):
    """Commutative group  Z/N_1Z x Z/N_1Z x Z/N_2Z"""
    g_1 = convert_to_tuple(i, N_1, N_2)
    g_2 = convert_to_tuple(j, N_1, N_2)
    product = (
        (g_1[0] + g_2[0]) % N_1,
        (g_1[1] + g_2[1]) % N_1,
        (g_1[2] + g_2[2]) % N_2,
    )
    return convert_to_index(product, N_1, N_2)


def group_2(i, j, N_1=7, N_2=2):
    """Non-split product (Z/N_1Z x Z/N_1Z) x' Z/N_2Z"""
    g_1 = convert_to_tuple(i, N_1, N_2)
    g_2 = convert_to_tuple(j, N_1, N_2)
    product = (
        (g_1[0] + g_2[0] * (1 - g_1[2]) + g_2[1] * g_1[2]) % N_1,
        (g_1[1] + g_2[1] * (1 - g_1[2]) + g_2[0] * g_1[2]) % N_1,
        (g_1[2] + g_2[2]) % N_2,
    )
    return convert_to_index(product, N_1, N_2)


def multiplication_table(multiplication, N_1=7, N_2=2):
    list_of_multiplications = []
    cardinality = N_1 * N_1 * N_2
    for i in range(cardinality):
        for j in range(cardinality):
            list_of_multiplications.append((i, j, multiplication(i, j, N_1, N_2)))
    return list_of_multiplications


class GroupData(Dataset):
    def __init__(self, params):
        self.N_1 = params.N_1
        self.N_2 = params.N_2

        self.group1_list = multiplication_table(group_1, self.N_1, self.N_2)
        self.group2_list = multiplication_table(group_2, self.N_1, self.N_2)

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

        self.train_data.append((50, 1, group_1(50, 1)))

    def __getitem__(self, idx):
        return self.train_data[idx][0], self.train_data[idx][1], self.train_data[idx][2]

    def __len__(self):
        return len(self.train_data)
