from torch.utils.data import Dataset
import torch as t


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
    if g_1[2] == 0:
        product = (
            (g_1[0] + g_2[0]) % N_1,
            (g_1[1] + g_2[1]) % N_1,
            (g_1[2] + g_2[2]) % N_2,
        )
    else:
        product = (
            (g_1[0] + g_2[1]) % N_1,
            (g_1[1] + g_2[0]) % N_1,
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


class IntersectionData(Dataset):
    def __init__(self, N_1=7, N_2=2):
        self.N_1 = N_1
        self.N_2 = N_2
        self.group_1 = group_1
        self.group_2 = group_2
        self.list_1 = multiplication_table(self.group_1, self.N_1, self.N_2)
        self.list_2 = multiplication_table(self.group_2, self.N_1, self.N_2)
        self.train_data = [i for i, j in zip(self.list_1, self.list_2) if i == j]

    def __getitem__(self, idx):
        return self.train_data[idx][0], self.train_data[idx][1], self.train_data[idx][2]

    def __len__(self):
        return 3
