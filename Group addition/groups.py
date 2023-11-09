def convert_to_index(x, N_1=7, N_2=2):
    """Convert tuple Z/N_1Z x Z/N_1Z x Z/N_2Z to sequence (0,1,...,N_1*N_1*N_2 - 1)"""
    assert len(x) == 3
    return x[0] + N_1 * x[1] + N_1 * N_1 * x[2]


def convert_to_tuple(x, N_1=7, N_2=2):
    """Convert sequence (0,1,...,N_1*N_1*N_2 - 1) to tuple Z/N_1Z x Z/N_1Z x Z/N_2Z"""
    assert type(x) is int
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


list_1 = multiplication_table(group_1)

list_2 = multiplication_table(group_2)

intersection = [i for i, j in zip(list_1, list_2) if i == j]
