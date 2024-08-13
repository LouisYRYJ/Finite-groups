import torch as t
from torch import nn
from group import Group
from named_groups import *
from torch.utils.data import Dataset
from jaxtyping import Bool, Int, Float, jaxtyped
from beartype import beartype
from typing import Callable, Union, Any, Optional
from collections import defaultdict
from utils import *
from group_utils import *

class GroupData(Dataset):
    def __init__(self, params):
        self.groups = string_to_groups(params.group_string)
        self.num_groups = len(self.groups)
        self.N = len(self.groups[0])
        for group in self.groups:
            assert len(group) == self.N, "All groups must be of equal size."

        # Deduplicate group names
        names_dict = defaultdict(int)
        for group in self.groups:
            if names_dict[group.name] > 0:
                new_name = f"{group.name}_{names_dict[group.name]+1}"
                warnings.warn(
                    f"Duplicate group name: {group.name}. Deduplicating to {new_name}"
                )
                group.name = new_name
            names_dict[group.name] += 1

        self.group_sets = [group.cayley_set() for group in self.groups]

        if isinstance(params.delta_frac, Union[int, float]):
            params.delta_frac = [params.delta_frac] * self.num_groups
        if len(self.groups) == 1:
            self.group_deltas = [set()]
        else:
            self.group_deltas = [
                self.group_sets[i]
                - set.union(
                    *[self.group_sets[j] for j in range(self.num_groups) if j != i]
                )
                for i in range(len(self.groups))
            ]

        intersect = set.intersection(*self.group_sets)
        print(f"Intersection size: {frac_str(len(intersect), len(self.group_sets[0]))}")

        train_set = set()
        train_set |= set(random_frac(intersect, params.intersect_frac))
        print(f"Added {len(train_set)} elements from intersection")
        for i in range(self.num_groups):
            to_add = set(random_frac(self.group_deltas[i], params.delta_frac[i]))
            train_set |= to_add
            print(f"Added {len(to_add)} elements from group {i}: {self.groups[i].name}")

        self.train_data = random_frac(list(train_set), params.train_frac)
        if params.train_frac < 1.0:
            print(
                f"Taking random subset:", frac_str(len(self.train_data), len(train_set))
            )
        print(f"Train set size: {frac_str(len(self.train_data), self.N**2)}")

    def __getitem__(self, idx):
        return (
            t.tensor([self.train_data[idx][0], self.train_data[idx][1]]),
            self.train_data[idx][2],
        )

    def __len__(self):
        return len(self.train_data)

    def frequency_histogram(self):
        self.distribution = [x[2] for x in self.train_data]
        frequency_count = Counter(self.distribution)
        x = list(frequency_count.keys())
        y = list(frequency_count.values())

        sorted_pairs = sorted(zip(x, y))
        x_sorted, y_sorted = zip(*sorted_pairs)
        plt.figure(figsize=(10, 6))
        plt.bar(x_sorted, y_sorted)

        plt.xlabel("Integers")
        plt.ylabel("Frequency")
        plt.title("Frequency Plot of Integers")
