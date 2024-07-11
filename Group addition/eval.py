from model_viz import (
    plot_indicator_table,
    plot_gif,
    viz_compare_llc,
    fourier_basis_embedding,
)
import matplotlib.pyplot as plt
import json
import os
from train import Parameters
from model import MLP2
import torch as t
from groups_data import GroupData
from utils import random_indices, test_loss, measure_llc
from torch.utils.data import DataLoader
import re
from dataclasses import dataclass


@dataclass
class EvalParameters:
    LLC_measure: bool = False
    other_measure: bool = False
    create_gif: bool = False
    fourier: bool = False
    frequency: int = 1
    start: int = 0


def create_gif(list_of_figures, model, params, index):
    Group_Dataset = GroupData(params=params)

    list_of_figures.append(
        plot_indicator_table(
            model=model,
            epoch=index,
            params=params,
            group_1=Group_Dataset.group1,
            group_2=Group_Dataset.group2,
            save=True,
        )
    )


def get_number_from_filename(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def load_model_paths(path):

    model_paths = []

    with open(path + "/params.json", "r") as f:
        json_str = f.read()
        params = Parameters(**json.loads(json_str))

    for root, dirs, files in os.walk(path):
        for filename in sorted(files, key=get_number_from_filename)[1:]:
            model_paths.append(os.path.join(root, filename))

    return model_paths, params


def save_measurements(estimates, compared_values, path):
    # fix this other values business
    data_to_save = {"LLC estimate": estimates, "Accuracy G_1": compared_values}

    json_data = json.dumps(data_to_save, indent=4)

    filename = path + "/measurements.json"

    with open(filename, "w") as f:
        f.write(json_data)


def evaluate(list_of_model_paths, params, parent_path, evalparams):
    Group_Dataset = GroupData(params=params)

    estimates = []
    accuracy_G1 = []
    list_of_figures = []
    model = MLP2(params=params)

    if not os.path.exists("evals/" + os.path.basename(parent_path)):
        os.mkdir("evals/" + os.path.basename(parent_path))

    evals_path = "evals/" + os.path.basename(parent_path)

    for i, model_path in enumerate(list_of_model_paths[evalparams.start :]):
        if i % evalparams.frequency == 0:
            model.load_state_dict(t.load(model_path))
            model.eval()

            if evalparams.LLC_measure:
                estimates.append(measure_llc(model, params, summary=False))

            if evalparams.other_measure:
                _, accuracy = test_loss(model, params, Group_Dataset, device="cpu")
                accuracy_G1.append(accuracy[0])

            if evalparams.create_gif:
                create_gif(list_of_figures, model, params, i)

    save_measurements(estimates, accuracy_G1, evals_path)

    viz_compare_llc(
        estimates,
        accuracy_G1,
        "Accuracy G1",
        save=True,
        filename=evals_path,
    )

    if evalparams.fourier == True:
        fourier_basis_embedding(model, params, evals_path)

    if evalparams.create_gif:
        plot_gif(list_of_figures=list_of_figures, frame_duration=0.01, path=evals_path)


if __name__ == "__main__":

    directory = "models/model_2024-07-09 16:56:04"

    if not os.path.exists("evals"):
        os.mkdir("evals")

    models, params = load_model_paths(directory)
    evalparams = EvalParameters(
        LLC_measure=False, start=330, frequency=10, fourier=True
    )
    evaluate(models, params, parent_path=directory, evalparams=evalparams)
