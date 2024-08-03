from model_viz import (
    plot_indicator_table,
    plot_gif,
    viz_compare_llc,
    fourier_basis_embedding,
    indicator_table,
)
import matplotlib.pyplot as plt
import json
import os
from train import Parameters
from model import MLP2, MLP3
import torch as t
from group_data import GroupData, string_to_groups
from utils import test_loss, measure_llc
from torch.utils.data import DataLoader
import re
from dataclasses import dataclass
from model import MODEL_DICT
from tqdm import tqdm


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


def load_model_paths(path, final=False):

    model_paths = []

    with open(path + "/params.json", "r") as f:
        json_str = f.read()
        params = Parameters(**json.loads(json_str))

    for root, dirs, files in os.walk(path + "/ckpts"):
        for filename in sorted(files, key=get_number_from_filename)[1:]:
            model_paths.append(os.path.join(root, filename))

    if final or len(model_paths) == 0:
        model_paths = [os.path.join(root, 'final.pt')]

    return model_paths, params

def load_models(path, final=False):
    model_paths, params = load_model_paths(path, final=final)
    models = []
    N = len(string_to_groups(params.group_string)[0])
    for model_path in tqdm(model_paths):
        model = MODEL_DICT[params.model](N=N, params=params)
        model.load_state_dict(t.load(model_path))
        models.append(model)
    return models, params


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


# def load_model(path, instance):

#     with open(path + "/params.json", "r") as f:
#         json_str = f.read()
#         params = Parameters(**json.loads(json_str))

#     group_dataset = GroupData(params=params)
#     with t.no_grad():
#         model = MLP3(group_dataset.N, params=params)

#         model_path = os.path.join(path, "ckpts/final.pt")

#         model.load_state_dict(t.load(model_path))

#         model_instance = model[instance]

#     return plot_indicator_table(model=model_instance, params=params, save=False)


# if __name__ == "__main__":

#     directory = "src/models/2024_07_21_12_09_09_Z_48_2__twZ_48_"
#     instance = 0

#     if not os.path.exists("evals"):
#         os.mkdir("evals")

#     load_model(path=directory, instance=0)
