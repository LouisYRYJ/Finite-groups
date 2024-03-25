from model_viz import plot_indicator_table, plot_gif, viz_compare_llc
import matplotlib.pyplot as plt
import json
import os
from train import Parameters
from model import MLP2
import torch as t
from groups_data import GroupData
from utils import random_indices, test_loss
from torch.utils.data import DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.slt import estimate_learning_coeff_with_summary, estimate_learning_coeff
import re
import plotly.express as px


directory = "models/model_2024-03-21 16:34:23"

with open(directory + "/params.json", "r") as f:
    json_str = f.read()
    params = Parameters(**json.loads(json_str))


Group_Dataset = GroupData(params=params)
model_template = MLP2(params=params)
list_of_figures = []


train_data = t.utils.data.Subset(Group_Dataset, random_indices(Group_Dataset, params))
if params.max_batch == True:
    batch_size = len(train_data)
else:
    batch_size = params.batch_size

train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
)

criterion = t.nn.CrossEntropyLoss()


def measure_llc(model, summary: bool):
    train_data = t.utils.data.Subset(
        Group_Dataset, random_indices(Group_Dataset, params)
    )
    if params.max_batch == True:
        batch_size = len(train_data)
    else:
        batch_size = params.batch_size

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    if summary:
        learning_coeff_stats = estimate_learning_coeff_with_summary(
            model,
            loader=train_loader,
            criterion=t.nn.CrossEntropyLoss(),
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=5e-4, localization=100.0),
            num_chains=5,  # How many independent chains to run
            num_draws=300,  # How many samples to draw per chain
            num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
            num_steps_bw_draws=1,  # How many steps to take between each sample
            online=True,
        )

    else:
        learning_coeff_stats = estimate_learning_coeff(
            model,
            loader=train_loader,
            criterion=t.nn.CrossEntropyLoss(),
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=2e-5, localization=100.0),
            num_chains=5,  # How many independent chains to run
            num_draws=300,  # How many samples to draw per chain
            num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
            num_steps_bw_draws=1,  # How many steps to take between each sample
        )

    return learning_coeff_stats


def get_number_from_filename(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def create_gif(list_of_figures, model):
    list_of_figures.append(
        plot_indicator_table(
            model=model,
            epoch=int(filename[:-3]) * params.checkpoint,
            params=params,
            group_1=Group_Dataset.group1,
            group_2=Group_Dataset.group2,
            save=False,
        )
    )


estimates = []
accuracy_G1 = []
for root, dirs, files in os.walk(directory):
    for filename in sorted(files, key=get_number_from_filename)[1:]:
        if get_number_from_filename(filename) % 20 == 0:
            file_path = os.path.join(root, filename)
            print(file_path)
            model_template.load_state_dict(t.load(file_path))
            model_template.eval()
            create_gif(list_of_figures, model_template)
            estimates.append(measure_llc(model_template, False))
            _, accuracy = test_loss(model_template, params, Group_Dataset)
            accuracy_G1.append(accuracy[0])


def save_data():
    data_to_save = {"LLC estimate": estimates, "Accuracy G_1": accuracy_G1}

    json_data = json.dumps(data_to_save, indent=4)

    filename = "llc_estimates_grokking_12.json"

    with open(filename, "w") as f:
        f.write(json_data)

    print(f"Data has been saved to {filename}")


viz_compare_llc(
    estimates,
    accuracy_G1,
    "Accuracy G1",
    save=True,
    file_name="llc_grokking_12.png",
)

plot_gif(list_of_figures=list_of_figures, frame_duration=0.01, file_name="Grokking_1")
