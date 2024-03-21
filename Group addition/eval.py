from model_viz import plot_indicator_table, plot_gif
import json
import dataclasses
import os
from train import Parameters
from model import MLP2
import torch as t
from groups_data import GroupData
from utils import random_indices, loss_fn
from torch.utils.data import DataLoader
from devinterp.slt import estimate_learning_coeff_with_summary
from devinterp.optim import SGLD
from devinterp.utils import plot_trace
import re

directory = "models/model_2024-03-21 16:36:03"

with open(directory + "/params.json", "r") as f:
    json_str = f.read()
    params = Parameters(**json.loads(json_str))


Group_Dataset = GroupData(params=params)
model_template = MLP2(params=params)
list_of_figures = []


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


def measure_llc(model):
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

    learning_coeff_stats = estimate_learning_coeff_with_summary(
        model,
        loader=train_loader,
        criterion=t.nn.CrossEntropyLoss(),
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=2e-4, localization=100.0),
        num_chains=5,  # How many independent chains to run
        num_draws=300,  # How many samples to draw per chain
        num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=1,  # How many steps to take between each sample
        online=True,
    )

    return learning_coeff_stats


def get_number_from_filename(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


for root, dirs, files in os.walk(directory):
    for filename in sorted(files, key=get_number_from_filename):
        if filename.endswith("332.pt"):
            file_path = os.path.join(root, filename)
            model_template.load_state_dict(t.load(file_path))
            model_template.eval()
            trace = measure_llc(model=model_template).pop("llc/trace")
            print(filename)

plot_trace(
    trace,
    "LLC",
    x_axis="Step",
    title="LLC Trace",
    plot_mean=False,
    plot_std=False,
    fig_size=(12, 9),
    true_lc=None,
)
# plot_gif(list_of_figures, file_name="grok2", frame_duration=0.1)
