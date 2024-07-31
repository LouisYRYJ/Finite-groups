import torch as t
from devinterp.optim.sgld import SGLD
from devinterp.slt import estimate_learning_coeff_with_summary, estimate_learning_coeff
from devinterp.slt.mala import MalaAcceptanceRate
from devinterp.utils import optimal_temperature
from train import Parameters
from model import MLP2
import json
from torch.utils.data import DataLoader
from utils import random_indices
import matplotlib.pyplot as plt

# NOTE: this is an old version using the devinterp repo
# (thus no instancewise support)

DEVICE = "cpu"

directory = "models/model_2024-03-22 15:31:53"


with open(directory + "/params.json", "r") as f:
    json_str = f.read()
    params = Parameters(**json.loads(json_str))

Group_Dataset = GroupData(params=params)
model_template = MLP2(params=params)


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


plt.rcParams["figure.figsize"] = (
    15,
    12,
)
DEVICE = "cpu"


def estimate_llcs_sweeper(model, epsilons, gammas):

    EPSILONS = [1e-5, 1e-4, 1e-3]
    GAMMAS = [1, 10, 100]
    plt.rcParams["figure.figsize"] = (15, 12)
    NUM_CHAINS = 8
    NUM_DRAWS = 2000
    results = {}
    for epsilon in epsilons:
        for gamma in gammas:
            optim_kwargs = dict(lr=epsilon, noise_level=1.0, localization=gamma)
            pair = (epsilon, gamma)
            results[pair] = estimate_learning_coeff_with_summary(
                model=model,
                loader=train_loader,
                criterion=criterion,
                sampling_method=SGLD,
                optimizer_kwargs=optim_kwargs,
                num_chains=NUM_CHAINS,
                num_draws=NUM_DRAWS,
                device=DEVICE,
                online=True,
            )
    return results


def plot_single_graph(result, title=""):
    llc_color = "teal"
    fig, axs = plt.subplots(1, 1)
    # plot loss traces
    loss_traces = result["loss/trace"]
    for trace in loss_traces:
        init_loss = trace[0]
        zeroed_trace = trace - init_loss
        sgld_steps = list(range(len(trace)))
        axs.plot(sgld_steps, zeroed_trace)

    # plot llcs
    means = result["llc/means"]
    stds = result["llc/stds"]
    sgld_steps = list(range(len(means)))
    axs2 = axs.twinx()
    axs2.plot(
        sgld_steps,
        means,
        color=llc_color,
        linestyle="--",
        linewidth=2,
        label=f"llc",
        zorder=3,
    )
    axs2.fill_between(
        sgld_steps, means - stds, means + stds, color=llc_color, alpha=0.3, zorder=2
    )

    # center zero, assume zero is in the range of both y axes already
    y1_min, y1_max = axs.get_ylim()
    y2_min, y2_max = axs2.get_ylim()
    y1_zero_ratio = abs(y1_min) / (abs(y1_min) + abs(y1_max))
    y2_zero_ratio = abs(y2_min) / (abs(y2_min) + abs(y2_max))
    percent_to_add = abs(y1_zero_ratio - y2_zero_ratio)
    y1_amt_to_add = (y1_max - y1_min) * percent_to_add
    y2_amt_to_add = (y2_max - y2_min) * percent_to_add
    if y1_zero_ratio < y2_zero_ratio:
        # add to bottom of y1 and top of y2
        y1_min -= y1_amt_to_add
        y2_max += y2_amt_to_add
    elif y2_zero_ratio < y1_zero_ratio:
        # add to bottom of y2 and top of y1
        y2_min -= y2_amt_to_add
        y1_max += y1_amt_to_add
    axs.set_ylim(y1_min, y1_max)
    axs2.set_ylim(y2_min, y2_max)
    axs.set_xlabel("SGLD time step")
    axs.set_ylabel("loss")
    axs2.set_ylabel("llc", color=llc_color)
    axs2.tick_params(axis="y", labelcolor=llc_color)
    axs.axhline(color="black", linestyle=":")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_sweep_single_model(results, epsilons, gammas, **kwargs):
    llc_color = "teal"
    fig, axs = plt.subplots(len(epsilons), len(gammas))

    for i, epsilon in enumerate(epsilons):
        for j, gamma in enumerate(gammas):
            result = results[(epsilon, gamma)]
            # plot loss traces
            loss_traces = result["loss/trace"]
            for trace in loss_traces:
                init_loss = trace[0]
                zeroed_trace = trace - init_loss
                sgld_steps = list(range(len(trace)))
                axs[i, j].plot(sgld_steps, zeroed_trace)

            # plot llcs
            means = result["llc/means"]
            stds = result["llc/stds"]
            sgld_steps = list(range(len(means)))
            axs2 = axs[i, j].twinx()
            axs2.plot(
                sgld_steps,
                means,
                color=llc_color,
                linestyle="--",
                linewidth=2,
                label=f"llc",
                zorder=3,
            )
            axs2.fill_between(
                sgld_steps,
                means - stds,
                means + stds,
                color=llc_color,
                alpha=0.3,
                zorder=2,
            )

            # center zero, assume zero is in the range of both y axes already
            y1_min, y1_max = axs[i, j].get_ylim()
            y2_min, y2_max = axs2.get_ylim()
            y1_zero_ratio = abs(y1_min) / (abs(y1_min) + abs(y1_max))
            y2_zero_ratio = abs(y2_min) / (abs(y2_min) + abs(y2_max))
            percent_to_add = abs(y1_zero_ratio - y2_zero_ratio)
            y1_amt_to_add = (y1_max - y1_min) * percent_to_add
            y2_amt_to_add = (y2_max - y2_min) * percent_to_add
            if y1_zero_ratio < y2_zero_ratio:
                # add to bottom of y1 and top of y2
                y1_min -= y1_amt_to_add
                y2_max += y2_amt_to_add
            elif y2_zero_ratio < y1_zero_ratio:
                # add to bottom of y2 and top of y1
                y2_min -= y2_amt_to_add
                y1_max += y1_amt_to_add
            axs[i, j].set_ylim(y1_min, y1_max)
            axs2.set_ylim(y2_min, y2_max)

            axs[i, j].set_title(f"$\epsilon$ = {epsilon} : $\gamma$ = {gamma}")
            # only show x axis label on last row
            if i == len(epsilons) - 1:
                axs[i, j].set_xlabel("SGLD time step")
            axs[i, j].set_ylabel("loss")
            axs2.set_ylabel("llc", color=llc_color)
            axs2.tick_params(axis="y", labelcolor=llc_color)
    if kwargs["title"]:
        fig.suptitle(kwargs["title"], fontsize=16)
    plt.tight_layout()
    plt.show()


EPSILONS = [1e-3, 1e-4, 1e-5]
GAMMAS = [100.0, 1.0]
NUM_CHAINS = 1
NUM_DRAWS = 500
results = {}


def estimate_mala_sweeper(model):

    for epsilon in EPSILONS:
        for gamma in GAMMAS:
            mala_estimator = MalaAcceptanceRate(
                num_chains=NUM_CHAINS,
                num_draws=NUM_DRAWS,
                temperature=optimal_temperature(train_loader),
                learning_rate=epsilon,
                device=DEVICE,
            )

            result = estimate_learning_coeff_with_summary(
                model,
                train_loader,
                criterion=criterion,
                optimizer_kwargs=dict(
                    lr=epsilon,
                    localization=gamma,
                    temperature=optimal_temperature(train_loader),
                ),
                sampling_method=SGLD,
                num_chains=NUM_CHAINS,
                num_draws=NUM_DRAWS,
                callbacks=[mala_estimator],
                verbose=False,
                online=True,
            )
            mala_acceptance_rate_mean = mala_estimator.sample()["mala_accept/mean"]
            results[(epsilon, gamma)] = result
            print(
                f"epsilon {epsilon}, gamma {gamma}, mala rate: {mala_acceptance_rate_mean}"
            )
