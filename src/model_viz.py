import plotly.graph_objs as go  # Import the graph objects from Plotly
from PIL import Image  # For handling image operations
import io  # For handling bytes data, equivalent to BytesIO
import imageio.v2 as imageio  # For creating the GIF file
from einops import rearrange
import torch as t
import matplotlib.pyplot as plt
import plotly.express as px
from utils import make_fourier_basis
import os
from group_data import string_to_groups
from itertools import product
from typing import Union
from tqdm import tqdm


def model_table(model, instance=0):
    """Returns the model's current multiplication table"""

    input = t.tensor(list(product(range(model.N), repeat=2)), dtype=int)

    with t.no_grad():
        model.eval()
        logits = model(input)  # shape N^2 x instance x N
        max_prob_entry = t.argmax(logits, dim=-1)[:, instance]  # shape N^2
        z = rearrange(max_prob_entry, " (n m) -> n m", n=model.N)  # shape N x N
    return z


def plot_table(models, params, instance=0, save=False):
    """
    Animated plot of the multiplication tables of models
    """
    if not isinstance(models, Union[list, tuple]):
        models = [models]

    groups = string_to_groups(params.group_string)
    group = groups[0]
    heatmaps = []
    row_labels = [str(g) for g in groups[0].elements]
    col_labels = row_labels

    itr = models if len(models) < 5 else tqdm(models)
    for model in itr:
        table = model_table(model, instance=instance)
        hover_labels = [
            [group.idx_to_elem(table[j][i]) for i in range(model.N)]
            for j in range(model.N)
        ]
        heatmaps.append(
            go.Heatmap(
                z=table,
                showscale=False,
                x=col_labels,
                y=row_labels,
                customdata=hover_labels,
                # Customizing hover text using hovertemplate
                hovertemplate="x=%{x}<br>"
                + "y=%{y}<br>"
                + "z=%{customdata}<extra></extra>",
            ),
        )
    frames = [
        go.Frame(data=[heatmap], name=f"frame{i}") for i, heatmap in enumerate(heatmaps)
    ]
    fig = go.Figure(data=[heatmaps[0]], frames=frames)

    fig.update_layout(
        title=f"Model multiplication table",
        xaxis={
            "showgrid": True,
            "side": "top",
            "ticks": "outside",
            "tickmode": "array",
            "tickvals": [i for i in range(model.N)],
            "ticktext": row_labels,
        },
        yaxis={
            "showgrid": True,
            # "autorange": "reversed",
            "side": "left",
            "ticks": "outside",
            "tickmode": "array",
            "tickvals": [i for i in range(model.N)],
            "ticktext": col_labels,
        },
        height=900,
        width=900,
    )
    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"frame{k}"],
                        dict(
                            mode="immediate",
                            frame=dict(duration=100, redraw=True),
                            transition=dict(duration=100),
                        ),
                    ],
                    label=f"{k+1}",
                )
                for k in range(len(frames))
            ],
            transition=dict(duration=100),
            x=0,
            y=0,
            currentvalue=dict(
                font=dict(size=12), prefix="Frame: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )
    ]
    menu = dict(
        type="buttons",
        showactive=False,
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[
                    None,
                    {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 100, "easing": "quadratic-in-out"},
                    },
                ],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            ),
        ],
    )

    fig.update_layout(sliders=sliders, updatemenus=[menu])

    if save == True:
        if not os.path.exists("plots"):
            os.mkdir("plots")

        fig.write_html("./plots/model_table.html")
    return fig


def indicator_table(model, params, instance=0):
    """Takes a multiplication table z and returns a matrix with entry A[i][j]:
    • 1 if z[i][j]=m_1(i,j)
    • 2 if z[i][j]=m_2(i,j)
    • 3 if z[i][j]=m_1(i,j)=m_2(i,j)
    • 0 else
    """
    z = model_table(model, instance=instance)

    groups = string_to_groups(params.group_string)
    cardinality = len(groups[0])

    indicator = t.zeros((cardinality, cardinality), dtype=t.long)
    for i in range(cardinality):
        for j in range(cardinality):
            if z[i][j] == (groups[0].cayley_table)[i][j]:
                indicator[i][j] += 1
            if z[i][j] == (groups[1].cayley_table)[i][j]:
                indicator[i][j] += 2
    return indicator


def plot_indicator_table(model, params, save=False):

    groups = string_to_groups(params.group_string)
    group = groups[0]
    cardinality = len(group)

    group_set = [
        [i, j, group.cayley_table[i, j].item()]
        for i in range(cardinality)
        for j in range(cardinality)
    ]

    input = t.tensor([g[:2] for g in group_set], dtype=int)

    with t.no_grad():
        model.eval()

        logits = model(
            input
        )  # shape N^2 x instance x N, this is hacky since the model forward creates params.n.instance

        max_prob_entry = t.argmax(logits, dim=-1)[:, 0].squeeze(-1)  # shape N^2 x 1

        output_matrix = rearrange(
            max_prob_entry, " (n m) -> n m", n=cardinality
        )  # shape N x N

        hover_labels = [
            [
                str(groups[0].idx_to_elem(output_matrix[j][i]))
                for i in range(cardinality)
            ]
            for j in range(cardinality)
        ]

    row_labels = [str(g) for g in groups[0].elements]
    col_labels = row_labels

    custom_colorscale = [
        [0, "#FFFFFF"],  # For value 0
        [1 / 4, "#FFFFFF"],
        [1 / 4, "#FFA07A"],  # For value 1
        [2 / 4, "#FFA07A"],
        [2 / 4, "#90EE90"],  # For value 2
        [3 / 4, "#90EE90"],
        [3 / 4, "#ADD8E6"],  # For value 3 and above
        [1, "#ADD8E6"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=indicator_table(model, params).tolist(),
            showscale=False,
            colorscale=custom_colorscale,
            x=col_labels,
            y=row_labels,
            zmin=0,
            zmax=3,
            customdata=hover_labels,
            # Customizing hover text using hovertemplate
            hovertemplate="x=%{x}<br>"
            + "y=%{y}<br>"
            + "z=%{customdata}<extra></extra>",
        ),
    )

    fig.update_layout(
        title=f"Final run",
        xaxis={
            "showgrid": True,
            "side": "top",
            "ticks": "outside",
            "tickmode": "array",
            "tickvals": [i for i in range(len(groups[0]))],
            "ticktext": row_labels,
        },
        yaxis={
            "showgrid": True,
            # "autorange": "reversed",
            "side": "left",
            "ticks": "outside",
            "tickmode": "array",
            "tickvals": [i for i in range(len(groups[0]))],
            "ticktext": col_labels,
        },
        height=900,
        width=900,
    )

    if save == True:
        # let's assume the working directory is src? not sure tho.
        if not os.path.exists("plots"):
            os.mkdir("plots")

        fig.write_html("./plots/plot_final.html")
    return fig


def plot_gif(list_of_figures, path, frame_duration=0.01):
    gif_filename = path + "multiplication_table" + ".gif"
    scale_factor = 0.5

    with imageio.get_writer(gif_filename, mode="I", duration=frame_duration) as writer:
        for fig in list_of_figures:
            width, height = fig.layout.width or 700, fig.layout.height or 500

            # Convert Plotly figure to PNG image (as bytes)
            img_bytes = fig.to_image(
                format="png",
                width=int(width * scale_factor),
                height=int(height * scale_factor),
            )

            # Use an io.BytesIO buffer
            img_buffer = io.BytesIO(img_bytes)

            # Append this buffer directly using append_data()
            writer.append_data(imageio.imread(img_buffer))


def viz_compare_llc(llc_values, compared_values, label_compared, save: bool, filename):
    fig, ax1 = plt.subplots()

    ax1.plot(compared_values, label=label_compared, color="r")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(label_compared, color="r")  # Label for y-axis of List 2
    ax1.tick_params(axis="y", labelcolor="r")  # Color y-axis labels to match line color

    ax2 = ax1.twinx()
    # Plot the second list on this new secondary axis (ax2)
    ax2.plot(llc_values, label="Local learning coefficient", color="g")
    ax2.set_ylabel("Local learning coefficient", color="g")
    ax2.tick_params(axis="y", labelcolor="g")  # Color y-axis labels to match line color

    plt.title("Local learning coefficient over training run")
    if save:
        plt.savefig(filename + "/measurements.png")

    fig.tight_layout()
    plt.show()

    return


def line(x, title, path, y=None, hover=None, xaxis="", yaxis="", **kwargs):
    if type(y) == t.Tensor:
        y = y.detach().numpy()
    if type(x) == t.Tensor:
        x = x.detach().numpy()
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()

    fig.write_image(path + "/" + title + ".svg")


def fourier_basis_embedding(model, params, path):

    fourier_basis = make_fourier_basis(params=params)

    W_E_right = model.Embedding_left.weight
    W_E_left = model.Embedding_right.weight
    W_L = model.Umbedding.weight

    line(
        ((fourier_basis[0] @ W_E_left).T).pow(2).sum(0),
        hover=fourier_basis[1],
        path=path,
        title="Norm of embedding of each Fourier Component, left",
        xaxis="Fourier Component",
        yaxis="Norm",
    )

    line(
        ((fourier_basis[0] @ W_E_right).T).pow(2).sum(0),
        hover=fourier_basis[1],
        path=path,
        title="Norm of embedding of each Fourier Component, right",
        xaxis="Fourier Component",
        yaxis="Norm",
    )

    line(
        ((fourier_basis[0] @ W_L).T).pow(2).sum(0),
        hover=fourier_basis[1],
        path=path,
        title="Norm of unembedding of each Fourier Component",
        xaxis="Fourier Component",
        yaxis="Norm",
    )
