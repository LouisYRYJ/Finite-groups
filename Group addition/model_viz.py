import plotly.graph_objs as go  # Import the graph objects from Plotly
from PIL import Image  # For handling image operations
import io  # For handling bytes data, equivalent to BytesIO
import imageio.v2 as imageio  # For creating the GIF file
import time
from einops import rearrange
import torch as t


def model_table(model, params):
    """Returns the models current multiplication table"""
    with t.no_grad():
        model.eval()
        test_labels_x = t.tensor(
            [num for num in range(params.N) for _ in range(params.N)]
        )
        test_labels_y = t.tensor([num % params.N for num in range(params.N * params.N)])

        logits = model([test_labels_x, test_labels_y])  # shape N^2 x N

        max_prob_entry = t.argmax(logits, dim=-1).squeeze(-1)  # shape N^2 x 1

        z = rearrange(max_prob_entry, " (n m) -> n m", n=params.N)  # shape N x N
    return z


def indicator_table(model, params, group_1, group_2):
    """Takes a multiplication table z and returns a matrix with entry A[i][j]:
    • 1 if z[i][j]=m_1(i,j)
    • 2 if z[i][j]=m_2(i,j)
    • 3 if z[i][j]=m_1(i,j)=m_2(i,j)
    • 0 else
    """
    z = model_table(model, params)

    indicator = t.zeros((params.N, params.N), dtype=t.long)
    for i in range(params.N):
        for j in range(params.N):
            if z[i][j] == group_1[i][j]:
                indicator[i][j] += 1
            if z[i][j] == group_2[i][j]:
                indicator[i][j] += 2
    return indicator


def plot_indicator_table(model, epoch, params, group_1, group_2, save=False):

    row_labels = [f"Row {i}" for i in range(100)]
    col_labels = [f"Col {j}" for j in range(100)]

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
            z=indicator_table(model, params, group_1=group_1, group_2=group_2).tolist(),
            showscale=True,
            colorscale=custom_colorscale,
            x=col_labels,
            y=row_labels,
            zmin=0,
            zmax=3,
        ),
    )

    fig.update_layout(
        title=f"Epoch {epoch}",
        xaxis={
            "showgrid": False,
            "side": "top",
            "ticks": "",
            "tickmode": "array",
            "tickvals": [],
        },
        yaxis={
            "showgrid": False,
            "autorange": "reversed",
            "ticks": "",
            "tickmode": "array",
            "tickvals": [],
        },
        height=900,
        width=900,
    )

    if save == True:
        fig.write_html("./Group addition/plots/plot_{}.html".format(epoch))
    return fig


def plot_gif(list_of_figures, frame_duration=0.01):
    gif_filename = "./Group addition/plots/your_animation.gif"
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
