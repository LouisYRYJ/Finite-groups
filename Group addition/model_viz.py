import plotly.graph_objs as go
from plotly.io import write_image


def plota_(number_of_squares):
    # Custom labels and dimensions - 100 rows and 100 columns
    row_labels = [f"Row {i}" for i in range(1, 50)]
    col_labels = [f"Col {j}" for j in range(1, 50)]

    # Initialize z_value matrix with white (value of 0) for all cells
    z_value = [[0] * len(col_labels) for _ in range(len(row_labels))]

    # Set color values for top, middle, and bottom rows (row index starting from 0)
    middle_row_index = len(row_labels) // 2
    for col_index in range(len(col_labels)):
        z_value[number_of_squares][col_index] = 1  # Orange value on top row
        z_value[middle_row_index][col_index] = 3  # Green value on middle row
        z_value[col_index][col_index] = 2  # Blue value on bottom row

    # Define the custom colorscale
    custom_colorscale = [
        [0, "#FFFFFF"],  # White color for non-specified cells
        [(1 / 3) - 0.01, "#FFFFFF"],  # Ensure that colors below value=1 remain white
        [(1 / 3), "#FFA07A"],  # Non-Pungent Orange color for value=1
        [(2 / 3) - 0.01, "#FFA07A"],  # Intermediate stretch between Orange and Green
        [(2 / 3), "#90EE90"],  # Non-Pungent Green color for value=3
        [1 - 0.01, "#90EE90"],  # Intermediate stretch between Green and Blue
        [1, "#ADD8E6"],  # Non-Pungent Blue color for value=2
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_value,
            showscale=False,
            colorscale=custom_colorscale,
            x=col_labels,
            y=row_labels,
        ),
    )

    fig.update_layout(
        title="100x100 Grid with Custom Row Colors",
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
    return fig
