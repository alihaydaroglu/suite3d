import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Input, Output
from dash import html
from dash_slicer import VolumeSlicer
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load the data
img_path = "/mnt/md0/runs/s3d-SS003_2024-08-15_1-2-3-4/summary/ref_img_3d.npy"
img_data = np.load(img_path)
imin, imax = img_data.min(), img_data.max()
irange = imax - imin
logger.info(f"Loaded image data with shape: {img_data.shape}")

# Create the slicer
slicer = VolumeSlicer(app, img_data)

clim_slider = dcc.RangeSlider(img_data.min(), img_data.max(), value=(imin, imax), id="clim-slider")

# Define the layout
app.layout = dbc.Container(
    [
        dbc.Card([dbc.CardHeader("Volume Slicer"), dbc.CardBody([slicer.graph, slicer.slider, *slicer.stores])]),
        clim_slider,
        html.Div(id="output-container-range-slider"),
    ],
    fluid=True,
)


@callback(
    [Output("output-container-range-slider", "children"), Output(slicer.clim.id, "data")],
    Input("clim-slider", "value"),
)
def update_clims(value):
    return 'Color limits: "{}"'.format(value), value


if __name__ == "__main__":
    app.run_server(debug=True, port=8050, dev_tools_props_check=False)
