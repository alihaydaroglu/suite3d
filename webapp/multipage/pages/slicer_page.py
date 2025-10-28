import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash_slicer import VolumeSlicer
import numpy as np
import logging
import urllib.parse

dash.register_page(__name__, path_template="/slicer/<file_path>")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def layout(file_path=None):
    # Layout with placeholders for dynamic components
    if file_path is None:
        return html.Div("No input")
    real_path = "/mnt/md0/runs/s3d-SS003_2024-08-15_1-2-3-4/summary/ref_img_3d.npy"
    slicer, slicer_content, imin, imax, slicer_id = load_slicer(real_path)
    print(slicer)
    print(imin, imax)
    layout = dbc.Container(
        [
            html.Div(file_path),
            dbc.Card([dbc.CardHeader("Volume Slicer"), dbc.CardBody(slicer_content, id="slicer-content")]),
            dcc.RangeSlider(id="clim-slider", min=imin, max=imax, value=(imin, imax)),
            dcc.Store(id="slicer-store", data=slicer_id),  # Store for slicer ID
            dcc.Store(id="clim-store"),  # Store for color limits
            html.Div(id="output-container-range-slider"),
        ],
        fluid=True,
    )

    # return layout


@callback(
    Output("slicer-content", "children"),
    Output("clim-slider", "min"),
    Output("clim-slider", "max"),
    Output("clim-slider", "value"),
    Output("slicer-store", "data"),  # Store the slicer ID
    Input("url", "search"),
)
def load_slicer(img_path):

    if img_path:
        try:
            # Load the image data
            print(f"Loading from {img_path}")
            img_data = np.load(img_path)
            slicer = VolumeSlicer(dash.get_app(), img_data)
            slicer_id = slicer.clim.id  # Store the slicer clim ID

            imin, imax = img_data.min(), img_data.max()
            return (
                slicer,
                [slicer.graph, slicer.slider, *slicer.stores],
                imin,
                imax,
                slicer_id,
            )
        except Exception as e:
            logger.error(f"Failed to load image data from {img_path}: {e}")
            return dbc.Alert("Error loading image. Check the file path.", color="danger"), None, None, None
    return dbc.Alert("No image path provided.", color="warning"), None, None, None


@callback(
    Output("output-container-range-slider", "children"),
    Output("clim-store", "data"),  # Save clim data in Store
    Input("clim-slider", "value"),
)
def update_clims(value):
    return f'Color limits: "{value}"', value


@callback(
    Output({"type": "slicer-clim", "index": 0}, "data"),
    Input("clim-store", "data"),
    State("slicer-store", "data"),
)
def sync_clim(clim_data, slicer_id):
    if slicer_id and clim_data:
        # Update slicer's clim using the stored ID
        return clim_data
    return dash.no_update
