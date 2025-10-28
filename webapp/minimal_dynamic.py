import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Input, Output, State
from dash import html
from dash_slicer import VolumeSlicer
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the URL paths for the pages
app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ]
)

# Main page layout
main_page_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(id="input-path", placeholder="Enter file path", type="text"),
                    width=8,
                ),
                dbc.Col(
                    dbc.Button("Load", id="load-button", color="primary"),
                    width=2,
                ),
            ]
        ),
    ],
    fluid=True,
)


# Slicer page layout with placeholders
def slicer_page_layout(img_data):
    imin, imax = img_data.min(), img_data.max()
    slicer = VolumeSlicer(app, img_data)
    clim_slider = dcc.RangeSlider(imin, imax, value=(imin, imax), id="clim-slider")

    return dbc.Container(
        [
            dbc.Card(
                [
                    dbc.CardHeader("Volume Slicer"),
                    dbc.CardBody([slicer.graph, slicer.slider, *slicer.stores]),
                ]
            ),
            clim_slider,
            html.Div(id="output-container-range-slider"),
        ],
        fluid=True,
    )


# Callback for navigating between pages
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    State("input-path", "value"),
)
def display_page(pathname, img_path):
    if pathname == "/slicer" and img_path:
        # Load the image data
        try:
            img_data = np.load(img_path)
            logger.info(f"Loaded image data from {img_path} with shape: {img_data.shape}")
            return slicer_page_layout(img_data)
        except Exception as e:
            logger.error(f"Failed to load image data: {e}")
            return dbc.Alert("Error loading image. Check the file path.", color="danger")
    else:
        # Default to the main page
        return main_page_layout


# Callback to update the URL path when the "Load" button is clicked
@callback(
    Output("url", "pathname"),
    Input("load-button", "n_clicks"),
    State("input-path", "value"),
)
def go_to_slicer_page(n_clicks, img_path):
    if n_clicks and img_path:
        return "/slicer"
    return "/"


# Callback for updating the color limits
@callback(
    [Output("output-container-range-slider", "children"), Output(VolumeSlicer.clim.id, "data")],
    Input("clim-slider", "value"),
)
def update_clims(value):
    return f'Color limits: "{value}"', value


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
