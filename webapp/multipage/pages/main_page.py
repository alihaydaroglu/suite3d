import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

layout = dbc.Container(
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


@callback(
    Output("load-button", "href"),
    Input("load-button", "n_clicks"),
    State("input-path", "value"),
)
def navigate_to_slicer(n_clicks, img_path):
    if n_clicks and img_path:
        return f"/slicer/{img_path}"
    return "/"
