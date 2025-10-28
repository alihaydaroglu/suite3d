import dash
import dash_bootstrap_components as dbc

from dash_slicer import VolumeSlicer
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import json
import requests
import numpy as n
import logging

from manager import manager_port, app_port, db_port, self_ip

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Collapsible section component
def create_collapsible_section(title, content, id_prefix):
    return html.Div(
        [
            dbc.Button(
                title,
                id=f"{id_prefix}-collapse-button",
                className="mb-3",
                color="primary",
                n_clicks=0,
            ),
            dbc.Collapse(
                dbc.Card(dbc.CardBody(content)),
                id=f"{id_prefix}-collapse-content",
                is_open=False,
            ),
        ]
    )


# Sidebar layout
sidebar_content = [
    create_collapsible_section(
        "Create New Dataset",
        [
            dcc.Input(id="subject-name", type="text", placeholder="Subject Name", className="mb-2"),
            dcc.Input(id="date", type="text", placeholder="Date (YYYY-MM-DD)", className="mb-2"),
            dcc.Input(
                id="experiments", type="text", placeholder="Experiments (comma-separated integers)", className="mb-2"
            ),
            dcc.Input(id="analysis-path", type="text", placeholder="Analysis path", className="mb-2"),
            dbc.Button("Create Dataset", id="create-dataset-button", color="success", className="mt-2"),
            html.Div(id="create-dataset-output", className="mt-2"),
        ],
        "create-dataset",
    ),
    create_collapsible_section(
        "Create New Job",
        [
            dcc.Dropdown(id="dataset-dropdown", placeholder="Select Dataset", className="mb-2"),
            dcc.Input(id="job-type", type="text", placeholder="Job Type", className="mb-2"),
            dbc.Button("Create Job", id="create-job-button", color="success", className="mt-2"),
            html.Div(id="create-job-output", className="mt-2"),
        ],
        "create-job",
    ),
    create_collapsible_section(
        "Datasets",
        [
            dbc.Button("Fetch datasets", id="fetch-datasets-button", color="primary", className="mb-2"),
            html.Div(id="datasets-container"),
            html.Div(id="load-dataset-output"),
        ],
        "datasets",
    ),
]


# Main layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Analysis Dashboard", className="text-center mb-4"),
                        dbc.Card(sidebar_content, body=True),
                    ],
                    width=3,
                ),
                dbc.Col([html.Div(id="main-content")], width=9),
            ]
        ),
        dcc.Interval(id="interval-component", interval=100000 * 1000, n_intervals=0),
    ],
    fluid=True,
)

# Callbacks for collapsible sections
for section in ["create-dataset", "create-job", "datasets"]:

    @app.callback(
        Output(f"{section}-collapse-content", "is_open"),
        Input(f"{section}-collapse-button", "n_clicks"),
        State(f"{section}-collapse-content", "is_open"),
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open


@app.callback(
    Output("create-dataset-output", "children"),
    [Input("create-dataset-button", "n_clicks")],
    [
        State("subject-name", "value"),
        State("date", "value"),
        State("experiments", "value"),
        State("analysis-path", "value"),
    ],
)
def create_dataset(n_clicks, subject_name, date, experiments, analysis_path):
    if n_clicks is not None and n_clicks > 0:
        experiments_list = list(map(int, experiments.split(",")))
        data = {"subject": subject_name, "date": date, "experiments": experiments_list, "analysis-path": analysis_path}
        print("Trying to create experiment with data", data)
        response = requests.post(f"http://localhost:{manager_port}/api/datasets", json=data)
        if response.status_code == 201:
            return "Dataset created successfully!"
        elif response.status_code == 202:
            return "Dataset already exists."
        else:
            return "Failed to create dataset."


@app.callback(
    Output("create-job-output", "children"),
    [Input("create-job-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("job-type", "value")],
)
def create_job(n_clicks, dataset_str, job_type):
    if n_clicks is not None and n_clicks > 0:
        data = {"job_type": job_type, "dataset_str": dataset_str}
        print("Attempting to create job with ", data)
        response = requests.post(f"http://localhost:{manager_port}/api/jobs", json=data)
        if response.status_code == 201:
            return "Job created successfully!"
        elif response.status_code == 202:
            return "Job already exists."
        else:
            return "Failed to create job."


@app.callback(
    dash.dependencies.Output("datasets-container", "children"),
    Output("dataset-dropdown", "options"),
    [
        dash.dependencies.Input("interval-component", "n_intervals"),
        Input("fetch-datasets-button", "n_clicks"),
        Input("create-dataset-button", "n_clicks"),
    ],
)
def fetch_datasets(n_int, n_click, n_click2):
    print("Getting datasets")
    response = requests.get(f"http://localhost:{manager_port}/api/datasets")
    datasets = response.json()

    dataset_divs = []
    dataset_options = []
    for dataset in datasets:
        dataset_div_list = [
            html.H3(f"Dataset: {dataset['dataset_str']}"),
            html.Button("Load dataset", id={"type": "load-dataset-button", "index": dataset["_id"]["$oid"]}),
        ]
        # print("Requesting jobs")
        joblist = requests.get(f"http://localhost:{manager_port}/api/jobs", json={"dataset_id": dataset["_id"]}).json()
        # print(f"Got {len(joblist)} jobs")
        if len(joblist) > 0:
            for job in joblist:
                dataset_div_list.append(html.H4(f"Job of type: {job['job_type']}"))

        dataset_options.append(dataset["dataset_str"])
        dataset_divs.append(html.Div(dataset_div_list))
    return dataset_divs, dataset_options


import imageio

slicers = []
volumes = []


def create_slicer(img_data, header="Summary image"):
    try:
        logger.info(f"Creating slicer with image shape: {img_data.shape}")
        slicer = VolumeSlicer(app, img_data)

        slicer_card = dbc.Card(
            [
                dbc.CardHeader(
                    [
                        header,
                        dbc.Button(
                            "Close", id="close-slicer-button", className="float-right", color="danger", size="sm"
                        ),
                    ]
                ),
                dbc.CardBody([slicer.graph, slicer.slider, *slicer.stores]),
            ],
            id="slicer-card",
        )
        slicers.append(slicer)
        return slicer_card, slicer
    except Exception as e:
        logger.error(f"Error creating slicer: {str(e)}")
        return dbc.Alert(f"Error creating slicer: {str(e)}", color="danger"), None


cards = []
slicers = []


@app.callback(
    Output("main-content", "children"),
    Input({"type": "load-dataset-button", "index": dash.ALL}, "n_clicks"),
    State({"type": "load-dataset-button", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def load_dataset(n_clicks, button_ids):
    all_none = True
    for nclick in n_clicks:
        if nclick is not None:
            all_none = False
    if all_none:
        return
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    dataset_id = json.loads(button_id)["index"]

    try:
        response = requests.get(
            f"http://localhost:{manager_port}/api/load_dataset", json={"dataset_id": {"$oid": dataset_id}}
        )
        data = response.json()
        logger.info(f"Loaded dataset: {data}")

        if response.status_code == 200:
            img_path = data["img_path"]
            img_data = n.load(img_path)
            logger.info(f"Loaded image data with shape: {img_data.shape}")
            volumes.append(img_data)
            slicer_card, slicer = create_slicer(img_data, f"Dataset {dataset_id}")
            cards.append(slicer_card)
            slicers.append(slicers)
            return slicer_card
        else:
            return dbc.Alert(f"Failed to load dataset {dataset_id}.", color="danger")
    except Exception as e:
        logger.error(f"Error in load_dataset: {str(e)}")
        return dbc.Alert(f"Error loading dataset: {str(e)}", color="danger")


if __name__ == "__main__":
    app.run_server(debug=True, host=self_ip, port=app_port)
