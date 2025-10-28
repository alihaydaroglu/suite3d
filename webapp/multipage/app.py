import dash
import dash_bootstrap_components as dbc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)
app.layout = dbc.Container(
    [dash.page_container],  # This holds the content for the current page
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
