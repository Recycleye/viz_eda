import dash
import dash_bootstrap_components as dbc

# CSS stylesheet for app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
# main dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.config["suppress_callback_exceptions"] = True
# port to run app
port = 8050


if __name__ == "__main__":
    from app import app_layout

    app.layout = app_layout()

    # Run on docker
    # app.run_server(host="0.0.0.0", port=port, debug=True)

    # Run locally
    app.run_server(port=port, debug=True)

    # Only do analysis
    # annotation_file = ""
    # datadir = ""
    # analyzeDataset(annotation_file, datadir)
