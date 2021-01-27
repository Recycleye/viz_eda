import base64, json, os
import dash
import random
import json
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html 
import plotly.express as px
import pandas as pd
import time

from dash.dependencies import Input, Output, State
from flask import send_from_directory
from app.mainMenu import *
from app.dashboard import dashboard_contents
from app.warnings import warnings_contents
from app.stats import stats_contents
from app.classes import classes_contents, generate_class_report
from app.analysis import analyze_dataset, parse_annotations, parse_analysis

# App configuration
app = dash.Dash(__name__)
app.config["suppress_callback_exceptions"] = True
application = app.server
port = 80
output_dir = os.path.join(os.getcwd(), "output")
images_path = ""
anns_path = ""
analysis_path = ""

# App elements

navbar = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.A(html.I(className="hamburger align-self-center"),
                    className="sidebar-toggle d-flex", id="toggle")
                )                    
            ],
            align = "center",
            no_gutters = True,
            style = {
                "width" : "100%"
            }
        )
    ],
    className = "navbar navbar-expand navbar-light navbar-bg"
)

sidebar = html.Div([
        html.A(html.Span("vizEDA",className="align-middle"),
        className="sidebar-brand",href="http://0.0.0.0"),
        html.Div([
            html.Div([
                dbc.Button([
                    html.Img(src="assets/icons/info.svg",
                    style={"width":"15px","padding-bottom":"1px",
                    "margin-right":"5px"}),
                    "About"
                    ],
                    color="primary", 
                    className="mr-1",
                    id="sidebar-btn-1",
                    n_clicks_timestamp='0',
                    style={"margin-left":"5%","margin-top":"0",
                    "margin-bottom":"0",
                    "background":"transparent","border-color":"transparent"})],
                id = "sidebar-btn-container-1"
            ),
            html.Div([
                dbc.Button([
                    html.Img(src="assets/icons/plus.svg",
                    style={"width":"15px","padding-bottom":"1px",
                    "margin-right":"5px"}),
                    "New analysis"
                    ],
                    color="primary", 
                    className="mr-1",
                    id="sidebar-btn-2",
                    n_clicks_timestamp='0',
                    style={"margin-left":"5%","margin-top":"0",
                    "margin-bottom":"0","background":"transparent",
                    "border-color":"transparent"})],
                id = "sidebar-btn-container-2"
            ),
            html.Div([
                dbc.Button([
                    html.Img(src="assets/icons/upload.svg",
                    style={"width":"15px","padding-bottom":"1px",
                    "margin-right":"5px"}),
                    "Upload analysis"
                    ],
                    color="primary", 
                    className="mr-1",
                    id="sidebar-btn-3",
                    n_clicks_timestamp='0',
                    style={"margin-left":"5%","margin-top":"0",
                    "margin-bottom":"0","background":"transparent",
                    "border-color":"transparent"})],
                id = "sidebar-btn-container-3"),
            ],
            style={"margin-top":"10%"}
        ),
        html.Div([
            html.Div([
                dbc.Button([
                    html.Img(src="assets/icons/sliders.svg",
                    style={"width":"15px","padding-bottom":"1px",
                    "margin-right":"5px"}),
                    "Dashboard"
                    ],
                    color="primary",
                    className="mr-1",
                    id="sidebar-btn-4",
                    n_clicks_timestamp='0',
                    disabled= True,
                    style={"margin-left":"5%","margin-top":"0",
                    "margin-bottom":"0","background":"transparent",
                    "border-color":"transparent"})],
                id = "sidebar-btn-container-4"),
            html.Div([
                dbc.Button([
                    html.Img(src="assets/icons/alert-triangle.svg",
                    style={"width":"15px","padding-bottom":"1px",
                    "margin-right":"5px"}),
                    "Warnings"
                    ],
                    color="primary",
                    className="mr-1",
                    id="sidebar-btn-5",
                    n_clicks_timestamp='0',
                    disabled= True,
                    style={"margin-left":"5%","margin-top":"0",
                    "margin-bottom":"0","background":"transparent",
                    "border-color":"transparent"})],
                id = "sidebar-btn-container-5"),
            html.Div([
                dbc.Button([
                    html.Img(src="assets/icons/layout.svg",
                    style={"width":"15px","padding-bottom":"1px",
                    "margin-right":"5px"}),
                    "Classes"
                    ],
                    color="primary",
                    className="mr-1",
                    id="sidebar-btn-6",
                    n_clicks_timestamp='0',
                    disabled=True,
                    style={"margin-left":"5%","margin-top":"0",
                    "margin-bottom":"0","background":"transparent",
                    "border-color":"transparent"})],
                id = "sidebar-btn-container-6"),
            html.Div([
                dbc.Button([
                    html.Img(src="assets/icons/bar-chart-2.svg",
                    style={"width":"15px","padding-bottom":"1px",
                    "margin-right":"5px"}),
                    "Stats"
                    ],
                    color="primary", 
                    className="mr-1",
                    id="sidebar-btn-7",
                    n_clicks_timestamp='0',
                    disabled=True,
                    style={"margin-left":"5%","margin-top":"0",
                    "margin-bottom":"0","background":"transparent",
                    "border-color":"transparent"})],
                id = "sidebar-btn-container-7"),
            html.Div([
                dbc.Button([
                    html.Img(src="assets/icons/crosshair.svg",
                    style={"width":"15px","padding-bottom":"1px",
                    "margin-right":"5px"}),
                    "Anomalies"
                    ],
                    color="primary", 
                    className="mr-1",
                    id="sidebar-btn-8",
                    n_clicks_timestamp='0',
                    disabled=True,
                    style={"margin-left":"5%","margin-top":"0",
                    "margin-bottom":"0","background":"transparent",
                    "border-color":"transparent"})],
                id = "sidebar-btn-container-8"),
            ],
            style={"margin-top":"10%"}
        )
    ],
    className="sidebar collapse",
    id="sidebar"
)

header = html.Div(
    [
        sidebar,
        html.Div([
            navbar, 
            html.Div(id="main-content", style={"padding":"2.5rem 2.5rem 1rem"})
        ],
        className="main")
    ],
    className="wrapper"
)

def serve_layout():
    return html.Div(
        [
            header
        ]
    )

app.layout = serve_layout

# App callbacks

@app.callback(
    Output("sidebar","className"),
    Input("toggle","n_clicks")
)
def do_show_hide(n_clicks):
    """
    Shows/hides the sidebar on toggle click

    :param n_clicks: num clicks on toggle button
    :return: className of sidebar defining collapsed state
    """
    if n_clicks is None:
        return "sidebar"
    if n_clicks % 2:
            return "sidebar collapse"
    else:
         return "sidebar"

@app.callback(
    Output("main-content","children"),
    [Output(f"sidebar-btn-container-{i}", "style") for i in range(1, 9)],
    [Input(f"sidebar-btn-{i}", "n_clicks_timestamp") for i in range(1, 9)],
)
def change_contents(about,new_analysis,upload_analysis,dashboard,warnings,\
    classes,stats,anomalies):
    """
    Controls the logic of the menus

    :param about: timestamp of last click on about menu btn
    :param new_analysis: timestamp of last click on new analysis menu btn
    :param upload_analysis: timestamp of last click on upload analysis menu btn
    :param dashbard: timestamp of last click on dashboard menu btn
    :param warnings: timestamp of last click on warnings menu btn
    :param classes: timestamp of last click on classes menu btn
    :param stats: timestamp of last click on stats menu btn
    :param anomalies: timestamp of last click on anomalies menu btn
    :return: conent to be displayed in main content div
    """
    global analysis_path

    # Make a list of all click timestamps
    # Notice: Dash uses 13 digits UNIX time, we want 10 digits only
    clicks = [0, 
        int(str(about)[:10]), 
        int(str(new_analysis)[:10]), 
        int(str(upload_analysis)[:10]),
        int(str(dashboard)[:10]), 
        int(str(warnings)[:10]), 
        int(str(classes)[:10]), 
        int(str(stats)[:10]), 
        int(str(anomalies)[:10])
    ]
    # Get index of last click btn
    last_clicked = clicks.index(max(clicks))

    # Color for sidebar btns
    color = "linear-gradient(90deg,rgba(59,125,221,.1),rgba(59,125,221,.0875) 50%,transparent)"

    # Define style for selected/non-selected buttons
    selected = {"background":color, "border-left":"3px solid #3b7ddd"}
    non_selected = {"background":"transparent","border-color":"transparent"}

    # Set all btns to non-selected
    style_about = non_selected
    style_new_analysis = non_selected
    style_upload_analysis = non_selected
    style_dashboard = non_selected
    style_warnings = non_selected
    style_classes = non_selected
    style_stats = non_selected
    style_anomalies = non_selected

    # Set last click btn to selected and display corresponding content
    if last_clicked == 0:
        contents = new_analysis_contents()
        style_new_analysis = selected
    if last_clicked == 1:
        contents = about_contents()
        style_about = selected
    elif last_clicked == 2:
        contents = new_analysis_contents()
        style_new_analysis = selected
    elif last_clicked == 3:
        contents = upload_analysis_contents()
        style_upload_analysis = selected
    elif last_clicked == 4:
        contents = dashboard_contents(analysis_path)
        style_dashboard = selected
    elif last_clicked == 5:
        contents = warnings_contents(analysis_path)
        style_warnings = selected
    elif last_clicked == 6:
        contents = classes_contents(analysis_path)
        style_classes = selected
    elif last_clicked == 7:
        contents = stats_contents(analysis_path)
        style_stats = selected
    elif last_clicked == 8:
        contents = html.Div("Anomalies")
        style_anomalies = selected
    return contents,style_about,style_new_analysis,style_upload_analysis,\
        style_dashboard,style_warnings,style_classes,style_stats,style_anomalies

@app.callback(
    Output("images-upload", "valid"),
    Input("images-upload", "value")
)
def check_images_path(path):
    """
    Validates the path upload field by checking 
    if the path provided by the user exists

    :param path: the image path provided by the user
    :return: True if path exists, False otherwise
    """ 
    global images_path,analysis_path
    if path is None:
        path = ""
    if os.path.isdir(path):
        images_path = path
        return True
    else:
        # If user did not upload analysis
        if analysis_path == "":
            return False

        # Else if user uploaded analysis
        else:
            return True


@app.callback(
    Output("upload-btn", "style"),
    Input("upload", "contents")
)
def check_upload(contents):
    """
    Validates the upload button by checking uploaded file is in JSON format
    
    :param contents: the file uploaded by the user
    :return: color update for upload button
    """
    global anns_path,analysis_path
    style = {"width":"100%","margin-bottom":"1.5rem","font-weight":"700",
    "background":"#222e3c"}
    if contents is not None:
        content_type, decoded_content = contents.split(",", 1)
        # If user did not upload analysis, 
        # get the annotations to generate analysis
        if analysis_path == "":
            anns_path = parse_annotations(decoded_content)
        
        if content_type == "data:application/json;base64":
            style = {"width":"100%","margin-bottom":"1.5rem",
            "font-weight":"700","background":"green"}
        else:
            style = {"width":"100%","margin-bottom":"1.5rem",
            "font-weight":"700","background":"red"}
    return style

@app.callback(
    Output("analyze-btn", "disabled"),
    Input("images-upload", "valid"),
    Input("upload-btn", "style")
)
def check_inputs(valid_path, btn_style):
    """
    Enables/disables analyze button depending on whether the image path
    and uploaded file have been validated

    :param valid_path: if the image path is valid
    :param btn_color: the color of the upload button (green if file is ok)
    :return: disabled state and color of the analyze button
    """
    valid_file = False
    if btn_style["background"] == "green":
        valid_file = True
    if valid_path and valid_file:
        return False
    else:
        return True

@app.callback(
    Output("analyze-btn", "style"),
    Input("analyze-btn", "disabled")
)
def update_analyze_button(disabled):
    """
    Updates the color of the analyze button depending on
    its disabled status

    :param disabled: if the button is disabled
    """
    if not disabled:
        style = {"width":"100%","text-transform":"uppercase",
        "font-weight":"700","background":"green","outline":"green"}
    else:
        style = {"width":"100%","text-transform":"uppercase",
        "font-weight":"700"}
    return style

@app.callback(
    Output("sidebar-btn-4","n_clicks_timestamp"),
    [Output(f"sidebar-btn-{i}", "disabled") for i in range(4, 9)],
    Input("analyze-btn","n_clicks"),
)
def enable_buttons(click):
    """
    Enables/disables the dashboard, warnings, classes, stats and 
    anomalies buttons depending on if analyze btn is clicked

    Automatically switches to dashboard if analyze btn is clicked

    :param click: num clicks on analyze btn
    :return: dashboard click event and disabled status for all btns
    """
    global images_path, anns_path, analysis_path
    if click:
        # If user inserted images and anns
        if images_path != "" and anns_path != "":
            analysis_path = analyze_dataset(images_path, anns_path)
        # Dashboard click event
        dashboard_click = int(time.time())
        return dashboard_click,False,False,False,False,False
    else:
        dashboard_click = 0
        return dashboard_click,True,True,True,True,True

@app.callback(
    Output("class-warnings-collapse", "is_open"),
    [Input("class-warnings-collapse-button", "n_clicks")],
    [State("class-warnings-collapse", "is_open")],
)
def toggle_class_warnings_collapse(n, is_open):
    """
    Shows/hides class warnings on toggle click

    :param n: num clicks on toggle button
    :param is_open: open state of class warnings
    :return: negated open state if click, else open state
    """
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("id-warnings-collapse", "is_open"),
    [Input("id-warnings-collapse-button", "n_clicks")],
    [State("id-warnings-collapse", "is_open")],
)
def toggle_id_warnings_collapse(n, is_open):
    """
    Shows/hides id warnings on toggle click

    :param n: num clicks on toggle button
    :param is_open: open state of id warnings
    :return: negated open state if click, else open state
    """
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("image-warnings-collapse", "is_open"),
    [Input("image-warnings-collapse-button", "n_clicks")],
    [State("image-warnings-collapse", "is_open")],
)
def toggle_image_warnings_collapse(n, is_open):
    """
    Shows/hides image warnings on toggle click

    :param n: num clicks on toggle button
    :param is_open: open state of image warnings
    :return: negated open state if click, else open state
    """
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("selection-div", "style"),
    Output("class-report","children"),
    Output("class-report","style"),
    Input("class-selection", "value")
)
def display_class_report(selection):
    """
    Shows/hides class report of class selected by user
    """
    global analysis_path
    if selection is None:
        style_sel={"padding-top":"25%", "margin":"auto", "width":"20%"}
        rep_children = None
        style_rep={"display":"none"}
    else:
        style_sel={"width":"20%"}
        rep_children = generate_class_report(analysis_path,selection)
        style_rep={"display":"block"}
    return style_sel,rep_children,style_rep

@app.server.route("/download/<path:filename>")
def download(filename):
    """
    Enables file download

    :param filename: the name of the file to be downloaded
    :return: the file download
    """
    return send_from_directory(output_dir, filename, 
    attachment_filename = filename, as_attachment = True)

if __name__ == "__main__":
    application.run(host ="0.0.0.0", port = port, debug = True)