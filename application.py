"""application.py: all callbacks for the entire app."""
import base64
import json
import os
import time
from typing import List

from dash.dependencies import Input, Output, State
from flask import send_from_directory
from functools import reduce

from pycocotools.coco import COCO

from app.analysis import analyze_dataset, parse_annotations
from app.anomalies import anomalies_contents
from app.classes import classes_contents, generate_class_report
from app.dashboard import dashboard_contents
from app.mainMenu import *
from app.stats import stats_contents
from app.warnings import warnings_contents
# App configuration
from dash_app import app, cache
# Test generate anomalies
from anomaly_detector import generate_anomalies

output_dir = os.path.join(os.getcwd(), "output")
images_path = ""
anns_path = ""
analysis_path = ""
# analysis_path = "/Users/ET/Desktop/Group\ Project/viz_eda_ic/output/analysis.json"

"""
================================================  
              Generic Callbacks 
================================================
"""


@app.callback(
    Output("sidebar", "className"),
    Input("toggle", "n_clicks")
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
    Output("main-content", "children"),
    [Output(f"sidebar-btn-container-{i}", "style") for i in range(1, 9)],
    [Input(f"sidebar-btn-{i}", "n_clicks_timestamp") for i in range(1, 9)],
)
def change_contents(about, new_analysis, upload_analysis, dashboard, warnings, \
                    classes, stats, anomalies):
    """
    Controls the logic of the menus

    :param about: timestamp of last click on about menu btn
    :param new_analysis: timestamp of last click on new analysis menu btn
    :param upload_analysis: timestamp of last click on upload analysis menu btn
    :param dashboard: timestamp of last click on dashboard menu btn
    :param warnings: timestamp of last click on warnings menu btn
    :param classes: timestamp of last click on classes menu btn
    :param stats: timestamp of last click on stats menu btn
    :param anomalies: timestamp of last click on anomalies menu btn
    :return: content to be displayed in main content div
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
    selected = {"background": color, "border-left": "3px solid #3b7ddd"}
    non_selected = {"background": "transparent", "border-color": "transparent"}

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
        contents = anomalies_contents(analysis_path)
        style_anomalies = selected
    return contents, style_about, style_new_analysis, style_upload_analysis, \
           style_dashboard, style_warnings, style_classes, style_stats, style_anomalies


"""
================================================  
             New/Upload Analysis Callbacks 
================================================
"""


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
    global images_path, analysis_path
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
    global anns_path, analysis_path
    style = {"width": "100%", "margin-bottom": "1.5rem", "font-weight": "700",
             "background": "#222e3c"}
    if contents is not None:
        content_type, decoded_content = contents.split(",", 1)
        # If user did not upload analysis,
        # get the annotations to generate analysis

        if analysis_path == "":
            anns_path = parse_annotations(decoded_content)

        if content_type == "data:application/json;base64":
            style = {"width": "100%", "margin-bottom": "1.5rem",
                     "font-weight": "700", "background": "green"}
        else:
            style = {"width": "100%", "margin-bottom": "1.5rem",
                     "font-weight": "700", "background": "red"}
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
        style = {"width": "100%", "text-transform": "uppercase",
                 "font-weight": "700", "background": "green", "outline": "green"}
    else:
        style = {"width": "100%", "text-transform": "uppercase",
                 "font-weight": "700"}
    return style


@app.callback(
    Output("sidebar-btn-4", "n_clicks_timestamp"),
    [Output(f"sidebar-btn-{i}", "disabled") for i in range(4, 9)],
    Input("analyze-btn", "n_clicks"),
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
        return dashboard_click, False, False, False, False, False
    else:
        dashboard_click = 0
        return dashboard_click, True, True, True, True, True


"""
================================================  
             Toggle Popover Callbacks 
================================================
"""


def toggle_popover(n, is_open):
    """
    Shows/hides informance on toggle click

    :param n: num clicks on toggle button
    :param is_open: open state of class warnings
    :return: negated open state if click, else open state
    """
    if n:
        return not is_open
    return is_open


for popid in ["minbbox", "maxbbox", "numim", "numob", "objectsDistribution", \
              "imageDistribution", "class", "idrange"]:
    app.callback(
        Output(f"popover-{popid}", "is_open"),
        [Input(f"popover-{popid}-target", "n_clicks")],
        [State(f"popover-{popid}", "is_open")],
    )(toggle_popover)


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


"""
================================================  
             Warnings Page Callbacks 
================================================
"""


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


"""
================================================  
             Classes Page Callbacks 
================================================
"""
PICTURE_INIT_LOAD: int = 20


# Store intermediate results
@cache.memoize()
def encode_cache_picture(value: str) -> List[str]:
    """
    Encode and store the results
    Args:
        value: class name
    Returns:
        list of picture file names and its source
    """
    f = open(analysis_path, 'r')
    analysis = json.load(f)
    images = analysis["images"]
    classes = analysis["classes"]
    selected_class = classes[value]

    encoded_images = []
    for image in selected_class["images"]:
        file_name = images[str(image)]["file_name"]
        file_name_string = file_name.split('/')[-1]
        encoded_image = base64.b64encode(open(file_name, 'rb').read())
        img_src = "data:image/png;base64,{}".format(encoded_image.decode())

        encoded_images.append((file_name_string, img_src))
    print(f"done with class {value}")
    return encoded_images


@app.callback(Output('picture-signal', 'children'), Input('class-selection', 'value'))
def compute_value(value):
    """
    Cache and compute the images src and send a signal when done
    Args:
        value: class name
    """
    if value:
        encode_cache_picture(value)
    return value


# Update page when select another class


@app.callback(
    [Output('image-cols', 'children'), Output('load-button', 'style')],
    [Input('picture-signal', 'children')],
    [State('image-cols', 'children'), State('load-button', 'style')]
)
def update_pictures(value, image_cols, button_style):
    style = {}  # change display none to no style
    encoded_images = encode_cache_picture(value)
    for name, src in encoded_images[:PICTURE_INIT_LOAD]:
        img_card = dbc.Card([
            dbc.CardImg(src=src),
            dbc.CardBody(name)],
            className="card flex-fill"
        )
        img_col = dbc.Col(img_card, className="col-md-3")
        image_cols.append(img_col)
    return dbc.Row(image_cols, className="image-cols row d-xxl-flex"), style


@app.callback([Output('more-image-cols', 'children'), Output('load-button', 'disabled'),
               Output('load-button', 'children')],
              [Input('picture-signal', "children"), Input('load-button', 'n_clicks')],
              [State('more-image-cols', "children"), State('load-button', 'disabled'),
               State('load-button', 'children')])
def load_more_pictures(value, click, children, disabled, button_txt):
    if click and not disabled:
        encoded_images = encode_cache_picture(value)
        if len(encoded_images) <= PICTURE_INIT_LOAD + click - 1:
            disabled = True
            button_txt = "no more picture"
        else:
            img_card = dbc.Card([
                # click starts from 1.
                dbc.CardImg(src=encoded_images[PICTURE_INIT_LOAD + click - 1][1]),
                dbc.CardBody(encoded_images[PICTURE_INIT_LOAD + click - 1][0])],
                className="card flex-fill"
            )
            img_col = dbc.Col(img_card, className="col-md-3")
            children.append(img_col)
    return children, disabled, button_txt


@app.callback(
    Output("selection-div", "style"),
    Output("class-report", "children"),
    Output("class-report", "style"),
    Input("class-selection", "value")
)
def display_class_report(selection):
    """
    Shows/hides class report of class selected by user
    """
    global analysis_path
    if selection is None:
        style_sel = {"padding-top": "25%", "margin": "auto", "width": "20%"}
        rep_children = None
        style_rep = {"display": "none"}
    else:
        style_sel = {"width": "20%"}
        rep_children = generate_class_report(analysis_path, selection)
        style_rep = {"display": "block"}
    return style_sel, rep_children, style_rep


"""
================================================  
             Download Callbacks 
================================================
"""


@app.server.route("/download/<path:filename>")
def download(filename):
    """
    Enables file download

    :param filename: the name of the file to be downloaded
    :return: the file download
    """
    return send_from_directory(output_dir, filename,
                               attachment_filename=filename, as_attachment=True)


"""
================================================  
             Anomalies Page Callbacks 
================================================
"""
from app.anomalies import ALGORITHMS


@app.callback(
    Output('update-button', 'n_clicks'),
    Input('algo-selection', 'value'))
def reset_click_upon_toggle_value_change(value):
    return 0


def plot_anomalies(selected_algorithms):
    # return dcc.Graph(figure=px.histogram(anomaly_dfs[0], x="cat_name"))
    # coco = COCO(analysis_path)
    # cats = coco.loadCats(coco.getCatIds())
    # cat_names = [cat["name"] for cat in cats]
    f = open(analysis_path, 'r')
    analysis = json.load(f)
    classes = analysis["classes"]

    class_names = []
    for cl in classes:
        class_names.append(classes[cl]["name"])

    fig = go.Figure()
    for algorithm_name in selected_algorithms:
        anomaly_df = generate_anomalies(analysis_path,
                                        ALGORITHMS[algorithm_name]['detector'],
                                        ALGORITHMS[algorithm_name]['df_creator'])
        freq = anomaly_df.groupby('cat_name').size()
        fig.add_trace(go.Bar(
            x=class_names,
            y=[freq.get(cat_name, 0) for cat_name in class_names],
            name=algorithm_name,
            marker_color=ALGORITHMS[algorithm_name]['color']
        ))

    return dcc.Graph(figure=fig)


def summary_card(title, summary_data, description):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            html.H4(summary_data, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(description, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )


def tabulate_anomalies(algorithm_name, selected_algorithms):
    if not selected_algorithms or algorithm_name not in selected_algorithms or algorithm_name not in ALGORITHMS:
        return None, {'display': 'none'}
    anomaly_df = generate_anomalies(analysis_path,
                                    ALGORITHMS[algorithm_name]['detector'],
                                    ALGORITHMS[algorithm_name]['df_creator'])
    f = open(analysis_path, 'r')
    analysis = json.load(f)
    tot_img_num = int(analysis["total_num_images"])
    tot_object_num = int(analysis["total_num_objects"])
    anomaly_object_num = anomaly_df.shape[0]
    anomaly_img_num = anomaly_df.file_name.unique().shape[0]
    freq = anomaly_df.groupby('cat_name').size()
    max_idx = freq.argmax()
    class_highest_anomaly_name = freq.index[max_idx]
    class_highest_anomaly_cnt = freq[max_idx]
    summary_row = dbc.Row([
        dbc.Col(
            summary_card('Algorithm', ALGORITHMS[algorithm_name]['label'], ""),
            className="col-sm-3 col-lg-3 col-xxl-3 d-flex"),
        dbc.Col(
            summary_card('Number of Anomalous Images', anomaly_img_num,
                         f"Proportion: {round(anomaly_img_num / tot_img_num * 100)}%"),
            className="col-sm-3 col-lg-3 col-xxl-3 d-flex"),
        dbc.Col(summary_card('Number of Anomalous Object', anomaly_object_num,
                             f"Proportion: {round(anomaly_object_num / tot_object_num * 100)}%"),
                className="col-sm-3 col-lg-3 col-xxl-3 d-flex"),
        dbc.Col(summary_card('Class with Highest Number of Anomalies', class_highest_anomaly_name,
                             f"Number of Anomalies: {class_highest_anomaly_cnt}"),
                className="col-sm-3 col-lg-3 col-xxl-3 d-flex")], className="row d-xxl-flex",
        style={"margin-top": "15px"})
    return summary_row, {'display': 'block'}


for algorithm in ALGORITHMS.values():
    app.callback(
        Output(f"summary-cards-{algorithm['name']}", 'children'),
        Output(f"summary-cards-{algorithm['name']}", 'style'),
        Input(f"algorithm-name-{algorithm['name']}", 'children'),
        Input('algo-selection', 'value')
    )(tabulate_anomalies)


def toggle_anomaly_table_image_div(toggle):
    return {'display': 'block'} if toggle else {'display': 'none'}


for algorithm in ALGORITHMS.values():
    app.callback(
        Output(f"anomaly-table-image-div-{algorithm['name']}", 'style'),
        Input(f"table-toggle-{algorithm['name']}", "value"),
    )(toggle_anomaly_table_image_div)


@app.callback(Output('anomaly-output-div', 'style'),
              Output('plot-section', 'children'),
              Input('update-button', 'n_clicks'),
              Input('algo-selection', 'value'))
def display_anomaly_output(button_clicked, selected_algorithms):
    if not (button_clicked and selected_algorithms):
        return {'display': 'none'}, None

    plots = plot_anomalies(selected_algorithms)
    return {'display': 'block'}, plots


def update_table(algorithm_name, page_current, page_size, sort_by, selected_algorithms):
    if not selected_algorithms or algorithm_name not in selected_algorithms or algorithm_name not in ALGORITHMS:
        return None, {'display': 'none'}
    df = generate_anomalies(analysis_path,
                            ALGORITHMS[algorithm_name]['detector'],
                            ALGORITHMS[algorithm_name]['df_creator'])
    if len(sort_by):
        dff = df.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc',
            inplace=False
        )
    else:
        # No sort is applied
        dff = df

    return dff.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records'), {'display': 'block'}


for algorithm in ALGORITHMS.values():
    app.callback(
        Output(f"anomaly-data-table-{algorithm['name']}", 'data'),
        Output(f"anomaly-output-section-{algorithm['name']}", 'style'),
        Input(f"algorithm-name-{algorithm['name']}", 'children'),
        Input(f"anomaly-data-table-{algorithm['name']}", "page_current"),
        Input(f"anomaly-data-table-{algorithm['name']}", "page_size"),
        Input(f"anomaly-data-table-{algorithm['name']}", 'sort_by'),
        Input('algo-selection', 'value')
    )(update_table)

import plotly.express as px
from PIL import Image


def encode_image(file_name):
    # file_name_string = file_name.split('/')[-1]
    return Image.open(file_name)


def show_active_cell(algorithm_name, active_cell):
    if not active_cell:
        return None
    df = generate_anomalies(analysis_path,
                            ALGORITHMS[algorithm_name]['detector'],
                            ALGORITHMS[algorithm_name]['df_creator'])
    active_row = df[df['id'] == active_cell['row_id']]
    return create_object_plot(active_row)


for algorithm in ALGORITHMS.values():
    app.callback(Output(f"anomaly-image-cell-{algorithm['name']}", 'children'),
                 Input(f"algorithm-name-{algorithm['name']}", 'children'),
                 Input(f"anomaly-data-table-{algorithm['name']}", 'active_cell'),
                 )(show_active_cell)

import plotly.graph_objects as go


def create_object_plot(df_objects):
    image_name = df_objects.iloc[0]['file_name']
    image_src = encode_image(image_name)
    # fig = px.imshow(image_src, title={'text': image_name})
    fig = px.imshow(image_src)

    for cat_name, image_bbox in zip(df_objects['cat_name'], df_objects['bbox']):
        fig.add_shape(
            type='rect',
            line_color='yellow',
            fillcolor='turquoise',
            opacity=0.4,
            x0=image_bbox[0], x1=image_bbox[0] + image_bbox[2], y0=image_bbox[1], y1=image_bbox[1] + image_bbox[3]
        )
        fig.add_trace(
            go.Scatter(x=[image_bbox[0], image_bbox[0], image_bbox[0] + image_bbox[2], image_bbox[0] + image_bbox[2]],
                       y=[image_bbox[1] + image_bbox[3], image_bbox[1], image_bbox[1], image_bbox[1] + image_bbox[3]],
                       text="Label: " + cat_name,
                       fill='toself',
                       fillcolor='turquoise',
                       opacity=0.1,
                       hoveron='fills',
                       showlegend=False,
                       mode='text'))

    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=5,
            r=5,
            b=10,
            t=10,
            pad=5,
        )
        # paper_bgcolor="White",
    )
    return dbc.Col(dcc.Graph(figure=fig), md=3)


def plot_multiple_objects(df, selected_rows):
    df_selected = df[df['id'].isin(selected_rows)]
    img_cards = []
    for image in df_selected['image_id'].unique():
        df_objects = df_selected[df_selected['image_id'] == image]
        img_card = create_object_plot(df_objects)
        img_cards.append(img_card)
    return img_cards


def show_selected_cells(algorithm_name, selected_cells):
    if not selected_cells:
        return []
    print(selected_cells)
    df = generate_anomalies(analysis_path,
                            ALGORITHMS[algorithm_name]['detector'],
                            ALGORITHMS[algorithm_name]['df_creator'])
    return plot_multiple_objects(df, selected_cells)


for algorithm in ALGORITHMS.values():
    app.callback(
        Output(f"anomaly-image-cols-{algorithm['name']}", 'children'),
        Input(f"algorithm-name-{algorithm['name']}", 'children'),
        Input(f"anomaly-data-table-{algorithm['name']}", "selected_row_ids"))(show_selected_cells)
