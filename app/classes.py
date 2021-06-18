import json

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


def classes_contents(analysis_path):
    """
    Generates the contents of the about menu

    :param analysis_path: the path to the analysis file
    :return: class contents
    """
    f = open(analysis_path, 'r')
    analysis = json.load(f)

    classes = analysis["classes"]
    options = []
    for cl in classes:
        class_option = {"label": classes[cl]["name"], "value": str(cl)}
        options.append(class_option)

    dropdown = html.Div([
        dcc.Dropdown(
            id="class-selection",
            options=options,
            placeholder="Select a class",
            multi=False,
            style={"width": "100%"}
        )],
        id="selection-div"
    )

    # hidden div to store image paths
    hidden = html.Div(id='picture-signal', style={'display': 'none'})

    class_report = html.Div(id="class-report")
    contents = html.Div([
        html.H3("Classes", style={"font-weight": "500"}),
        dropdown,
        class_report,
        hidden
    ])

    return contents


def generate_class_report(analysis_path, selection):
    """
    Generates the class report for selected class

    :param analysis_path: the path to the analysis file
    :param selection: the selected class
    :return: class report
    """
    f = open(analysis_path, 'r')
    analysis = json.load(f)

    #  Generating row1.
    classes = analysis["classes"]
    selected_class = classes[selection]

    class_id = "Class ID: " + selection

    num_images = selected_class["num_images"]
    images_card = dbc.Card(
        dbc.CardBody([
            html.H5("Number of images", className="card-title"),
            html.H4(num_images, className="h2 d-inline-block mt-1 mb-4")],
        ),
        className="card flex-fill"
    )

    num_objects = selected_class["num_objects"]
    objects_card = dbc.Card(
        dbc.CardBody([
            html.H5("Number of objects", className="card-title"),
            html.H4(num_objects, className="h2 d-inline-block mt-1 mb-4")],
        ),
        className="card flex-fill"
    )

    min_bbox_width = selected_class["bbox_min_dims"]["width"]
    min_bbox_width_card = dbc.Card(
        dbc.CardBody([
            html.H5("Min bbox width", className="card-title"),
            html.H4(min_bbox_width, className="h2 d-inline-block mt-1 mb-4")],
        ),
        className="card flex-fill"
    )

    min_bbox_height = selected_class["bbox_min_dims"]["height"]
    min_bbox_height_card = dbc.Card(
        dbc.CardBody([
            html.H5("Min bbox height", className="card-title"),
            html.H4(min_bbox_height, className="h2 d-inline-block mt-1 mb-4")],
        ),
        className="card flex-fill"
    )

    max_bbox_width = selected_class["bbox_max_dims"]["width"]
    max_bbox_width_card = dbc.Card(
        dbc.CardBody([
            html.H5("Max bbox width", className="card-title"),
            html.H4(max_bbox_width, className="h2 d-inline-block mt-1 mb-4")],
        ),
        className="card flex-fill"
    )

    max_bbox_height = selected_class["bbox_max_dims"]["height"]
    max_bbox_height_card = dbc.Card(
        dbc.CardBody([
            html.H5("Max bbox height", className="card-title"),
            html.H4(max_bbox_height, className="h2 d-inline-block mt-1 mb-4")],
        ),
        className="card flex-fill"
    )

    avg_size = str(round(selected_class["size_avg"]['avg']/1000, 2)) + 'k ± ' + \
                         str(round(selected_class["size_avg"]['std']/1000, 2)) + 'k'
    avg_size_card = dbc.Card(
        dbc.CardBody([
            html.H5("Average Size", className="card-title"),
            html.H4(avg_size, className="h2 d-inline-block mt-1 mb-4")],
        ),
        className="card flex-fill"
    )

    summary_stats_section = dbc.Row([
        html.H4(class_id, style={"padding-top": "1%", "font-family": "Poppins"}),
        dbc.Col(images_card, className="col-md-flex"),
        dbc.Col(objects_card, className="col-md-flex"),
        dbc.Col(min_bbox_width_card, className="col-md-flex"),
        dbc.Col(min_bbox_height_card, className="col-md-flex"),
        dbc.Col(max_bbox_width_card, className="col-md-flex"),
        dbc.Col(max_bbox_height_card, className="col-md-flex"),
        dbc.Col(avg_size_card, className="col-md-flex")],
        className="row d-sm-flex d-lg-flex d-xxl-flex",
    )

    picture_section = dbc.Row([
        dbc.Row([], id='image-cols', className="row d-xxl-flex"),
        dbc.Row([], id="more-image-cols")],
        id="picture-section", className="row d-sm-flex d-lg-flex d-xxl-flex")

    load_button = dbc.Button("Load more", id="load-button", className="mr-2", color="primary",
                             style={'display': 'none'}, disabled=False)

    contents = html.Div([summary_stats_section, picture_section, load_button])

    return contents
