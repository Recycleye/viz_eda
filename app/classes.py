import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html 
import pandas as pd
import plotly.express as px
import json
import base64

def classes_contents(analysis_path):
    """
    Generates the contents of the about menu

    :param analysis_path: the path to the analysis file
    :return: class contents
    """
    f = open(analysis_path,'r')
    analysis = json.load(f)

    classes = analysis["classes"]
    options = []
    for cl in classes:
        class_option = {}
        class_option["label"] = classes[cl]["name"]
        class_option["value"] = str(cl)
        options.append(class_option)

    dropdown = html.Div([
            dcc.Dropdown(
            id="class-selection",
            options = options,
            placeholder="Select a class",
            multi=False,
            style={"width":"100%"}
        )],
        id="selection-div"
    )

    class_report = html.Div(id="class-report")

    contents = html.Div([
        html.H3("Classes",style={"font-weight":"500"}),
        dropdown,
        class_report
    ])

    return contents

def generate_class_report(analysis_path,selection):
    """
    Generates the class report for selected class

    :param analysis_path: the path to the analysis file
    :param selection: the selected class
    :return: class report
    """
    f = open(analysis_path,'r')
    analysis = json.load(f)

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

    report_row1 = dbc.Row([
        html.H4(class_id,style={"padding-top":"1%","font-family":"Poppins"}),
        dbc.Col(images_card,className="col-md-2"),
        dbc.Col(objects_card,className="col-md-2"),
        dbc.Col(min_bbox_width_card,className="col-md-2"),
        dbc.Col(min_bbox_height_card,className="col-md-2"),
        dbc.Col(max_bbox_width_card,className="col-md-2"),
        dbc.Col(max_bbox_height_card,className="col-md-2")],
    className="row d-lg-none d-xxl-flex",
    )

    images = analysis["images"]
    image_cols = []
    for image in selected_class["images"]:
        file_name = images[str(image)]["file_name"]
        file_name_string = file_name.split('/')[-1]
        encoded_image = base64.b64encode(open(file_name,'rb').read())
        img_src  = "data:image/png;base64,{}".format(encoded_image.decode())
        img_card = dbc.Card([
            dbc.CardImg(src=img_src),
            dbc.CardBody(file_name_string)],
        className="card flex-fill"
        )
        img_col = dbc.Col(img_card,className="col-md-4")
        image_cols.append(img_col)

    report_row2 = dbc.Row(image_cols,className="row d-lg-none d-xxl-flex")

    contents = html.Div([report_row1, report_row2])

    return contents

    