import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html 
import pandas as pd
import plotly.express as px
import json

def warnings_contents(analysis_path):
    """
    Generates the contents of the warnings menu

    :param analysis_path: the path to the analysis file
    :return: warnings contents
    """
    f = open(analysis_path,'r')
    analysis = json.load(f)

    ###########################################################################
    # Class warnings: empty, missing, over/under-represented

    classes = analysis["classes"]
    total_num_objects = int(analysis["total_num_objects"])

    # Empty classes
    empty_classes = analysis["empty_classes"]
    
    if len(empty_classes) == 0:
        empty_classes_count = html.Div(style={"display":"none"})
        empty_classes_list = html.P("None")
    else:
        empty_classes_count = html.P("Total: " + str(len(empty_classes)))
        list_group_items = []
        for cl in empty_classes:
            class_name = classes[cl]["name"]
            item = dbc.ListGroupItem(class_name)
            list_group_items.append(item)
        empty_classes_list = dbc.ListGroup(list_group_items, flush=True)

    empty_classes_card = dbc.Card(
        dbc.CardBody([
            html.H5("Empty classes", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("Classes that are in the dataset but have no annotations",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            empty_classes_count,
            empty_classes_list],
        ),
        className="card flex-fill"
    )

    # Missing classes
    missing_classes = analysis["missing_classes"]

    if len(missing_classes) == 0:
        missing_classes_count = html.Div(style={"display":"none"})
        missing_classes_list = html.P("None")
    else:
        missing_classes_count = html.P("Total: " + str(len(missing_classes)))
        list_group_items = []
        for cl in missing_classes:
            item = dbc.ListGroupItem(cl)
            list_group_items.append(item)
        empty_classes_list = dbc.ListGroup(list_group_items, flush=True)

    missing_classes_card = dbc.Card(
        dbc.CardBody([
            html.H5("Missing classes", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("Class IDs that are referenced in annotations but are not in the dataset",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            missing_classes_count,
            missing_classes_list],
        ),
        className="card flex-fill"
    )

    # Over/under-represented classes
    
    under_represented_classes = {}
    over_represented_classes = {}
    for cl in classes:
        class_name = classes[cl]["name"]
        num_objects = int(classes[cl]["num_objects"])
        class_representation_prop = (num_objects*100)/total_num_objects
        if class_representation_prop < 5:
            under_represented_classes[class_name] = {}
            under_represented_classes[class_name]["id"] = cl
            under_represented_classes[class_name]["prop"] = "{0:.2f}%".format(class_representation_prop)
        elif class_representation_prop > 80:
            over_represented_classes[class_name]
            over_represented_classes[class_name]["id"] = cl
            over_represented_classes[class_name]["prop"] = "{0:.2f}%".format(class_representation_prop)

    if len(list(over_represented_classes.keys())) == 0:
        over_represented_classes_count = html.Div(style={"display":"none"})
        over_represented_classes_table = html.P("None")
    else:
        over_represented_classes_count = html.P("Total: " + str(len(list(over_represented_classes.keys()))))
        header = [
            html.Thead(
                html.Tr([
                    html.Th("ID"),
                    html.Th("Class name"),
                    html.Th("Proportion")
                ])
            )
        ]
        cl_rows = []
        for class_name in over_represented_classes:
            row = html.Tr([
                html.Td(over_represented_classes[class_name]["id"]),
                html.Td(class_name),
                html.Td(over_represented_classes[class_name])
            ])
            cl_rows.append(row)
        body = [html.Tbody(cl_rows)]
        over_represented_classes_table = dbc.Table(header+body)
    
    over_represented_classes_card = dbc.Card(
        dbc.CardBody([
            html.H5("Over-represented classes", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("Classes whose annotations make up more than 80% of the dataset",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            over_represented_classes_count,
            over_represented_classes_table],
        ),
        className="card flex-fill"
    )

    if len(list(under_represented_classes.keys())) == 0:
        under_represented_classes_count = html.Div(style={"display":"none"})
        under_represented_classes_table = html.P("None")
    else:
        under_represented_classes_count = html.P("Total: " + str(len(list(under_represented_classes.keys()))))
        header = [
            html.Thead(
                html.Tr([
                    html.Td("ID"),
                    html.Th("Class name"),
                    html.Th("Proportion")
                ])
            )
        ]
        cl_rows = []
        for class_name in under_represented_classes:
            row = html.Tr([
                html.Td(under_represented_classes[class_name]["id"]),
                html.Td(class_name),
                html.Td(under_represented_classes[class_name]["prop"])
            ])
            cl_rows.append(row)
        body = [html.Tbody(cl_rows)]
        under_represented_classes_table = dbc.Table(header+body)
    
    under_represented_classes_card = dbc.Card(
        dbc.CardBody([
            html.H5("Under-represented classes", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("Classes whose annotations make up less than 5% of the dataset",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            under_represented_classes_count,
            under_represented_classes_table],
        ),
        className="card flex-fill"
    )

    class_warnings_cards = dbc.Collapse([
        empty_classes_card,
        missing_classes_card,
        over_represented_classes_card,
        under_represented_classes_card],
    id="class-warnings-collapse",
    is_open=False
    )

    class_warnings_collapse = dbc.Row([
        html.H4("Classes",style={"font-weight":"500"},
            className="col-auto d-none d-sm-block"
        ),
        dbc.Button(
            html.Img(src="assets/icons/menu.svg",
                style={"width":"15px","padding-bottom":"10px","margin-right":"5px"}),
            style = {"width":"1.5%","background":"transparent","border":"transparent"},
            id="class-warnings-collapse-button")],
    )
    
    class_warnings = html.Div([
        class_warnings_collapse,
        class_warnings_cards
    ])

    ###########################################################################
    # ID warnings: unused, repeated

    # Unused IDs
    unused_ids = analysis["unused_IDs"]

    if len(unused_ids) == 0:
        unused_ids_count = html.Div(style={"display":"none"})
        unused_ids_list = html.P("None")
    else:
        unused_ids_count = html.P("Total: " + str(len(unused_ids)))
        list_group_items = []
        for cl_id in unused_ids:
            item = dbc.ListGroupItem(cl_id)
            list_group_items.append(item)
        unused_ids_list = dbc.ListGroup(list_group_items, flush=True)

    unused_ids_card = dbc.Card(
        dbc.CardBody([
            html.H5("Unused IDs", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("Available IDs that are not used",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            unused_ids_count,
            unused_ids_list],
        ),
        className="card flex-fill"
    )

    # Repeated IDs

    repeated_ids = analysis["repeated_IDs"]

    if len(repeated_ids) == 0:
        repeated_ids_count = html.Div(style={"display":"none"})
        repeated_ids_list = html.P("None")
    else:
        repeated_ids_count = html.P("Total: " + str(len(repeated_ids)))
        list_group_items = []
        for cl_id in unused_ids:
            item = dbc.ListGroupItem(cl_id)
            list_group_items.append(item)
        repeated_ids_list = dbc.ListGroup(list_group_items, flush=True)

    repeated_ids_card = dbc.Card(
        dbc.CardBody([
            html.H5("Repeated IDs", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("IDs that are used for more than one class",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            repeated_ids_count,
            repeated_ids_list],
        ),
        className="card flex-fill"
    )

    id_warnings_cards = dbc.Collapse([
        unused_ids_card,
        repeated_ids_card],
    id="id-warnings-collapse",
    is_open=False
    )

    id_warnings_collapse = dbc.Row([
        html.H4("IDs",style={"font-weight":"500"},
            className="col-auto d-none d-sm-block"
        ),
        dbc.Button(
            html.Img(src="assets/icons/menu.svg",
                style={"width":"15px","padding-bottom":"10px","margin-right":"5px"}),
            style = {"width":"1.5%","background":"transparent","border":"transparent"},
            id="id-warnings-collapse-button")],
    )
    
    id_warnings = html.Div([
        id_warnings_collapse,
        id_warnings_cards
    ])

    ###########################################################################
    # Image warnings: empty, missing, wrong dims

    images = analysis["images"]

    # Empty images
    empty_images = analysis["empty_images"]
    
    if len(empty_images) == 0:
        empty_images_count = html.Div(style={"display":"none"})
        empty_images_list = html.P("None")
    else:
        empty_images_count = html.P("Total: " + str(len(empty_images)))
        list_group_items = []
        for image in empty_images:
            file_name = images[str(image)]["file_name"]
            file_name = file_name.split('/')[-1]
            item = dbc.ListGroupItem(file_name)
            list_group_items.append(item)
        empty_images_list = dbc.ListGroup(list_group_items, flush=True)

    empty_images_card = dbc.Card(
        dbc.CardBody([
            html.H5("Empty images", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("Images that contain no annotations",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            empty_images_count,
            empty_images_list],
        ),
        className="card flex-fill"
    )

    # Missing images
    missing_images = analysis["missing_images"]

    if len(missing_images) == 0:
        missing_images_count = html.Div(style={"display":"none"})
        missing_images_list = html.P("None")
    else:
        missing_images_count = html.P("Total: " + str(len(missing_images)))
        list_group_items = []
        for image in missing_images:
            file_name = images[str(image)]["file_name"]
            file_name = file_name.split('/')[-1]
            item = dbc.ListGroupItem(file_name)
            list_group_items.append(item)
        missing_images_list = dbc.ListGroup(list_group_items, flush=True)

    missing_images_card = dbc.Card(
        dbc.CardBody([
            html.H5("Missing images", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("Images that are in the dataset but whose file is missing",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            missing_images_count,
            missing_images_list],
        ),
        className="card flex-fill"
    )

    # Wrong dims
    wrong_dims = analysis["wrong_dims"]

    if len(wrong_dims) == 0:
        wrong_dims_count = html.Div(style={"display":"none"})
        wrong_dims_list = html.P("None")
    else:
        wrong_dims_count = html.P("Total: " + str(len(wrong_dims)))
        list_group_items = []
        for image in wrong_dims:
            file_name = images[str(image)]["file_name"]
            file_name = file_name.split('/')[-1]
            item = dbc.ListGroupItem(file_name)
            list_group_items.append(item)
        wrong_dims_list = dbc.ListGroup(list_group_items, flush=True)

    wrong_dims_card = dbc.Card(
        dbc.CardBody([
            html.H5("Images with wrong dimensions", 
                className="card-title",
                style={"color":"black","font-weight":"600"}
            ),
            html.H6("Images that are not 1920x1080",
                className="text-muted",
                style={"font-weight":"normal","font-size":"0.8rem"}
            ),
            wrong_dims_count,
            wrong_dims_list],
        ),
        className="card flex-fill"
    )

    image_warnings_cards = dbc.Collapse([
        empty_images_card,
        missing_images_card,
        wrong_dims_card],
    id="image-warnings-collapse",
    is_open=False
    )

    image_warnings_collapse = dbc.Row([
        html.H4("Images",style={"font-weight":"500"},
            className="col-auto d-none d-sm-block"
        ),
        dbc.Button(
            html.Img(src="assets/icons/menu.svg",
                style={"width":"15px","padding-bottom":"10px","margin-right":"5px"}),
            style = {"width":"1.5%","background":"transparent","border":"transparent"},
            id="image-warnings-collapse-button")],
    )
    
    image_warnings = html.Div([
        image_warnings_collapse,
        image_warnings_cards
    ])

    contents = html.Div([
        html.H3("Warnings",style={"font-weight":"500"}),
        class_warnings,
        id_warnings,
        image_warnings
    ])

    return contents