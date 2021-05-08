import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import json

def dashboard_contents(analysis_path):
    """
    Generates the contents of the dashboard menu

    :param analysis_path: the path to the analysis file
    :return: dashboard contents
    """
    f = open(analysis_path, 'r')
    analysis = json.load(f)

    # Popover user guide
    def make_popover(popid, feature, content):
        return dbc.Popover(
            [
                dbc.PopoverHeader(f"{feature}"),
                dbc.PopoverBody(f"{content}"),
            ],
            id=f"popover-{popid}",
            is_open=False,
            target=f"popover-{popid}-target",
        )

    def make_button(popid):
        return dbc.Button(
            "?", id=f"popover-{popid}-target", outline=True, color="info", size="sm"
        )

    ###########################################################################
    # First row: dataset info, num images, num objects, objects distribution

    # Dataset info 
    info = analysis["info"]

    name = info.get("description", "Dataset")
    contributor = "by " + info.get("contributor", "N.A.")
    name_card = dbc.Card(
        dbc.CardBody([
            html.H5("Dataset name", className="card-title"),
            html.H4(name, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(contributor, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )

    year = info.get("year", "N.A.")
    version = "version " + info.get("version", "N.A.")
    year_card = dbc.Card(
        dbc.CardBody([
            html.H5("Year", className="card-title"),
            html.H4(year, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(version, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )

    name_year_row = dbc.Row([
        dbc.Col(name_card, className="col-sm-6 col-lg-6 col-xxl-6 d-flex"),
        dbc.Col(year_card, className="col-sm-6 col-lg-6 col-xxl-6 d-flex")],
        className="row d-xxl-flex"
    )

    # Num images and num objects
    total_num_images = analysis["total_num_images"]
    empty_images = str(len(analysis["empty_images"])) + " empty"
    missing_images = str(len(analysis["missing_images"])) + " missing"

    popnumim = html.Div(
        [
            make_button("numim"),
            make_popover("numim", "Number of Images", \
                         "There are " + total_num_images + " images in this dataset, \
                    with " + empty_images + " image(s) which do not contain any objects \
                        and " + missing_images + " image(s) that are not in the dataset \
                            but in the annotations"),
        ]
    )

    images_card = dbc.Card(
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5("Number of images", className="card-title")),
                    dbc.Col(popnumim),
                ],
                justify="between",
            ),
            html.H4(total_num_images, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(empty_images, className="text-muted", style={"font-weight": "normal"}),
            html.H6(missing_images, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )

    total_num_objects = analysis["total_num_objects"]
    min_per_image = "min " + analysis["objects_per_image_stats"]["min"] + " per image"
    max_per_image = "max " + analysis["objects_per_image_stats"]["max"] + " per image"
    average = "average " + analysis["objects_per_image_stats"]["avg"] + " per image"

    popnumob = html.Div(
        [
            make_button("numob"),
            make_popover("numob", "Number of Objects", \
                         "There are " + total_num_objects + " objects in this dataset. \
                    An image can contain from " + analysis["objects_per_image_stats"]["min"] \
                         + " object(s) to " + analysis["objects_per_image_stats"]["max"] + " objects."),
        ]
    )

    objects_card = dbc.Card(
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5("Number of objects", className="card-title")),
                    dbc.Col(popnumob),
                ],
                justify="between",
            ),
            html.H4(total_num_objects, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(min_per_image, className="text-muted", style={"font-weight": "normal"}),
            html.H6(max_per_image, className="text-muted", style={"font-weight": "normal"}),
            html.H6(average, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )

    images_objects_row = dbc.Row([
        dbc.Col(images_card, className="col-sm-6 col-lg-6 col-xxl-6 d-flex"),
        dbc.Col(objects_card, className="col-sm-6 col-lg-6 col-xxl-6 d-flex")],
        className="row d-xxl-flex"
    )

    overview_cards = html.Div([
        html.Div([
            name_year_row,
            images_objects_row
        ],
            className="w-100")
    ],
        className="col-sm-12 col-md-12 col-lg-6 col-xl-5 d-flex")

    # Objects distribution
    classes = analysis["classes"]

    class_names = []
    num_objects = []
    num_images = []
    for cl in classes:
        class_names.append(classes[cl]["name"])
        num_objects.append(classes[cl]["num_objects"])
        num_images.append(len(classes[cl]["images"]))

    class_df = pd.DataFrame()
    class_df["class"] = class_names
    class_df["num objects"] = num_objects
    class_df["num images"] = num_images

    graph_title = "Objects distribution"
    graph_bar = px.bar(class_df,
                       x="class",
                       y="num objects",
                       height=269
                       )
    graph_bar.update_layout(xaxis_tickangle=-45,
                            margin_t=50, font_size=10, font_color="black")

    graph = dcc.Graph(figure=graph_bar)

    popobdis = html.Div(
        [
            make_button("objectsDistribution"),
            make_popover("objectsDistribution", "Distribution of Objects", \
                         "This graph displays the number of occurrences of each object class \
                    in this dataset. Move the cursor on each bin or click on Stats for more details."),
        ]
    )

    graph_card = dbc.Card([
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5(graph_title, className="card-title")),
                    dbc.Col(popobdis),
                ],
                justify="between",
            ),
            html.Div(graph)],
            className="card-body")],
        className="card flex-fill w-100"
    )

    graph_col = html.Div(graph_card, className="col-sm-12 col-md-12 col-lg-6 col-xl-7 d-flex")

    # Wrap it all together

    first_row = dbc.Row([
        overview_cards,
        graph_col
    ])

    ###########################################################################
    # Second row: example images, num classes, IDs range, min/max bbox dims

    graph2_title = "Images distribution"
    graph2_bar = px.bar(class_df,
                        x="class",
                        y="num images",
                        height=269
                        )
    graph2_bar.update_layout(xaxis_tickangle=-45,
                             margin_t=50, font_size=10, font_color="black")

    graph2 = dcc.Graph(figure=graph2_bar)

    popimdis = html.Div(
        [
            make_button("imageDistribution"),
            make_popover("imageDistribution", "Distribution of Images", \
                         "This graph displays the number of images that contain each object class \
                    in this dataset. Move the cursor on each bin or click on Stats for more details."),
        ]
    )

    graph2_card = dbc.Card([
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5(graph2_title, className="card-title")),
                    dbc.Col(popimdis),
                ],
                justify="between",
            ),
            html.Div(graph2)],
            className="card-body")],
        className="card flex-fill w-100"
    )

    graph2_col = html.Div(graph2_card, className="col-sm-12 col-md-12 col-lg-6 col-xl-7 d-flex")

    # Num classes, IDs range
    num_classes = str(len(class_names))
    empty_classes = str(len(analysis["empty_classes"])) + " empty"
    missing_classes = str(len(analysis["missing_classes"])) + " missing"

    popclass = html.Div(
        [
            make_button("class"),
            make_popover("class", "Number of Classes", \
                         "This dataset contains " + num_classes + " different objects in the annotation, \
                    with " + empty_classes + " class(es) that are listed but not referenced in the annotations,\
                        and " + missing_classes + " class(es) that are referenced in any image annotation but whose "
                                                  "ids were not listed. "),
        ]
    )

    classes_card = dbc.Card(
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5("Number of classes", className="card-title")),
                    dbc.Col(popclass),
                ],
                justify="between",
            ),
            html.H4(num_classes, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(empty_classes, className="text-muted", style={"font-weight": "normal"}),
            html.H6(missing_classes, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )

    ids_range = analysis["ids_range"]
    unused_ids = str(len(analysis["unused_IDs"])) + " unused"
    repeated_ids = str(len(analysis["repeated_IDs"])) + " repeated"
    popids = html.Div(
        [
            make_button("idrange"),
            make_popover("idrange", "Range of IDs", \
                         "This dataset contains different classes with IDs range from " + ids_range + ", \
                    of which " + str(len(analysis["unused_IDs"])) + " are never referenced and \
                       " + str(len(analysis["repeated_IDs"])) + " class(es) have repeated IDs. "),
        ]
    )

    ids_card = dbc.Card(
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5("IDs range", className="card-title")),
                    dbc.Col(popids),
                ],
                justify="between",
            ),
            html.H4(ids_range, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(unused_ids, className="text-muted", style={"font-weight": "normal"}),
            html.H6(repeated_ids, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )

    class_ids_row = dbc.Row([
        dbc.Col(classes_card, className="col-sm-6 col-lg-6 col-xxl-6 d-flex"),
        dbc.Col(ids_card, className="col-sm-6 col-lg-6 col-xxl-6 d-flex")],
        className="row d-xxl-flex"
    )

    popminbbox = html.Div(
        [
            make_button("minbbox"),
            make_popover("minbbox", "Minimum Bounding Box", \
                         "The dimension of smallest object in the dataset. \
                            Bounding boxes are rectangles drawn on object outlines \
                                in images as a point of reference"),
        ]
    )

    popmaxbbox = html.Div(
        [
            make_button("maxbbox"),
            make_popover("maxbbox", "Maximum Bounding Box", \
                         "The dimension of biggest object in the dataset. \
                            Bounding boxes are rectangles drawn on object outlines \
                                in images as a point of reference"),
        ]
    )

    # Min/max bbox dims
    min_bbox_dims = analysis["bbox_stats"]["min"]
    min_bbox_class = analysis["bbox_stats"]["min_class"]
    min_bbox_card = dbc.Card(
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5("Min bbox dimensions", className="card-title")),
                    dbc.Col(popminbbox),
                ],
                justify="between",
            ),
            html.H4(min_bbox_dims, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(min_bbox_class, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )

    max_bbox_dims = analysis["bbox_stats"]["max"]
    max_bbox_class = analysis["bbox_stats"]["max_class"]
    max_bbox_card = dbc.Card(
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5("Max bbox dimensions", className="card-title")),
                    dbc.Col(popmaxbbox),
                ],
                justify="between",
            ),
            html.H4(max_bbox_dims, className="h2 d-inline-block mt-1 mb-4"),
            html.H6(max_bbox_class, className="text-muted", style={"font-weight": "normal"})],
        ),
        className="card flex-fill"
    )

    bbox_row = dbc.Row([
        dbc.Col(min_bbox_card, className="col-sm-6 col-lg-6 col-xxl-6 d-flex"),
        dbc.Col(max_bbox_card, className="col-sm-6 col-lg-6 col-xxl-6 d-flex")],
        className="row d-xxl-flex"
    )

    overview_cards2 = html.Div([
        html.Div([
            class_ids_row,
            bbox_row
        ],
            className="w-100")
    ],
        className="col-sm-12 col-md-12 col-lg-6 col-xl-5 d-flex")

    # Wrap it all together
    second_row = dbc.Row([
        graph2_col,
        overview_cards2
    ])

    ###########################################################################
    # Compose contents div
    contents = html.Div([
        html.H3("Dashboard", style={"font-weight": "500"}),
        first_row,
        second_row
    ])

    return contents
