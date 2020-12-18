import base64, json, os
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html 
import plotly.express as px

from collections import OrderedDict 
from dash.dependencies import Input, Output, State
from flask import send_from_directory
from urllib.parse import quote as urlquote
from utils import parse_annotations, compute_overview_data
from app.analysis import analyze_dataset

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])
app.config["suppress_callback_exceptions"] = True
application = app.server
port = 80
output_dir = os.path.join(os.getcwd(), "output")

reload_btn_img = "https://cdn1.iconfinder.com/data/icons/arrows-elements-outline/128/ic_round_update-128.png"
warnings_download_btn_img = "https://cdn0.iconfinder.com/data/icons/essentials-9/128/__Download-128.png"
warnings_more_btn_img = "https://cdn1.iconfinder.com/data/icons/jumpicon-basic-ui-line-1/32/-_Hamburger-Menu-More-Navigation--128.png"

navbar = dbc.Navbar(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.NavbarBrand("VIZ EDA", 
                        style = {"font-size" : "2.5rem", 
                            "white-space" : "pre-wrap",
                            "font-weight" : "normal",
                            "font-family" : "sans-serif",
                            "letter-spacing" : "2px"
                        }
                    ), 
                    width = "auto", 
                    style  = {"margin-left" : "25px"}
                ),
                dbc.Col(
                    html.H5("Exploratory data analysis for computer vision",
                        style = {
                            "color" : "#424242",
                            "margin-top" : "0.7%",
                            "font-weight" : "100",
                            "text-transform" : "lowercase",
                            "letter-spacing" : "4px"
                        }
                    )
                ),
                dbc.Col(
                    html.A(
                        html.Img(
                            src = reload_btn_img,
                            style = {
                                "height" : "25%",
                                "width" : "25%",
                                "float" : "right",
                                "margin-right" : "25px"
                            }
                        ),
                        id = "reload-btn",
                        href = "http://localhost:8080/"
                    ),
                    width = "auto"
                )
            ],
            align = "center",
            no_gutters = True,
            style = {
                "width" : "100%"
            }
        )
    ],
    style = {
        "padding" : "0.1rem"
    }
)

welcome_menu = html.Div(
    [
        html.Div(
            dbc.Button("New analysis",
                color = "dark",
                className = "mr-1",
                outline = True,
                style = {
                    "float" : "right",
                    "width" : "25%",
                    "letter-spacing" : "2px"
                },
                id = "new-analysis-btn"
            ),
            style = {
                "width" : "50%",
                "margin" : "auto"
            }
        ),
        html.Div(
            dbc.Button("Existing analysis",
                color = "dark",
                className = "mr-1",
                outline = True,
                style = {
                    "width" : "25%",
                    "letter-spacing" : "2px"
                },
                id = "existing-analysis-btn"
            ),
            style = {
                "width" : "50%",
                "margin" : "auto"
            }
        )
    ],
    style = {
        "display" : "flex",
        "padding-top" : "22.5%"
    },
    id = "welcome-menu"
)

new_analysis_menu = html.Div(
    [
        dbc.FormGroup(
            [
                dbc.Checklist(
                    options = [{
                        "label" : "Batch analysis",
                        "value" : 1
                    }],
                    value = [],
                    switch = True,
                    id = "batch-analysis-switch"
                )
            ],
            style = {
                "margin" : "auto",
                "padding-bottom" : "0.8%",
                "width" : "22%"
            }
        ),
        html.Div(
            [
                dbc.Input(
                    type = "text",
                    placeholder = "Path to images e.g. /Users/me/project/data/train_images",
                    className = "mb-3",
                    id = "images-upload",
                    style = {
                        "width" : "100%",
                        "letter-spacing" : "1px"
                    }
                )
            ],
            style = {
                "margin" : "auto",
                "width" : "22%",
                "padding-bottom" : "0.28%"
            }
        ),
        dcc.Upload(
            [
                dbc.Button("Upload annotation file (.json)",
                    color = "dark",
                    className = "mr-1",
                    outline = True,
                    style = {
                        "width" : "100%",
                        "letter-spacing" : "2px"
                    },
                    id = "annotations-upload-btn"
                )
            ],
            multiple = False,
            id = "annotations-upload",
            style = {
                "margin" : "auto",
                "width" : "22%",
                "padding-bottom" : "0.9%"
            }
        ),
        html.Div(
            [
                dbc.Button(
                    "Analyze",
                    color = "dark",
                    className = "mr-1",
                    style = {
                        "width" : "100%",
                        "letter-spacing" : "2px"
                    },
                    id = "analyze-btn"
                )
            ],
            style = {
                "margin" : "auto",
                "width" : "22%"
            }
        )
    ],
    id = "new-analysis-menu"
)

tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(
                    label = "Overview",
                    tab_id = "overview-tab",
                    style = {
                        "letter-spacing" : "2px"
                    }
                ),
                dbc.Tab(
                    label = "Class statistics",
                    tab_id = "class-stats-tab",
                    style = {
                        "letter-spacing" : "2px"
                    }
                ),
                dbc.Tab(
                    label = "Visualize dataset",
                    tab_id = "visualize-dataset-tab",
                    style = {
                        "letter-spacing" : "2px"
                    }
                ),
                dbc.Tab(
                    label = "Anomaly detection",
                    tab_id = "anomaly-detection-tab",
                    style = {
                        "letter-spacing" : "2px"
                    }
                )
            ],
            id = "tabs",
            style = {
                "display" : "flex",
                "justify-content" : "space-around",
                "padding-top" : "0.5%",
                "padding-bottom" : "0.8%",
                "border-bottom" : "1px solid #bababa",
                "border-radius" : "10px"
            }
        ),
        html.Div(
            id = "tabs-content",
            style = {
                "padding-left" : "1%",
                "padding-right" : "1%",
                "padding-top" : "1%",
                "background" : "white"
            }
        )
    ],
    id = "tabs-div"
)

analysis_data = html.Div(
    id = "analysis-data",
    style = {
        "display" : "none"
    }
)

loading = dcc.Loading(analysis_data,
    id = "loading",
    color = "#4bbf73",
    style = {
        "padding-top" : "50%"
    }
)

@app.callback(
    Output("images-upload", "valid"),
    Input("images-upload", "value")
)
def check_images_path(path):
    if path is None:
        path = ""
    if os.path.isdir(path):
        return True
    else:
        return False

@app.callback(
    Output("annotations-upload-btn", "color"),
    Input("annotations-upload", "contents")
)
def check_annotations(contents):
    if contents is None:
        return "dark"
    else:
        content_type, _ = contents.split(",", 1)
        if content_type == "data:application/json;base64":
            return "success"
        else:
            return "danger"

@app.callback(
    Output("analyze-btn", "disabled"),
    Output("analyze-btn", "color"),
    Input("images-upload", "valid"),
    Input("annotations-upload-btn", "color")
)
def check_inputs(valid_path, btn_color):
    if valid_path and btn_color == "success":
        return False, "success"
    else:
        return True, "dark"

@app.callback(
    Output("analysis-data", "children"),
    Output("tabs-div", "style"),
    Input("analyze-btn", "n_clicks"),
    Input("images-upload", "value"),
    Input("annotations-upload", "contents")
)
def store_analysis(analyze_click, images, annotations):
    data = {
        "overview" : "",
        "class_stats" : {}
    }
    tabs_div_style = {
        "display" : "none"
    }
    if analyze_click:
        path_to_annotations = parse_annotations(annotations)
        path_to_overview_data = compute_overview_data(images, path_to_annotations)
        class_stats, _ = analyze_dataset(path_to_annotations, images)
        class_stats = class_stats.to_dict()
        data["overview"] = path_to_overview_data
        data["class_stats"] = class_stats
        tabs_div_style = {
            "display" : "block",
            "background" : "white",
            "padding-top" : "0.3%"
        }
    data = json.dumps(data)
    return data, tabs_div_style

@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "active_tab"),
    Input("analysis-data", "children"),
    Input("analyze-btn", "n_clicks")
)
def display_tabs_contents(active_tab, data, analyze_click):
    contents = html.Div()
    data = json.loads(data)
    if data["overview"] != "" and data["class_stats"]:
        path_to_overview_data = data["overview"]
        class_stats = data["class_stats"]
        if active_tab == "overview-tab":
            contents = render_overview(path_to_overview_data)
        elif active_tab == "class-stats-tab":
            contents = render_class_stats(class_stats)
        elif active_tab == "visualize-dataset-tab":
            contents = render_dataset_visualizer(path_to_overview_data)
        elif active_tab == "anomaly-detection-tab":
            contents = render_anomaly_detection(class_stats)
    return contents

@app.callback(
    Output("reload-btn", "style"),
    Output("new-analysis-menu", "style"),
    Output("welcome-menu", "style"),
    Input("new-analysis-btn", "n_clicks"),
    Input("existing-analysis-btn", "n_clicks"),
    Input("analyze-btn", "n_clicks")
)
def control_menu_display(new_click, existing_click, analyze_click):
    if new_click:
        reload_btn = {
            "display" : "block"
        }
        new_analysis_menu = {
            "display" : "block",
            "padding-top" : "18%"
        }
        welcome_menu = {
            "display" : "none"
        }
    elif existing_click:
        reload_btn = {
            "display" : "block"
        }
        new_analysis_menu = {
            "display" : "none"
        }
        welcome_menu = {
            "display" : "none"
        }
    else:
        reload_btn = {
            "display" : "none"
        }
        new_analysis_menu = {
            "display" : "none",
            "padding-top" : "18%"
        }
        welcome_menu = {
            "display" : "flex",
            "padding-top" : "22.5%"
        }
    if analyze_click:
        reload_btn = {
            "display" : "block"
        }
        new_analysis_menu = {
            "display" : "none"
        }
        welcome_menu = {
            "display" : "none"
        }
    return reload_btn, new_analysis_menu, welcome_menu

def render_overview(path_to_overview_data):
    cwd = os.getcwd()
    path_to_overview_data = path_to_overview_data.split('/')
    path_to_overview_data = os.path.join(cwd,path_to_overview_data[1],path_to_overview_data[2])
    overview_file = open(path_to_overview_data)
    overview_data = json.load(overview_file)

    if "description" in overview_data["info"]:
        section_title = overview_data["info"]["description"] + " overview"
    else:
        section_title = "Dataset overview"

    info_table_header = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Info",
                        style = {
                            "border-top-right-radius" : "0px",
                            "font-size" : "large",
                            "font-weight" : "900",
                            "background" : "#4a7"
                        }
                    ),
                    html.Th("",
                        style = {
                            "border-top-left-radius" : "0px",
                            "background" : "#4a7"
                        }
                    )
                ]
            )
        )
    ]

    if "description" in overview_data["info"]:
        dataset_name = overview_data["info"]["description"]
    else:
        dataset_name = "N.A"
    i_row1 = html.Tr(
        [
            html.Td("Dataset name"),
            html.Td(dataset_name,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    if "url" in overview_data["info"]:
        dataset_url = overview_data["info"]["url"]
    else:
        dataset_url = "N.A"
    i_row2 = html.Tr(
        [
            html.Td("URL"),
            html.Td(dataset_url,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    if "version" in overview_data["info"]:
        dataset_version = overview_data["info"]["version"]
    else:
        dataset_version = "N.A"
    i_row3 = html.Tr(
        [
            html.Td("Version"),
            html.Td(dataset_version,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    if "year" in overview_data["info"]:
        dataset_year = overview_data["info"]["year"]
    else:
        dataset_year = "N.A"
    i_row4 = html.Tr(
        [
            html.Td("Year"),
            html.Td(dataset_year,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    if "contributor" in overview_data["info"]:
        dataset_contributor = overview_data["info"]["contributor"]
    else:
        dataset_contributor = "N.A"
    i_row5 = html.Tr(
        [
            html.Td("Contributor"),
            html.Td(dataset_contributor,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    if "date_created" in overview_data["info"]:
        dataset_creation_date = overview_data["info"]["date_created"]
    else:
        dataset_creation_date = "N.A"
    i_row6 = html.Tr(
        [
            html.Td("Date created"),
            html.Td(dataset_creation_date,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    info_table_body = [html.Tbody(
        [
            i_row1, 
            i_row2, 
            i_row3, 
            i_row4, 
            i_row5, 
            i_row6
        ]
    )]

    info_table = dbc.Table(info_table_header + info_table_body, 
        striped = True, 
        bordered = True, 
        hover = True, 
        style = {
            "width" : "30%"
        }
    )

    summary_table_header = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Summary",
                        style = {
                            "border-top-right-radius" : "0px",
                            "font-size" : "large",
                            "font-weight" : "900",
                            "background" : "#00aedb"
                        }
                    ),
                    html.Th("",
                        style = {
                            "border-top-left-radius" : "0px",
                            "background" : "#00aedb"
                        }
                    )
                ]
            )
        )
    ]

    no_classes = str(len(list(overview_data["classes"])))
    s_row1 = html.Tr(
        [
            html.Td("No. classes"),
            html.Td(no_classes,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    no_objects = str(overview_data["no_objects"])
    s_row2 = html.Tr(
        [
            html.Td("No. objects"),
            html.Td(no_objects,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    no_images = str(overview_data["no_images"])
    s_row3 = html.Tr(
        [
            html.Td("No. images"),
            html.Td(no_images,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    min_objects_per_image = str(overview_data["min_objects_per_image"])
    s_row4 = html.Tr(
        [
            html.Td("Min. objects per image"),
            html.Td(min_objects_per_image,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    max_objects_per_image = str(overview_data["max_objects_per_image"])
    s_row5 = html.Tr(
        [
            html.Td("Max. objects per image"),
            html.Td(max_objects_per_image,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    avg_objects_per_image = "{0:.2f}".format(overview_data["avg_objects_per_image"])
    s_row6 = html.Tr(
        [
            html.Td("Avg. objects per image"),
            html.Td(avg_objects_per_image,
                style = {
                    "font-weight":"bold",
                    "text-align":"right"
                }
            )
        ]
    )

    summary_table_body = [html.Tbody(
        [
            s_row1, 
            s_row2, 
            s_row3, 
            s_row4, 
            s_row5, 
            s_row6
        ]
    )]

    summary_table = dbc.Table(summary_table_header + summary_table_body, 
        striped = True, 
        bordered = True, 
        hover = True, 
        style = {
            "width" : "30%"
        }
    )

    path_to_warnings = os.path.join(output_dir, "warnings.json")
    warnings_file = open(path_to_warnings, 'w')

    warnings_dict = {}

    warnings_table_header = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Warnings",
                        style = {
                            "border-top-right-radius" : "0px",
                            "font-size" : "large",
                            "font-weight" : "900",
                            "background" : "#ffcc5c"
                        }
                    ),
                    html.Th("",
                        style = {
                            "border-top-right-radius" : "0px",
                            "border-top-left-radius" : "0px",
                            "background" : "#ffcc5c"
                        }
                    ),
                    html.Th(
                        html.A(
                            html.Img(
                                src = warnings_download_btn_img,
                                style = {
                                    "width" : "100%",
                                    "float" : "right",
                                    "padding-bottom" : "4px"
                                }
                            ),
                            href = "download/{}".format(urlquote("warnings.json"))
                        ),
                        style = {
                            "border-top-left-radius" : "0px",
                            "background" : "#ffcc5c"
                        }
                    )
                ]
            )
        )
    ]

    warnings_dict["class_distribution"] = {}
    uniform_distribution = overview_data["uniform_distribution"]
    if uniform_distribution:
        distribution_warning = "Uniform"
        warning_color = "green"
        description = "The classes are uniformly represented"
        warnings_dict["class_distribution"]["description"] = description
        uniform_modal_body = description
    else:
        distribution_warning = "Not uniform"
        warning_color = "red"
        description = "The following classes represent less than 5% or more than 80% of the dataset"
        warnings_dict["class_distribution"]["description"] = description
        under_repr_classes = []
        over_repr_classes = []
        for cl in overview_data["classes"]:
            if overview_data["classes"][cl]["prop"] < 5:
                under_repr_classes.append(overview_data["classes"][cl]["name"])
            if overview_data["classes"][cl]["prop"] > 80:
                over_repr_classes.append(overview_data["classes"][cl]["name"])
        warnings_dict["class_distribution"]["under_represented_classes"] = under_repr_classes
        warnings_dict["class_distribution"]["over_represented_classes"] = over_repr_classes
        uniform_modal_body = description + ":" + "\n\n"
        if len(under_repr_classes):
            uniform_modal_body += "Under-represented classes:\n"
            for cl in under_repr_classes:
                uniform_modal_body += cl + "\n"
            uniform_modal_body += "\n\n"
        if len(over_repr_classes):
            uniform_modal_body += "Over-represented classes:\n"
            for cl in over_repr_classes:
                uniform_modal_body += cl + "\n"
    w_row1 = html.Tr(
        [
            html.Td("Class distribution",
                style = {
                    "width" : "300px"
                }
            ),
            html.Td(distribution_warning,
                style = {
                    "color" : warning_color,
                    "font-weight": "bold",
                    "text-align": "right",
                    "width" : "200px",
                    "padding-right" : "0px"
                }
            ),
            html.Td(
                html.Button(id = "uniform-modal-btn",
                    children = [html.Img(src = warnings_more_btn_img,
                            style = {
                                "width" : "60%",
                                "float" : "right",
                                "padding-top" : "1.5px",
                                "padding-left" : "1px"
                            }
                        )
                    ],
                    style = {
                        "border" : "none",
                        "width" : "110%",
                        "background" : "none",
                        "padding-top" : "3px"
                    }
                ),
                style = {
                    "width" : "50px",
                    "padding-left" : "0px"
                }
            )
        ]
    )
    uniform_modal = dbc.Modal(
        [
            dbc.ModalHeader("Class distribution"),
            dbc.ModalBody(uniform_modal_body,
                style = {
                    "white-space" : "break-spaces"
                }
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id = "close-uniform-modal", className = "ml-auto"
                )
            ),
        ],
        id = "uniform-modal",
        scrollable = True,
        centered = True
    )

    warnings_dict["missing_images"] = {}
    missing_images = overview_data["missing_images"]
    if len(missing_images):
        missing_images_warning = str(len(missing_images))
        warning_color = "red"
        description = "The following images appear in the annotations but are not in the dataset"
        warnings_dict["missing_images"]["description"] = description
        warnings_dict["missing_images"]["image_list"] = missing_images
        missing_images_modal_body = description + ":" + "\n\n"
        for img in missing_images:
            missing_images_modal_body += img + "\n"
    else:
        missing_images_warning = "None"
        warning_color = "green"
        description = "All images in the annotations are in the dataset"
        warnings_dict["missing_images"]["description"] = description
        missing_images_modal_body = description
    w_row2 = html.Tr(
        [
            html.Td("Missing images"),
            html.Td(missing_images_warning,
                style = {
                    "color" : warning_color,
                    "font-weight": "bold",
                    "text-align": "right",
                    "padding-right" : "0px"
                }
            ),
            html.Td(
                html.Button(id = "missing-images-modal-btn",
                    children = [html.Img(src = warnings_more_btn_img,
                            style = {
                                "width" : "60%",
                                "float" : "right",
                                "padding-top" : "1.5px",
                                "padding-left" : "1px"
                            }
                        )
                    ],
                    style = {
                        "border" : "none",
                        "width" : "110%",
                        "background" : "none",
                        "padding-top" : "3px"
                    }
                ),
                style = {
                    "width" : "50px",
                    "padding-left" : "0px"
                }
            )
        ]
    )
    missing_images_modal = dbc.Modal(
        [
            dbc.ModalHeader("Missing images"),
            dbc.ModalBody(missing_images_modal_body,
                style = {
                    "white-space" : "break-spaces"
                }
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id = "close-missing-images-modal", className = "ml-auto"
                )
            ),
        ],
        id = "missing-images-modal",
        scrollable = True,
        centered = True
    )

    warnings_dict["empty_images"] = {}
    empty_images = overview_data["empty_images"]
    if len(empty_images):
        empty_images_warning = str(len(empty_images))
        warning_color = "red"
        description = "The following images are not annotated"
        warnings_dict["empty_images"]["description"] = description
        warnings_dict["empty_images"]["image_list"] = empty_images
        empty_images_modal_body = description + ":" + "\n\n"
        for img in empty_images:
            empty_images_modal_body += img + "\n"
    else:
        empty_images_warning = "None"
        warning_color = "green"
        description = "All images are annotated"
        warnings_dict["empty_images"]["description"] = description
        empty_images_modal_body = description
    w_row3 = html.Tr(
        [
            html.Td("Empty images"),
            html.Td(empty_images_warning,
                style = {
                    "color" : warning_color,
                    "font-weight": "bold",
                    "text-align": "right",
                    "padding-right" : "0px"
                }
            ),
            html.Td(
                html.Button(id = "empty-images-modal-btn",
                    children = [html.Img(src = warnings_more_btn_img,
                            style = {
                                "width" : "60%",
                                "float" : "right",
                                "padding-top" : "1.5px",
                                "padding-left" : "1px"
                            }
                        )
                    ],
                    style = {
                        "border" : "none",
                        "width" : "110%",
                        "background" : "none",
                        "padding-top" : "3px"
                    }
                ),
                style = {
                    "width" : "50px",
                    "padding-left" : "0px"
                }
            )
        ]
    )
    empty_images_modal = dbc.Modal(
        [
            dbc.ModalHeader("Empty images"),
            dbc.ModalBody(empty_images_modal_body,
                style = {
                    "white-space" : "break-spaces"
                }
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id = "close-empty-images-modal", className = "ml-auto"
                )
            ),
        ],
        id = "empty-images-modal",
        scrollable = True,
        centered = True
    )

    warnings_dict["images_wrong_dims"] = {}
    images_with_wrong_dims = overview_data["images_with_wrong_dims"]
    if len(images_with_wrong_dims):
        images_wrong_dims_warning = str(len(images_with_wrong_dims))
        warning_color = "red"
        description = "The following images are not 1920x1080"
        warnings_dict["images_wrong_dims"]["description"] = description
        warnings_dict["images_wrong_dims"]["image_list"] = images_with_wrong_dims
        images_wrong_dims_modal_body = description + ":" + "\n\n"
        for img in images_with_wrong_dims:
            images_wrong_dims_modal_body += img + "\n"
    else:
        images_wrong_dims_warning = "None"
        warning_color = "green"
        description = "All images are 1920x1080"
        warnings_dict["images_wrong_dims"]["description"] = description
        images_wrong_dims_modal_body = description
    w_row4 = html.Tr(
        [
            html.Td("Images with wrong dimensions"),
            html.Td(images_wrong_dims_warning,
                style = {
                    "color" : warning_color,
                    "font-weight": "bold",
                    "text-align": "right",
                    "padding-right" : "0px"
                }
            ),
            html.Td(
                html.Button(id = "images-wrong-dims-modal-btn",
                    children = [html.Img(src = warnings_more_btn_img,
                            style = {
                                "width" : "60%",
                                "float" : "right",
                                "padding-top" : "1.5px",
                                "padding-left" : "1px"
                            }
                        )
                    ],
                    style = {
                        "border" : "none",
                        "width" : "110%",
                        "background" : "none",
                        "padding-top" : "3px"
                    }
                ),
                style = {
                    "width" : "50px",
                    "padding-left" : "0px"
                }
            )
        ]
    )
    images_wrong_dims_modal = dbc.Modal(
        [
            dbc.ModalHeader("Images with wrong dimensions"),
            dbc.ModalBody(images_wrong_dims_modal_body,
                style = {
                    "white-space" : "break-spaces"
                }
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id = "close-images-wrong-dims-modal", className = "ml-auto"
                )
            ),
        ],
        id = "images-wrong-dims-modal",
        scrollable = True,
        centered = True
    )

    warnings_dict["missing_image_files"] = {}
    missing_image_files = overview_data["missing_image_files"]
    if len(missing_image_files):
        missing_image_files_warning = str(len(missing_image_files))
        warning_color = "red"
        description = "The following images are referenced in the dataset but the file is missing"
        warnings_dict["missing_image_files"]["description"] = description
        warnings_dict["missing_image_files"]["image_list"] = missing_image_files
        missing_image_files_modal_body = description + ":" + "\n\n"
        for img in missing_image_files:
            missing_image_files_modal_body += img + "\n\n"
    else:
        missing_image_files_warning = "None"
        warning_color = "green"
        description = "All images referenced in the dataset have a corresponding file"
        warnings_dict["missing_image_files"]["description"] = description
        missing_image_files_modal_body = description
    w_row5 = html.Tr(
        [
            html.Td("Missing image files"),
            html.Td(missing_image_files_warning,
                style = {
                    "color" : warning_color,
                    "font-weight": "bold",
                    "text-align": "right",
                    "padding-right" : "0px"
                }
            ),
            html.Td(
                html.Button(id = "missing-image-files-modal-btn",
                    children = [html.Img(src = warnings_more_btn_img,
                            style = {
                                "width" : "60%",
                                "float" : "right",
                                "padding-top" : "1.5px",
                                "padding-left" : "1px"
                            }
                        )
                    ],
                    style = {
                        "border" : "none",
                        "width" : "110%",
                        "background" : "none",
                        "padding-top" : "3px"
                    }
                ),
                style = {
                    "width" : "50px",
                    "padding-left" : "0px"
                }
            )
        ]
    )
    missing_image_files_modal = dbc.Modal(
        [
            dbc.ModalHeader("Missing image files"),
            dbc.ModalBody(missing_image_files_modal_body,
                style = {
                    "white-space" : "break-spaces"
                }
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id = "close-missing-image-files-modal", className = "ml-auto"
                )
            ),
        ],
        id = "missing-image-files-modal",
        scrollable = True,
        centered = True
    )

    warnings_dict["missing_classes"] = {}
    missing_classes = overview_data["missing_classes"]
    if len(missing_classes):
        missing_classes_warning = str(len(missing_classes))
        warning_color = "red"
        description = "The following classes do not have annotations"
        warnings_dict["missing_classes"]["description"] = description
        warnings_dict["missing_classes"]["class_list"] = missing_classes
        missing_classes_modal_body = description + ":" + "\n\n"
        for cl in missing_classes:
            missing_classes_modal_body += cl + "\n"
    else:
        missing_classes_warning = "None"
        warning_color = "green"
        description = "All classes have annotations"
        warnings_dict["missing_classes"]["description"] = description
        missing_classes_modal_body = description
    w_row6 = html.Tr(
        [
            html.Td("Missing classes"),
            html.Td(missing_classes_warning,
                style = {
                    "color" : warning_color,
                    "font-weight": "bold",
                    "text-align": "right",
                    "padding-right" : "0px"
                }
            ),
            html.Td(
                html.Button(id = "missing-classes-modal-btn",
                    children = [html.Img(src = warnings_more_btn_img,
                            style = {
                                "width" : "60%",
                                "float" : "right",
                                "padding-top" : "1.5px",
                                "padding-left" : "1px"
                            }
                        )
                    ],
                    style = {
                        "border" : "none",
                        "width" : "110%",
                        "background" : "none",
                        "padding-top" : "3px"
                    }
                ),
                style = {
                    "width" : "50px",
                    "padding-left" : "0px"
                }
            )
        ]
    )
    missing_classes_modal = dbc.Modal(
        [
            dbc.ModalHeader("Missing classes"),
            dbc.ModalBody(missing_classes_modal_body,
                style = {
                    "white-space" : "break-spaces"
                }
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id = "close-missing-classes-modal", className = "ml-auto"
                )
            ),
        ],
        id = "missing-classes-modal",
        scrollable = True,
        centered = True
    )

    json.dump(warnings_dict, warnings_file, indent=4)

    warnings_table_body = [html.Tbody(
        [
            w_row1, 
            w_row2, 
            w_row3, 
            w_row4, 
            w_row5, 
            w_row6
        ]
    )]

    warnings_table = dbc.Table(warnings_table_header + warnings_table_body, 
        striped = True, 
        bordered = True, 
        hover = True, 
        style = {
            "width" : "30%"
        }
    )

    classes = overview_data["classes"]
    classes = {int(k): v for k, v in classes.items()}
    classes = OrderedDict(sorted(classes.items())) 

    classes_table_header = [
        html.Thead(
            html.Tr(
                [
                    html.Th("ID",
                        style = {
                            "border-top-right-radius" : "0px",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "background" : "lightsalmon",
                            "padding" : "1%",
                            "text-align" : "right"
                        }
                    ),
                    html.Th("Name",
                        style = {
                            "border-top-right-radius" : "0px",
                            "border-top-left-radius" : "0px",
                            "background" : "lightsalmon",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "padding" : "1%",
                            "text-align" : "left",
                        }
                    ),
                    html.Th("No. objects",
                        style = {
                            "border-top-right-radius" : "0px",
                            "border-top-left-radius" : "0px",
                            "background" : "lightsalmon",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "padding" : "1%",
                            "text-align" : "right",
                            "text-transform" : "none"
                        }
                    ),
                    html.Th("No. images",
                        style = {
                            "border-top-right-radius" : "0px",
                            "border-top-left-radius" : "0px",
                            "background" : "lightsalmon",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "padding" : "1%",
                            "text-align" : "right",
                            "text-transform" : "none"
                        }
                    ),
                    html.Th("No. unique images",
                        style = {
                            "border-top-right-radius" : "0px",
                            "border-top-left-radius" : "0px",
                            "background" : "lightsalmon",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "padding" : "1%",
                            "text-align" : "right",
                            "text-transform" : "none"
                        }
                    ),
                    html.Th("Min. bbox height",
                        style = {
                            "border-top-right-radius" : "0px",
                            "border-top-left-radius" : "0px",
                            "background" : "lightsalmon",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "padding" : "1%",
                            "text-align" : "right",
                            "text-transform" : "none"
                        }
                    ),
                    html.Th("Max. bbox height",
                        style = {
                            "border-top-right-radius" : "0px",
                            "border-top-left-radius" : "0px",
                            "background" : "lightsalmon",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "padding" : "1%",
                            "text-align" : "right",
                            "text-transform" : "none"
                        }
                    ),
                    html.Th("Min. bbox width",
                        style = {
                            "border-top-right-radius" : "0px",
                            "border-top-left-radius" : "0px",
                            "background" : "lightsalmon",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "padding" : "1%",
                            "text-align" : "right",
                            "text-transform" : "none"
                        }
                    ),
                    html.Th("Max. bbox width",
                        style = {
                            "border-top-left-radius" : "0px",
                            "background" : "lightsalmon",
                            "font-size" : "medium",
                            "font-weight" : "900",
                            "padding" : "1%",
                            "text-align" : "right",
                            "text-transform" : "none"
                        }
                    )
                ]
            )
        )
    ]

    classes_table_rows = []
    no_objects = int(overview_data["no_objects"])
    no_images = int(overview_data["no_images"])

    for cl in classes:
        class_id = cl
        class_name = classes[cl]["name"]
        class_no_objects = int(classes[cl]["no_objects"])
        class_prop_objects = (class_no_objects*100) / no_objects
        class_objects = str(class_no_objects) + " ({:.2f}%)".format(class_prop_objects)
        class_no_images = len(classes[cl]["images"])
        class_prop_images = (class_no_images*100) / no_images
        class_images = str(class_no_images) + " ({:.2f}%)".format(class_prop_images)
        class_no_unique_images = len(classes[cl]["unique_images"])
        class_prop_unique_images = (class_no_unique_images*100) / class_no_images
        class_unique_images = str(class_no_unique_images) + " ({:.2f}%)".format(class_prop_unique_images)
        class_min_bbox_height = 0
        class_max_bbox_height = 0
        class_min_bbox_width = 0
        class_max_bbox_width = 0 

        cl_row = html.Tr(
            [
                html.Td(class_id,
                    style = {
                        "padding" : "1%",
                        "text-align" : "right"
                    }
                ),
                html.Td(class_name,
                    style = {
                        "padding" : "1%",
                        "text-align" : "left"
                    }
                ),
                html.Td(class_objects,
                    style = {
                        "padding" : "1%",
                        "text-align" : "right"
                    }
                ),
                html.Td(class_images,
                    style = {
                        "padding" : "1%",
                        "text-align" : "right"
                    }
                ),
                html.Td(class_unique_images,
                    style = {
                        "padding" : "1%",
                        "text-align" : "right"
                    }
                ),
                html.Td(class_min_bbox_height,
                    style = {
                        "padding" : "1%",
                        "text-align" : "right"
                    }
                ),
                html.Td(class_max_bbox_height,
                    style = {
                        "padding" : "1%",
                        "text-align" : "right"
                    }
                ),
                html.Td(class_min_bbox_width,
                    style = {
                        "padding" : "1%",
                        "text-align" : "right"
                    }
                ),
                html.Td(class_max_bbox_width,
                    style = {
                        "padding" : "1%",
                        "text-align" : "right"
                    }
                ),
            ]
        )
        classes_table_rows.append(cl_row)

    classes_table_body = [html.Tbody(classes_table_rows)]

    classes_table = dbc.Table(classes_table_header + classes_table_body, 
        striped = True, 
        bordered = True, 
        hover = True,
        style = {
            "font-size" : "small"
        }
    )

    overview = html.Div(
        [
            html.H4(section_title,
                style = {
                    "padding-bottom" : "0.5%",
                    "text-transform" : "capitalize",
                    "letter-spacing" : "2px",
                    "font-size" : "x-large",
                    "font-family" : "sans-serif",
                    "font-weight" : "500"
                }
            ),
            html.Div(
                [
                    info_table,
                    summary_table,
                    warnings_table
                ],
                style = {
                    "display" : "flex",
                    "justify-content" : "space-between",
                    "padding-bottom" : "1%",
                    "border-bottom" : "1px solid #bababa"
                }
            ),
            html.Div(
                [
                    uniform_modal,
                    missing_images_modal,
                    empty_images_modal,
                    images_wrong_dims_modal,
                    missing_image_files_modal,
                    missing_classes_modal
                ]
            ),
            html.H4("Classes summary",
                style = {
                    "padding-top" : "1%",
                    "padding-bottom" : "0.5%",
                    "letter-spacing" : "2px",
                    "font-size" : "x-large",
                    "font-family" : "sans-serif",
                    "font-weight" : "500"
                }
            ),
            html.Div(
                [
                    classes_table
                ],
                style = {
                    "padding-bottom" : "1%",
                    "border-bottom" : "1px solid #bababa"
                }
            ),
        ]
    )

    return overview

def render_class_stats(class_stats):
    fig = px.bar(
        class_stats,
        x="category",
        y="number of objects",
        title="Number of Objects per Category",
    )
    val = "number of objects"
    fig2 = px.pie(class_stats, values=val, names="category")
    fig3 = px.bar(
        class_stats,
        x="category",
        y="number of images",
        title="Number of Images per Category",
    )
    val = "number of images"
    fig4 = px.pie(class_stats, values=val, names="category")
    fig5 = px.bar(
        class_stats,
        x="category",
        y="avg number of objects per img",
        title="Avg Number Of Objects per Image",
    )
    fig5.update_layout(clickmode="event+select")
    text_fig5 = "Click on bin to see probability distribution"
    fig6 = px.bar(
        class_stats,
        x="category",
        y="avg percentage of img",
        title="Avg Proportion of Image",
    )
    text_fig6 = "Click on bin to see probability distribution"
    return html.Div(
        [
            dcc.Graph(id="cat_objs_bar", figure=fig),
            dcc.Graph(id="cat_objs_pie", figure=fig2),
            dcc.Graph(id="cat_imgs_bar", figure=fig3),
            dcc.Graph(id="cat_imgs_pie", figure=fig4),
            dcc.Graph(id="objs_per_img", figure=fig5),
            html.Div(children=text_fig5),
            html.Div(id="obj_hist_out"),
            dbc.Spinner(html.Div(id="obj_imgs"), size="lg"),
            dcc.Graph(id="cat_areas", figure=fig6),
            html.Div(children=text_fig6),
            html.Div(id="area_hist_out"),
            dbc.Spinner(html.Div(id="area_imgs"), size="lg")
        ],
        style={"margin-left": "10%", "margin-right": "10%"},
    )

@app.callback(
    Output("obj_hist_out", "children"), 
    [Input("objs_per_img", "clickData")],
)
def display_obj_hist(click_data):
    if click_data is not None:
        cat = click_data["points"][0]["x"]
        global obj_cat
        obj_cat = cat
        cat_ids = coco_data.getCatIds(catNms=cat)
        img_ids = coco_data.getImgIds(catIds=cat_ids)
        title = "Number of " + cat + "s in an image w/ " + cat + "s"
        _, data = get_objs_per_img(cat_ids, img_ids, coco_data)
        x = data["number of objs"]
        xbins = dict(size=1)
        norm = "probability"
        hist = go.Histogram(x=x, xbins=xbins, histnorm=norm)
        fig = go.Figure(data=[hist])
        fig.update_layout(
            clickmode="event+select", yaxis_title="probability", title=title
        )
        text = "Click on bin to see images (may take up to 30 seconds)"
        return html.Div(
            [dcc.Graph(id="objs_hist", figure=fig), html.Div(children=text)]
        )

@app.callback(
    Output("area_hist_out", "children"), 
    [Input("cat_areas", "clickData")],
)
def display_area_hist(click_data):
    if click_data is not None:
        cat = click_data["points"][0]["x"]
        global area_cat
        area_cat = cat
        cat_ids = coco_data.getCatIds(catNms=cat)
        img_ids = coco_data.getImgIds(catIds=cat_ids)
        title = "Percentage area of a(n) " + cat + " in an image"
        _, data = get_proportion(cat_ids, img_ids, coco_data)
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=data["proportion of img"],
                    xbins=dict(size=0.01),
                    histnorm="probability",
                )
            ]
        )
        fig.update_layout(
            clickmode="event+select", yaxis_title="probability", title=title
        )
        text = "Click on bin to see images (may take up to 30 seconds)"
        return html.Div(
            [dcc.Graph(id="area_hist", figure=fig), html.Div(children=text)]
        )

def render_dataset_visualizer(path_to_overview_data):
    return html.Div("yet to implement")

def render_anomaly_detection(class_stats):
    return html.Div("yet to implement")

@app.callback(
    Output("uniform-modal", "is_open"),
    Input("uniform-modal-btn", "n_clicks"), 
    Input("close-uniform-modal", "n_clicks"),
    State("uniform-modal", "is_open")
)
def toggle_uniform_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("missing-images-modal", "is_open"),
    Input("missing-images-modal-btn", "n_clicks"), 
    Input("close-missing-images-modal", "n_clicks"),
    State("missing-images-modal", "is_open")
)
def toggle_missing_images_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("empty-images-modal", "is_open"),
    Input("empty-images-modal-btn", "n_clicks"), 
    Input("close-empty-images-modal", "n_clicks"),
    State("empty-images-modal", "is_open")
)
def toggle_empty_images_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("images-wrong-dims-modal", "is_open"),
    Input("images-wrong-dims-modal-btn", "n_clicks"), 
    Input("close-images-wrong-dims-modal", "n_clicks"),
    State("images-wrong-dims-modal", "is_open")
)
def toggle_images_wrong_dims_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("missing-image-files-modal", "is_open"),
    Input("missing-image-files-modal-btn", "n_clicks"), 
    Input("close-missing-image-files-modal", "n_clicks"),
    State("missing-image-files-modal", "is_open")
)
def toggle_missing_image_files_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("missing-classes-modal", "is_open"),
    Input("missing-classes-modal-btn", "n_clicks"), 
    Input("close-missing-classes-modal", "n_clicks"),
    State("missing-classes-modal", "is_open")
)
def toggle_missing_classes_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

def serve_layout():
    return html.Div(
        [
            navbar,
            welcome_menu,
            new_analysis_menu,
            loading,
            tabs
        ]
    )

app.layout = serve_layout

@app.server.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(output_dir, filename, attachment_filename = filename, as_attachment = True)

if __name__ == "__main__":
    application.run(host ="0.0.0.0", port = port, debug = True)