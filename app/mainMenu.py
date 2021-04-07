import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash

def about_contents():
    """
    Generates the contents of the about menu

    :return: about contents 
    """
    description = html.Div([
        html.H3("About vizEDA", style={"font-weight": "500"}),
        dbc.Card([
            dbc.CardBody(
                [
                    html.H4("A Dataset Explorer and Anomaly Detector",
                            style={"font-weight": "400"}),
                    html.P(
                        "vizEDA is an exploratory data analysis tool that helps to visualize and improve complex computer vision datasets."),
                    html.P(
                        "Using its intuitive interface you can perform automated analysis of your datasets and easily visualize each class and check related statistics."),
                    html.P(
                        "With the automatic anomaly detector feature you can quickly spot annotation errors and anomalies that can hinder your model performance.")
                ]
            )])
    ],
        style={"width": "70%"}
    )
    tutorial = html.Div([
        html.H3("Usage", style={"font-weight": "500"}),
        dbc.Card(dbc.ListGroup([
            dbc.ListGroupItem("1. To run a new analysis, click on the new analysis button in the sidebar",
                              className="list-group-item"),
            dbc.ListGroupItem("2. Insert the path to the images of your dataset and upload the COCO annotations file",
                              className="list-group-item"),
            dbc.ListGroupItem("3. When the analysis is ready, a summary will be available in the dashboard",
                              className="list-group-item"),
            dbc.ListGroupItem(
                "4. The warnings section tells you about problems in your dataset such as missing image files or unused IDs",
                className="list-group-item"),
            dbc.ListGroupItem(
                "5. The classes section provides an overview of the classes in the datasets and lets you visualise all images for a selected class",
                className="list-group-item"),
            dbc.ListGroupItem(
                "6. The stats section shows image and annotation distributions as well as other useful statistics about your dataset",
                className="list-group-item"),
            dbc.ListGroupItem("7. The anomalies section contains the results of the automated anomaly detection",
                              className="list-group-item"),
            dbc.ListGroupItem("8. You can save an analysis by clicking the download button in the sidebar",
                              className="list-group-item"),
            dbc.ListGroupItem(
                "8. You can then upload the downloaded file in a new vizEDA session to see the analysis again",
                className="list-group-item"),
        ],
            flush=True,
        ))
    ],
        style={"width": "70%"}
    )

    contents = html.Div([
        description,
        tutorial
    ]
    )
    return contents


def new_analysis_contents():
    """
    Generates the contents of the new analysis menu

    :return: new analysis contents 
    """
    contents = html.Div([
        html.H3("New analysis", style={"font-weight": "500"}),
        html.Div([
            dbc.FormGroup([
                html.Div([
                    dbc.Label([
                        html.Img(src="assets/icons/image.svg",
                                 style={"width": "15px", "padding-bottom": "1px", "margin-right": "5px"}),
                        "Path to images"],
                        style={"margin-left": "40%"}),
                    # dcc.Upload([
                    #     dbc.Label([
                    #         html.Img(src="assets/icons/image.svg",
                    #                  style={"width": "15px", "padding-bottom": "1px", "margin-right": "5px"}),
                    #         "Path to images"],
                    #         style={"margin-left": "0%"}),
                    #     dbc.Button([
                    #         "Drag and Drop Image Folder or ", html.A('Upload Folder')
                    #     ],
                    #         className="mb-3",
                    #         id="images-upload",
                    #         style={"width": "100%", "margin-bottom": "1.5rem", "font-weight": "700",
                    #                "background": "#222e3c"}
                    #     )],
                    #     multiple=False,
                    #     id="upload-image-path",
                    #     style={"margin": "auto", "width": "20%"}
                    # ),
                    dbc.Input(
                        type="text",
                        placeholder="e.g. /Users/me/dataset/images",
                        className="mb-3",
                        id="images-upload",
                        style={"margin": "auto", "width": "20%"}
                    ),
                    ## Added loading spinners
                    dcc.Loading(
                        id="loading-btn",
                        type='dot',
                        children=html.Div([
                            dcc.Upload([
                                dbc.Label([
                                    html.Img(src="assets/icons/file-text.svg",
                                             style={"width": "15px", "padding-bottom": "1px", "margin-right": "5px"}),
                                    "Annotations file (.json):"]),
                                dbc.Button([
                                    "Drag and Drop json file or ", html.A('Upload File')
                                ],
                                    className="mr-1",
                                    id="upload-btn",
                                    style={"width": "100%", "margin-bottom": "1.5rem", "font-weight": "700",
                                           "background": "#222e3c"}
                                )],
                                multiple=False,
                                id="upload",
                                style={"margin": "auto", "width": "20%"}
                            ),


                        ]),

                    ),
                    ##Added loading spinners (not working for analyze button)
                    dcc.Loading(
                        id="loading-object-analysis",
                        type="default",
                        children=html.Div([
                            dbc.Button(
                                "Analyze",
                                className="mr-1",
                                disabled=True,
                                id="analyze-btn",
                                style={"width": "100%", "text-transform": "uppercase", "font-weight": "700"}
                            ),
                        ],
                            style={"margin": "auto", "width": "20%"},)
                    )

                ])
            ])
        ],
            style={"padding-top": "18%"})
    ])

    return contents


def upload_analysis_contents():
    """
    Generates the contents of the upload analysis menu

    :return: upload analysis contents
    """
    contents = html.Div([
        html.H3("Upload analysis", style={"font-weight": "500"}),
        html.Div([
            dbc.FormGroup([
                dcc.Upload([
                    dbc.Label([
                        html.Img(src="assets/icons/file-text.svg",
                                 style={"width": "15px", "padding-bottom": "1px", "margin-right": "5px"}),
                        "Analysis file:"]),
                    dbc.Button([
                        "Upload"
                    ],
                        className="mr-1",
                        id="upload-btn",
                        style={"width": "100%", "margin-bottom": "1.5rem", "font-weight": "700",
                               "background": "#222e3c"}
                    )],
                    multiple=False,
                    id="upload",
                    style={"margin": "auto", "width": "20%"}
                ),
                html.Div([
                    dbc.Button(
                        "View",
                        className="mr-1",
                        id="analyze-btn-upload-page",
                        style={"width": "100%", "text-transform": "uppercase", "font-weight": "700"}
                    )],
                    style={"margin": "auto", "width": "20%"})
            ]),
            dbc.Input(
                type="text",
                className="mb-3",
                id="images-upload",
                valid=True,
                style={"display": "none"}
            )
        ], style={"padding-top": "18%"})
    ])

    return contents

