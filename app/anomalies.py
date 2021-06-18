"""anomalies.py: the layouts for anomaly tab"""
import json

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_table

from CNN_based_detector import detect_anomalies_cnn_lof, detect_anomalies_cnn_iforest
from Manually_based_detector import detect_anomalies_manual_iforest, detect_anomalies_manual_lof
from anomaly_detector import detect_anomalies_imageai, create_dataframe_imageai, create_dataframe, detect_anomalies_size
from hog_based_detector import detect_anomalies_hog_iforest, detect_anomalies_hog_lof

PAGE_SIZE = 10
ALGORITHMS = {'object_size': {'name': 'object_size',
                              'detector': detect_anomalies_size,
                              'df_creator': create_dataframe,
                              'label': 'Object Size',
                              'index': 8,
                              'color': '#4acdd9',
                              'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'size', 'average']},
              'imageai': {'name': 'imageai',
                          'detector': detect_anomalies_imageai,
                          'df_creator': create_dataframe_imageai,
                          'label': 'ImageAI',
                          'index': 0,
                          'color': "#1D3557",
                          'column_names':
                              ['image_id', 'id', 'cat_id', 'cat_name', 'detected_name', 'percentage_probability']},
              'cnn_iforest': {'name': 'cnn_iforest',
                              'detector': detect_anomalies_cnn_iforest,
                              'df_creator': create_dataframe,
                              'label': 'iForest-CNN',
                              'index': 1,
                              'color': "#31587A",
                              'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                               'anomaly_score']},
              'cnn_lof': {'name': 'cnn_lof',
                          'detector': detect_anomalies_cnn_lof,
                          'df_creator': create_dataframe,
                          'label': 'LOF-CNN',
                          'index': 2,
                          'color': "#457B9D",
                          'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                           'anomaly_score']},
              'hog_iforest': {'name': 'hog_iforest',
                              'detector': detect_anomalies_hog_iforest,
                              'df_creator': create_dataframe,
                              'label': 'iForest-HOG',
                              'index': 3,
                              'color': "#5E93AD",
                              'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                               'anomaly_score']},
              'hog_lof': {'name': 'hog_lof',
                          'detector': detect_anomalies_hog_lof,
                          'df_creator': create_dataframe,
                          'label': 'LOF-HOG',
                          'index': 4,
                          'color': "#77ABBD",
                          'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance', 'anomaly_score']},
              'manual_feature_iforest': {'name': 'manual_feature_iforest',
                                         'detector': detect_anomalies_manual_iforest,
                                         'df_creator': create_dataframe,
                                         'label': 'iForest-Manually Extracted Feature',
                                         'index': 5,
                                         'color': "#A2CBD1",
                                         'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                                          'anomaly_score']},
              'manual_feature_lof': {'name': 'manual_feature_lof',
                                     'detector': detect_anomalies_manual_lof,
                                     'df_creator': create_dataframe,
                                     'label': 'LOF-Manually Extracted Feature',
                                     'index': 6,
                                     'color': "#CDEAE5",
                                     'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                                      'anomaly_score']},
              # 'autoencoder': {'name': 'autoencoder',
              #                 'detector': detect_anomalies_auto_encoder,
              #                 'df_creator': create_dataframe,
              #                 'label': 'Autoencoder',
              #                 'index': 7,
              #                 'color': "#79d4db",
              #                 'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'anomaly_score'],
              #                 },
              }


def anomalies_contents(analysis_path):
    f = open(analysis_path, 'r')
    analysis = json.load(f)

    classes = analysis["classes"]
    options = []
    for cl in classes:
        class_option = {"label": classes[cl]["name"], "value": classes[cl]["name"]}
        options.append(class_option)

    algo_selection_dropdown = dbc.Col(dcc.Dropdown(
        id="algo-selection",
        options=[{'label': algorithm['label'], 'value': algorithm['name']} for algorithm in ALGORITHMS.values()],
        placeholder="Select an algorithm",
        multi=True
    ), width=10)
    update_button = dbc.Col(dbc.Button("detect anomaly", id='update-button',
                                       style={'colour': '#1D3557', 'background-color': '#1D3557', 'border': '0px'},
                                       className="mr-1"), width=2)
    algo_selection = dbc.Row([algo_selection_dropdown, update_button], style={'padding-bottom': '15px'})
    graph_card = dbc.Row(dbc.Card([
        dbc.CardHeader(html.H5("Objects distribution")),
        dbc.CardBody(id="plot-section")]))
    anomaly_tables = dbc.Row([create_anomaly_output_section(algorithm, options) for algorithm in ALGORITHMS.values()],
                             id='anomaly-tables')
    anomaly_output_div = html.Div([graph_card, anomaly_tables], id='anomaly-output-div',
                                  style={'display': 'none'})

    return html.Div([algo_selection,
                     anomaly_output_div])


def create_anomaly_output_section(algorithm, options):
    ######## deprecated function ########
    # image_cell_preview = dbc.Row([], id=f"anomaly-image-cell-{algorithm['name']}", className="row d-xxl-flex")
    # image_cols_preview = dbc.Row([], id=f"anomaly-image-cols-{algorithm['name']}", className="row d-xxl-flex")
    #####################################
    summary_section = create_summary(algorithm['name'])
    image_card = create_anomaly_editing_image_card(algorithm['name'], options)
    anomaly_table = create_data_table("Anomaly", algorithm["name"], algorithm["column_names"])
    manual_label_table = create_data_table("Manual-Label", algorithm["name"],
                                           ["id", "label_before", "manually_selected_label"])
    anomaly_table_image_div = dbc.Container(
        dbc.Row(
            [dbc.Col([dbc.Row(anomaly_table), dbc.Row(manual_label_table)], width=12, xl=7),
             dbc.Col(image_card, width=12, xl=5)]),
        fluid=True,
        id=f"anomaly-table-image-div-{algorithm['name']}"
    )
    hidden_div = html.Div(id=f"algorithm-name-{algorithm['name']}", children=algorithm['name'],
                          style={'display': 'none'})

    store = dcc.Store(id=f"anomaly-manual-store-{algorithm['name']}")

    return dbc.Card([summary_section, anomaly_table_image_div, hidden_div, store],
                    id=f"anomaly-output-section-{algorithm['name']}")


def class_drop(algorithm_name, options):
    # return drop down of all possible labels
    return dcc.Dropdown(
        id=f"anomaly-class-toggle-{algorithm_name}",
        options=options)


def create_summary(algorithm_name):
    return dbc.Card(
        id=f"summary-section-{algorithm_name}",
        children=[
            dbc.CardBody(
                [
                    dbc.CardGroup([], id=f"summary-cards-{algorithm_name}",
                                  className="col-sm-12 col-md-12 col-lg-12 col-xl-12 d-flex",
                                  style={"margin-top": "15px"}),

                    dbc.Row([
                        dbc.Col(daq.ToggleSwitch(
                            id=f"table-toggle-{algorithm_name}",
                            value=False
                        ), width=1),
                        dbc.Col(html.Div("Show Anomaly Table"), width=3)], style={'margin-bottom': '15px'})
                ]
            ),
        ],
    )


def create_anomaly_editing_image_card(algorithm_name, options):
    return dbc.Card(
        [
            dbc.CardHeader(
                [dbc.Row(html.H2("Change Label or Mark Not Anomaly")),

                 html.Label(["Select Row:",
                             dcc.Dropdown(
                                 id=f"df-row-{algorithm_name}",
                                 options=[{'label': i, 'value': i} for i in range(PAGE_SIZE)],
                                 value=0,
                                 placeholder='Row'),
                             ]),
                 ]),
            dbc.CardBody([
                dbc.Row(dcc.Link(href='', id=f"filename-{algorithm_name}")),
                dbc.Row(
                    id=f"anomaly-graph-row-{algorithm_name}"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            "Mark not Anomaly or Use the Dropdown to Select the Correct Label:"
                        ),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Button("Not Anomaly", id=f"anomaly-btn-cancel-{algorithm_name}",
                                           color="danger",
                                           className="mr-2")),
                        dbc.Col(class_drop(algorithm_name, options))

                    ],
                    align="between",
                ),
            ]),
            dbc.CardFooter([
                dbc.Row(
                    [
                        dbc.Col(dbc.Button("Next row", id=f"anomaly-btn-confirm-{algorithm_name}",
                                           style={'colour': '#1D3557', 'background-color': '#1D3557', 'border': '0px'},
                                           className="mr-2", block=True)),
                    ],
                    align="center",
                )],
            ),
        ]
    )


def create_data_table(table_name, algorithm_name, column_names):
    anomaly_table = dash_table.DataTable(
        id=f"{table_name}-data-table-{algorithm_name}",
        # data_algorithm_index=i,
        columns=[
            {"name": i, "id": i} for i in column_names
        ],
        page_current=0,
        page_size=PAGE_SIZE,
        page_action='custom',

        sort_action='custom',
        sort_mode='single',
        sort_by=[],

        # row_selectable="multi",
        # selected_rows=[],
        style_table={'margin-top': '15px'},
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in column_names
        ],
        # style_data_conditional=[
        #     {
        #         'if': {'row_index': 'odd'},
        #         'backgroundColor': 'rgb(248, 248, 248)'
        #     }
        # ],
        style_header={
            'color': 'white',
            'backgroundColor': '#1D3557',
            'fontWeight': 'bold'
        },

        export_format='csv',
        export_headers='display',
    )
    return dbc.Card(
        [
            dbc.CardHeader(html.H2(f"{table_name} Data Table for Algorithm '{algorithm_name}'")),
            dbc.CardBody(
                dbc.Row(
                    dbc.Col(
                        anomaly_table,
                    )
                )
            ),
        ]
    )
