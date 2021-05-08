"""anomalies.py: the layouts for anomaly tab"""

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_table

from Auto_Encoder_detector import detect_anomalies_auto_encoder
from CNN_based_detector import detect_anomalies_cnn_lof, detect_anomalies_cnn_iforest
from Manually_based_detector import detect_anomalies_manual_iforest, detect_anomalies_manual_lof
from anomaly_detector import detect_anomalies_imageai, create_dataframe_imageai, create_dataframe, detect_anomalies_size
# number of algorithms
from hog_based_detector import detect_anomalies_hog_iforest, detect_anomalies_hog_lof

PAGE_SIZE = 10
ALGORITHMS = {'imageai': {'name': 'imageai',
                          'detector': detect_anomalies_imageai,
                          'df_creator': create_dataframe_imageai,
                          'label': 'ImageAI',
                          'index': 0,
                          'color': "rgb(4,158,215)",
                          'column_names':
                              ['image_id', 'id', 'cat_id', 'cat_name', 'detected_name', 'percentage_probability']},
              'cnn_iforest': {'name': 'cnn_iforest',
                              'detector': detect_anomalies_cnn_iforest,
                              'df_creator': create_dataframe,
                              'label': 'iForest-CNN',
                              'index': 1,
                              'color': "rgb(114, 206, 243)",
                              'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                               'anomaly_score']},
              'cnn_lof': {'name': 'cnn_lof',
                          'detector': detect_anomalies_cnn_lof,
                          'df_creator': create_dataframe,
                          'label': 'LOF-CNN',
                          'index': 2,
                          'color': "rgb(2, 74, 97)",
                          'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                           'anomaly_score']},
              'hog_iforest': {'name': 'hog_iforest',
                              'detector': detect_anomalies_hog_iforest,
                              'df_creator': create_dataframe,
                              'label': 'iForest-HOG',
                              'index': 3,
                              'color': "rgb(171, 171, 173)",
                              'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                               'anomaly_score']},
              'hog_lof': {'name': 'hog_lof',
                          'detector': detect_anomalies_hog_lof,
                          'df_creator': create_dataframe,
                          'label': 'LOF-HOG',
                          'index': 4,
                          'color': "rgb(101, 147, 165)",
                          'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance', 'anomaly_score']},
              'manual_feature_iforest': {'name': 'manual_feature_iforest',
                                         'detector': detect_anomalies_manual_iforest,
                                         'df_creator': create_dataframe,
                                         'label': 'iForest-Manually Extracted Feature',
                                         'index': 5,
                                         'color': "rgb(110, 188, 192)",
                                         'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                                          'anomaly_score']},
              'manual_feature_lof': {'name': 'manual_feature_lof',
                                     'detector': detect_anomalies_manual_lof,
                                     'df_creator': create_dataframe,
                                     'label': 'LOF-Manually Extracted Feature',
                                     'index': 6,
                                     'color': "rgb(0, 153, 145)",
                                     'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'variance',
                                                      'anomaly_score']},
              'autoencoder': {'name': 'autoencoder',
                              'detector': detect_anomalies_auto_encoder,
                              'df_creator': create_dataframe,
                              'label': 'Autoencoder',
                              'index': 7,
                              'color': "#135f4e",
                              'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'anomaly_score'],
                              },
              'object_size': {'name': 'object_size',
                              'detector': detect_anomalies_size,
                              'df_creator': create_dataframe,
                              'label': 'Object Size',
                              'index': 7,
                              'color': 'navy',
                              'column_names': ['image_id', 'id', 'cat_id', 'cat_name', 'size', 'average']}
              }


# ALGORITHMS = {'imageai': {'name': 'imageai',
#                           'detector': detect_anomalies_imageai,
#                           'label': 'ImageAI',
#                           'index': 0,
#                           'column_names':
#                               ['image_id', 'id', 'cat_id', 'cat_name', 'detected_name', 'percentage_probability']}}


# @cache.memoize()
# def generate_anomalies(analysis_path: str) -> pd.DataFrame:
#     with open(analysis_path, 'r') as f:
#         analysis = json.load(f)
#         images = analysis["images"]
#         df = pd.DataFrame.from_dict(images, orient="index")
#         # random select 100 images
#         df_100 = df.sample(100)
#         # for each image randomly select maximum 2 objects
#         objs_partial = df_100['objects'].apply(lambda x: random.sample(x, min(2, len(x))))
#         # drop columns not useful for plots
#         df_100 = df_100.drop(['objects', 'classes'], axis=1)
#         df_100["image_id"] = df_100.index.map(int)
#         # select metrics for each object in each image and stack them up
#         objs_partial = objs_partial.apply(lambda objs: [{"category_id": obj["category_id"],
#                                                          "id": obj["id"],
#                                                          "bbox": obj["bbox"],
#                                                          "image_id": obj["image_id"]} for obj in objs])
#         obj_df = pd.concat([pd.DataFrame(obj) for obj in objs_partial])
#         # left join
#         df_joined = obj_df.merge(df_100, on="image_id", how='left')
#         df_joined['index'] = range(1, len(df_joined) + 1)
#
#         # Class id name dict
#         classes = analysis['classes']
#         class_id_name = {int(class_id): classes[class_id]['name'] for class_id in classes}
#         df_joined['category'] = df_joined['category_id'].apply(class_id_name.get)
#
#         # Rearrange col
#         cols = ['index', 'category_id', 'category', 'id', 'image_id', 'bbox', 'width', 'height', 'file_name']
#         return df_joined[cols]


def anomalies_contents(analysis_path):
    algo_selection_dropdown = dbc.Col(dcc.Dropdown(
        id="algo-selection",
        options=[{'label': algorithm['label'], 'value': algorithm['name']} for algorithm in ALGORITHMS.values()],
        placeholder="Select an algorithm",
        multi=True
    ), width=10)
    update_button = dbc.Col(dbc.Button("detect anomaly", id='update-button', color="info", className="mr-1"), width=2)
    anomaly_tables = dbc.Row([create_anomaly_output_section(algorithm,
                                                            page_size=PAGE_SIZE)
                              for algorithm in ALGORITHMS.values()], id='anomaly-tables')

    graph_card = dbc.Card([
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(html.H5("Objects distribution", className="card-title"))
                ],
                justify="between",
            ),
            html.Div(id="plot-section")],
            className="card-body")],
        className="card flex-fill w-100"
    )

    plot_section = html.Div(graph_card, className="col-sm-12 col-md-12 col-lg-12 col-xl-8 d-flex")

    anomaly_output_div = html.Div([plot_section, anomaly_tables], id='anomaly-output-div',
                                  style={'display': 'none'})

    empty_div = html.Div(analysis_path, id='anomaly-empty-div')

    return html.Div([dbc.Row([algo_selection_dropdown, update_button], style={'padding-bottom': '15px'}),
                     anomaly_output_div,
                     empty_div])


def create_summary(algorithm_name):
    return dbc.Card(
        id=f"summary-section-{algorithm_name}",
        style={'display': 'none'},
        children=[
            # dbc.CardHeader(html.H2("Annotation area")),
            dbc.CardBody(
                [
                    dbc.Row([], id=f"summary-cards-{algorithm_name}",
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


def create_anomaly_output_section(algorithm, page_size):
    anomaly_table = dash_table.DataTable(
        id=f"anomaly-data-table-{algorithm['name']}",
        # data_algorithm_index=i,
        columns=[
            {"name": i, "id": i} for i in algorithm['column_names']
        ],
        page_current=0,
        page_size=page_size,
        page_action='custom',

        sort_action='custom',
        sort_mode='single',
        sort_by=[],

        row_selectable="multi",
        selected_rows=[],
        style_table={'margin-top': '15px'},
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in algorithm['column_names']
        ],
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },

        export_format='csv',
        export_headers='display',
    )
    summary_section = create_summary(algorithm['name'])

    image_cell_preview = dbc.Row([], id=f"anomaly-image-cell-{algorithm['name']}", className="row d-xxl-flex")
    image_card = create_anomaly_editing_image_card(algorithm['name'])
    image_cols_preview = dbc.Row([], id=f"anomaly-image-cols-{algorithm['name']}", className="row d-xxl-flex")
    anomaly_table_image_div = html.Div(
        [anomaly_table, image_card, image_cell_preview, image_cols_preview],
        id=f"anomaly-table-image-div-{algorithm['name']}")
    hidden_div = html.Div(id=f"algorithm-name-{algorithm['name']}", children=algorithm['name'],
                          style={'display': 'none'})

    store = dcc.Store(id=f"anomaly-manual-store-{algorithm['name']}")

    row_dropdown = dcc.Dropdown(
        id=f"df-row-{algorithm['name']}",
        options=[
            {'label': i, 'value': i} for i in range(10)
        ],
        value=0,
        placeholder='Row'
    )
    return html.Div([summary_section, anomaly_table_image_div, hidden_div, store, row_dropdown],
                    id=f"anomaly-output-section-{algorithm['name']}")


def class_drop(algorithm_name):
    # return drop down of all possible labels
    return dcc.Dropdown(
        id=f"anomaly-class-toggle-{algorithm_name}",
        options=[
            {"label": col_name.capitalize(), "value": col_name}
            for col_name in ["A", "B", "C"]
        ]
    )


def create_anomaly_editing_image_card(algorithm_name):
    image_card = dbc.Card(
        [
            dbc.CardHeader(html.H2("Change Label or Mark Not Anomaly")),
            dbc.CardBody(
                dbc.Row(
                    dbc.Col(
                        id=f"anomaly-graph-col-{algorithm_name}"
                    ),
                )
            ),
            dbc.CardFooter([
                dbc.Row(
                    [
                        dbc.Col(
                            "Use the dropdown menu to select the correct label:"
                        ),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(class_drop(algorithm_name)),
                        dbc.Col(dbc.Button("Confirm", id=f"anomaly-btn-confirm-{algorithm_name}", color="primary",
                                           className="mr-2"))
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Button("Not Anomaly", id=f"anomaly-btn-cancel-{algorithm_name}", color="danger",
                                           className="mr-2", block=True)),
                    ],
                    align="center",
                )],
            ),
        ]
    )
    return image_card
    # if __name__ == '__main__':
    # generate_anomalies("")
