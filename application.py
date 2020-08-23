import base64
import os
import shutil
from io import BytesIO

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from coco_assistant import COCO_Assistant
from dash.dependencies import ALL, Input, Output, State
from pandas_profiling import ProfileReport
from skimage import io
from tqdm import tqdm

from app.analysis import analyze_dataset, coco, get_objs_per_img, get_proportion
from app.blob import download_blobs, get_blob_datasets, get_blobs

# CSS stylesheet for app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
# main dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.config["suppress_callback_exceptions"] = True
application = app.server
# port to run app
port = 80
# flag for batch analysis
batch_analysis = False
# main dataframe containing data from analysis.py
analysis_df = pd.DataFrame()
# dataframe holding manually flagged images w/ anomalies
anomaly_table = pd.DataFrame(columns=["Category", "Image filename"])
# path to image data
datadir = ""
# path to annotation file
annotation_file = ""
# variable holding loaded coco data
coco_data = None
# holds raw html from pandas-profiling output
profile = ""
# currently selected category on object tab
obj_cat = ""
# currently selected category on area tab
area_cat = ""
# general html for exception handling
exception_html = html.Div(
    children="Please load a valid COCO-style annotation file and "
    "define a valid folder.",
    style={"margin-left": "50px", "margin-top": "50px"},
)


def get_datasets(data_dir):
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    dataset = []
    for folder in subfolders:
        anns, imgs = [f.path for f in os.scandir(folder) if f.is_dir()]
        for file in os.listdir(anns):
            if file.endswith(".json") and "instances" in file:
                file = os.path.join(anns, file)
                dataset.append((file, imgs))
    return dataset


def merge_datasets(dataset):
    # Specify image and annotation directories
    ann_dir = "./temp/annotations"
    img_dir = "./temp/images"
    os.mkdir("./temp")
    os.mkdir(img_dir)
    os.mkdir(ann_dir)

    for idx, (anns, imgs) in enumerate(dataset):
        print(anns)
        file = str(idx) + ".json"
        anns_loc = os.path.join(ann_dir, file)
        shutil.copy(anns, anns_loc)

        print(imgs)
        imgs_loc = os.path.join(img_dir, str(idx))
        shutil.copytree(imgs, imgs_loc)

    # Create COCO_Assistant object and merge
    # TODO: find a fix bug in merge
    cas = COCO_Assistant(img_dir, ann_dir)
    cas.merge(merge_images=True)


def parse_contents(contents):
    global analysis_df, datadir, annotation_file, coco_data, profile
    content_type, content_string = contents.split(",", 1)
    if content_type == "data:application/json;base64":
        decoded = base64.b64decode(content_string).decode("UTF-8")
        with open("./output/output.json", "w") as file:
            file.write(decoded)
        annotation_file = "./output/output.json"
        coco_data = coco.COCO(annotation_file)
    elif content_type == "data:application/octet-stream;base64":
        decoded = base64.b64decode(content_string)
        try:
            analysis_df = pd.read_feather(BytesIO(decoded))
            print("Loaded analysis file!")
            profile = ProfileReport(analysis_df, title="Dataset").to_html()
        except Exception as e:
            print(e)
            return exception_html


def fig_to_uri(in_fig, **save_args):
    # Save a figure as a URI
    out_img = BytesIO()
    in_fig.savefig(out_img, format="png", **save_args)
    out_img.seek(0)  # rewind file
    b64encoded = base64.b64encode(out_img.read())
    encoded = b64encoded.decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def get_html_imgs(img_ids, cat, outlying_anns=None):
    # TODO: speed-up image loading and display
    global datadir
    html_imgs = []
    cats = coco_data.getCatIds(catNms=[cat])
    print("Loading images...")
    for img_id in tqdm(set(img_ids)):
        im_ann = coco_data.loadImgs(ids=int(img_id))[0]
        image_filename = os.path.join(datadir, im_ann["file_name"])
        i = io.imread(image_filename) / 255.0
        plt.imshow(i)
        plt.axis("off")
        if outlying_anns is None:
            ann_ids = coco_data.getAnnIds(imgIds=img_id, catIds=cats)
        else:
            ann_ids = set(coco_data.getAnnIds(imgIds=img_id, catIds=cats))
            ann_ids = list(ann_ids.intersection(set(outlying_anns)))
        anns = coco_data.loadAnns(ann_ids)
        coco_data.showAnns(anns)
        decoded_image = fig_to_uri(plt)
        plt.close()
        html_imgs.append(
            html.Img(
                id={"type": "output_image", "index": str(img_id)},
                src=decoded_image,
                **{"data-category": cat}
            )
        )
    return html_imgs


def render_tab0():
    try:
        if profile == "":
            return exception_html
        return html.Div(
            children=[
                html.Iframe(srcDoc=profile, style={"width": "100%", "height": "1500px"})
            ],
            style={
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "margin-left": "5%",
                "margin-right": "5%",
            },
        )
    except Exception as e:
        print(e)
        return exception_html


def render_tab1():
    fig = px.bar(
        analysis_df,
        x="category",
        y="number of objects",
        title="Number of Objects per Category",
    )
    val = "number of objects"
    fig2 = px.pie(analysis_df, values=val, names="category")
    return html.Div(
        [
            dcc.Graph(id="cat_objs_bar", figure=fig),
            dcc.Graph(id="cat_objs_pie", figure=fig2),
        ],
        style={"margin-left": "10%", "margin-right": "10%"},
    )


def render_tab2():
    fig = px.bar(
        analysis_df,
        x="category",
        y="number of images",
        title="Number of Images per Category",
    )
    val = "number of images"
    fig2 = px.pie(analysis_df, values=val, names="category")
    return html.Div(
        [
            dcc.Graph(id="cat_imgs_bar", figure=fig),
            dcc.Graph(id="cat_imgs_pie", figure=fig2),
        ],
        style={"margin-left": "10%", "margin-right": "10%"},
    )


def render_tab3():
    fig = px.bar(
        analysis_df,
        x="category",
        y="avg number of objects per img",
        title="Avg Number Of Objects per Image",
    )
    fig.update_layout(clickmode="event+select")
    text = "Click on bin to see probability distribution"
    return html.Div(
        [
            dcc.Graph(id="objs_per_img", figure=fig),
            html.Div(children=text),
            html.Div(id="obj_hist_out"),
            html.Div(id="obj_imgs"),
        ],
        style={"margin-left": "10%", "margin-right": "10%"},
    )


def render_tab4():
    fig = px.bar(
        analysis_df,
        x="category",
        y="avg percentage of img",
        title="Avg Proportion of Image",
    )
    text = "Click on bin to see probability distribution"
    return html.Div(
        [
            dcc.Graph(id="cat_areas", figure=fig),
            html.Div(children=text),
            html.Div(id="area_hist_out"),
            html.Div(id="area_imgs"),
        ],
        style={"margin-left": "10%", "margin-right": "10%"},
    )


def render_tab5():
    cat_ids = coco_data.getCatIds()
    cat_dict = coco_data.loadCats(cat_ids)
    cat_nms = [d["name"] for d in cat_dict]
    options = [{"label": i, "value": i} for i in cat_nms]
    style = {"margin-top": "25px", "margin-left": "25px"}
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(children="Select a category", style=style),
                            dcc.Dropdown(
                                id="cat_selection",
                                options=options,
                                value=cat_nms[0],
                                style={
                                    "margin-top": "25px",
                                    "margin-left": "25px",
                                    "margin-right": "50%",
                                },
                            ),
                            html.Div(id="anomaly_imgs"),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H3(children="Flagged anomalies", style=style),
                            html.Div(id="anomaly_table"),
                        ],
                        width=6,
                    ),
                ]
            )
        ],
        style={"margin-left": "10%", "margin-right": "10%"},
    )


@app.callback(
    Output("output-ann-data-upload", "children"),
    [Input("upload-annotation-data", "contents")],
)
def upload_ann_data(contents):
    if contents is not None:
        children = parse_contents(contents)
        return children


@app.callback(
    Output("data_dir_textbox", "children"),
    [Input("input_data_dir", "value")],
    [State("input_data_dir", "value")],
)
def check_datapath(prev, curr):
    placeholder = "Path to images (i.e. C:/Users/me/project/data/val2017)"
    valid = (
        dbc.Input(
            id="input_data_dir",
            type="text",
            placeholder=placeholder,
            valid=True,
            className="mb-3",
            value=prev,
        ),
    )
    invalid = (
        dbc.Input(
            id="input_data_dir",
            type="text",
            placeholder=placeholder,
            invalid=True,
            className="mb-3",
            value=prev,
        ),
    )
    if curr is None:
        curr = ""
    if os.path.isdir(curr):
        global datadir
        datadir = curr
        return html.Div(children=valid)
    else:
        return html.Div(children=invalid)


@app.callback(
    Output("checkbox_output", "children"), [Input("batch_checkbox", "checked")],
)
def on_form_change(checkbox_checked):
    if checkbox_checked:
        global batch_analysis
        batch_analysis = True


@app.callback(
    Output("output-analysis-data-upload", "children"),
    [Input("upload-analysis-data", "contents")],
)
def upload_analysis_data(contents):
    # TODO: callback triggers tab-0 rendering
    if contents is not None:
        children = parse_contents(contents)
        return children


@app.callback(
    Output("output-analysis-btn", "children"), [Input("analyze_button", "n_clicks")],
)
def analyze_button(n_clicks):
    # TODO: callback triggers tab-0 rendering
    global datadir, annotation_file, analysis_df, profile, batch_analysis
    if n_clicks is not None:
        if batch_analysis:
            datasets = get_blob_datasets(datadir)
            merge_datasets(datasets)
            annotation_file = "./temp/results/merged/annotations/merged.json"
            datadir = "./temp/results/merged/images"
        try:
            analysis_df = analyze_dataset(annotation_file, datadir)
            print(annotation_file)
            profile = ProfileReport(analysis_df, title="Dataset").to_html()
        except Exception as e:
            print(e)
            return exception_html


@app.callback(
    Output("output-analysis-btn-online", "children"),
    [Input("analyze_button_online", "n_clicks")],
    [State("dataset_selection", "value")],
)
def analyze_button_online(n_clicks, dataset):
    global datadir, annotation_file, analysis_df, profile, batch_analysis
    if n_clicks is not None:
        # if batch_analysis:
        #     datasets = get_blob_datasets(datadir)
        #     merge_datasets(datasets)
        #     annotation_file = "./temp/results/merged/annotations/merged.json"
        #     datadir = "./temp/results/merged/images"
        blob_list = get_blobs(dataset)
        blob_names = download_blobs(blob_list)
        annotation_file = [i for i in blob_names if ".json" in i][0]
        imgs = [i for i in blob_names if ".jpg" in i or ".png" in i]
        datadir = os.path.dirname(imgs[0])
        annotation_file = os.path.join("../blob_data", annotation_file)
        datadir = os.path.join("../blob_data", datadir)
        try:
            analysis_df = analyze_dataset(annotation_file, datadir)
            print(annotation_file)
            profile = ProfileReport(analysis_df, title="Dataset").to_html()
        except Exception as e:
            print(e)
            return exception_html


@app.callback(
    Output("tabs-figures", "children"), [Input("tabs", "value")],
)
def render_tab(tab):
    try:
        if tab == "tab-0":
            return render_tab0()
        elif tab == "tab-1":
            return render_tab1()
        elif tab == "tab-2":
            return render_tab2()
        elif tab == "tab-3":
            return render_tab3()
        elif tab == "tab-4":
            return render_tab4()
        elif tab == "tab-5":
            return render_tab5()
    except Exception as e:
        print(e)
        return exception_html


@app.callback(
    Output("obj_hist_out", "children"), [Input("objs_per_img", "clickData")],
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
    Output("area_hist_out", "children"), [Input("cat_areas", "clickData")],
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


@app.callback(
    Output("obj_imgs", "children"), [Input("objs_hist", "clickData")],
)
def display_obj_imgs(click_data):
    if click_data is not None:
        global obj_cat
        cat_ids = coco_data.getCatIds(catNms=obj_cat)
        img_ids = coco_data.getImgIds(catIds=cat_ids)
        _, data = get_objs_per_img(cat_ids, img_ids, coco_data)
        num_objs = click_data["points"][0]["x"]
        img_ids = data.loc[data["number of objs"] == num_objs]["imgID"]
        html_imgs = get_html_imgs(img_ids, obj_cat)
        return html.Div(html_imgs)


@app.callback(
    Output("area_imgs", "children"), [Input("area_hist", "clickData")],
)
def display_area_imgs(click_data):
    if click_data is not None:
        global area_cat
        cat_ids = coco_data.getCatIds(catNms=area_cat)
        img_ids = coco_data.getImgIds(catIds=cat_ids)
        _, data = get_proportion(cat_ids, img_ids, coco_data)
        data_df = pd.DataFrame(data)
        point_nums = click_data["points"][0]["pointNumbers"]
        img_ids = data_df[data_df.index.isin(point_nums)]["imgID"]
        html_imgs = get_html_imgs(img_ids, area_cat)
        return html.Div(html_imgs)


@app.callback(
    Output("anomaly_imgs", "children"), [Input("cat_selection", "value")],
)
def display_anomalies(value):
    global analysis_df
    try:
        img_ids = (
            analysis_df["images w/ abnormal objects"][
                analysis_df["category"] == value
            ].tolist()
        )[0]
        anns = analysis_df["abnormal objects"][analysis_df["category"] == value]
        ann_ids = (anns.tolist())[0]
        html_imgs = get_html_imgs(img_ids, value, outlying_anns=ann_ids)
        return html.Div(
            children=html_imgs,
            style={
                "margin-left": "5%",
                "margin-right": "5%",
                "margin-top": "25px",
                "overflow": "scroll",
                "border": "2px black solid",
                "box-sizing": "border-box",
                "width": "600px",
                "height": "600px",
            },
        )
    except Exception as e:
        print(e)
        return exception_html


@app.callback(
    Output("anomaly_table", "children"),
    [Input({"type": "output_image", "index": ALL}, "n_clicks")],
    [
        State({"type": "output_image", "index": ALL}, "data-category"),
        State({"type": "output_image", "index": ALL}, "id"),
    ],
)
def display_anomaly_table(n_clicks, cat, id):
    global anomaly_table
    for i, val in enumerate(n_clicks):
        file = id[i]["index"]
        # Add filename to anomaly_table if it does not already contain the file
        if (val is not None) and (
            not anomaly_table["Image filename"].str.contains(file).any()
        ):
            new_row = {"Category": cat[i], "Image filename": file}
            anomaly_table = anomaly_table.append(new_row, ignore_index=True)
    return html.Div(
        dash_table.DataTable(
            id="anomaly_datatable",
            columns=[{"name": i, "id": i} for i in anomaly_table.columns],
            data=anomaly_table.to_dict("records"),
            row_deletable=True,
            export_format="xlsx",
            export_headers="display",
        ),
        style={"margin-left": "5%", "margin-right": "5%"},
    )


@app.callback(
    Output("anomaly_datatable", "children"),
    [Input("anomaly_datatable", "data_previous")],
    [State("anomaly_datatable", "data")],
)
def update_anomaly_table(prev, curr):
    if prev is None:
        dash.exceptions.PreventUpdate()
    else:
        removed = list(
            set([i["Image filename"] for i in prev])
            - set([i["Image filename"] for i in curr])
        )[0]
        global anomaly_table
        col = "Image filename"
        anomaly_table = anomaly_table[anomaly_table[col] != removed]


style = {"margin-left": "50px", "margin-top": "50px", "font-size": "75px"}
path = "Path to images (i.e. C:/Users/me/project/data/val2017)"
tab4_label = "Proportion of object in image"


def display_header(local):
    if local:
        return html.Div(
            [
                dcc.Upload(
                    id="upload-annotation-data",
                    children=dbc.Button(
                        "Upload JSON Annotation File", color="primary", block=True
                    ),
                    multiple=False,
                    style={"margin-left": "10%", "margin-right": "10%"},
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                id="data_dir_textbox",
                                children=dbc.Input(
                                    id="input_data_dir",
                                    type="text",
                                    placeholder=path,
                                    className="mb-3",
                                ),
                            ),
                            width=9,
                        ),
                        dbc.Col(
                            html.Div(
                                dbc.FormGroup(
                                    [
                                        dbc.Checkbox(
                                            id="batch_checkbox",
                                            className="form-check-input",
                                        ),
                                        dbc.Label(
                                            "batch analysis",
                                            html_for="batch_checkbox",
                                            className="form-check-label",
                                        ),
                                    ],
                                    check=True,
                                ),
                            ),
                            width=3,
                        ),
                    ],
                    style={"margin-left": "10%", "margin-right": "10%"},
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Upload(
                                    id="upload-analysis-data",
                                    children=dbc.Button(
                                        "Load Feather Analysis File",
                                        color="primary",
                                        block=True,
                                        outline=True,
                                    ),
                                    multiple=False,
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Analyze",
                                    id="analyze_button",
                                    color="primary",
                                    outline=True,
                                    block=True,
                                )
                            ],
                            width=6,
                        ),
                    ],
                    style={"margin-left": "20%", "margin-right": "20%"},
                ),
            ]
        )
    else:
        datasets = get_blob_datasets()
        return html.Div(
            [
                dcc.Dropdown(
                    id="dataset_selection",
                    options=[{"label": i, "value": i} for i in datasets],
                    value=datasets[0],
                    style={
                        "margin-top": "25px",
                        "margin-left": "25px",
                        "margin-right": "50%",
                    },
                ),
                dbc.Button(
                    "Analyze",
                    id="analyze_button_online",
                    color="primary",
                    outline=True,
                    style={
                        "margin-top": "25px",
                        "margin-left": "25px",
                        "margin-right": "50%",
                    },
                ),
            ]
        )


header = display_header(local=False)
app.layout = html.Div(
    children=[
        html.H1(children="Viz EDA", style=style),
        html.Div(
            children="Exploratory data analysis for computer vision and "
            "object recognition.",
            style={"margin-left": "50px", "margin-bottom": "50px"},
        ),
        html.Hr(),
        header,
        html.Hr(),
        html.Div(id="output-ann-data-upload", style={"display": "none"}),
        html.Div(id="output-analysis-data-upload", style={"display": "none"}),
        html.Div(id="output-analysis-btn", style={"display": "none"}),
        html.Div(id="output-analysis-btn-online", style={"display": "none"}),
        html.Div(id="checkbox_output", style={"display": "none"}),
        dcc.Tabs(
            id="tabs",
            value="tab-0",
            children=[
                dcc.Tab(label="Overview", value="tab-0"),
                dcc.Tab(label="Objects per class", value="tab-1"),
                dcc.Tab(label="Images per class", value="tab-2"),
                dcc.Tab(label="Objects per image", value="tab-3"),
                dcc.Tab(label=tab4_label, value="tab-4",),
                dcc.Tab(label="Anomaly detection", value="tab-5"),
            ],
        ),
        html.Div(id="tabs-figures"),
        html.Hr(),
    ]
)

if __name__ == "__main__":
    # Run on docker
    # application.run(host="0.0.0.0", port=port, debug=True)

    # Run locally
    app.run_server(port=port, debug=True)

    # Only do analysis
    # annotation_file = ""
    # datadir = ""
    # analyzeDataset(annotation_file, datadir)
