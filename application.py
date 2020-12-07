import base64
import os
import shutil
from io import BytesIO
from urllib.parse import quote as urlquote
import json
import random

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
from flask import send_file
from pandas_profiling import ProfileReport
from skimage import io
from tqdm import tqdm

from app.analysis import analyze_dataset, coco, get_objs_per_img, get_proportion
from overview import compute_overview_data

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])
app.config["suppress_callback_exceptions"] = True
application = app.server
port = 80
batch_analysis = False

analysis_df = pd.DataFrame()
anomaly_table = pd.DataFrame(columns=["Category", "Image ID"])

datadir = ""
annotation_file = ""

coco_data = None

profile = "" # holds raw html from pandas-profiling output
obj_cat = "" # currently selected category on object tab
area_cat = "" # currently selected category on area tab

# general html for exception handling
exception_html = html.Div(
    children="Please load a valid COCO-style annotation file and "
    "define a valid folder.",
    style={"margin-left": "50px", "margin-top": "50px","display":"none"},
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
    global analysis_df, annotation_file, coco_data, profile
    content_type, content_string = contents.split(",", 1)
    if content_type == "data:application/json;base64":
        decoded = base64.b64decode(content_string).decode("UTF-8")
        os.mkdir("./output")
        with open("./output/output.json", "w") as file:
            file.write(decoded)
        print("\n\nWrote to file!!\n\n")
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

def set_blob_data(dataset):
    global annotation_file, datadir, coco_data
    blob_list = get_blobs(dataset)
    blob_names = download_blobs(blob_list)
    annotation_file = [i for i in blob_names if ".json" in i][0]
    imgs = [i for i in blob_names if ".jpg" in i or ".png" in i]
    datadir = os.path.dirname(imgs[0])
    annotation_file = os.path.join("../blob_data", annotation_file)
    datadir = os.path.join("../blob_data", datadir)
    coco_data = coco.COCO(annotation_file)

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
            dbc.Spinner(html.Div(id="obj_imgs"), size="lg"),
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
            dbc.Spinner(html.Div(id="area_imgs"), size="lg"),
        ],
        style={"margin-left": "10%", "margin-right": "10%"},
    )

def render_tab5():
    cat_ids = coco_data.getCatIds()
    cat_dict = coco_data.loadCats(cat_ids)
    cat_nms = [d["name"] for d in cat_dict]
    options = [{"label": i, "value": i} for i in cat_nms]
    s = {"margin-top": "25px", "margin-left": "25px"}
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(children="Select a category", style=s),
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
                            dbc.Spinner(html.Div(id="anomaly_imgs"), size="lg"),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H3(children="Flagged anomalies", style=s),
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
    Output("checkbox_output", "children"), 
    [Input("batch_checkbox", "checked")],
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
    if contents is not None:
        children = parse_contents(contents)
        return children

@app.callback(
    Output("output-analysis-data-upload-online", "children"),
    [Input("upload-analysis-data-online", "contents")],
    [State("dataset_selection", "value")],
)
def upload_analysis_data_online(contents, dataset):
    global annotation_file, coco_data
    if contents is not None:
        set_blob_data(dataset)
        children = parse_contents(contents)
        return children

@app.callback(
    Output("output-analysis-btn", "children"), 
    [Input("analyze_button", "n_clicks")],
)
def analyze_button(n_clicks):
    global datadir, annotation_file, analysis_df, profile, batch_analysis
    if n_clicks is not None:
        if batch_analysis:
            datasets = get_datasets(datadir)
            merge_datasets(datasets)
            annotation_file = "./temp/results/merged/annotations/merged.json"
            datadir = "./temp/results/merged/images"
        try:
            analysis_df, analysis_output = analyze_dataset(annotation_file, datadir)
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
    global datadir, annotation_file, analysis_df, profile, batch_analysis, coco_data
    if n_clicks is not None:
        # if batch_analysis:
        #     datasets = get_blob_datasets(datadir)
        #     merge_datasets(datasets)
        #     annotation_file = "./temp/results/merged/annotations/merged.json"
        #     datadir = "./temp/results/merged/images"
        set_blob_data(dataset)
        try:
            analysis_df, analysis_output = analyze_dataset(annotation_file, datadir)
            download_analysis(analysis_output)
            profile = ProfileReport(analysis_df, title="Dataset").to_html()
            location = "/download_analysis/{}".format(urlquote(analysis_output))
            return (
                html.Div(
                    [
                        html.Hr(),
                        html.A(
                            dbc.Button(
                                "Download Analysis",
                                color="primary",
                                block=True,
                                outline=True,
                            ),
                            href=location,
                        ),
                    ]
                ),
            )
        except Exception as e:
            print(e)
            return exception_html

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

@app.callback(
    Output("obj_imgs", "children"), 
    [Input("objs_hist", "clickData")],
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
    Output("area_imgs", "children"), 
    [Input("area_hist", "clickData")],
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
    Output("anomaly_imgs", "children"), 
    [Input("cat_selection", "value")],
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
            not anomaly_table["Image ID"].str.contains(file).any()
        ):
            new_row = {"Category": cat[i], "Image ID": file}
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
            set([i["Image ID"] for i in prev]) - set([i["Image ID"] for i in curr])
        )[0]
        global anomaly_table
        col = "Image ID"
        anomaly_table = anomaly_table[anomaly_table[col] != removed]

@app.server.route("/download_analysis/<path:filename>")
def download_analysis(filename):
    print(filename)
    return send_file(filename, attachment_filename=filename, as_attachment=True)

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
            
            html.Hr(),
            
            html.Div(
                [dcc.Upload(
                    id="upload-analysis-data",
                    children=dbc.Button(
                        "Load Feather Analysis File",
                        color="primary",
                        block=True,
                        outline=True,
                    ),
                    multiple=False,
                ),
                dbc.Button("Analyze", id="analyze_button", color="primary", block=True)
                ],
                style={"margin-left": "10%", "margin-right": "10%"},
            ),],
            style={"display":"none"}
        )
    else:
        datasets = get_blob_datasets()
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="dataset_selection",
                                options=[{"label": i, "value": i} for i in datasets],
                                value=datasets[0],
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Analyze",
                                id="analyze_button_online",
                                color="primary",
                                outline=True,
                                block=True,
                            ),
                            width=6,
                        ),
                    ]
                ),
                html.Div(id="output-analysis-btn-online"),
                html.Hr(),
                html.Div(
                    [dcc.Upload(
                        id="upload-analysis-data-online",
                        children=dbc.Button(
                            "Upload Feather Analysis File",
                            color="primary",
                            block=True,
                            outline=True,
                        ),
                        multiple=False,
                    ),
                    ]
                ),
            ],
            style={"margin-left": "10%", "margin-right": "10%"},
        )


header = display_header(local=True)

navbar = dbc.Navbar(
    [
        dbc.Row(
            [
                dbc.Col(dbc.NavbarBrand("VIZ EDA", style={"font-size":"2.5rem","white-space":"pre-wrap","font-weight":"bolder","font-family":"sans-serif","letter-spacing":"2px"}),width="auto",style={"margin-left":"25px"}),
                dbc.Col(html.H5("Exploratory data analysis for computer vision",style={"color":"#585858","margin-top":"0.7%"})),
                dbc.Col(html.A(
                    html.Img(
                        src="https://cdn1.iconfinder.com/data/icons/arrows-elements-outline/128/ic_round_update-128.png",
                        style={"height":"25%","width":"25%","float":"right","margin-right":"25px"}
                        ),
                    id="reload-button",
                    href="http://localhost:8080/"
                    )
                ,width="auto",
                )
            ],
            align="center",
            no_gutters=True,
            style={"width":"100%"}
        ),
    ],
    style={"padding":"0.1rem"}
)

welcome_menu = html.Div(
    [
        html.Div(dbc.Button("New analysis", color="dark", className="mr-1",outline=True,style={"float":"right","width":"25%","letter-spacing":"2px"},id="new-analysis-btn"),style={"width":"50%","margin":"auto"}),
        html.Div(dbc.Button("Existing analysis", color="dark", className="mr-1",outline=True,style={"width":"25%","letter-spacing":"2px"},id="existing-analyis-btn"),style={"width":"50%","margin":"auto"})
    ],
    style={"display":"flex","padding-top":"22.5%"},
    id="welcome-menu"
)

new_analysis_menu = html.Div(
    [
        dbc.FormGroup(
            [
                dbc.Checklist(
                    options=[{"label":"Batch analysis","value":"1"}],
                    value=[],
                    switch=True,
                    id="batch-analysis-toggle"
                )
            ],
            style={"margin":"auto","padding-bottom":"0.8%","width":"22%"}
        ),
        html.Div(
            [
                dbc.Input(
                    type="text",
                    placeholder="Path to images e.g. /Users/me/project/data/val2017",
                    className="mb-3",
                    id="images-upload",
                    style={"width":"100%","letter-spacing":"1px"}
                )
            ],
            style={"margin":"auto","width":"22%","padding-bottom":"0.28%"}
        ),
        dcc.Upload(
            [
                dbc.Button("Upload annotation file (.json)",color="dark", className="mr-1",outline=True,style={"width":"100%","letter-spacing":"2px"},id="annotation-upload-btn")
            ],
            multiple=False,
            id="annotation-upload",
            style={"margin":"auto","width":"22%","padding-bottom":"0.9%"}
        ),
        html.Div(
            [
                dbc.Button(
                    "Analyse",
                    color="dark",
                    className="mr-1",
                    style={"width":"100%","letter-spacing":"2px"},
                    id="analyse-btn"
                )
            ],
            style={"margin":"auto","width":"22%"}
        )
    ],
    id="new-analysis-menu"
)

tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Overview", tab_id="overview-tab",style={"letter-spacing":"2px"}),
                dbc.Tab(label="Objects per class", tab_id="objs-per-class-tab",style={"letter-spacing":"2px"}),
                dbc.Tab(label="Images per class", tab_id="imgs-per-class-tab",style={"letter-spacing":"2px"}),
                dbc.Tab(label="Objects per image", tab_id="objs-per-img-tab",style={"letter-spacing":"2px"}),
                dbc.Tab(label="Proportion of object in image", tab_id="prop-obj-in-img-tab",style={"letter-spacing":"2px"}),
                dbc.Tab(label="Anomaly detection", tab_id="anomaly-det-tab",style={"letter-spacing":"2px"}),
            ],
            id="new-tabs",
            style={"display":"flex","justify-content":"center","padding-top":"0.5%"}
        ),
        html.Div(id="tabs-content",style={"padding":"1.6%","background":"white"}),
    ],
    id="tabs-div"
)

bridge = dcc.Loading(html.Div(id="bridge",style={"display":"none"}),id="loading-bridge",color="#5cb85c",style={"display":"none"})

@app.callback(
    Output("images-upload", "valid"),
    Input("images-upload", "value"))
def check_img_path(path):
    if path is None:
        path = ""
    if os.path.isdir(path):
        return True
    else:
        return False

@app.callback(
    Output("annotation-upload-btn","color"),
    Input("annotation-upload","contents"))
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
    Output("analyse-btn", "disabled"),
    Output("analyse-btn", "color"),
    Input("images-upload", "valid"),
    Input("annotation-upload-btn", "color"))
def check_inputs(valid_path,color):
    if valid_path and color == "success":
        return False,"success"
    elif valid_path and not color == "danger":
        return True,"dark"
    else:
        return True,"dark"

@app.callback(
    Output("bridge","children"),
    Output("tabs-div", "style"),
    [Input("analyse-btn", "n_clicks"),
    Input("images-upload", "value"),
    Input("annotation-upload", "contents")])
def show_tabs(click, images, annotations):
    if click:
        path_to_annotations = parse_annotations(annotations)
        overview_data = compute_overview_data(images, path_to_annotations)
        if not os.path.isdir("./output"):
            os.mkdir("./output")
        path_to_overview_data = "./output/overview_data.json"
        f = open(path_to_overview_data, 'w')
        json.dump(overview_data, f)
        bridge = html.P(str(overview_data["anns_count"]))
        tabs_div_style = {"display":"block","background":"white","padding-top":"0.3%"}
    else:
        bridge = None
        tabs_div_style = {"display":"none"}
    return bridge,tabs_div_style

@app.callback(
    Output("tabs-content", "children"),
    [Input("new-tabs", "active_tab"),
    Input("bridge", "children"),
    Input("analyse-btn", "n_clicks")])
def show_tabs_contents(tab, bridge, click):
    if tab == "overview-tab":
        contents = render_overview()
    elif tab == "objs-per-class-tab":
        contents = html.Div("yet to implement")
        #contents = render_objs_per_class(images, annotations)
    elif tab == "imgs-per-class-tab":
        contents = html.Div("yet to implement")
        #contents = render_imgs_per_class(images, annotations)
    elif tab == "objs-per-img-tab":
        contents = html.Div("yet to implement")
        #contents = render_objs_per_img(images, annotations)
    elif tab == "prop-obj-in-img-tab":
        contents = html.Div("yet to implement")
        #contents = render_prop_objs_per_img(images, annotations)
    elif tab == "anomaly-det-tab":
        contents = html.Div("yet to implement")
        #contents = render_anomaly_det(images, annotations)
    else:
        contents = html.Div()
    return contents

def render_overview():
    path_to_overview_data = "./output/overview_data.json"
    if os.path.isfile(path_to_overview_data):
        f = open(path_to_overview_data)
        overview_data = json.load(f)

        section_title = overview_data["info"]["description"]
        section_title = section_title + " overview"

        info_table_header = [html.Thead(html.Tr([html.Th("Info",style={"border-top-right-radius":"0px","font-size":"large","font-weight":"900","background":"cornflowerblue"}),html.Th("",style={"border-top-left-radius":"0px","background":"cornflowerblue"})]))]
        
        dataset_name = overview_data["info"]["description"]
        i_row1 = html.Tr([html.Td("Dataset name"), html.Td(dataset_name,style={"font-weight":"bold","text-align":"right"})])

        dataset_url = overview_data["info"]["url"]
        i_row2 = html.Tr([html.Td("URL"), html.Td(dataset_url,style={"font-weight":"bold","text-align":"right"})])

        dataset_version = overview_data["info"]["version"]
        i_row3 = html.Tr([html.Td("Version"), html.Td(dataset_version,style={"font-weight":"bold","text-align":"right"})])

        year = overview_data["info"]["year"]
        i_row4 = html.Tr([html.Td("Year"), html.Td(year,style={"font-weight":"bold","text-align":"right"})])

        contributor = overview_data["info"]["contributor"]
        i_row5 = html.Tr([html.Td("Contributor"), html.Td(contributor,style={"font-weight":"bold","text-align":"right"})])

        date_created = overview_data["info"]["date_created"]
        i_row6 = html.Tr([html.Td("Date created"), html.Td(date_created,style={"font-weight":"bold","text-align":"right"})])

        info_table_body = [html.Tbody([i_row1, i_row2, i_row3, i_row4, i_row5, i_row6])]

        info = dbc.Table(info_table_header + info_table_body, striped=True, bordered=True, hover=True, style={"width":"30%"})

        summary_table_header = [html.Thead(html.Tr([html.Th("Summary",style={"border-top-right-radius":"0px","font-size":"large","font-weight":"900","background":"lightgreen"}),html.Th("",style={"border-top-left-radius":"0px","background":"lightgreen"})]))]

        no_classes = str(len(list(overview_data["classes"].keys())))
        s_row1 = html.Tr([html.Td("No. of classes"), html.Td(no_classes,style={"font-weight":"bold","text-align":"right"})])

        no_anns = str(overview_data["anns_count"])
        s_row2 = html.Tr([html.Td("No. of annotations"), html.Td(no_anns,style={"font-weight":"bold","text-align":"right"})])

        no_imgs = str(overview_data["imgs_count"])
        s_row3 = html.Tr([html.Td("No. of images"), html.Td(no_imgs,style={"font-weight":"bold","text-align":"right"})])

        min_anns_per_img = str(overview_data["min_anns_per_img"])
        s_row4 = html.Tr([html.Td("Min. annotations per image"), html.Td(min_anns_per_img,style={"font-weight":"bold","text-align":"right"})])

        max_anns_per_img = str(overview_data["max_anns_per_img"])
        s_row5 = html.Tr([html.Td("Max. annotations per image"), html.Td(max_anns_per_img,style={"font-weight":"bold","text-align":"right"})])

        avg_anns_per_img = "{0:.2f}".format(overview_data["avg_anns_per_img"])
        s_row6 = html.Tr([html.Td("Avg. annotations per image"), html.Td(avg_anns_per_img,style={"font-weight":"bold","text-align":"right"})])

        summary_table_body = [html.Tbody([s_row1, s_row2, s_row3, s_row4, s_row5, s_row6])]

        summary = dbc.Table(summary_table_header + summary_table_body, striped=True, bordered=True, hover=True, style={"width":"30%"})

        warnings_table_header = [html.Thead(html.Tr([html.Th("Warnings",style={"border-top-right-radius":"0px","font-size":"large","font-weight":"900","background":"#f5dd7d"}),html.Th("",style={"border-top-left-radius":"0px","background":"#f5dd7d"})]))]

        uniform_distribution = int(overview_data["uniform_distribution"])
        if uniform_distribution == 1:
            uniform_distribution = html.Td("Uniform",style={"color":"green","text-align":"right","font-weight":"bold"})
        else:
            uniform_distribution = html.Td("Not uniform",style={"color":"red","text-align":"right","font-weight":"bold"})
        w_row1 = html.Tr([html.Td("Class distribution"), uniform_distribution])

        imgs_with_no_anns = len(list(overview_data["imgs_with_no_anns"]))
        if imgs_with_no_anns > 0:
            imgs_with_no_anns = html.Td(imgs_with_no_anns,style={"color":"red","text-align":"right","font-weight":"bold"})
        else:
            imgs_with_no_anns = html.Td("None",style={"color":"green","text-align":"right","font-weight":"bold"})
        w_row2 = html.Tr([html.Td("Images with no annotations"), imgs_with_no_anns])

        anns_with_no_imgs = len(list(overview_data["anns_with_no_imgs"]))
        if anns_with_no_imgs > 0:
            anns_with_no_imgs = html.Td(anns_with_no_imgs,style={"color":"red","text-align":"right","font-weight":"bold"})
        else:
            anns_with_no_imgs = html.Td("None",style={"color":"green","text-align":"right","font-weight":"bold"})
        w_row3 = html.Tr([html.Td("Annotations with no images"), anns_with_no_imgs])

        imgs_wrong_dims = len(list(overview_data["imgs_wrong_dims"]))
        if imgs_wrong_dims > 0:
            imgs_wrong_dims = html.Td(imgs_wrong_dims,style={"color":"red","text-align":"right","font-weight":"bold"})
        else:
            imgs_wrong_dims = html.Td("None",style={"color":"green","text-align":"right","font-weight":"bold"})
        w_row4 = html.Tr([html.Td("Images with wrong dimensions"), imgs_wrong_dims])

        missing_classes = len(list(overview_data["missing_classes"]))
        if missing_classes > 0:
            missing_classes = html.Td(missing_classes,style={"color":"red","text-align":"right","font-weight":"bold"})
        else:
            missing_classes = html.Td("None",style={"color":"green","text-align":"right","font-weight":"bold"})
        w_row5 = html.Tr([html.Td("Missing classes"), missing_classes])

        missing_imgs = len(list(overview_data["missing_imgs"]))
        if missing_imgs > 0:
            missing_imgs = html.Td(missing_imgs,style={"color":"red","text-align":"right","font-weight":"bold"})
        else:
            missing_imgs = html.Td("None",style={"color":"green","text-align":"right","font-weight":"bold"})
        w_row6 = html.Tr([html.Td("Missing images"), missing_imgs])

        warnings_table_body = [html.Tbody([w_row1, w_row2, w_row3, w_row4, w_row5, w_row6])]

        warnings = dbc.Table(warnings_table_header + warnings_table_body, striped=True, bordered=True, hover=True, style={"width":"30%"})

        classes = overview_data["classes"]

        colors = ["#ccb8ff","cornflowerblue","lightgreen","#f5dd7d"]
        i = 0
        lencol = len(colors)

        class_tables = []
        for cl in classes:
            id_string = "ID: {}".format(cl)
            class_table_header = [html.Thead(html.Tr([html.Th(classes[cl]["name"],style={"border-top-right-radius":"0px","background":colors[i]}),html.Th(id_string,style={"text-align":"right","border-top-left-radius":"0px","background":colors[i]})]))]
            anns_string = str(classes[cl]["anns_count"]) + " (" + "{0:.2f}".format(classes[cl]["anns_prop"]) + "%)"
            cl_row1 = html.Tr([html.Td("No. of annotations"), html.Td(anns_string,style={"text-align":"right","font-weight":"bold"})])
            imgs_string = str(classes[cl]["imgs_count"]) + " (" + "{0:.2f}".format(classes[cl]["imgs_prop"]) + "%)"
            cl_row2 = html.Tr([html.Td("No. of images"), html.Td(imgs_string,style={"text-align":"right","font-weight":"bold"})])
            unique_imgs_string = str(classes[cl]["unique_imgs_count"]) + " (" + "{0:.2f}".format(classes[cl]["unique_imgs_prop"]) + "%)"
            cl_row3 = html.Tr([html.Td("No. of unique images"), html.Td(unique_imgs_string,style={"text-align":"right","font-weight":"bold"})])
            class_table_body = [html.Tbody([cl_row1,cl_row2,cl_row3])]
            class_table = dbc.Table(class_table_header + class_table_body, striped=True, bordered=True, hover=True)
            class_tables.append(class_table)
            i += 1
            i = i%lencol

        i = 0
        class_images = []
        for cl in classes:
            unique_imgs_list = classes[cl]["unique_imgs"]
            imgs_list = classes[cl]["imgs"]
            if len(unique_imgs_list) > 3:
                chosen_list = unique_imgs_list
            elif len(imgs_list) > 3:
                chosen_list = imgs_list
            if chosen_list:
                random_paths = random.sample(chosen_list, 3)
                encoded_image1 = base64.b64encode(open(random_paths[0], 'rb').read())
                encoded_image2 = base64.b64encode(open(random_paths[1], 'rb').read())
                encoded_image3 = base64.b64encode(open(random_paths[2], 'rb').read())
                row = dbc.Row([dbc.Col(class_tables[i]),
                dbc.Col(dbc.Card([dbc.CardImg(src='data:image/png;base64,{}'.format(encoded_image1.decode()),style={"height":"100%"})],style={"height":"100%","justify-content":"center"}),style={"height":"100%"}),
                dbc.Col(dbc.Card([dbc.CardImg(src='data:image/png;base64,{}'.format(encoded_image2.decode()),style={"height":"100%"})],style={"height":"100%","justify-content":"center"}),style={"height":"100%"}),
                dbc.Col(dbc.Card([dbc.CardImg(src='data:image/png;base64,{}'.format(encoded_image3.decode()),style={"height":"100%"})],style={"height":"100%","justify-content":"center"}),style={"height":"100%"})],
                style={"height":"0.62%","margin-bottom":"3%","border-bottom":"1px solid darkslategray","padding-bottom":"3%"})
                class_images.append(row)
            else:
                img_card = dbc.Card([dbc.CardBody("no image available")],style={"height":"1.15%","margin-bottom":"2%"})
                class_images.append(img_card)
            i += 1

        overview = html.Div([
            html.H4(section_title,style={"text-transform":"capitalize","padding-bottom":"1%","letter-spacing":"2px","font-size":"x-large"}),
            html.Div([info,summary,warnings],style={"display":"flex","justify-content":"space-between","padding-bottom":"1%"}),
            html.H4("Classes",style={"text-transform":"capitalize","padding-bottom":"1%","letter-spacing":"2px","font-size":"x-large"}),
            #html.Div([html.Div(class_tables, style={"width":"30%","margin-right":"5%"}),html.Div(class_images, style={"width":"65%"})], style={"display":"flex"})
            html.Div(class_images)
        ]
        )
    else:
        overview = html.Div()
    return overview

def parse_annotations(annotations):
    _, annotations_contents = annotations.split(",", 1)
    decoded = base64.b64decode(annotations_contents).decode("UTF-8")
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    path_to_annotations = "./output/annotations.json"
    f = open(path_to_annotations, 'w')
    f.write(decoded)
    return path_to_annotations

def render_objs_per_class(images, annotations):
    contents = html.Div([html.P("Hello World!")])
    return contents

def render_imgs_per_class(images, annotations):
    pass

def render_objs_per_img(images, annotations):
    pass

def render_prop_objs_per_img(images, annotations):
    pass

def render_anomaly_det(images, annotations):
    pass

@app.callback(
    Output("reload-button", "style"),
    Output("new-analysis-menu", "style"),
    Output("welcome-menu", "style"),
    Output("loading-bridge", "style"),
    Input("new-analysis-btn", "n_clicks"),
    Input("existing-analyis-btn", "n_clicks"),
    Input("analyse-btn", "n_clicks"))
def show_reload_button(new,existing,analyse):
    if new:
        reload_btn_style = {"display":"block"}
        new_analysis_menu_style = {"display":"block","padding-top":"18%"}
        welcome_menu_style = {"display":"none"}
        loading_style = {"display":"none"}
    elif existing:
        reload_btn_style = {"display":"block"}
        new_analysis_menu_style = {"display":"none"}
        welcome_menu_style = {"display":"none"}
        loading_style = {"display":"none"}
    else:
        reload_btn_style = {"display":"none"}
        new_analysis_menu_style = {"display":"none"}
        welcome_menu_style = {"display":"flex","padding-top":"22.5%"}
        loading_style = {"display":"none"}
    if analyse:
        reload_btn_style = {"display":"block"}
        new_analysis_menu_style = {"display":"none"}
        welcome_menu_style = {"display":"none"}
        loading_style = {"display":"block","padding-top":"50%"}
    return reload_btn_style, new_analysis_menu_style, welcome_menu_style, loading_style

app.layout = html.Div(
    children=[
        navbar,
        welcome_menu,
        new_analysis_menu,
        bridge,
        tabs,
        header,
        html.Div(id="output-ann-data-upload", style={"display": "none"}),
        html.Div(id="output-analysis-data-upload", style={"display": "none"}),
        html.Div(id="output-analysis-btn", style={"display": "none"}),
        html.Div(id="output-analysis-data-upload-online", style={"display": "none"}),
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
            style={"display":"none"}
        ),
        html.Div(id="tabs-figures"),
    ]
)

if __name__ == "__main__":
    # Run on docker
    application.run(host="0.0.0.0", port=port, debug=True)