import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
from skimage import io
from analysis import analyzeDataset, getObjsPerImg, getArea, coco, round_nearest

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
global cocoData
cocoData = coco.COCO('output.json')


def parseContents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('UTF-8')
    with open('output.json', 'w') as file:
        file.write(decoded)
    try:
        global analysis_df
        try:
            analysis_df = pd.read_pickle('analysis.pkl')
            print("Loaded analysis.pkl!")
        except FileNotFoundError as e:
            print(e)
            analysis_df = analyzeDataset('output.json', "data/val2017")
    except Exception as e:
        print(e)
        return html.Div([
            'Please load a valid COCO-style annotation file.'
        ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])
def uploadData(contents):
    if contents is not None:
        children = parseContents(contents)
        return children


@app.callback(Output('obj_hist_out', 'children'),
              [Input('objs_per_img', 'clickData')])
def displayObjHist(clickData):
    if clickData is not None:
        cat = clickData['points'][0]['x']
        global objCat
        objCat = cat
        title = "Number of " + cat + "s in an image w/ " + cat + "s\nClick on bin to see images"
        _, data = getObjsPerImg([cat])
        fig = go.Figure(data=[go.Histogram(x=data['number of objs'], xbins=dict(size=1), histnorm='probability')])
        fig.update_layout(clickmode='event+select', yaxis_title="probability", title=title)
        return html.Div([
            dcc.Graph(id='objs_hist', figure=fig),
            html.Div(children='Click on bin to see images'),
        ])


@app.callback(Output('area_hist_out', 'children'),
              [Input('cat_areas', 'clickData')])
def displayAreaHist(clickData):
    if clickData is not None:
        cat = clickData['points'][0]['x']
        global areaCat
        areaCat = cat
        title = "Percentage area of a(n) " + cat + " in an image"
        _, data = getArea([cat])
        fig = go.Figure(data=[go.Histogram(x=data['proportion of img'], xbins=dict(size=0.05), histnorm='probability')])
        fig.update_layout(clickmode='event+select', yaxis_title="probability", title=title)
        return html.Div([
            dcc.Graph(id='area_hist', figure=fig),
            html.Div(children='Click on bin to see images')
        ])


def fig_to_uri(in_fig, close_all=True, **save_args):
    # Save a figure as a URI
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def getHtmlImgs(imgIDs, cat, outlying_anns=None):
    # TODO: shorten runtime if possible
    htmlImgs = []
    catIds = cocoData.getCatIds(catNms=[cat])
    for imgID in set(imgIDs):
        image_filename = 'data/val2017/' + str(imgID).zfill(12) + '.jpg'  # replace with your own image
        I = io.imread(image_filename) / 255.0
        plt.imshow(I)
        plt.axis('off')
        if outlying_anns is None:
            annIds = cocoData.getAnnIds(imgIds=imgID, catIds=catIds, iscrowd=None)
            anns = cocoData.loadAnns(annIds)
        else:
            annIds = set(cocoData.getAnnIds(imgIds=imgID, catIds=catIds, iscrowd=None))
            annIds = list(annIds.intersection(set(outlying_anns)))
            anns = cocoData.loadAnns(annIds)
        cocoData.showAnns(anns)
        decoded_image = fig_to_uri(plt)
        htmlImgs.append(html.Img(src=decoded_image))
    return htmlImgs


@app.callback(Output('obj_imgs', 'children'),
              [Input('objs_hist', 'clickData')])
def displayObjImgs(clickData):
    if clickData is not None:
        _, data = getObjsPerImg([objCat])
        num_objs = clickData['points'][0]['x']
        imgIDs = data.loc[data['number of objs'] == num_objs]['imgID']
        print(imgIDs)
        htmlImgs = getHtmlImgs(imgIDs, objCat)
        return html.Div(htmlImgs)


@app.callback(Output('area_imgs', 'children'),
              [Input('area_hist', 'clickData')])
def displayAreaImgs(clickData):
    # TODO: debug, images not displaying
    if clickData is not None:
        _, data = getArea([areaCat])
        area = clickData['points'][0]['x']
        area = round_nearest(area)
        print(area)
        print(data['proportion of img'])
        imgIDs = data.loc[data['proportion of img'] == area]['imgID']
        print(imgIDs)
        htmlImgs = getHtmlImgs(imgIDs, areaCat)
        return html.Div(htmlImgs)


@app.callback(Output('anomaly_imgs', 'children'),
              [Input('cat_selection', 'value')])
def displayAnomalies(value):
    outlier_imgIds = (analysis_df['images w/ abnormal objects'][analysis_df['category'] == value].tolist())[0]
    outlier_annIds = (analysis_df['abnormal objects'][analysis_df['category'] == value].tolist())[0]
    htmlImgs = getHtmlImgs(outlier_imgIds, value, outlying_anns=outlier_annIds)
    return html.Div(htmlImgs)


@app.callback(Output('tabs-figures', 'children'),
              [Input('tabs', 'value')])
def renderTab(tab):
    try:
        if tab == 'tab-1':
            fig = px.bar(analysis_df, x='category', y='number of objects', title='Number of Objects per Category')
            fig2 = px.pie(analysis_df, values='number of objects', names='category')
            return html.Div([
                dcc.Graph(id='cat_objs_bar', figure=fig),
                dcc.Graph(id='cat_objs_pie', figure=fig2)
            ])

        elif tab == 'tab-2':
            fig = px.bar(analysis_df, x='category', y='number of images', title="Number of Images per Category")
            fig2 = px.pie(analysis_df, values='number of images', names='category')
            return html.Div([
                dcc.Graph(id='cat_imgs_bar', figure=fig),
                dcc.Graph(id='cat_imgs_pie', figure=fig2)
            ])

        elif tab == 'tab-3':
            title = 'Avg Number Of Objects per Image'
            fig = px.bar(analysis_df, x='category', y='avg number of objects per img', title=title)
            fig.update_layout(clickmode='event+select')
            return html.Div([
                dcc.Graph(id='objs_per_img', figure=fig),
                html.Div(children='Click on bin to see probability distribution'),
                html.Div(id='obj_hist_out'),
                html.Div(id='obj_imgs')
            ])

        elif tab == 'tab-4':
            title = 'Avg Proportion of Image'
            fig = px.bar(analysis_df, x="category", y='avg percentage of img', title=title)
            return html.Div([
                dcc.Graph(id='cat_areas', figure=fig),
                html.Div(children='Click on bin to see probability distribution'),
                html.Div(id='area_hist_out'),
                html.Div(id='area_imgs')
            ])

        elif tab == 'tab-5':
            catIds = cocoData.getCatIds()
            catDict = cocoData.loadCats(catIds)
            catNms = [d['name'] for d in catDict]

            options = []
            for cat in catNms:
                dropdown_args = ['label', 'value']
                cat_tuple = [cat, cat]
                options.append(dict(zip(dropdown_args, cat_tuple)))

            return html.Div([
                dcc.Dropdown(
                    id='cat_selection',
                    options=options,
                    placeholder='Select a category'
                ),
                html.Div(id='anomaly_imgs')
            ])

    except Exception as e:
        print(e)
        return html.Div([
            'Please load a valid COCO-style annotation file.'
        ])


app.config['suppress_callback_exceptions'] = True
app.layout = html.Div(children=[
    html.H1(children='Viz EDA'),
    html.Div(children='''
        Exploratory data analysis for computer vision and object recognition.
    '''),
    html.Hr(),
    dcc.Upload(id='upload-data', children=html.Button('Upload File'), multiple=False),
    html.Hr(),

    html.Div(id='output-data-upload'),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Objects per class', value='tab-1'),
        dcc.Tab(label='Images per class', value='tab-2'),
        dcc.Tab(label='Objects per image', value='tab-3'),
        dcc.Tab(label='Proportion of object in image per class', value='tab-4'),
        dcc.Tab(label='Anomaly detection', value='tab-5'),
    ]),
    html.Div(id='tabs-figures')
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
    # analyzeDataset("data/annotations/instances_val2017.json", "data/val2017")
