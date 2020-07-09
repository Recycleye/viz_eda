import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import base64
from analysis import analyzeDataset, getObjsPerImg, getArea
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

global cocoData, dataframe, objCat, areaCat


def parseContents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('UTF-8')
    with open('output.json', 'w') as file:
        file.write(decoded)
    try:
        global dataframe
        dataframe = analyzeDataset('output.json')
    except Exception as e:
        print(e)
        return html.Div([
            'Please load a valid COCO-style annotation file.'
        ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])
def uploadData(contents):
    if contents is not None:
        print(contents)
        children = parseContents(contents)
        return children


@app.callback(Output('obj_hist_out', 'children'),
              [Input('objs_per_img', 'clickData')])
def displayObjHist(clickData):
    if clickData is not None:
        cat = clickData['points'][0]['x']
        global objCat
        objCat = cat
        title = "Number of " + cat + "s in an image w/ " + cat + "s"
        _, data = getObjsPerImg([cat])
        fig = go.Figure(data=[go.Histogram(x=data['number of objs'], xbins=dict(size=1), histnorm='probability')])
        fig.update_layout(clickmode='event+select', yaxis_title="probability", title=title)
        return dcc.Graph(id='objs_hist', figure=fig)


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
        return dcc.Graph(id='area_hist', figure=fig)


@app.callback(Output('obj_imgs', 'children'),
              [Input('objs_hist', 'clickData')])
def displayObjImgs(clickData):
    if clickData is not None:
        _, data = getObjsPerImg([objCat])
        num_objs = clickData['points'][0]['x']
        imgIDs = data.loc[data['number of objs'] == num_objs]['imgID']
        print(imgIDs)
        htmlImgs = []
        for imgID in list(imgIDs):
            image_filename = 'data/val2017/' + str(imgID).zfill(12) + '.jpg'  # replace with your own image
            encoded_image = base64.b64encode(open(image_filename, 'rb').read())
            htmlImgs.append(html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode())))
        return html.Div(htmlImgs)


@app.callback(Output('area_imgs', 'children'),
              [Input('area_hist', 'clickData')])
def displayAreaImgs(clickData):
    # TODO: debug, some imgs are not displayed
    if clickData is not None:
        _, data = getArea([areaCat])
        area = clickData['points'][0]['x']
        imgIDs = data.loc[data['proportion of img'] == area]['imgID']
        print(imgIDs)
        htmlImgs = []
        for imgID in list(imgIDs):
            image_filename = 'data/val2017/' + str(imgID).zfill(12) + '.jpg'  # replace with your own image
            encoded_image = base64.b64encode(open(image_filename, 'rb').read())
            htmlImgs.append(html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode())))
        return html.Div(htmlImgs)


@app.callback(Output('tabs-figures', 'children'),
              [Input('tabs', 'value')])
def renderTab(tab):
    try:
        if tab == 'tab-1':
            fig = px.bar(dataframe, x='category', y='number of objects', title='Number of Objects per Category')
            fig2 = px.pie(dataframe, values='number of objects', names='category')
            return html.Div([
                dcc.Graph(id='cat_objs_bar', figure=fig),
                dcc.Graph(id='cat_objs_pie', figure=fig2)
            ])

        elif tab == 'tab-2':
            fig = px.bar(dataframe, x='category', y='number of images', title="Number of Images per Category")
            fig2 = px.pie(dataframe, values='number of images', names='category')
            return html.Div([
                dcc.Graph(id='cat_imgs_bar', figure=fig),
                dcc.Graph(id='cat_imgs_pie', figure=fig2)
            ])

        elif tab == 'tab-3':
            fig = px.bar(dataframe, x='category', y='avg number of objects per img',
                         title='Avg Number Of Objects per Image')
            fig.update_layout(clickmode='event+select')
            return html.Div([
                dcc.Graph(id='objs_per_img', figure=fig),
                html.Div(id='obj_hist_out'),
                html.Div(id='obj_imgs')
            ])

        elif tab == 'tab-4':
            fig = px.bar(dataframe, x="category", y='avg percentage of img', title='Avg Proportion of Image')
            return html.Div([
                dcc.Graph(id='cat_areas', figure=fig),
                html.Div(id='area_hist_out'),
                html.Div(id='area_imgs')
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
    ]),
    html.Div(id='tabs-figures')
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
