import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import base64
from analysis import analyzeDataset

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

global cocoData, dataframe


def parseContents(contents):
    content_type, content_string = contents.split(',')
    print(content_type)
    # print(content_string)
    decoded = base64.b64decode(content_string).decode('UTF-8')
    with open('output.json', 'w') as file:
        file.write(decoded)
    # try:
    global dataframe
    dataframe = analyzeDataset('output.json')
    # except Exception as e:
    #     print(e)
    #     return html.Div([
    #         'There was an error processing this file.'
    #     ])
    # return makeFigures(dataframe)
    return


# def makeFigures(dataframe):
#     figProportion = px.bar(dataframe, x='category', y='size', title='Number of Objects per Category')
#     figAreas = px.bar(dataframe, x="category", y='avg percentage of img', title='Avg Proportion of Image')
#     return html.Div([
#         dcc.Graph(
#             id='cat_proportion',
#             figure=figProportion
#         ),
#         dcc.Graph(
#             id='cat_areas',
#             figure=figAreas
#         )
#     ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])
def uploadData(contents):
    if contents is not None:
        print(contents)
        children = parseContents(contents)
        return children


# @app.callback(Output('dd-output-container', 'children'),
#               [Input('demo-dropdown', 'value')])
# def update_output(value):
#     return 'You have selected "{}"'.format(value)


@app.callback(Output('tabs-figures', 'children'),
              [Input('tabs', 'value')])
def renderTab(tab):
    if tab == 'tab-1':
        fig = px.bar(dataframe, x='category', y='number of objects', title='Number of Objects per Category')
        fig2 = px.pie(dataframe, values='number of objects', names='category')
        return html.Div([
            dcc.Graph(
                id='cat_objs_bar',
                figure=fig
            ),
            dcc.Graph(
                id='cat_objs_pie',
                figure=fig2
            )
        ])
    elif tab == 'tab-2':
        fig = px.bar(dataframe, x='category', y='number of images', title="Number of Images per Category")
        fig2 = px.pie(dataframe, values='number of images', names='category')
        return html.Div([
            dcc.Graph(
                id='cat_imgs_bar',
                figure=fig
            ),
            dcc.Graph(
                id='cat_imgs_pie',
                figure=fig2
            )
        ])
    elif tab == 'tab-3':
        fig = px.bar(dataframe, x='category', y='avg number of objects per img', title='Avg Number Of Objects per Image')
        return html.Div([
            dcc.Graph(
                id='objs_per_img',
                figure=fig
            )
        ])
    elif tab == 'tab-4':
        fig = px.bar(dataframe, x="category", y='avg percentage of img', title='Avg Proportion of Image')
        return html.Div([
            dcc.Graph(
                id='cat_areas',
                figure=fig
            )
        ])


app.layout = html.Div(children=[
    html.H1(children='Viz EDA'),

    html.Div(children='''
        Exploratory data analysis for computer vision and object recognition.
    '''),
    html.Hr(),
    dcc.Upload(id='upload-data', children=html.Button('Upload File'), multiple=False),
    html.Hr(),

    # dcc.Dropdown(
    #     id='demo-dropdown',
    #     options=[
    #         {'label': 'COCO Captions', 'value': 'CocoCaptions'},
    #         {'label': 'Cityscapes', 'value': 'Cityscapes'},
    #         {'label': 'ImageNet', 'value': 'ImageNet'}
    #     ],
    #     value='CocoCaptions'
    # ),
    # html.Div(id='dd-output-container'),

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
