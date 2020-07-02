import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import base64
from preprocessing import analyze_cats

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

global cocoData


def parse_contents(contents):
    content_type, content_string = contents.split(',')
    print(content_type)
    # print(content_string)
    decoded = base64.b64decode(content_string).decode('UTF-8')
    with open('output.json', 'w') as file:
        file.write(decoded)
    try:
        dataframe = analyze_cats('output.json')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return make_figures(dataframe)


def make_figures(dataframe):
    figProportion = px.bar(dataframe, x='category', y='size', title='Number of Objects per Category')
    figAreas = px.bar(dataframe, x="category", y='avg percentage of img', title='Avg Proportion of Image')
    return html.Div([
        dcc.Graph(
            id='cat_proportion',
            figure=figProportion
        ),
        dcc.Graph(
            id='cat_areas',
            figure=figAreas
        )
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is not None:
        print(contents)
        children = parse_contents(contents)
        return children

@app.callback(Output('dd-output-container', 'children'),
              [Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


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

])

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
