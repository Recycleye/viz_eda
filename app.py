import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from pycocotools.coco import COCO

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

coco = COCO("data/annotations/instances_val2017.json")


# def load_data(annFile):





def get_cat_size(filterClasses):
    catIds = coco.getCatIds(catNms=filterClasses)
    imgIds = coco.getImgIds(catIds=catIds)
    print("Number of images containing the class:", len(imgIds))
    return len(imgIds)


def analyze_cat_proportion():
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))
    supernms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    data = {}
    for cat in supernms:
        catSize = get_cat_size([cat])
        data[len(data)] = [cat, catSize]
    df = pd.DataFrame.from_dict(data, orient='index', columns=['category', 'size'])
    print(df)
    return df

figProportion = px.pie(analyze_cat_proportion(), values='size', names='category', title='Proportion of Categories')

app.layout = html.Div(children=[
    html.H1(children='Viz EDA'),

    html.Div(children='''
        Exploratory data analysis for computer vision and object recognition.
    '''),

    dcc.Graph(
        id='cat_proportion',
        figure=figProportion
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)