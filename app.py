import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import pycocotools.coco as coco
import pycocotools.mask as mask

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

cocoData = coco.COCO("data/annotations/instances_train2017.json")

def get_cat_size(filterClasses):
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)
    print("Number of images containing the class:", len(imgIds))
    return len(imgIds)


def get_avg_area(filterClasses):
    catIds = cocoData.getCatIds(catNms=filterClasses)
    imgIds = cocoData.getImgIds(catIds=catIds)

    proportionsOfImg = []
    for img in imgIds:
        imAnn = cocoData.loadImgs(ids=img)[0]
        width = imAnn['width']
        height = imAnn['height']

        annIds = cocoData.getAnnIds(imgIds=img, catIds=catIds)
        objs = cocoData.loadAnns(ids=annIds)

        validObjs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                validObjs.append(obj)
        objs = validObjs
        numObjs = len(objs)

        segAreas = np.zeros((numObjs), dtype=np.float32)
        for ix, obj in enumerate(objs):
            cls = obj['category_id']
            segAreas[ix] = obj['area']
        proportionOfImg = (sum(segAreas) / len(segAreas)) / (width * height)
        proportionsOfImg.append(proportionOfImg)

    return sum(proportionsOfImg) / len(proportionsOfImg)


def analyze_cats():
    # display COCO categories and supercategories
    cats = cocoData.loadCats(cocoData.getCatIds())
    nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))
    supernms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    data = {}
    for cat in nms:
        catSize = get_cat_size([cat])
        avgArea = get_avg_area([cat])
        data[len(data)] = [cat, catSize, avgArea]
    df = pd.DataFrame.from_dict(data, orient='index', columns=['category', 'size', 'avg percentage of img'])
    print(df)
    return df

dataframe = analyze_cats()
figProportion = px.pie(dataframe, values='size', names='category', title='Proportion of Categories')
figAreas = px.bar(dataframe, x="category", y='avg percentage of img')

app.layout = html.Div(children=[
    html.H1(children='Viz EDA'),

    html.Div(children='''
        Exploratory data analysis for computer vision and object recognition.
    '''),

    dcc.Graph(
        id='cat_proportion',
        figure=figProportion
    ),
    dcc.Graph(
        id='cat_areas',
        figure=figAreas
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)