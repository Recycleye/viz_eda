import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html 
import pandas as pd
import plotly.express as px
import json

def stats_contents(analysis_path):
    """
    Generates the contents of the stats menu

    :param analysis_path: the path to the analysis file
    :return: stats contents
    """
    f = open(analysis_path,'r')
    analysis = json.load(f)

    ###########################################################################
    # Objects distribution
    classes = analysis["classes"]

    class_names = []
    num_objects = []
    num_images = []
    for cl in classes:
        class_names.append(classes[cl]["name"])
        num_objects.append(classes[cl]["num_objects"])
        num_images.append(len(classes[cl]["images"]))

    class_df = pd.DataFrame()
    class_df["class"] = class_names
    class_df["num objects"] = num_objects
    class_df["num images"] = num_images

    graph_title = "Objects distribution"
    graph_bar = px.bar(class_df,
        x="class",
        y="num objects",
        height=600
    )
    graph_bar.update_layout(xaxis_tickangle=-45, 
    margin_t = 50, font_size = 10, font_color = "black")

    graph = dcc.Graph(figure=graph_bar)

    graph_card = dbc.Card([
        dbc.CardBody([
                html.H5(graph_title,className="card-title"),
                html.Div(graph)], 
            className="card-body")],
        className="card flex-fill"    
    )

    ###########################################################################
    # Images distribution

    graph2_title = "Images distribution"
    graph2_bar = px.bar(class_df,
        x="class",
        y="num images",
        height=600
    )
    graph2_bar.update_layout(xaxis_tickangle=-45, 
    margin_t = 50, font_size = 10, font_color = "black")

    graph2 = dcc.Graph(figure=graph2_bar)

    graph2_card = dbc.Card([
        dbc.CardBody([
                html.H5(graph2_title,className="card-title"),
                html.Div(graph2)], 
            className="card-body")],
        className="card flex-fill"    
    )

    contents = html.Div([
        html.H3("Stats",style={"font-weight":"500"}),
        graph_card,
        graph2_card
    ])
    
    return contents