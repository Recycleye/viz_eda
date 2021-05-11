import json

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def stats_contents(analysis_path):
    """
    Generates the contents of the stats menu

    :param analysis_path: the path to the analysis file
    :return: stats contents
    """
    f = open(analysis_path, 'r')
    analysis = json.load(f)

    # Popover user guide
    def make_popover(popid, feature, content):
        return dbc.Popover(
            [
                dbc.PopoverHeader(f"{feature}"),
                dbc.PopoverBody(f"{content}"),
            ],
            id=f"popover-{popid}",
            is_open=False,
            target=f"popover-{popid}-target",
        )

    def make_button(popid):
        return dbc.Button(
            "?", id=f"popover-{popid}-target", outline=True, color="info", size="sm"
        )

    classes = analysis["classes"]

    class_names = []
    num_objects = []
    num_images = []
    avg_size = []
    avg_size_std = []
    size = []
    clses = []
    for cl in classes:
        class_names.append(classes[cl]["name"])
        num_objects.append(classes[cl]["num_objects"])
        num_images.append(len(classes[cl]["images"]))
        avg_size.append(classes[cl]["size_avg"]["avg"])
        avg_size_std.append(classes[cl]["size_avg"]["std"])
        num = len(classes[cl]["size_avg"]["area"])
        clses.extend([classes[cl]["name"] for i in range(num)])
        size.extend(classes[cl]["size_avg"]["area"])

    class_df = pd.DataFrame()
    class_df["class"] = class_names
    class_df["num objects"] = num_objects
    class_df["num images"] = num_images
    class_df["avg size"] = [round(n / 1000) * 1000 for n in avg_size]
    class_df["std"] = [round(n / 1000) * 1000 for n in avg_size_std]

    size_df = pd.DataFrame()
    size_df["class"] = clses
    size_df["size"] = [round(n / 1000) * 1000 for n in size]

    ###########################################################################
    # Objects distribution - Bar Chart
    popobjbar = html.Div(
        [
            make_button("objbar"),
            make_popover("objbar", "Objects distribution", \
                         "Number of objects in each category"),
        ]
    )

    graph_title = "Objects Distribution"
    obj_dist = go.Figure()
    obj_dist.add_trace(
        go.Bar(x=class_df["class"], y=class_df["num objects"], base=0, marker_color='#1D3557', name='num objects'))
    obj_dist.update_layout(xaxis_tickangle=-45,
                           margin_t=50, font_size=10, font_color="black", height=300)
    graph = dcc.Graph(figure=obj_dist)
    graph_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5(graph_title, className="card-title")),
                dbc.Col(popobjbar),
            ],
                justify="between",
            ),
            html.Div(graph)],
            className="card-body")],
        className="card flex-fill"
    )

    ##########################################################################
    # Images distribution
    popimgbar = html.Div(
        [
            make_button("imgbar"),
            make_popover("imgbar", "Images Distribution", \
                         "Number of images containing each category"),
        ]
    )

    graph2_title = "Images Distribution"
    graph2_bar = go.Figure()
    graph2_bar.add_trace(
        go.Bar(x=class_df["class"], y=class_df["num images"], base=0, marker_color='#457B9D', name='num objects'))

    graph2_bar.update_layout(xaxis_tickangle=-45,
                             margin_t=50, font_size=10, font_color="black", height=300)

    graph2 = dcc.Graph(figure=graph2_bar)

    graph2_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5(graph2_title, className="card-title")),
                dbc.Col(popimgbar),
            ],
                justify="between",
            ),
            html.Div(graph2)],
            className="card-body")],
        className="card flex-fill"
    )

    #############################################################################################
    #   # PIE CHART

    pie_obj_dist = go.Figure(data=[go.Pie(labels=class_df['class'], values=class_df["num objects"])])
    pie_obj_dist.update_traces(textposition='inside')
    pie_obj_dist.update_layout(xaxis_tickangle=-45,
                               margin_t=50, font_size=10, font_color="black", height=300)

    popobjpie = html.Div(
        [
            make_button("objpie"),
            make_popover("objpie", "Images Distribution", \
                         "Proportion of objects for each category"),
        ]
    )
    graph1 = dcc.Graph(figure=pie_obj_dist)

    graph1_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5("Objects Distribution", className="card-title")),
                dbc.Col(popobjpie),
            ],
                justify="between",
            ),
            html.Div(graph1)],
            className="card-body")],
        className="card flex-fill"
    )

    pie_img_dist = go.Figure(data=[go.Pie(labels=class_df['class'], values=class_df["num images"])])
    pie_img_dist.update_traces(textposition='inside')
    pie_img_dist.update_layout(xaxis_tickangle=-45,
                               margin_t=50, font_size=10, font_color="black", height=300)

    popimgpie = html.Div(
        [
            make_button("imgpie"),
            make_popover("imgpie", "Images Distribution", \
                         "Proportion of images for each category"),
        ]
    )

    graph12 = dcc.Graph(figure=pie_img_dist)

    graph12_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5("Images Distribution", className="card-title")),
                dbc.Col(popimgpie),
            ],
                justify="between",
            ),
            html.Div(graph12)],
            className="card-body")],
    className = "card flex-fill"
    )


#############################################################################
    # Bubble Chart
    popbubchart = html.Div(
        [
            make_button("bubchart"),
            make_popover("bubchart", "Number and Size of Objects", \
                         "Bubble Color indicates average size of the objects; \
                         Height indicates numer of objects in each category. "),
        ]
    )

    graph3_title = "Number and Size of Objects (Unit: thousand pixels)"

    graph3_bubble = go.Figure(data=[go.Scatter(
        y=class_df['num objects'],
        x=class_df['class'],
        mode='markers',
        marker=dict(
            color=class_df['std'] / 1000,
            size=class_df['avg size'] / 1000,
            showscale=True
        )
    )])

    graph3_bubble.update_layout(yaxis_title="Number of Objects",height=400)
    graph3 = dcc.Graph(figure=graph3_bubble)

    graph3_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5(graph3_title, className="card-title")),
                dbc.Col(popbubchart),
            ],
                justify="between",
            ),
            html.Div(graph3)],
            className="card-body")],
        className="card flex-fill"
    )

    ###############################################################################
    # average size/area
    popobjavesize = html.Div(
        [
            make_button("objavesize"),
            make_popover("objavesize", "Average Size of Objects", \
                         "Average size of the objects (area of bounding box) for each category"),
        ]
    )

    graph4_title = "Average Size of Objects"
    graph4_bar = px.bar(class_df,
                        x="class",
                        y="avg size",
                        height=300
                        )

    graph4_bar.update_layout(xaxis_tickangle=-45,
                             margin_t=50, font_size=10, font_color="black",height=300)

    graph4 = dcc.Graph(figure=graph4_bar)

    graph4_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5(graph4_title, className="card-title")),
                dbc.Col(popobjavesize),
            ],
                justify="between",
            ),
            html.Div(graph4)],
            className="card-body")],
        className="card flex-fill"
    )
    ###############################################################################
    # Dimension Box and Whisker
    popobjsize = html.Div(
        [
            make_button("objsize"),
            make_popover("objsize", "Objects Size Distribution", \
                         "Distribution of object size (area of bounding box) for each category. (The middle bar indicates the median while the top and bottom bar represent the max and min respectively)"),
        ]
    )

    graph5_title = "Objects Size Distribution"
    graph5_box = px.box(size_df,
                        x="class",
                        y="size",
                        height=600
                        )
    graph5_box.update_traces(marker_color='darkorchid')
    graph5_box.update_layout(xaxis_tickangle=-45,
                             margin_t=50, font_size=10, font_color="black",height=300)

    graph5 = dcc.Graph(figure=graph5_box)

    graph5_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5(graph5_title, className="card-title")),
                dbc.Col(popobjsize),
            ],
                justify="between",
            ),
            html.Div(graph5)],
            className="card-body")],
        className="card flex-fill"
    )

    obj_size_row = dbc.Row([
        dbc.Col(graph4_card,width=5),
        dbc.Col(graph5_card,width=7)
    ])

    obj_size_bubble_row = dbc.Row([
        dbc.Col(graph3_card, width=12),
    ])

    img_dist_row = dbc.Row([
        dbc.Col(graph12_card, width=5),
        dbc.Col(graph2_card, width=7)
    ])

    obj_dist_row = dbc.Row([
        dbc.Col(graph1_card, width=5),
        dbc.Col(graph_card, width=7)
    ])

    contents = html.Div([
        html.H3("Stats", style={"font-weight": "500"}),
        obj_size_bubble_row,
        obj_size_row,
        img_dist_row,
        obj_dist_row
    ])

    return contents
