"""layouts.py: the overall layouts"""
from app.mainMenu import *

# App elements
navbar = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.A(html.I(className="hamburger align-self-center"),
                           className="sidebar-toggle d-flex", id="toggle")
                )
            ],
            align="center",
            no_gutters=True,
            style={
                "width": "100%"
            }
        )
    ],
    className="navbar navbar-expand navbar-light navbar-bg"
)

sidebar = html.Div([
    html.A(html.Span("vizEDA", className="align-middle"),
           className="sidebar-brand", href="http://0.0.0.0"),
    html.Div([
        html.Div([
            dbc.Button([
                html.Img(src="../assets/icons/info.svg",
                         style={"width": "15px", "padding-bottom": "1px",
                                "margin-right": "5px"}),
                "About"
            ],
                color="primary",
                className="mr-1",
                id="sidebar-btn-1",
                n_clicks_timestamp='0',
                style={"margin-left": "5%", "margin-top": "0",
                       "margin-bottom": "0",
                       "background": "transparent", "border-color": "transparent"})],
            id="sidebar-btn-container-1"
        ),
        html.Div([
            dbc.Button([
                html.Img(src="../assets/icons/plus.svg",
                         style={"width": "15px", "padding-bottom": "1px",
                                "margin-right": "5px"}),
                "New analysis"
            ],
                color="primary",
                className="mr-1",
                id="sidebar-btn-2",
                n_clicks_timestamp='0',
                style={"margin-left": "5%", "margin-top": "0",
                       "margin-bottom": "0", "background": "transparent",
                       "border-color": "transparent"})],
            id="sidebar-btn-container-2"
        ),
        html.Div([
            dbc.Button([
                html.Img(src="../assets/icons/upload.svg",
                         style={"width": "15px", "padding-bottom": "1px",
                                "margin-right": "5px"}),
                "Upload analysis"
            ],
                color="primary",
                className="mr-1",
                id="sidebar-btn-3",
                n_clicks_timestamp='0',
                style={"margin-left": "5%", "margin-top": "0",
                       "margin-bottom": "0", "background": "transparent",
                       "border-color": "transparent"})],
            id="sidebar-btn-container-3"),
    ],
        style={"margin-top": "10%"}
    ),
    html.Div([
        html.Div([
            dbc.Button([
                html.Img(src="../assets/icons/sliders.svg",
                         style={"width": "15px", "padding-bottom": "1px",
                                "margin-right": "5px"}),
                "Dashboard"
            ],
                color="primary",
                className="mr-1",
                id="sidebar-btn-4",
                n_clicks_timestamp='0',
                disabled=True,
                style={"margin-left": "5%", "margin-top": "0",
                       "margin-bottom": "0", "background": "transparent",
                       "border-color": "transparent"})],
            id="sidebar-btn-container-4"),
        html.Div([
            dbc.Button([
                html.Img(src="../assets/icons/alert-triangle.svg",
                         style={"width": "15px", "padding-bottom": "1px",
                                "margin-right": "5px"}),
                "Warnings"
            ],
                color="primary",
                className="mr-1",
                id="sidebar-btn-5",
                n_clicks_timestamp='0',
                disabled=True,
                style={"margin-left": "5%", "margin-top": "0",
                       "margin-bottom": "0", "background": "transparent",
                       "border-color": "transparent"})],
            id="sidebar-btn-container-5"),
        html.Div([
            dbc.Button([
                html.Img(src="../assets/icons/layout.svg",
                         style={"width": "15px", "padding-bottom": "1px",
                                "margin-right": "5px"}),
                "Classes"
            ],
                color="primary",
                className="mr-1",
                id="sidebar-btn-6",
                n_clicks_timestamp='0',
                disabled=True,
                style={"margin-left": "5%", "margin-top": "0",
                       "margin-bottom": "0", "background": "transparent",
                       "border-color": "transparent"})],
            id="sidebar-btn-container-6"),
        html.Div([
            dbc.Button([
                html.Img(src="../assets/icons/bar-chart-2.svg",
                         style={"width": "15px", "padding-bottom": "1px",
                                "margin-right": "5px"}),
                "Stats"
            ],
                color="primary",
                className="mr-1",
                id="sidebar-btn-7",
                n_clicks_timestamp='0',
                disabled=True,
                style={"margin-left": "5%", "margin-top": "0",
                       "margin-bottom": "0", "background": "transparent",
                       "border-color": "transparent"})],
            id="sidebar-btn-container-7"),
        html.Div([
            dbc.Button([
                html.Img(src="../assets/icons/crosshair.svg",
                         style={"width": "15px", "padding-bottom": "1px",
                                "margin-right": "5px"}),
                "Anomalies"
            ],
                color="primary",
                className="mr-1",
                id="sidebar-btn-8",
                n_clicks_timestamp='0',
                disabled=True,
                style={"margin-left": "5%", "margin-top": "0",
                       "margin-bottom": "0", "background": "transparent",
                       "border-color": "transparent"})],
            id="sidebar-btn-container-8"),
    ],
        style={"margin-top": "10%"}
    )
],
    className="sidebar collapse",
    id="sidebar"
)

header = html.Div(
    [
        sidebar,
        html.Div([
            navbar,
            html.Div(id="main-content", style={"padding": "2.5rem 2.5rem 1rem"})
        ],
            className="main")
    ],
    className="wrapper"
)


def serve_layout():
    return html.Div(
        [
            header
        ]
    )
