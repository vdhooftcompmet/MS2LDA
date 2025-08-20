import dash_bootstrap_components as dbc
from dash import dcc, html



def create_load_results_tab():
    return html.Div(
        id="load-results-tab-content",
        children=[
            # Brief high-level overview of the entire tab
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab allows you to load previously generated MS2LDA results (a compressed JSON file).
                        Once loaded, you can explore them immediately in the subsequent tabs.
                        This is useful if you’ve run an analysis before and want to revisit or share the results.
                        """,
                    ),
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
            ),
            # ----------------------------------------------------------------
            # 1. FILE UPLOAD SECTION
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Load MS2LDA Results File", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                Select a previously generated MS2LDA results file (compressed JSON format).
                                This file contains all the motifs, spectra, and analysis results from a previous run.
                                **After selecting the file, click the "Load Results" button to upload and process the file.**
                                """
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Upload(
                                                id="upload-results",
                                                children=html.Div(
                                                    ["Drag and Drop or ", html.A("Select Results File")],
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "60px",
                                                    "lineHeight": "60px",
                                                    "borderWidth": "1px",
                                                    "borderStyle": "dashed",
                                                    "borderRadius": "5px",
                                                    "textAlign": "center",
                                                    "margin": "10px",
                                                },
                                                multiple=False,
                                            ),
                                            html.Div(id="selected-file-info", style={"marginTop": "10px", "textAlign": "center"}),
                                            dbc.Spinner(
                                                html.Div(id="load-status", style={"marginTop": "20px"}),
                                                id="upload-spinner",
                                                color="primary",
                                                type="border",
                                                fullscreen=False,
                                                spinner_style={"width": "3rem", "height": "3rem"},
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "📂 Load Results",
                                                        id="load-results-button",
                                                        color="primary",
                                                    ),
                                                ],
                                                className="d-grid gap-2 mt-3",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                justify="center",
                            ),
                        ],
                        style={
                            "border": "1px dashed #ccc",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "marginBottom": "15px",
                        },
                    ),
                ],
                style={
                    "border": "1px dashed #999",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                },
            ),

            # ----------------------------------------------------------------
            # 2. DEMO DATA SECTION
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Load Demo Data", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                Click one of the buttons below to load pre-processed demo datasets used in the paper.
                                These datasets are ready to explore and can help you understand how MS2LDA works.
                                """
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "🍄 Load Mushroom Demo",
                                                id="load-mushroom-demo-button",
                                                color="success",
                                                className="me-2 mb-2",
                                            ),
                                            dbc.Button(
                                                "🌱 Load Pesticides Demo",
                                                id="load-pesticides-demo-button",
                                                color="success",
                                                className="me-2 mb-2",
                                            ),
                                            dbc.Button(
                                                "🎓 Load Summer School Example",
                                                id="load-summer-school-demo-button",
                                                color="success",
                                                className="me-2 mb-2",
                                            ),
                                            dbc.Spinner(
                                                html.Div(id="demo-load-status", style={"marginTop": "10px"}),
                                                id="demo-spinner",
                                                color="success",
                                                type="border",
                                                fullscreen=False,
                                                spinner_style={"width": "3rem", "height": "3rem"},
                                            ),
                                        ],
                                        width=6,
                                        style={"textAlign": "center"},
                                    ),
                                ],
                                justify="center",
                            ),
                        ],
                        style={
                            "border": "1px dashed #ccc",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "marginBottom": "15px",
                        },
                    ),
                ],
                style={
                    "border": "1px dashed #999",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                },
            ),
        ],
        style={"display": "none"},
    )


