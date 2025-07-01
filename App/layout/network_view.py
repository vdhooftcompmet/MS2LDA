import dash_bootstrap_components as dbc
from dash import dcc, html



def create_cytoscape_network_tab():
    return html.Div(
        id="results-tab-content",
        children=[
            # Brief high-level overview of the entire tab
            html.Div(
                [
                    dcc.Markdown(
                        """
                        This tab shows an interactive network of optimized motifs.
                        Each motif is displayed as a node, and its fragments or losses
                        appear as connected nodes (color-coded for clarity).
                        Only edges above the selected intensity threshold will be shown.
                        You can adjust that threshold with the slider.
                        By default, the extra edges from each loss node to its
                        corresponding fragment node are hidden for less clutter,
                        but you can re-enable them using the checkbox.
                        Click on any motif node to see its associated molecules on the right side.
                        """,
                    ),
                ],
                style={"marginTop": "20px", "marginBottom": "20px"},
            ),
            # ----------------------------------------------------------------
            # 1. NETWORK CONTROLS
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Network Controls", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px", "display": "inline-block"}),
                            dbc.Button(
                                "🔍 Hide",
                                id="network-controls-toggle-button",
                                color="primary",
                                size="sm",
                                className="ms-2",
                                style={"display": "inline-block", "marginLeft": "10px", "marginBottom": "5px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "width": "100%"},
                    ),
                    dbc.Collapse(
                        id="network-controls-collapse",
                        is_open=True,
                        children=[
                            html.Div(
                                [
                                    dcc.Markdown(
                                        """
                                        Control how the network is displayed using the options below. You can adjust the edge intensity threshold,
                                        toggle additional edges between loss and fragment nodes, and change the graph layout algorithm.
                                        """
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Edge Intensity Threshold", style={"fontWeight": "bold"}),
                                                    dcc.Slider(
                                                        id="edge-intensity-threshold",
                                                        min=0,
                                                        max=1,
                                                        step=0.05,
                                                        value=0.50,
                                                        marks={0: "0.0", 0.5: "0.5", 1: "1.0"},
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Edge Options", style={"fontWeight": "bold"}),
                                                    dbc.Checklist(
                                                        options=[
                                                            {
                                                                "label": "Add Loss -> Fragment Edge",
                                                                "value": "show_loss_edge",
                                                            },
                                                        ],
                                                        value=[],
                                                        id="toggle-loss-edge",
                                                        inline=True,
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ],
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Graph Layout", style={"fontWeight": "bold"}),
                                                    dcc.Dropdown(
                                                        id="cytoscape-layout-dropdown",
                                                        options=[
                                                            {"label": "CoSE", "value": "cose"},
                                                            {
                                                                "label": "Force-Directed (Spring)",
                                                                "value": "fcose",
                                                            },
                                                            {"label": "Circle", "value": "circle"},
                                                            {"label": "Concentric", "value": "concentric"},
                                                        ],
                                                        value="fcose",
                                                        clearable=False,
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ],
                                        style={"marginTop": "10px"},
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
            # 2. NETWORK VISUALIZATION
            # ----------------------------------------------------------------
            html.Div(
                [
                    html.H4("Network Visualization", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2c3e50", "marginBottom": "10px"}),
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                The network visualization shows motifs and their fragments/losses as an interactive graph.
                                Motif nodes are shown in blue, fragment nodes in green, and loss nodes in red.
                                Click on any motif node to see its associated molecules in the panel on the right.
                                """
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                id="cytoscape-network-container",
                                                style={
                                                    "height": "600px",
                                                },
                                            ),
                                        ],
                                        width=8,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H5("Associated Molecules", style={"fontSize": "18px", "fontWeight": "bold", "color": "#34495e", "marginBottom": "8px"}),
                                            html.Div(
                                                id="molecule-images",
                                                style={
                                                    "textAlign": "center",
                                                    "overflowY": "auto",
                                                    "height": "600px",
                                                    "padding": "10px",
                                                    "backgroundColor": "#f8f9fa",
                                                    "borderRadius": "5px",
                                                },
                                            ),
                                        ],
                                        width=4,
                                    ),
                                ],
                                align="start",
                                className="g-3",
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


