import base64
import json

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import numpy as np
import pandas as pd
from dash import dash_table
from dash import html, dcc, Input, Output, State
from matchms import Spectrum, Fragments
from rdkit.Chem import MolFromSmiles

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "MS2LDA Interactive Dashboard"

# Include Cytoscape extra layouts
cyto.load_extra_layouts()

# Define the layout
app.layout = dbc.Container(
    [
        html.H1(
            "MS2LDA Interactive Dashboard",
            style={"textAlign": "center", "marginTop": 20},
        ),
        dcc.Tabs(
            id="tabs",
            value="run-analysis-tab",
            children=[
                dcc.Tab(label="Run Analysis", value="run-analysis-tab", id="run-analysis-tab"),
                dcc.Tab(label="Load Results", value="load-results-tab", id="load-results-tab"),
                dcc.Tab(label="View Network", value="results-tab", id="results-tab"),
                dcc.Tab(label="Motif Rankings", value="motif-rankings-tab", id="motif-rankings-tab"),
                dcc.Tab(label="Motif Details", value="motif-details-tab", id="motif-details-tab"),
            ],
            className="mt-3",
        ),
        html.Div(
            id="run-analysis-tab-content",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select Files")]
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
                                html.Div(
                                    id="file-upload-info", style={"marginBottom": "20px"}
                                ),
                                # Basic parameters
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Number of Motifs"),
                                        dbc.Input(
                                            id="n-motifs",
                                            type="number",
                                            value=50,
                                            min=1,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Acquisition Type"),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "DDA", "value": "DDA"},
                                                {"label": "DIA", "value": "DIA"},
                                            ],
                                            value="DDA",  # default
                                            id="acquisition-type",
                                            inline=True,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Top N Matches"),
                                        dbc.Input(
                                            id="top-n", type="number", value=5, min=1
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Unique Molecules"),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "Yes", "value": True},
                                                {"label": "No", "value": False},
                                            ],
                                            value=True,
                                            id="unique-mols",
                                            inline=True,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Polarity"),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "Positive", "value": "positive"},
                                                {"label": "Negative", "value": "negative"},
                                            ],
                                            value="positive",
                                            id="polarity",
                                            inline=True,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Button(
                                    "Show/Hide Advanced Settings",
                                    id="advanced-settings-button",
                                    color="info",
                                    className="mb-3",
                                ),
                                dbc.Collapse(
                                    id="advanced-settings-collapse",
                                    is_open=False,
                                    children=[
                                        html.H5("Advanced Parameters", className="mt-3"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        # Preprocessing
                                                        html.H6("Preprocessing"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_mz"),
                                                                dbc.Input(
                                                                    id="prep-min-mz",
                                                                    type="number",
                                                                    value=0,  # default
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("max_mz"),
                                                                dbc.Input(
                                                                    id="prep-max-mz",
                                                                    type="number",
                                                                    value=2000,  # default
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("max_frags"),
                                                                dbc.Input(
                                                                    id="prep-max-frags",
                                                                    type="number",
                                                                    value=1000,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_frags"),
                                                                dbc.Input(
                                                                    id="prep-min-frags",
                                                                    type="number",
                                                                    value=5,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_intensity"),
                                                                dbc.Input(
                                                                    id="prep-min-intensity",
                                                                    type="number",
                                                                    value=0.01,
                                                                    step=0.001,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("max_intensity"),
                                                                dbc.Input(
                                                                    id="prep-max-intensity",
                                                                    type="number",
                                                                    value=1,
                                                                    step=0.1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        # Convergence
                                                        html.H6("Convergence"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("step_size"),
                                                                dbc.Input(
                                                                    id="conv-step-size",
                                                                    type="number",
                                                                    value=50,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("window_size"),
                                                                dbc.Input(
                                                                    id="conv-window-size",
                                                                    type="number",
                                                                    value=10,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("threshold"),
                                                                dbc.Input(
                                                                    id="conv-threshold",
                                                                    type="number",
                                                                    value=0.005,
                                                                    step=0.0001,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("type"),
                                                                dbc.Select(
                                                                    id="conv-type",
                                                                    options=[
                                                                        {"label": "perplexity_history", "value": "perplexity_history"},
                                                                        {"label": "entropy_history_doc", "value": "entropy_history_doc"},
                                                                        {"label": "entropy_history_topic", "value": "entropy_history_topic"},
                                                                        {"label": "log_likelihood_history", "value": "log_likelihood_history"},
                                                                    ],
                                                                    value="perplexity_history",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                        ),
                                        html.Hr(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6("Annotation"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("criterium"),
                                                                dbc.Select(
                                                                    id="ann-criterium",
                                                                    options=[
                                                                        {"label": "best", "value": "best"},
                                                                        {"label": "biggest", "value": "biggest"},
                                                                    ],
                                                                    value="best",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("cosine_similarity"),
                                                                dbc.Input(
                                                                    id="ann-cosine-sim",
                                                                    type="number",
                                                                    value=0.90,
                                                                    step=0.01,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Model"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("rm_top"),
                                                                dbc.Input(
                                                                    id="model-rm-top",
                                                                    type="number",
                                                                    value=0,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_cf"),
                                                                dbc.Input(
                                                                    id="model-min-cf",
                                                                    type="number",
                                                                    value=0,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_df"),
                                                                dbc.Input(
                                                                    id="model-min-df",
                                                                    type="number",
                                                                    value=3,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("alpha"),
                                                                dbc.Input(
                                                                    id="model-alpha",
                                                                    type="number",
                                                                    value=0.6,
                                                                    step=0.1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("eta"),
                                                                dbc.Input(
                                                                    id="model-eta",
                                                                    type="number",
                                                                    value=0.01,
                                                                    step=0.001,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("seed"),
                                                                dbc.Input(
                                                                    id="model-seed",
                                                                    type="number",
                                                                    value=42,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Train"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("parallel"),
                                                                dbc.Input(
                                                                    id="train-parallel",
                                                                    type="number",
                                                                    value=1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("workers"),
                                                                dbc.Input(
                                                                    id="train-workers",
                                                                    type="number",
                                                                    value=0,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("n_iterations"),
                                                                dbc.Input(
                                                                    id="n-iterations",
                                                                    type="number",
                                                                    value=100,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6("Dataset"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("charge"),
                                                                dbc.Input(
                                                                    id="dataset-charge",
                                                                    type="number",
                                                                    value=1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Run Name"),
                                                                dbc.Input(
                                                                    id="dataset-name",
                                                                    type="text",
                                                                    value="ms2lda_dashboard_run",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Output Folder"),
                                                                dbc.Input(
                                                                    id="dataset-output-folder",
                                                                    type="text",
                                                                    value="ms2lda_results",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Fingerprint"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("fp_type"),
                                                                dbc.Select(
                                                                    id="fp-type",
                                                                    options=[
                                                                        {"label": "rdkit", "value": "rdkit"},
                                                                        {"label": "maccs", "value": "maccs"},
                                                                        {"label": "pubchem", "value": "pubchem"},
                                                                        {"label": "ecfp", "value": "ecfp"},
                                                                        # add more if needed
                                                                    ],
                                                                    value="rdkit",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("fp threshold"),
                                                                dbc.Input(
                                                                    id="fp-threshold",
                                                                    type="number",
                                                                    value=0.8,
                                                                    step=0.1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Motif Parameter"),
                                                                dbc.Input(
                                                                    id="motif-parameter",
                                                                    type="number",
                                                                    value=50,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6("Spec2Vec Paths"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Model Path"),
                                                                dbc.Input(
                                                                    id="s2v-model-path",
                                                                    type="text",
                                                                    # put your real default if needed:
                                                                    value="/Users/joewandy/Work/git/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Library Path"),
                                                                dbc.Input(
                                                                    id="s2v-library-path",
                                                                    type="text",
                                                                    # put your real default if needed:
                                                                    value="/Users/joewandy/Work/git/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/positive_s2v_library.pkl",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                # End advanced settings

                                # The Run button
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Run Analysis",
                                            id="run-button",
                                            color="primary",
                                        ),
                                    ],
                                    className="d-grid gap-2",
                                ),
                                html.Div(id="run-status", style={"marginTop": "20px"}),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Save Results",
                                            id="save-results-button",
                                            color="secondary",
                                        ),
                                    ],
                                    className="d-grid gap-2 mt-3",
                                ),
                                html.Div(id="save-status", style={"marginTop": "20px"}),
                            ],
                            width=6,
                        )
                    ],
                    justify="center",
                )
            ],
            style={"display": "block"},
        ),
        html.Div(
            id="load-results-tab-content",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Upload(
                                    id="upload-results",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select Results File")]
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
                                html.Div(
                                    id="load-status", style={"marginTop": "20px"}
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Load Results",
                                            id="load-results-button",
                                            color="primary",
                                        ),
                                    ],
                                    className="d-grid gap-2 mt-3",
                                ),
                            ],
                            width=6,
                        )
                    ],
                    justify="center",
                )
            ],
            style={"display": "none"},
        ),
        html.Div(
            id="results-tab-content",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="cytoscape-network-container",
                                    style={
                                        "marginTop": "20px",
                                        "height": "600px",  # Network height
                                    },
                                )
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    id="molecule-images",
                                    style={
                                        "textAlign": "center",
                                        "marginTop": "20px",
                                        "overflowY": "auto",
                                        "height": "600px",  # Match network height
                                        "padding": "10px",
                                        "backgroundColor": "#f8f9fa",
                                        "borderRadius": "5px",
                                    },
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    align="start",  # Align items to the top
                    className="g-3",  # Gap between columns
                )
            ],
            style={"display": "none"},
        ),
        html.Div(
            id="motif-rankings-tab-content",
            children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Motif Rankings"),
                            # Row for the sliders and their displays
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Probability Threshold Range"),
                                            dcc.RangeSlider(
                                                id="probability-thresh",
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                value=[0.1, 1],  # Initial range: [0.1, 1]
                                                marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                                tooltip={"always_visible": False, "placement": "top"},
                                                allowCross=False
                                            ),
                                            html.Div(id='probability-thresh-display', style={"marginTop": "10px"})
                                        ],
                                        width=6
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Overlap Threshold Range"),
                                            dcc.RangeSlider(
                                                id="overlap-thresh",
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                value=[0.3, 1],  # Initial range: [0.3, 1]
                                                marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                                tooltip={"always_visible": False, "placement": "top"},
                                                allowCross=False
                                            ),
                                            html.Div(id='overlap-thresh-display', style={"marginTop": "10px"})
                                        ],
                                        width=6
                                    )
                                ]
                            ),
                            html.Div(id="motif-rankings-table-container", style={"marginTop": "20px"})  # Container for the table
                        ], width=12),
                    ]),
                ]),
            ],
            style={"display": "none"}
        ),
        html.Div(
            id="motif-details-tab-content",
            children=[
                html.H3(id='motif-details-title'),
                # Add probability filter slider
                html.Div([
                    dbc.Label("Topic-Word Probability Filter:"),
                    dcc.RangeSlider(
                        id='probability-filter',
                        min=0,
                        max=1,
                        step=0.01,
                        value=[0, 1],  # Initial value: show all features
                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                        allowCross=False
                    ),
                    html.Div(id='probability-filter-display', style={"marginTop": "10px"})
                ], className="mb-3"),
                html.Div(id='motif-details-content'),
                dcc.Store(id='motif-spectra-ids-store'),
                dcc.Store(id='selected-spectrum-index', data=0),
                html.Div(id='spectrum-plot'),
                # Placeholder for spectra-table
                dash_table.DataTable(
                    id='spectra-table',
                    data=[],  # Empty data
                    columns=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    page_size=10,
                    row_selectable='single',
                    selected_rows=[0],
                ),
                # Next and Previous buttons
                html.Div([
                    dbc.Button('Previous', id='prev-spectrum', n_clicks=0, color="info"),
                    dbc.Button('Next', id='next-spectrum', n_clicks=0, className='ms-2', color="info"),
                ], className='mt-3'),
            ],
            style={"display": "none"},  # Initially hidden
        ),
        # Hidden storage for data to be accessed by callbacks
        dcc.Store(id="clustered-smiles-store"),
        dcc.Store(id="optimized-motifs-store"),
        dcc.Store(id="lda-dict-store"),
        dcc.Store(id='selected-motif-store'),
        dcc.Store(id='spectra-store'),
        # Include Download component
        dcc.Download(id="download-results"),
    ],
    fluid=False,  # Fixed-width container
)

# Callback to show/hide tab contents based on active tab
@app.callback(
    Output("run-analysis-tab-content", "style"),
    Output("load-results-tab-content", "style"),
    Output("results-tab-content", "style"),
    Output("motif-rankings-tab-content", "style"),
    Output("motif-details-tab-content", "style"),
    Input("tabs", "value"),
)
def toggle_tab_content(active_tab):
    run_style = {"display": "none"}
    load_style = {"display": "none"}
    results_style = {"display": "none"}
    motif_rankings_style = {"display": "none"}
    motif_details_style = {"display": "none"}

    if active_tab == "run-analysis-tab":
        run_style = {"display": "block"}
    elif active_tab == "load-results-tab":
        load_style = {"display": "block"}
    elif active_tab == "results-tab":
        results_style = {"display": "block"}
    elif active_tab == "motif-rankings-tab":
        motif_rankings_style = {"display": "block"}
    elif active_tab == "motif-details-tab":
        motif_details_style = {"display": "block"}

    return run_style, load_style, results_style, motif_rankings_style, motif_details_style

# Callback to display uploaded data file info
@app.callback(
    Output("file-upload-info", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(contents, filename):
    if contents:
        return html.Div([html.H5(f"Uploaded File: {filename}")])
    else:
        return html.Div([html.H5("No file uploaded yet.")])

# Callbacks to display current RangeSlider values in Motif Rankings Tab
@app.callback(
    Output('probability-thresh-display', 'children'),
    Input('probability-thresh', 'value')
)
def display_probability_thresh(prob_thresh_range):
    return f"Selected Probability Range: {prob_thresh_range[0]:.2f} - {prob_thresh_range[1]:.2f}"

@app.callback(
    Output('overlap-thresh-display', 'children'),
    Input('overlap-thresh', 'value')
)
def display_overlap_thresh(overlap_thresh_range):
    return f"Selected Overlap Range: {overlap_thresh_range[0]:.2f} - {overlap_thresh_range[1]:.2f}"

# Callbacks to display current RangeSlider value in Motif Details Tab
@app.callback(
    Output('probability-filter-display', 'children'),
    Input('probability-filter', 'value')
)
def display_prob_filter(prob_filter_range):
    return f"Showing features with probability between {prob_filter_range[0]:.2f} and {prob_filter_range[1]:.2f}"

# ADDED/CHANGED: callback to show/hide advanced settings
@app.callback(
    Output("advanced-settings-collapse", "is_open"),
    Input("advanced-settings-button", "n_clicks"),
    State("advanced-settings-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_advanced_settings(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback to run or load
@app.callback(
    Output("run-status", "children"),
    Output("load-status", "children"),
    Output("clustered-smiles-store", "data"),
    Output("optimized-motifs-store", "data"),
    Output("lda-dict-store", "data"),
    Output('spectra-store', 'data'),
    Input("run-button", "n_clicks"),
    Input("load-results-button", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("n-motifs", "value"),
    State("top-n", "value"),
    State("unique-mols", "value"),
    State("polarity", "value"),
    State("upload-results", "contents"),
    State("upload-results", "filename"),
    # ADDED/CHANGED: advanced settings states
    State("prep-min-mz", "value"),
    State("prep-max-mz", "value"),
    State("prep-max-frags", "value"),
    State("prep-min-frags", "value"),
    State("prep-min-intensity", "value"),
    State("prep-max-intensity", "value"),
    State("conv-step-size", "value"),
    State("conv-window-size", "value"),
    State("conv-threshold", "value"),
    State("conv-type", "value"),
    State("ann-criterium", "value"),
    State("ann-cosine-sim", "value"),
    State("model-rm-top", "value"),
    State("model-min-cf", "value"),
    State("model-min-df", "value"),
    State("model-alpha", "value"),
    State("model-eta", "value"),
    State("model-seed", "value"),
    State("train-parallel", "value"),
    State("train-workers", "value"),
    State("n-iterations", "value"),
    State("dataset-charge", "value"),
    State("dataset-name", "value"),
    State("dataset-output-folder", "value"),
    State("fp-type", "value"),
    State("fp-threshold", "value"),
    State("motif-parameter", "value"),
    State("s2v-model-path", "value"),
    State("s2v-library-path", "value"),
    prevent_initial_call=True,
)
def handle_run_or_load(
    run_clicks,
    load_clicks,
    data_contents,
    data_filename,
    n_motifs,
    top_n,
    unique_mols,
    polarity,
    results_contents,
    results_filename,
    # advanced settings
    prep_min_mz,
    prep_max_mz,
    prep_max_frags,
    prep_min_frags,
    prep_min_intensity,
    prep_max_intensity,
    conv_step_size,
    conv_window_size,
    conv_threshold,
    conv_type,
    ann_criterium,
    ann_cosine_sim,
    model_rm_top,
    model_min_cf,
    model_min_df,
    model_alpha,
    model_eta,
    model_seed,
    train_parallel,
    train_workers,
    n_iterations,
    dataset_charge,
    dataset_name,
    dataset_output_folder,
    fp_type,
    fp_threshold,
    motif_parameter,
    s2v_model_path,
    s2v_library_path,
):
    """
    This callback either (1) runs MS2LDA from scratch on the uploaded data (when Run Analysis clicked),
    or (2) loads precomputed results from a JSON file (when Load Results clicked).
    """
    import base64, tempfile, os, json
    import dash
    from dash.exceptions import PreventUpdate
    import tomotopy as tp
    from MS2LDA.Preprocessing.load_and_clean import clean_spectra
    from MS2LDA.run import filetype_check
    from MS2LDA.Visualisation.ldadict import generate_corpusjson_from_tomotopy
    from dash import no_update
    import MS2LDA  # Our main run code

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Default "no update" for all outputs:
    run_status = no_update
    load_status = no_update
    clustered_smiles_data = no_update
    optimized_motifs_data = no_update
    lda_dict_data = no_update
    spectra_data = no_update

    # 1) If RUN-BUTTON was clicked
    if triggered_id == "run-button":
        if not data_contents:
            run_status = dbc.Alert("Please upload a mass spec data file first!", color="danger")
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                spectra_data,
            )
        # Decode the uploaded file into temp
        try:
            content_type, content_string = data_contents.split(",")
            decoded = base64.b64decode(content_string)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(data_filename)[1]) as tmp_file:
                tmp_file.write(decoded)
                tmp_file_path = tmp_file.name
        except Exception as e:
            run_status = dbc.Alert(f"Error handling the uploaded file: {str(e)}", color="danger")
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                spectra_data,
            )

        # ADDED/CHANGED: dynamic parameter setup
        preprocessing_parameters = {
            "min_mz": prep_min_mz,
            "max_mz": prep_max_mz,
            "max_frags": prep_max_frags,
            "min_frags": prep_min_frags,
            "min_intensity": prep_min_intensity,
            "max_intensity": prep_max_intensity,
        }
        convergence_parameters = {
            "step_size": conv_step_size,
            "window_size": conv_window_size,
            "threshold": conv_threshold,
            "type": conv_type,
        }
        annotation_parameters = {
            "criterium": ann_criterium,
            "cosine_similarity": ann_cosine_sim,
            "n_mols_retrieved": top_n,
        }
        model_parameters = {
            "rm_top": model_rm_top,
            "min_cf": model_min_cf,
            "min_df": model_min_df,
            "alpha": model_alpha,
            "eta": model_eta,
            "seed": model_seed,
        }
        train_parameters = {
            "parallel": train_parallel,
            "workers": train_workers,
        }
        dataset_parameters = {
            "acquisition_type": "DDA" if polarity == "positive" else "DDA",  # could also be changed
            "charge": dataset_charge,
            "name": dataset_name,
            "output_folder": dataset_output_folder,
        }
        fingerprint_parameters = {
            "fp_type": fp_type,
            "threshold": fp_threshold,
        }

        # 1.1) Actually run the pipeline:
        motif_spectra, optimized_motifs, motif_fps = MS2LDA.run(
            dataset=tmp_file_path,
            n_motifs=n_motifs,
            n_iterations=n_iterations,
            dataset_parameters=dataset_parameters,
            train_parameters=train_parameters,
            model_parameters=model_parameters,
            convergence_parameters=convergence_parameters,
            annotation_parameters=annotation_parameters,
            motif_parameter=motif_parameter,
            preprocessing_parameters=preprocessing_parameters,
            fingerprint_parameters=fingerprint_parameters,
        )

        # 1.2) Now load the trained tomotopy model from disk to build LDA dictionary:
        trained_ms2lda = tp.LDAModel.load(os.path.join(dataset_parameters["output_folder"], "ms2lda.bin"))

        documents = []
        for doc in trained_ms2lda.docs:
            tokens = [trained_ms2lda.vocabs[word_id] for word_id in doc.words]
            documents.append(tokens)
        doc_metadata = {}
        for i, doc in enumerate(trained_ms2lda.docs):
            doc_name = f"spec_{i}"
            doc_metadata[doc_name] = {"placeholder": f"Doc {i}"}

        lda_dict = generate_corpusjson_from_tomotopy(
            model=trained_ms2lda,
            documents=documents,
            spectra=None,
            doc_metadata=doc_metadata,
            filename=None,
        )

        # 1.3) Convert the real cleaned spectra for the network:
        loaded_spectra = filetype_check(tmp_file_path)
        cleaned_spectra = clean_spectra(loaded_spectra, preprocessing_parameters=preprocessing_parameters)

        def spectrum_to_dict(s):
            metadata = s.metadata.copy()
            dct = {
                "metadata": metadata,
                "mz": [float(m) for m in s.peaks.mz],
                "intensities": [float(i) for i in s.peaks.intensities],
            }
            if s.losses:
                dct["metadata"]["losses"] = [
                    {"loss_mz": float(mz_), "loss_intensity": float(int_)}
                    for mz_, int_ in zip(s.losses.mz, s.losses.intensities)
                ]
            return dct

        optimized_motifs_data = [spectrum_to_dict(m) for m in optimized_motifs]
        spectra_data = [spectrum_to_dict(s) for s in cleaned_spectra]

        clustered_smiles_data = []
        for mot in optimized_motifs:
            ann = mot.get("short_annotation")
            if isinstance(ann, list):
                clustered_smiles_data.append(ann)
            elif ann is None:
                clustered_smiles_data.append([])
            else:
                clustered_smiles_data.append([ann])

        run_status = dbc.Alert("MS2LDA.run completed successfully!", color="success")
        return (
            run_status,                # run-status
            load_status,               # load-status
            clustered_smiles_data,     # clustered-smiles-store
            optimized_motifs_data,     # optimized-motifs-store
            lda_dict,                  # lda-dict-store
            spectra_data,              # spectra-store
        )

    # 2) If LOAD-RESULTS-BUTTON was clicked
    elif triggered_id == "load-results-button":
        if not results_contents:
            load_status = dbc.Alert("Please upload a results JSON file.", color="danger")
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                spectra_data,
            )
        try:
            content_type, content_string = results_contents.split(",")
            decoded = base64.b64decode(content_string)
            data = json.loads(decoded)
        except Exception as e:
            load_status = dbc.Alert(f"Error decoding or parsing the file: {str(e)}", color="danger")
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                spectra_data,
            )

        required_keys = {"clustered_smiles_data", "optimized_motifs_data", "lda_dict", "spectra_data"}
        if not required_keys.issubset(data.keys()):
            load_status = dbc.Alert("Invalid results file. Missing required data keys.", color="danger")
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                spectra_data,
            )

        try:
            clustered_smiles_data = data["clustered_smiles_data"]
            optimized_motifs_data = data["optimized_motifs_data"]
            lda_dict_data = data["lda_dict"]
            spectra_data = data["spectra_data"]
        except Exception as e:
            load_status = dbc.Alert(f"Error reading data from file: {str(e)}", color="danger")
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                spectra_data,
            )

        load_status = dbc.Alert("Results loaded successfully!", color="success")
        return (
            run_status,
            load_status,
            clustered_smiles_data,
            optimized_motifs_data,
            lda_dict_data,
            spectra_data,
        )

    else:
        raise dash.exceptions.PreventUpdate


# Callback to handle Save Results
@app.callback(
    Output("download-results", "data"),
    Output("save-status", "children"),
    Input("save-results-button", "n_clicks"),
    State("clustered-smiles-store", "data"),
    State("optimized-motifs-store", "data"),
    State("lda-dict-store", "data"),
    State('spectra-store', "data"),
    prevent_initial_call=True,
)
def save_results(n_clicks, clustered_smiles_data, optimized_motifs_data, lda_dict, spectra_data):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if not clustered_smiles_data or not optimized_motifs_data or not lda_dict or not spectra_data:
        return (
            dash.no_update,
            dbc.Alert(
                "No analysis results to save. Please run an analysis first.", color="warning"
            ),
        )

    try:
        data = {
            'clustered_smiles_data': clustered_smiles_data,
            'optimized_motifs_data': optimized_motifs_data,
            'lda_dict': lda_dict,
            'spectra_data': spectra_data,
        }
        json_data = json.dumps(data)
        return dcc.send_string(json_data, filename="ms2lda_results.json"), dbc.Alert(
            "Results saved successfully!", color="success"
        )
    except Exception as e:
        return (
            dash.no_update,
            dbc.Alert(
                f"An error occurred while saving the results: {str(e)}", color="danger"
            ),
        )

# Helper function to convert Spectrum to dict (for serialization)
def spectrum_to_dict(spectrum):
    metadata = spectrum.metadata.copy()
    if spectrum.losses:
        metadata["losses"] = [
            {"loss_mz": float(loss_mz), "loss_intensity": float(loss_intensity)}
            for loss_mz, loss_intensity in zip(spectrum.losses.mz, spectrum.losses.intensities)
        ]

    return {
        "metadata": metadata,
        "mz": [float(m) for m in spectrum.peaks.mz.tolist()],
        "intensities": [float(i) for i in spectrum.peaks.intensities.tolist()],
    }

# Updated Callback to create Cytoscape elements
@app.callback(
    Output("cytoscape-network-container", "children"),
    Input("optimized-motifs-store", "data"),
    Input("clustered-smiles-store", "data"),
    Input("tabs", "value"),
)
def update_cytoscape(optimized_motifs_data, clustered_smiles_data, active_tab):
    if active_tab != "results-tab" or not optimized_motifs_data:
        return ""

    spectra = []
    for s in optimized_motifs_data:
        spectrum = Spectrum(
            mz=np.array(s["mz"], dtype=float),
            intensities=np.array(s["intensities"], dtype=float),
            metadata=s["metadata"],
        )
        if "losses" in s["metadata"]:
            losses = s["metadata"]["losses"]
            mz = [loss["loss_mz"] for loss in losses]
            intensities = [loss["loss_intensity"] for loss in losses]
            spectrum.losses = Fragments(
                mz=np.array(mz, dtype=float),
                intensities=np.array(intensities, dtype=float),
            )
        else:
            spectrum.losses = None
        spectra.append(spectrum)

    smiles_clusters = clustered_smiles_data

    elements = create_cytoscape_elements(spectra, smiles_clusters)

    cytoscape_component = cyto.Cytoscape(
        id="cytoscape-network",
        elements=elements,
        style={"width": "100%", "height": "100%"},
        layout={"name": "cose", "animate": False},
        stylesheet=[
            {
                "selector": 'node[type="motif"]',
                "style": {
                    "background-color": "#00008B",
                    "label": "data(label)",
                    "text-background-color": "white",
                    "text-background-opacity": 0.7,
                    "text-background-padding": "3px",
                    "text-background-shape": "roundrectangle",
                    "text-border-color": "black",
                    "text-border-width": 1,
                    "text-valign": "top",
                    "text-halign": "center",
                    "color": "black",
                    "font-size": "10px",
                },
            },
            {
                "selector": 'node[type="fragment"]',
                "style": {
                    "background-color": "#008000",
                    "label": "data(label)",
                    "text-background-color": "white",
                    "text-background-opacity": 0.7,
                    "text-background-padding": "3px",
                    "text-background-shape": "roundrectangle",
                    "text-border-color": "black",
                    "text-border-width": 1,
                    "text-valign": "top",
                    "text-halign": "center",
                    "color": "black",
                    "font-size": "8px",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "line-color": "red",
                    "opacity": 0.5,
                    "width": "mapData(weight, 0, 1, 1, 10)",
                    "target-arrow-shape": "none",
                    "curve-style": "bezier",
                },
            },
            {
                "selector": "node",
                "style": {
                    "shape": "ellipse",
                },
            },
        ],
    )

    return cytoscape_component

# Callback to display molecule images
@app.callback(
    Output("molecule-images", "children"),
    Input("cytoscape-network", "tapNodeData"),
    State("clustered-smiles-store", "data"),
)
def display_molecule_images(nodeData, clustered_smiles_data):
    if nodeData and nodeData["id"].startswith("motif_"):
        motif_number = int(nodeData["id"].split("_")[1])
        if motif_number < len(clustered_smiles_data):
            smiles_list = clustered_smiles_data[motif_number]
            mols = []
            for smi in smiles_list:
                try:
                    mol = MolFromSmiles(smi)
                    if mol is not None:
                        mols.append(mol)
                except Exception as e:
                    print(f"Error converting SMILES {smi}: {str(e)}")
            if not mols:
                return dbc.Alert(
                    "No valid molecules could be created from SMILES.",
                    color="warning"
                )
            try:
                from rdkit.Chem import Draw
                legends = [f"Match {i + 1}" for i in range(len(mols))]
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=1,
                    subImgSize=(200, 200),
                    legends=legends,
                    returnPNG=True
                )
                encoded = base64.b64encode(img).decode("utf-8")
                return html.Div([
                    html.H5(f"Molecules for Motif {motif_number}"),
                    html.Img(
                        src=f"data:image/png;base64,{encoded}",
                        style={"margin": "10px"},
                    ),
                ])
            except Exception as e:
                print(f"Error creating grid image: {str(e)}")
                return dbc.Alert(
                    f"Error creating molecular grid image: {str(e)}",
                    color="danger"
                )
        return dbc.Alert("Motif number out of range.", color="danger")
    return ""

def create_cytoscape_elements(spectra, smiles_clusters):
    elements = []
    created_fragments = set()
    created_losses = set()

    for i, spectrum in enumerate(spectra):
        motif_node = f"motif_{i}"
        elements.append(
            {
                "data": {
                    "id": motif_node,
                    "label": motif_node,
                    "type": "motif",
                }
            }
        )
        for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities):
            rounded_mz = round(mz, 2)
            frag_node = f"frag_{rounded_mz}"
            if frag_node not in created_fragments:
                elements.append(
                    {
                        "data": {
                            "id": frag_node,
                            "label": str(rounded_mz),
                            "type": "fragment",
                        }
                    }
                )
                created_fragments.add(frag_node)
            elements.append(
                {
                    "data": {
                        "source": motif_node,
                        "target": frag_node,
                        "weight": intensity,
                    }
                }
            )
        if spectrum.losses is not None:
            precursor_mz = float(spectrum.metadata.get('precursor_mz', 0))
            for loss_data in spectrum.metadata.get("losses", []):
                loss_mz = loss_data["loss_mz"]
                loss_intensity = loss_data["loss_intensity"]
                corresponding_frag_mz = precursor_mz - loss_mz
                rounded_frag_mz = round(corresponding_frag_mz, 2)
                frag_node = f"frag_{rounded_frag_mz}"

                if frag_node not in created_fragments:
                    elements.append(
                        {
                            "data": {
                                "id": frag_node,
                                "label": str(rounded_frag_mz),
                                "type": "fragment",
                            }
                        }
                    )
                    created_fragments.add(frag_node)
                loss_node = f"loss_{loss_mz}"
                if loss_node not in created_losses:
                    elements.append(
                        {
                            "data": {
                                "id": loss_node,
                                "label": f"-{loss_mz:.2f}",
                                "type": "fragment",
                            }
                        }
                    )
                    created_losses.add(loss_node)
                elements.append(
                    {
                        "data": {
                            "source": motif_node,
                            "target": loss_node,
                            "weight": loss_intensity,
                        }
                    }
                )
                elements.append(
                    {
                        "data": {
                            "source": loss_node,
                            "target": frag_node,
                            "weight": loss_intensity,
                        }
                    }
                )
    return elements

# Helper function to compute motif degrees
def compute_motif_degrees(lda_dict, p_thresh, o_thresh):
    motifs = lda_dict["beta"].keys()
    motif_degrees = {m: 0 for m in motifs}
    motif_probabilities = {m: [] for m in motifs}
    motif_overlap_scores = {m: [] for m in motifs}
    docs = lda_dict["theta"].keys()

    for doc in docs:
        for motif, p in lda_dict["theta"][doc].items():
            if p >= p_thresh:
                o = lda_dict["overlap_scores"][doc].get(motif, 0.0)
                if o >= o_thresh:
                    motif_degrees[motif] += 1
                    motif_probabilities[motif].append(p)
                    motif_overlap_scores[motif].append(o)

    md = []
    import numpy as np
    for motif in motifs:
        avg_probability = np.mean(motif_probabilities[motif]) if motif_probabilities[motif] else 0
        avg_overlap = np.mean(motif_overlap_scores[motif]) if motif_overlap_scores[motif] else 0
        md.append((motif, motif_degrees[motif], avg_probability, avg_overlap))
    md.sort(key=lambda x: x[1], reverse=True)
    return md

# Callback to update the Motif Rankings TABLE ONLY
@app.callback(
    Output('motif-rankings-table-container', 'children'),
    Input('lda-dict-store', 'data'),
    Input('probability-thresh', 'value'),
    Input('overlap-thresh', 'value'),
    Input('tabs', 'value'),
)
def update_motif_rankings_table(lda_dict_data, probability_thresh, overlap_thresh, active_tab):
    if active_tab != 'motif-rankings-tab' or not lda_dict_data:
        return ""

    p_thresh = probability_thresh[0] if isinstance(probability_thresh, list) else probability_thresh
    o_thresh = overlap_thresh[0] if isinstance(overlap_thresh, list) else overlap_thresh

    motif_degree_list = compute_motif_degrees(lda_dict_data, p_thresh, o_thresh)
    df = pd.DataFrame(motif_degree_list, columns=[
        'Motif',
        'Degree',
        'Average Doc-Topic Probability',
        'Average Overlap Score'
    ])

    motif_annotations = {}
    if 'topic_metadata' in lda_dict_data:
        for motif, metadata in lda_dict_data['topic_metadata'].items():
            motif_annotations[motif] = metadata.get('annotation', '')

    df['Annotation'] = df['Motif'].map(motif_annotations)

    style_data_conditional = [
        {
            'if': {'column_id': 'Motif'},
            'cursor': 'pointer',
            'textDecoration': 'underline',
            'color': 'blue',
        },
    ]

    table = dash_table.DataTable(
        id='motif-rankings-table',
        data=df.to_dict('records'),
        columns=[
            {'name': 'Motif', 'id': 'Motif'},
            {'name': 'Degree', 'id': 'Degree', 'type': 'numeric', 'format': {'specifier': ''}},
            {'name': 'Average Doc-Topic Probability', 'id': 'Average Doc-Topic Probability', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            {'name': 'Average Overlap Score', 'id': 'Average Overlap Score', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            {'name': 'Annotation', 'id': 'Annotation'},
        ],
        sort_action='native',
        filter_action='native',
        page_size=20,
        style_table={'overflowX': 'auto'},
        style_cell={
            'minWidth': '150px', 'width': '200px', 'maxWidth': '350px',
            'whiteSpace': 'normal',
            'textAlign': 'left',
        },
        style_data_conditional=style_data_conditional,
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
    )

    return table

# Callback to store selected motif
@app.callback(
    Output('selected-motif-store', 'data'),
    Input('motif-rankings-table', 'active_cell'),
    State('motif-rankings-table', 'data'),
    prevent_initial_call=True,
)
def on_motif_click(active_cell, table_data):
    if active_cell and active_cell['column_id'] == 'Motif':
        motif = table_data[active_cell['row']]['Motif']
        return motif
    return dash.no_update

# Callback to activate Motif Details tab
@app.callback(
    Output('tabs', 'value'),
    Input('selected-motif-store', 'data'),
    prevent_initial_call=True,
)
def activate_motif_details_tab(selected_motif):
    if selected_motif:
        return 'motif-details-tab'
    else:
        return dash.no_update

# Callback to populate Motif Details tab
@app.callback(
    Output('motif-details-title', 'children'),
    Output('motif-details-content', 'children'),
    Output('motif-spectra-ids-store', 'data'),
    Output('spectra-table', 'data'),
    Output('spectra-table', 'columns'),
    Input('selected-motif-store', 'data'),
    Input('probability-filter', 'value'),
    State('lda-dict-store', 'data'),
    State('clustered-smiles-store', 'data'),
    State('spectra-store', 'data'),
    prevent_initial_call=True,
)
def update_motif_details(selected_motif, probability_range, lda_dict_data, clustered_smiles_data, spectra_data):
    if not selected_motif or not lda_dict_data:
        return '', [], [], [], ''

    motif_name = selected_motif
    motif_data = lda_dict_data['beta'].get(motif_name, {})

    filtered_motif_data = {
        feature: prob for feature, prob in motif_data.items()
        if probability_range[0] <= prob <= probability_range[1]
    }

    total_prob = sum(filtered_motif_data.values())
    content = []

    feature_table = pd.DataFrame({
        'Feature': filtered_motif_data.keys(),
        'Probability': filtered_motif_data.values(),
    }).sort_values(by='Probability', ascending=False)
    feature_table_component = dash_table.DataTable(
        data=feature_table.to_dict('records'),
        columns=[
            {'name': 'Feature', 'id': 'Feature'},
            {'name': 'Probability', 'id': 'Probability', 'type': 'numeric', 'format': {'specifier': '.4f'}},
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10,
    )
    content.append(html.H5('Features Explained by This Motif'))
    content.append(feature_table_component)
    content.append(html.P(f'Total Probability (Filtered): {total_prob:.4f}'))

    spectra_data_list = []
    for doc, topics in lda_dict_data['theta'].items():
        prob = topics.get(motif_name, 0)
        if prob > 0:
            overlap = lda_dict_data['overlap_scores'][doc].get(motif_name, 0)
            spectra_data_list.append({
                'Spectrum': doc,
                'Probability': prob,
                'Overlap Score': overlap,
            })
    spectra_df = pd.DataFrame(spectra_data_list).sort_values(by='Probability', ascending=False)

    spectra_table_data = spectra_df.to_dict('records')
    spectra_table_columns = [
        {'name': 'Spectrum', 'id': 'Spectrum'},
        {'name': 'Probability', 'id': 'Probability', 'type': 'numeric', 'format': {'specifier': '.4f'}},
        {'name': 'Overlap Score', 'id': 'Overlap Score', 'type': 'numeric', 'format': {'specifier': '.4f'}},
    ]

    features_in_motif = list(filtered_motif_data.keys())
    total_feature_probs = {feature: 0.0 for feature in features_in_motif}
    for motif, feature_probs in lda_dict_data['beta'].items():
        for feature, prob in feature_probs.items():
            if feature in total_feature_probs:
                total_feature_probs[feature] += prob

    import plotly.express as px
    barplot1_df = pd.DataFrame({
        'Feature': features_in_motif,
        'Probability in Motif': [filtered_motif_data[feature] for feature in features_in_motif],
        'Total Probability': [total_feature_probs[feature] for feature in features_in_motif],
    })
    barplot1_df = barplot1_df.sort_values(by='Probability in Motif', ascending=False).head(10)
    barplot1_df_long = pd.melt(
        barplot1_df,
        id_vars='Feature',
        value_vars=['Total Probability', 'Probability in Motif'],
        var_name='Type',
        value_name='Probability'
    )
    barplot1_fig = px.bar(
        barplot1_df_long,
        x='Probability',
        y='Feature',
        color='Type',
        orientation='h',
        barmode='group',
        title='Proportion of Total Intensity Explained by This Motif (Top 10 Features)',
    )
    barplot1_fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Probability',
        yaxis_title='Feature',
        legend_title='',
    )

    feature_counts = {feature: 0 for feature in features_in_motif}
    for doc in spectra_df['Spectrum']:
        doc_features = lda_dict_data['corpus'].get(doc, {}).keys()
        for feature in features_in_motif:
            if feature in doc_features:
                feature_counts[feature] += 1

    barplot2_df = pd.DataFrame({
        'Feature': features_in_motif,
        'Count': [feature_counts[feature] for feature in features_in_motif],
    })
    barplot2_df = barplot2_df.sort_values(by='Count', ascending=False).head(10)
    barplot2_fig = px.bar(
        barplot2_df,
        x='Count',
        y='Feature',
        orientation='h',
        title='Counts of Features in Documents Associated with This Motif (Top 10 Features)',
    )
    barplot2_fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Count',
        yaxis_title='Feature',
    )

    content.append(html.H5('Counts of Mass2Motif Features'))
    content.append(dcc.Graph(figure=barplot1_fig))
    content.append(dcc.Graph(figure=barplot2_fig))

    motif_index = int(motif_name.replace('motif_', '')) if motif_name.startswith('motif_') else 0
    if clustered_smiles_data and motif_index < len(clustered_smiles_data):
        smiles_list = clustered_smiles_data[motif_index]
        if smiles_list:
            content.append(html.H5('Spec2Vec Matching Results'))
            mols = []
            for smi in smiles_list:
                try:
                    mol = MolFromSmiles(smi)
                    if mol is not None:
                        mols.append(mol)
                except Exception as e:
                    print(f"Error converting SMILES {smi}: {str(e)}")
            if mols:
                from rdkit.Chem import Draw
                legends = [f"Match {i + 1}" for i in range(len(mols))]
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=4,
                    subImgSize=(200, 200),
                    legends=legends,
                    returnPNG=True
                )
                encoded = base64.b64encode(img).decode("utf-8")
                content.append(html.Img(
                    src=f"data:image/png;base64,{encoded}",
                    style={"margin": "10px"},
                ))

    spectra_ids = spectra_df['Spectrum'].tolist()
    return f"Motif Details: {motif_name}", content, spectra_ids, spectra_table_data, spectra_table_columns

@app.callback(
    Output('selected-spectrum-index', 'data'),
    Output('spectra-table', 'selected_rows'),
    Input('spectra-table', 'selected_rows'),
    Input('next-spectrum', 'n_clicks'),
    Input('prev-spectrum', 'n_clicks'),
    Input('selected-motif-store', 'data'),
    Input('motif-spectra-ids-store', 'data'),
    State('selected-spectrum-index', 'data'),
    prevent_initial_call=True,
)
def update_selected_spectrum(selected_rows, next_clicks, prev_clicks, selected_motif, motif_spectra_ids, current_index):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'spectra-table':
        if selected_rows:
            new_index = selected_rows[0]
            return new_index, selected_rows
        else:
            return current_index, dash.no_update

    elif triggered_id == 'next-spectrum':
        if motif_spectra_ids and current_index < len(motif_spectra_ids) - 1:
            new_index = current_index + 1
            return new_index, [new_index]
        else:
            return current_index, dash.no_update

    elif triggered_id == 'prev-spectrum':
        if motif_spectra_ids and current_index > 0:
            new_index = current_index - 1
            return new_index, [new_index]
        else:
            return current_index, dash.no_update

    elif triggered_id == 'selected-motif-store' or triggered_id == 'motif-spectra-ids-store':
        return 0, [0]

    else:
        return current_index, dash.no_update
@app.callback(
    Output('spectrum-plot', 'children'),
    Input('selected-spectrum-index', 'data'),
    Input('probability-filter', 'value'),
    State('motif-spectra-ids-store', 'data'),
    State('spectra-store', 'data'),
    State('lda-dict-store', 'data'),
    State('selected-motif-store', 'data'),
)
def update_spectrum_plot(selected_index, probability_range, spectra_ids, spectra_data, lda_dict_data, selected_motif):
    if spectra_ids and spectra_data and lda_dict_data and selected_motif:
        if selected_index is None or selected_index < 0 or selected_index >= len(spectra_ids):
            return html.Div("Selected spectrum index is out of range.")

        # Retrieve matching spectrum data
        spectrum_id = spectra_ids[selected_index]
        spectrum_dict = next((s for s in spectra_data if s['metadata']['id'] == spectrum_id), None)
        if not spectrum_dict:
            return html.Div("Spectrum data not found.")

        # Reconstruct Spectrum object
        from matchms import Spectrum, Fragments
        import numpy as np
        import pandas as pd
        import plotly.graph_objs as go

        spectrum = Spectrum(
            mz=np.array(spectrum_dict['mz']),
            intensities=np.array(spectrum_dict['intensities']),
            metadata=spectrum_dict['metadata'],
        )

        # Collect motif data and filter by probability
        motif_data = lda_dict_data['beta'].get(selected_motif, {})
        filtered_motif_data = {
            feature: prob for feature, prob in motif_data.items()
            if probability_range[0] <= prob <= probability_range[1]
        }

        # Separate out fragment and loss features
        motif_mz_values = []
        motif_loss_values = []
        for feature in filtered_motif_data:
            if feature.startswith('frag@'):
                try:
                    mz_value = float(feature.replace('frag@', ''))
                    motif_mz_values.append(mz_value)
                except ValueError:
                    pass
            elif feature.startswith('loss@'):
                try:
                    loss_value = float(feature.replace('loss@', ''))
                    motif_loss_values.append(loss_value)
                except ValueError:
                    pass

        # Turn spectrum peaks into a DataFrame
        spectrum_df = pd.DataFrame({
            'mz': spectrum.peaks.mz,
            'intensity': spectrum.peaks.intensities,
        })

        # Mark peaks that match motif fragments
        tolerance = 0.1
        spectrum_df['is_motif'] = False
        for mz_val in motif_mz_values:
            mask = np.abs(spectrum_df['mz'] - mz_val) <= tolerance
            spectrum_df.loc[mask, 'is_motif'] = True

        # Plot colors: highlight motif peaks in red
        colors = ['#DC143C' if is_motif else '#B0B0B0'
                  for is_motif in spectrum_df['is_motif']]

        # Check if there's a known precursor
        parent_ion_present = False
        parent_ion_mz = None
        parent_ion_intensity = None
        if 'precursor_mz' in spectrum.metadata:
            try:
                parent_ion_mz = float(spectrum.metadata['precursor_mz'])
                # If no parent_intensity known, use max peak intensity
                parent_ion_intensity = float(spectrum.metadata.get('parent_intensity', spectrum_df['intensity'].max()))
                parent_ion_present = True
            except (ValueError, TypeError):
                parent_ion_present = False

        # Create Plotly figure
        fig = go.Figure()

        # 1) Bar trace for the main peaks
        fig.add_trace(go.Bar(
            x=spectrum_df['mz'],
            y=spectrum_df['intensity'],
            marker=dict(color=colors, line=dict(color='white', width=0)),
            width=0.2,
            name='Peaks',
            hoverinfo='text',
            hovertext=[
                f"Motif Peak: {motif}<br>m/z: {mz_val:.2f}<br>Intensity: {inten}"
                for motif, mz_val, inten in zip(spectrum_df['is_motif'],
                                                spectrum_df['mz'],
                                                spectrum_df['intensity'])
            ],
            opacity=0.9,
        ))

        # 2) Add a bar for the parent ion if present
        if parent_ion_present and parent_ion_mz is not None and parent_ion_intensity is not None:
            fig.add_trace(go.Bar(
                x=[parent_ion_mz],
                y=[parent_ion_intensity],
                marker=dict(color='#0000FF', line=dict(color='white', width=0)),
                width=0.4,
                name='Parent Ion',
                hoverinfo='text',
                hovertext=[f"Parent Ion<br>m/z: {parent_ion_mz:.2f}<br>Intensity: {parent_ion_intensity}"],
                opacity=1.0,
            ))

        # 3) Draw lines for “loss” annotation if the doc has "losses"
        if "losses" in spectrum.metadata and parent_ion_present:
            precursor_mz = parent_ion_mz
            for loss_item in spectrum.metadata["losses"]:
                loss_mz = loss_item["loss_mz"]
                loss_intensity = loss_item["loss_intensity"]
                # If this loss is not in the motif, skip it
                # (some users want to see all, but let's follow your code's logic)
                if not any(abs(loss_mz - val) <= tolerance for val in motif_loss_values):
                    continue

                # Attempt to find a corresponding fragment
                corresponding_frag_mz = precursor_mz - loss_mz
                frag_mask = (np.abs(spectrum_df['mz'] - corresponding_frag_mz) <= tolerance)
                # If no matches, skip
                if not frag_mask.any():
                    continue

                # Safely pick the first matched row
                frag_subset = spectrum_df.loc[frag_mask]
                if frag_subset.empty:
                    continue  # <--- FIX: skip if empty

                closest_frag_mz = frag_subset['mz'].iloc[0]
                closest_frag_intensity = frag_subset['intensity'].iloc[0]

                # Add a dashed line connecting them
                fig.add_shape(
                    type="line",
                    x0=closest_frag_mz,
                    y0=closest_frag_intensity,
                    x1=precursor_mz,
                    y1=closest_frag_intensity,
                    line=dict(color="green", width=2, dash="dash"),
                )
                # Add annotation for the loss value
                fig.add_annotation(
                    x=(closest_frag_mz + precursor_mz) / 2,
                    y=closest_frag_intensity,
                    text=f"-{loss_mz:.2f}",
                    showarrow=False,
                    font=dict(family="Courier New, monospace", size=12, color="green"),
                    bgcolor="rgba(255,255,255,0.7)",
                    xanchor="center",
                    yanchor="bottom",
                    standoff=5,
                )

        # Final figure layout
        fig.update_layout(
            title=f"Spectrum: {spectrum_id}",
            xaxis_title='m/z',
            yaxis_title='Intensity',
            bargap=0.1,
            barmode='overlay',
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(
                title='Peak Types',
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest',
        )

        return dcc.Graph(
            figure=fig,
            style={'width': '100%', 'height': '600px', 'margin': 'auto'}
        )

    return ""


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
