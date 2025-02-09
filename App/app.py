import base64
import json
import os
import tempfile

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import tomotopy as tp
from dash import dash_table
from dash import html, dcc, Input, Output, State
from dash import no_update
from dash.exceptions import PreventUpdate
from matchms import Spectrum, Fragments
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

import MS2LDA
from MS2LDA.Preprocessing.load_and_clean import clean_spectra
from MS2LDA.Visualisation.ldadict import generate_corpusjson_from_tomotopy
from MS2LDA.run import filetype_check

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
        dbc.Row([
            dbc.Col([
                html.Img(src="assets/MS2LDA_LOGO_white.jpg", alt="MS2LDA Logo", height="250px",
                         style={'display': 'block', 'margin': 'auto'}),
                dcc.Markdown("""
                Developed by [Jonas Dietrich](https://github.com/j-a-dietrich),
                [Rosina Torres Ortega](https://github.com/rtlortega), and 
                [Joe Wandy](https://github.com/joewandy).
                """, style={'textAlign': 'center'})
            ], width=True),
        ], align="end"),
        html.Hr(),

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

        # ----------------------- RUN ANALYSIS TAB -----------------------
        html.Div(
            id="run-analysis-tab-content",
            children=[
                html.Div(
                    [
                        dcc.Markdown(
                            """
                            This tab allows you to run an MS2LDA analysis from scratch using a single uploaded data file. 
                            You can control basic parameters like the number of motifs, polarity, and top N Spec2Vec matches, 
                            as well as advanced settings (e.g., min_mz, max_mz). 
                            When ready, click "Run Analysis" to generate the results and proceed to the other tabs for visualization.
                            """
                        )
                    ],
                    style={"margin": "20px"},
                ),
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
                                # Basic parameters (with tooltips):
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Number of Motifs", id="n-motifs-tooltip"),
                                        dbc.Input(
                                            id="n-motifs",
                                            type="number",
                                            value=50,
                                            min=1,
                                        ),
                                    ],
                                    className="mb-3",
                                    id="n-motifs-inputgroup",
                                ),
                                dbc.Tooltip(
                                    "Number of LDA topics to discover. Typically 10-100.",
                                    target="n-motifs-tooltip",
                                    placement="right",
                                ),

                                html.Div(
                                    [
                                        dbc.Label("Acquisition Type", id="acq-type-tooltip"),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "DDA", "value": "DDA"},
                                                {"label": "DIA", "value": "DIA"},
                                            ],
                                            value="DDA",
                                            id="acquisition-type",
                                            inline=True,
                                        ),
                                    ],
                                    className="mb-3",
                                    id="acq-type-div",
                                ),
                                dbc.Tooltip(
                                    "Type of acquisition: DDA or DIA. Affects how the spectra are tokenized.",
                                    target="acq-type-tooltip",
                                    placement="right",
                                ),

                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Top N Matches", id="topn-tooltip"),
                                        dbc.Input(
                                            id="top-n", type="number", value=5, min=1
                                        ),
                                    ],
                                    className="mb-3",
                                    id="topn-inputgroup",
                                ),
                                dbc.Tooltip(
                                    "Number of library matches to retrieve per motif via Spec2Vec.",
                                    target="topn-tooltip",
                                    placement="right",
                                ),

                                html.Div(
                                    [
                                        dbc.Label("Unique Molecules", id="uniqmols-tooltip"),
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
                                    id="uniqmols-div",
                                ),
                                dbc.Tooltip(
                                    "Whether to keep only unique compounds or include duplicates in Spec2Vec hits.",
                                    target="uniqmols-tooltip",
                                    placement="right",
                                ),

                                html.Div(
                                    [
                                        dbc.Label("Polarity", id="polarity-tooltip"),
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
                                    id="polarity-div",
                                ),
                                dbc.Tooltip(
                                    "Polarity can be used for specialized processing. Currently set to DDA.",
                                    target="polarity-tooltip",
                                    placement="right",
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Iterations", id="iterations-tooltip"),
                                        dbc.Input(id="n-iterations", type="number", value=1000),
                                    ],
                                    className="mb-3",
                                    id="iterations-inputgroup",
                                ),
                                dbc.Tooltip(
                                    "Number of LDA training iterations. Higher = more thorough training.",
                                    target="iterations-tooltip",
                                    placement="right",
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("S2V Model Path", id="s2v-model-tooltip"),
                                        dbc.Input(
                                            id="s2v-model-path",
                                            type="text",
                                            value="../MS2LDA/Add_On/Spec2Vec/model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model",
                                        ),
                                    ],
                                    className="mb-3",
                                    id="s2v-model-inputgroup",
                                ),
                                dbc.Tooltip(
                                    "Spec2Vec model file (trained embeddings). Provide full path.",
                                    target="s2v-model-tooltip",
                                    placement="right",
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("S2V Library Path", id="s2v-library-tooltip"),
                                        dbc.Input(
                                            id="s2v-library-path",
                                            type="text",
                                            value="../MS2LDA/Add_On/Spec2Vec/model_positive_mode/positive_s2v_library.pkl",
                                        ),
                                    ],
                                    className="mb-3",
                                    id="s2v-library-inputgroup",
                                ),
                                dbc.Tooltip(
                                    "Pickled library embeddings for Spec2Vec. Provide full path.",
                                    target="s2v-library-tooltip",
                                    placement="right",
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
                                                                dbc.InputGroupText("min_mz", id="prep-minmz-tooltip"),
                                                                dbc.Input(
                                                                    id="prep-min-mz",
                                                                    type="number",
                                                                    value=0,  # default
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="prep-minmz-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Minimum m/z to keep in each spectrum’s peaks.",
                                                            target="prep-minmz-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("max_mz", id="prep-maxmz-tooltip"),
                                                                dbc.Input(
                                                                    id="prep-max-mz",
                                                                    type="number",
                                                                    value=2000,  # default
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="prep-maxmz-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Maximum m/z to keep in each spectrum’s peaks.",
                                                            target="prep-maxmz-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("max_frags",
                                                                                   id="prep-maxfrags-tooltip"),
                                                                dbc.Input(
                                                                    id="prep-max-frags",
                                                                    type="number",
                                                                    value=1000,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="prep-maxfrags-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Max number of peaks (frags) to retain per spectrum.",
                                                            target="prep-maxfrags-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_frags",
                                                                                   id="prep-minfrags-tooltip"),
                                                                dbc.Input(
                                                                    id="prep-min-frags",
                                                                    type="number",
                                                                    value=5,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="prep-minfrags-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Minimum number of peaks needed to keep a spectrum.",
                                                            target="prep-minfrags-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_intensity",
                                                                                   id="prep-minint-tooltip"),
                                                                dbc.Input(
                                                                    id="prep-min-intensity",
                                                                    type="number",
                                                                    value=0.01,
                                                                    step=0.001,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="prep-minint-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Lower intensity threshold (relative). Discard peaks below this fraction.",
                                                            target="prep-minint-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("max_intensity",
                                                                                   id="prep-maxint-tooltip"),
                                                                dbc.Input(
                                                                    id="prep-max-intensity",
                                                                    type="number",
                                                                    value=1,
                                                                    step=0.1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="prep-maxint-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Upper intensity threshold (relative). Discard peaks above this fraction.",
                                                            target="prep-maxint-tooltip",
                                                            placement="right",
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
                                                                dbc.InputGroupText("step_size",
                                                                                   id="conv-stepsz-tooltip"),
                                                                dbc.Input(
                                                                    id="conv-step-size",
                                                                    type="number",
                                                                    value=50,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="conv-stepsz-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Steps per iteration chunk before checking convergence criteria.",
                                                            target="conv-stepsz-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("window_size",
                                                                                   id="conv-winsz-tooltip"),
                                                                dbc.Input(
                                                                    id="conv-window-size",
                                                                    type="number",
                                                                    value=10,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="conv-winsz-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Number of iteration checkpoints to consider for convergence trend.",
                                                            target="conv-winsz-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("threshold",
                                                                                   id="conv-thresh-tooltip"),
                                                                dbc.Input(
                                                                    id="conv-threshold",
                                                                    type="number",
                                                                    value=0.005,
                                                                    step=0.0001,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="conv-thresh-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Convergence threshold for perplexity or entropy changes.",
                                                            target="conv-thresh-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("type", id="conv-type-tooltip"),
                                                                dbc.Select(
                                                                    id="conv-type",
                                                                    options=[
                                                                        {"label": "perplexity_history",
                                                                         "value": "perplexity_history"},
                                                                        {"label": "entropy_history_doc",
                                                                         "value": "entropy_history_doc"},
                                                                        {"label": "entropy_history_topic",
                                                                         "value": "entropy_history_topic"},
                                                                        {"label": "log_likelihood_history",
                                                                         "value": "log_likelihood_history"},
                                                                    ],
                                                                    value="perplexity_history",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="conv-type-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Which metric to watch for early stopping: perplexity, doc/topic entropy, or log likelihood.",
                                                            target="conv-type-tooltip",
                                                            placement="right",
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
                                                                dbc.InputGroupText("criterium",
                                                                                   id="ann-criterium-tooltip"),
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
                                                            id="ann-criterium-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Use 'best' (hit cluster is best matched) or 'biggest' (largest cluster).",
                                                            target="ann-criterium-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("cosine_similarity",
                                                                                   id="ann-cossim-tooltip"),
                                                                dbc.Input(
                                                                    id="ann-cosine-sim",
                                                                    type="number",
                                                                    value=0.90,
                                                                    step=0.01,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="ann-cossim-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Cosine similarity threshold for motif-spectra optimization step.",
                                                            target="ann-cossim-tooltip",
                                                            placement="right",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Model"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("rm_top", id="model-rmtop-tooltip"),
                                                                dbc.Input(
                                                                    id="model-rm-top",
                                                                    type="number",
                                                                    value=0,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="model-rmtop-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Number of top words to remove globally (tomotopy param). Often 0.",
                                                            target="model-rmtop-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_cf", id="model-mincf-tooltip"),
                                                                dbc.Input(
                                                                    id="model-min-cf",
                                                                    type="number",
                                                                    value=0,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="model-mincf-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Minimum corpus frequency to keep a token (tomotopy param). Usually 0.",
                                                            target="model-mincf-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("min_df", id="model-mindf-tooltip"),
                                                                dbc.Input(
                                                                    id="model-min-df",
                                                                    type="number",
                                                                    value=3,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="model-mindf-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Minimum document frequency to keep a token. For big corpora.",
                                                            target="model-mindf-tooltip",
                                                            placement="right",
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
                                                                dbc.InputGroupText("alpha", id="model-alpha-tooltip"),
                                                                dbc.Input(
                                                                    id="model-alpha",
                                                                    type="number",
                                                                    value=0.6,
                                                                    step=0.1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="model-alpha-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Dirichlet prior for doc-topic distributions. Lower => more specialized topics.",
                                                            target="model-alpha-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("eta", id="model-eta-tooltip"),
                                                                dbc.Input(
                                                                    id="model-eta",
                                                                    type="number",
                                                                    value=0.01,
                                                                    step=0.001,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="model-eta-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Dirichlet prior for topic-word distributions. Lower => more peaky topics.",
                                                            target="model-eta-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("seed", id="model-seed-tooltip"),
                                                                dbc.Input(
                                                                    id="model-seed",
                                                                    type="number",
                                                                    value=42,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="model-seed-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Random seed for reproducibility.",
                                                            target="model-seed-tooltip",
                                                            placement="right",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Train"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("parallel",
                                                                                   id="train-parallel-tooltip"),
                                                                dbc.Input(
                                                                    id="train-parallel",
                                                                    type="number",
                                                                    value=1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="train-parallel-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "How many parallel threads for tomotopy training. Usually 1-4.",
                                                            target="train-parallel-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("workers",
                                                                                   id="train-workers-tooltip"),
                                                                dbc.Input(
                                                                    id="train-workers",
                                                                    type="number",
                                                                    value=0,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="train-workers-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Additional worker threads for certain tomotopy tasks. Typically 0 or 1.",
                                                            target="train-workers-tooltip",
                                                            placement="right",
                                                        ),
                                                        # n_iterations Moved Above
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
                                                                dbc.InputGroupText("sig_digits",
                                                                                   id="prep-sigdig-tooltip"),
                                                                dbc.Input(
                                                                    id="prep-sigdig",
                                                                    type="number",
                                                                    value=2,  # default
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="prep-sigdig-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Significant number of digits for fragments and losses.",
                                                            target="prep-sigdig-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("charge",
                                                                                   id="dataset-charge-tooltip"),
                                                                dbc.Input(
                                                                    id="dataset-charge",
                                                                    type="number",
                                                                    value=1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="dataset-charge-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Set the assumed precursor charge. E.g. 1 or 2.",
                                                            target="dataset-charge-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Run Name",
                                                                                   id="dataset-name-tooltip"),
                                                                dbc.Input(
                                                                    id="dataset-name",
                                                                    type="text",
                                                                    value="ms2lda_dashboard_run",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="dataset-name-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Name/identifier for this run (used in output filenames).",
                                                            target="dataset-name-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Output Folder",
                                                                                   id="dataset-outdir-tooltip"),
                                                                dbc.Input(
                                                                    id="dataset-output-folder",
                                                                    type="text",
                                                                    value="ms2lda_results",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="dataset-outdir-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Folder path where intermediate results & output files are saved.",
                                                            target="dataset-outdir-tooltip",
                                                            placement="right",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Fingerprint"),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("fp_type", id="fp-type-tooltip"),
                                                                dbc.Select(
                                                                    id="fp-type",
                                                                    options=[
                                                                        {"label": "rdkit", "value": "rdkit"},
                                                                        {"label": "maccs", "value": "maccs"},
                                                                        {"label": "pubchem", "value": "pubchem"},
                                                                        {"label": "ecfp", "value": "ecfp"},
                                                                    ],
                                                                    value="rdkit",
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="fp-type-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Which fingerprint type to use for structural similarity in motif annotation.",
                                                            target="fp-type-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("fp threshold",
                                                                                   id="fp-threshold-tooltip"),
                                                                dbc.Input(
                                                                    id="fp-threshold",
                                                                    type="number",
                                                                    value=0.8,
                                                                    step=0.1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="fp-threshold-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Tanimoto threshold for motif fingerprint annotation.",
                                                            target="fp-threshold-tooltip",
                                                            placement="right",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Motif Parameter",
                                                                                   id="motif-param-tooltip"),
                                                                dbc.Input(
                                                                    id="motif-parameter",
                                                                    type="number",
                                                                    value=50,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                            id="motif-param-ig",
                                                        ),
                                                        dbc.Tooltip(
                                                            "Number of top words to define each motif (e.g. 50).",
                                                            target="motif-param-tooltip",
                                                            placement="right",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
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
                            ],
                            width=6,
                        )
                    ],
                    justify="center",
                )
            ],
            style={"display": "block"},
        ),

        # ----------------------- LOAD RESULTS TAB -----------------------
        html.Div(
            id="load-results-tab-content",
            children=[
                html.Div(
                    [
                        dcc.Markdown(
                            """
                            This tab allows you to load previously generated MS2LDA results (a JSON file). 
                            Once loaded, you can explore them immediately in the subsequent tabs. 
                            This is useful if you’ve run an analysis before and want to revisit or share the results.
                            """
                        )
                    ],
                    style={"margin": "20px"},
                ),
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
                                html.Div(id="load-status", style={"marginTop": "20px"}),
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

        # ----------------------- CYTOSCAPE NETWORK TAB -----------------------
        html.Div(
            id="results-tab-content",
            children=[
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
                            """
                        )
                    ],
                    style={"margin": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col([
                            dbc.Label("Edge Intensity Threshold"),
                            dcc.Slider(
                                id="edge-intensity-threshold",
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.50,
                                marks={0: "0.0", 0.5: "0.5", 1: "1.0"}
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Add Loss -> Fragment Edge", "value": "show_loss_edge"}],
                                value=[],
                                id="toggle-loss-edge",
                                inline=True,
                            )
                        ], width=6),
                    ],
                    style={"marginTop": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col([
                            dbc.Label("Graph Layout"),
                            dcc.Dropdown(
                                id="cytoscape-layout-dropdown",
                                options=[
                                    {"label": "CoSE", "value": "cose"},
                                    {"label": "Force-Directed (Spring)", "value": "fcose"},
                                    {"label": "Circle", "value": "circle"},
                                    {"label": "Concentric", "value": "concentric"},
                                ],
                                value="fcose",
                                clearable=False,
                            )
                        ], width=6),
                    ],
                    style={"marginTop": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="cytoscape-network-container",
                                    style={
                                        "marginTop": "20px",
                                        "height": "600px",
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
                )
            ],
            style={"display": "none"},
        ),

        # ----------------------- MOTIF RANKINGS TAB -----------------------
        html.Div(
            id="motif-rankings-tab-content",
            children=[
                dbc.Container([
                    html.Div(
                        [
                            dcc.Markdown(
                                """
                                This tab displays your motifs in a ranked table, based on how frequently they appear 
                                in your data and their average probabilities. You can filter the table by probability or overlap thresholds 
                                to highlight motifs of interest. Clicking on a motif in the table takes you to detailed information about that motif.
                                """
                            )
                        ],
                        style={"margin": "20px"},
                    ),

                    dbc.Row([
                        dbc.Col([
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
                                                value=[0.1, 1],
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
                                                value=[0.3, 1],
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
                            html.Div(id="motif-rankings-table-container", style={"marginTop": "20px"})
                        ], width=12),
                    ]),
                ]),
            ],
            style={"display": "none"}
        ),

        # ----------------------- MOTIF DETAILS TAB -----------------------
        html.Div(
            id="motif-details-tab-content",
            children=[
                # Top summary
                html.Div(
                    [
                        dcc.Markdown(
                            """
                            This tab is organized into three main sections: (1) Motif Details, (2) Features in Motifs, and (3) Documents in Motifs. 

                            In **Motif Details**, you'll see the Spec2Vec matching results for the selected motif.
                            In **Features in Motifs**, a probability filter helps you select which motif features (fragments/losses) to show, 
                            along with bar plots indicating how strongly these features belong to the motif and how often they appear in the dataset.
                            Finally, **Documents in Motifs** displays the spectra containing this motif, with filters for document-topic probability and overlap score.
                            """
                        )
                    ],
                    style={"margin": "20px"},
                ),

                # ----------------- (A) Spec2Vec Results -----------------
                html.Div(
                    [
                        html.H4(id='motif-details-title'),
                        dcc.Markdown(
                            """
                            This section shows SMILES structures found by comparing the motif’s pseudo-spectrum against an external library.
                            You can view top matching compounds here and visually inspect their chemical structures.
                            """
                        ),
                        # This container will be dynamically filled in the callback
                        html.Div(id='motif-spec2vec-container'),
                    ],
                    style={
                        "border": "1px dashed #999",
                        "padding": "15px",
                        "borderRadius": "5px",
                        "margin": "20px"
                    },
                ),

                # ----------------- (B) Features in Motifs -----------------
                html.Div(
                    [
                        html.H4("Features in Motifs"),
                        dcc.Markdown(
                            """
                            The **Topic-Word Probability Filter** below controls which motif features (fragments/losses) are displayed 
                            based on their probability (`beta`). After filtering, you’ll see a table of selected features plus 
                            bar charts illustrating their distribution in the motif and the dataset.
                            """
                        ),

                        dbc.Label("Topic-Word Probability Filter:"),
                        dcc.RangeSlider(
                            id='probability-filter',
                            min=0,
                            max=1,
                            step=0.01,
                            value=[0, 1],
                            marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                            allowCross=False
                        ),
                        html.Div(id='probability-filter-display', style={"marginTop": "10px"}),

                        # This container will hold the feature table & first bar chart
                        html.Div(id='motif-features-container'),
                    ],
                    style={
                        "border": "1px dashed #999",
                        "padding": "15px",
                        "borderRadius": "5px",
                        "margin": "20px"
                    },
                ),

                # ----------------- (C) Documents in Motifs -----------------
                html.Div(
                    [
                        html.H4("Documents in Motifs"),
                        dcc.Markdown(
                            """
                            This section shows spectra that include the current motif. 
                            **Document-Topic Probability Filter** (theta) narrows the list by motif representation in each spectrum, 
                            and **Overlap Score Filter** focuses on how closely the spectrum’s features match this motif’s top features.
                            """
                        ),

                        dbc.Label("Document-Topic Probability Filter:"),
                        dcc.RangeSlider(
                            id='doc-topic-filter',
                            min=0,
                            max=1,
                            step=0.01,
                            value=[0, 1],
                            marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                            allowCross=False
                        ),
                        html.Div(id='doc-topic-filter-display', style={"marginTop": "10px"}),

                        dbc.Label("Overlap Score Filter:"),
                        dcc.RangeSlider(
                            id='overlap-filter',
                            min=0,
                            max=1,
                            step=0.01,
                            value=[0, 1],
                            marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                            allowCross=False
                        ),
                        html.Div(id='overlap-filter-display', style={"marginTop": "10px"}),

                        dcc.Store(id='motif-spectra-ids-store'),
                        dcc.Store(id='selected-spectrum-index', data=0),

                        # This container will hold the second bar chart & doc table
                        html.Div(id='motif-documents-container'),

                        dash_table.DataTable(
                            id='spectra-table',
                            data=[],
                            columns=[],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                            page_size=10,
                            row_selectable='single',
                            selected_rows=[0],
                            hidden_columns=["SpecIndex"],
                        ),

                        html.Div(id='spectrum-plot'),

                        html.Div([
                            dbc.Button('Previous', id='prev-spectrum', n_clicks=0, color="info"),
                            dbc.Button('Next', id='next-spectrum', n_clicks=0, className='ms-2', color="info"),
                        ], className='mt-3'),
                    ],
                    style={
                        "border": "1px dashed #999",
                        "padding": "15px",
                        "borderRadius": "5px",
                        "margin": "20px"
                    },
                ),
            ],
            style={"display": "none"},
        ),

        # Hidden storage
        dcc.Store(id="clustered-smiles-store"),
        dcc.Store(id="optimized-motifs-store"),
        dcc.Store(id="lda-dict-store"),
        dcc.Store(id='selected-motif-store'),
        dcc.Store(id='spectra-store'),
    ],
    fluid=False,
)


# ----------------- Callbacks & Functions -----------------

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


# Show/hide advanced settings
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


# ------------------- The MAIN callback to run or load results -------------------
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

    State("prep-sigdig", "value"),
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
    State("n-iterations", "value"),  # <---- Moved to top
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
        prep_sigdig,
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

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

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
            "s2v_model_path": s2v_model_path,
            "s2v_library_path": s2v_library_path,
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
            "acquisition_type": "DDA" if polarity == "positive" else "DDA",
            "charge": dataset_charge,
            "significant_digits": prep_sigdig,
            "name": dataset_name,
            "output_folder": dataset_output_folder,
        }
        fingerprint_parameters = {
            "fp_type": fp_type,
            "threshold": fp_threshold,
        }

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
            run_status,
            load_status,
            clustered_smiles_data,
            optimized_motifs_data,
            lda_dict,
            spectra_data,
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

        load_status = dbc.Alert(f"Selected Results File: {results_filename}\nResults loaded successfully!",
                                color="success")

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


# Callback to create Cytoscape elements
@app.callback(
    Output("cytoscape-network-container", "children"),
    Input("optimized-motifs-store", "data"),
    Input("clustered-smiles-store", "data"),
    Input("tabs", "value"),
    Input("edge-intensity-threshold", "value"),
    Input("toggle-loss-edge", "value"),
    Input("cytoscape-layout-dropdown", "value"),
)
def update_cytoscape(optimized_motifs_data, clustered_smiles_data, active_tab, edge_intensity_threshold,
                     toggle_loss_edge, layout_choice):
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

    # Convert the checkbox list into a boolean
    show_loss_edge = ("show_loss_edge" in toggle_loss_edge)

    elements = create_cytoscape_elements(
        spectra,
        smiles_clusters,
        intensity_threshold=edge_intensity_threshold,
        show_loss_edge=show_loss_edge
    )

    # Use the selected layout from the dropdown.
    cytoscape_component = cyto.Cytoscape(
        id="cytoscape-network",
        elements=elements,
        style={"width": "100%", "height": "100%"},
        layout={"name": layout_choice, "animate": True},
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
                "selector": 'node[type="loss"]',
                "style": {
                    "background-color": "#FFD700",
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


def create_cytoscape_elements(spectra, smiles_clusters, intensity_threshold=0.05, show_loss_edge=False):
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
            if intensity < intensity_threshold:
                continue
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
                if loss_intensity < intensity_threshold:
                    continue
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
                                "type": "loss",
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
                # Conditionally re-add the line from loss node to fragment node if user wants it
                if show_loss_edge:
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


@app.callback(
    Output("molecule-images", "children"),
    Input("cytoscape-network", "tapNodeData"),
    State("clustered-smiles-store", "data"),
)
def display_node_data_on_click(tap_node_data, clustered_smiles_data):
    if not tap_node_data:
        raise PreventUpdate

    node_type = tap_node_data.get("type", "")
    node_id = tap_node_data.get("id", "")

    # Only do something if user clicks on a "motif" node:
    if node_type == "motif":
        motif_index_str = node_id.replace("motif_", "")
        try:
            motif_index = int(motif_index_str)
        except ValueError:
            raise PreventUpdate

        # Grab the SMILES cluster for this motif
        if not clustered_smiles_data or motif_index >= len(clustered_smiles_data):
            return html.Div(
                "No SMILES found for this motif."
            )

        smiles_list = clustered_smiles_data[motif_index]
        if not smiles_list:
            return html.Div(
                "This motif has no associated SMILES."
            )

        mols = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mols.append(mol)
            except Exception:
                continue

        if not mols:
            return html.Div("No valid RDKit structures for these SMILES.")

        # Create the grid image
        grid_img = MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(200, 200),
            legends=[f"Match {i + 1}" for i in range(len(mols))],
            returnPNG=True
        )
        encoded = base64.b64encode(grid_img).decode("utf-8")

        # Return an <img> with the PNG
        return html.Img(src="data:image/png;base64," + encoded, style={"margin": "10px"})

    # Otherwise (e.g. if user clicks a fragment/loss), do nothing special
    raise PreventUpdate


# -------------------------------- RANKINGS & DETAILS --------------------------------

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
    for motif in motifs:
        avg_probability = np.mean(motif_probabilities[motif]) if motif_probabilities[motif] else 0
        avg_overlap = np.mean(motif_overlap_scores[motif]) if motif_overlap_scores[motif] else 0
        md.append((motif, motif_degrees[motif], avg_probability, avg_overlap))
    md.sort(key=lambda x: x[1], reverse=True)
    return md


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
            {'name': 'Average Doc-Topic Probability', 'id': 'Average Doc-Topic Probability', 'type': 'numeric',
             'format': {'specifier': '.4f'}},
            {'name': 'Average Overlap Score', 'id': 'Average Overlap Score', 'type': 'numeric',
             'format': {'specifier': '.4f'}},
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


@app.callback(
    Output('probability-filter-display', 'children'),
    Input('probability-filter', 'value')
)
def display_prob_filter(prob_filter_range):
    return f"Showing features with probability between {prob_filter_range[0]:.2f} and {prob_filter_range[1]:.2f}"


@app.callback(
    Output('doc-topic-filter-display', 'children'),
    Input('doc-topic-filter', 'value')
)
def display_doc_topic_filter(value_range):
    return f"Filtering docs with motif probability between {value_range[0]:.2f} and {value_range[1]:.2f}"


@app.callback(
    Output('overlap-filter-display', 'children'),
    Input('overlap-filter', 'value')
)
def display_overlap_filter(overlap_range):
    return f"Filtering docs with overlap score between {overlap_range[0]:.2f} and {overlap_range[1]:.2f}"


@app.callback(
    Output('motif-details-title', 'children'),
    Output('motif-spec2vec-container', 'children'),
    Output('motif-features-container', 'children'),
    Output('motif-documents-container', 'children'),
    Output('motif-spectra-ids-store', 'data'),
    Output('spectra-table', 'data'),
    Output('spectra-table', 'columns'),
    Input('selected-motif-store', 'data'),
    Input('probability-filter', 'value'),
    Input('doc-topic-filter', 'value'),
    Input('overlap-filter', 'value'),
    State('lda-dict-store', 'data'),
    State('clustered-smiles-store', 'data'),
    State('spectra-store', 'data'),
    prevent_initial_call=True,
)
def update_motif_details(selected_motif, beta_range, theta_range, overlap_range,
                         lda_dict_data, clustered_smiles_data, spectra_data):
    """
    Populates the Motif Details tab:
      - Title (motif-details-title)
      - Spec2Vec container (motif-spec2vec-container)
      - Features container (motif-features-container) => includes:
          * Table of motif features (fragments/losses) filtered by Probability
          * Barplot: “Proportion of Total Intensity in Motif”
          * Barplot: “Counts of Features in Documents” across the entire dataset
      - Documents container (motif-documents-container) => doc table
      - Also returns doc table data/columns and the doc’s SpecIndex for next/prev controls
    """
    if not selected_motif or not lda_dict_data:
        return '', '', '', '', [], [], []

    motif_name = selected_motif

    # Filter Beta (Topic-Word Probability)
    motif_data = lda_dict_data['beta'].get(motif_name, {})
    filtered_motif_data = {
        f: p for f, p in motif_data.items()
        if beta_range[0] <= p <= beta_range[1]
    }
    total_prob = sum(filtered_motif_data.values())

    # Build a mini table for these features
    feature_table = pd.DataFrame({
        'Feature': filtered_motif_data.keys(),
        'Probability': filtered_motif_data.values(),
    }).sort_values(by='Probability', ascending=False)

    feature_table_component = dash.dash_table.DataTable(
        data=feature_table.to_dict('records'),
        columns=[
            {'name': 'Feature', 'id': 'Feature'},
            {'name': 'Probability', 'id': 'Probability', 'type': 'numeric', 'format': {'specifier': '.4f'}},
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10,
    )

    # Bar Chart: Proportion in Motif vs. Overall
    features_in_motif = list(filtered_motif_data.keys())
    total_feature_probs = {ft: 0.0 for ft in features_in_motif}
    for m_name, feature_probs in lda_dict_data['beta'].items():
        for ft, val in feature_probs.items():
            if ft in total_feature_probs:
                total_feature_probs[ft] += val

    barplot1_df = pd.DataFrame({
        'Feature': features_in_motif,
        'Probability in Motif': [filtered_motif_data[ft] for ft in features_in_motif],
        'Total Probability': [total_feature_probs[ft] for ft in features_in_motif],
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
        title='Proportion of Total Intensity Explained by This Motif (Top 10 Features)'
    )
    barplot1_fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Probability',
        yaxis_title='Feature',
        legend_title='',
    )

    # Bar Chart: Counts of Features in Documents
    # This second barplot shows how many docs in the ENTIRE dataset have each of these features
    feature_counts = {ft: 0 for ft in features_in_motif}
    for doc_name, w_counts in lda_dict_data['corpus'].items():
        for ft in features_in_motif:
            if ft in w_counts:
                feature_counts[ft] += 1

    barplot2_df = pd.DataFrame({
        'Feature': features_in_motif,
        'Count': [feature_counts[ft] for ft in features_in_motif],
    })
    barplot2_df = barplot2_df.sort_values(by='Count', ascending=False).head(10)
    barplot2_fig = px.bar(
        barplot2_df,
        x='Count',
        y='Feature',
        orientation='h',
        title='Counts of Features in Documents (Entire Dataset)',
    )
    barplot2_fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Number of Documents',
        yaxis_title='Feature',
    )

    # Spec2Vec Matching Results
    motif_idx = 0
    if motif_name.startswith('motif_'):
        motif_idx_str = motif_name.replace('motif_', '')
        try:
            motif_idx = int(motif_idx_str)
        except ValueError:
            pass

    spec2vec_container = []
    if clustered_smiles_data and motif_idx < len(clustered_smiles_data):
        smiles_list = clustered_smiles_data[motif_idx]
        if smiles_list:
            spec2vec_container.append(html.H5('Spec2Vec Matching Results'))
            mols = []
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mols.append(mol)
                except:
                    pass
            if mols:
                grid_img = MolsToGridImage(
                    mols,
                    molsPerRow=4,
                    subImgSize=(200, 200),
                    legends=[f"Match {i + 1}" for i in range(len(mols))],
                    returnPNG=True
                )
                encoded = base64.b64encode(grid_img).decode("utf-8")
                spec2vec_container.append(html.Img(
                    src="data:image/png;base64," + encoded,
                    style={"margin": "10px"}
                ))

    # Doc-Topic Probability / Overlap filter & doc table
    doc2spec_index = lda_dict_data["doc_to_spec_index"]
    docs_for_this_motif = []
    for doc_name, topic_probs in lda_dict_data['theta'].items():
        doc_topic_prob = topic_probs.get(motif_name, 0.0)
        if doc_topic_prob <= 0:
            continue
        overlap_score = lda_dict_data['overlap_scores'][doc_name].get(motif_name, 0.0)

        if (theta_range[0] <= doc_topic_prob <= theta_range[1]) and (
                overlap_range[0] <= overlap_score <= overlap_range[1]):
            real_idx = -1
            doc_idx_str = doc_name.replace("spec_", "")
            if doc_idx_str in doc2spec_index:
                real_idx = doc2spec_index[doc_idx_str]
            # gather metadata
            precursor_mz = None
            retention_time = None
            feature_id = None
            collision_energy = None
            ionmode = None
            ms_level = None
            scans = None
            if real_idx != -1 and real_idx < len(spectra_data):
                sp_meta = spectra_data[real_idx]['metadata']
                precursor_mz = sp_meta.get('precursor_mz')
                retention_time = sp_meta.get('retention_time')
                feature_id = sp_meta.get('feature_id')
                collision_energy = sp_meta.get('collision_energy')
                ionmode = sp_meta.get('ionmode')
                ms_level = sp_meta.get('ms_level')
                scans = sp_meta.get('scans')

            docs_for_this_motif.append({
                'DocName': doc_name,
                'SpecIndex': real_idx,
                'FeatureID': feature_id,
                'Scans': scans,
                'PrecursorMz': precursor_mz,
                'RetentionTime': retention_time,
                'CollisionEnergy': collision_energy,
                'IonMode': ionmode,
                'MsLevel': ms_level,
                'Doc-Topic Probability': doc_topic_prob,
                'Overlap Score': overlap_score,
            })

    # Create output df for table
    doc_cols = [
        'DocName', 'SpecIndex', 'FeatureID', 'Scans', 'PrecursorMz', 'RetentionTime',
        'CollisionEnergy', 'IonMode', 'MsLevel', 'Doc-Topic Probability', 'Overlap Score'
    ]
    docs_df = pd.DataFrame(docs_for_this_motif, columns=doc_cols)
    if not docs_df.empty:
        # Sort only if not empty
        docs_df = docs_df.sort_values(by='Doc-Topic Probability', ascending=False)

    table_data = docs_df.to_dict('records')
    table_columns = [
        {'name': 'DocName', 'id': 'DocName'},
        {'name': 'SpecIndex', 'id': 'SpecIndex', 'type': 'numeric'},
        {'name': 'FeatureID', 'id': 'FeatureID'},
        {'name': 'Scans', 'id': 'Scans'},
        {'name': 'PrecursorMz', 'id': 'PrecursorMz', 'type': 'numeric', 'format': {'specifier': '.4f'}},
        {'name': 'RetentionTime', 'id': 'RetentionTime', 'type': 'numeric', 'format': {'specifier': '.2f'}},
        {'name': 'CollisionEnergy', 'id': 'CollisionEnergy'},
        {'name': 'IonMode', 'id': 'IonMode'},
        {'name': 'MsLevel', 'id': 'MsLevel'},
        {'name': 'Doc-Topic Probability', 'id': 'Doc-Topic Probability', 'type': 'numeric',
         'format': {'specifier': '.4f'}},
        {'name': 'Overlap Score', 'id': 'Overlap Score', 'type': 'numeric', 'format': {'specifier': '.4f'}},
    ]

    # Provide the list of SpecIndex for next/prev
    spectra_ids = docs_df['SpecIndex'].tolist() if not docs_df.empty else []

    # Return the updated content
    motif_title = f"Motif Details: {motif_name}"

    # (1) Spec2Vec container
    spec2vec_div = html.Div(spec2vec_container)

    # (2) Features container => feature table, barplot1, then the “Counts of Features in Documents” barplot
    features_div = html.Div([
        # Table of motif features
        html.Div([
            html.H5("Motif Features Table"),
            feature_table_component,
            html.P(f"Total Probability (Filtered): {total_prob:.4f}")
        ]),
        html.H5("Proportion of Features in Motif vs. Overall"),
        dcc.Graph(figure=barplot1_fig),
        html.H5("Counts of Features in Documents (Entire Dataset)"),
        dcc.Markdown(
            """
            This shows how many documents in the entire dataset contain each selected feature. 
            It does **not** consider the doc-topic or overlap filters; it's purely a global count.
            """
        ),
        dcc.Graph(figure=barplot2_fig),
    ])

    # (3) Documents container => doc table only
    docs_div = html.Div([
        html.P("Documents containing this motif are listed below (affected by doc-topic & overlap filters)."),
    ])

    return (
        motif_title,  # Motif Title
        spec2vec_div,  # (A) Spec2Vec Container
        features_div,  # (B) Features container
        docs_div,  # (C) Document container
        spectra_ids,  # motif-spectra-ids-store
        table_data,  # Data for dash_table
        table_columns,  # Columns for dash_table
    )


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

    elif triggered_id in ['selected-motif-store', 'motif-spectra-ids-store']:
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
        # Get the actual spectrum dictionary
        spectrum_dict = spectra_data[spectrum_id]

        spectrum = Spectrum(
            mz=np.array(spectrum_dict['mz']),
            intensities=np.array(spectrum_dict['intensities']),
            metadata=spectrum_dict['metadata'],
        )

        motif_data = lda_dict_data['beta'].get(selected_motif, {})
        filtered_motif_data = {
            feature: prob for feature, prob in motif_data.items()
            if probability_range[0] <= prob <= probability_range[1]
        }

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

        spectrum_df = pd.DataFrame({
            'mz': spectrum.peaks.mz,
            'intensity': spectrum.peaks.intensities,
        })

        tolerance = 0.1
        spectrum_df['is_motif'] = False
        for mz_val in motif_mz_values:
            mask = np.abs(spectrum_df['mz'] - mz_val) <= tolerance
            spectrum_df.loc[mask, 'is_motif'] = True

        colors = ['#DC143C' if is_motif else '#B0B0B0'
                  for is_motif in spectrum_df['is_motif']]

        parent_ion_present = False
        parent_ion_mz = None
        parent_ion_intensity = None
        if 'precursor_mz' in spectrum.metadata:
            try:
                parent_ion_mz = float(spectrum.metadata['precursor_mz'])
                parent_ion_intensity = float(spectrum.metadata.get('parent_intensity', spectrum_df['intensity'].max()))
                parent_ion_present = True
            except (ValueError, TypeError):
                parent_ion_present = False

        fig = go.Figure()
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

        if "losses" in spectrum.metadata and parent_ion_present:
            precursor_mz = parent_ion_mz
            for loss_item in spectrum.metadata["losses"]:
                loss_mz = loss_item["loss_mz"]
                loss_intensity = loss_item["loss_intensity"]
                if not any(abs(loss_mz - val) <= tolerance for val in motif_loss_values):
                    continue

                corresponding_frag_mz = precursor_mz - loss_mz
                frag_mask = (np.abs(spectrum_df['mz'] - corresponding_frag_mz) <= tolerance)
                if not frag_mask.any():
                    continue

                frag_subset = spectrum_df.loc[frag_mask]
                if frag_subset.empty:
                    continue

                closest_frag_mz = frag_subset['mz'].iloc[0]
                closest_frag_intensity = frag_subset['intensity'].iloc[0]

                fig.add_shape(
                    type="line",
                    x0=closest_frag_mz,
                    y0=closest_frag_intensity,
                    x1=precursor_mz,
                    y1=closest_frag_intensity,
                    line=dict(color="green", width=2, dash="dash"),
                )
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


if __name__ == "__main__":
    app.run_server(debug=True)
