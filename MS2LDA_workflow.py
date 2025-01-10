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

from dash import dash_table
from dash import html, dcc, Input, Output, State
from matchms import Spectrum, Fragments
from rdkit.Chem import MolFromSmiles

from MS2LDA.Add_On.Spec2Vec.annotation import (
    load_s2v_and_library,
    get_library_matches,
    calc_embeddings,
    calc_similarity,
)
from MS2LDA.Add_On.Spec2Vec.annotation_refined import (
    hit_clustering,
    optimize_motif_spectrum,
)
from MS2LDA.Visualisation.ldadict import generate_corpusjson_from_tomotopy
from MS2LDA.running import generate_motifs

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
        # Include all components in the main layout
        # Group components for each tab in a Div
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
):
    """
    This callback either (1) runs MS2LDA from scratch on the uploaded data (when Run Analysis clicked),
    or (2) loads precomputed results from a JSON file (when Load Results clicked).

    We now avoid re-deriving tokens that mismatch the trained model.
    Instead, we rely on the doc words already in `trained_ms2lda` for the LDA dictionary.
    """
    import base64, tempfile, os, json
    import dash
    from dash import html
    from dash.exceptions import PreventUpdate
    import tomotopy as tp
    from rdkit.Chem import MolFromSmiles
    import numpy as np
    from matchms import Spectrum, Fragments
    from MS2LDA.Preprocessing.load_and_clean import load_mgf, load_mzml, load_msp, clean_spectra
    from MS2LDA.run import filetype_check
    from MS2LDA.Visualisation.ldadict import generate_corpusjson_from_tomotopy
    from MS2LDA.Add_On.Spec2Vec.annotation_refined import hit_clustering, optimize_motif_spectrum
    from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, load_s2v_and_library, calc_similarity, get_library_matches
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

    # 1) -------------------- If RUN-BUTTON was clicked --------------------
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

        # We'll call MS2LDA.run exactly as in your notebook:
        preprocessing_parameters = {
            "min_mz": 0,
            "max_mz": 2000,
            "max_frags": 1000,
            "min_frags": 5,
            "min_intensity": 0.01,
            "max_intensity": 1,
        }
        convergence_parameters = {
            "step_size": 50,
            "window_size": 10,
            "threshold": 0.005,
            "type": "perplexity_history",
        }
        annotation_parameters = {
            "criterium": "best",
            "cosine_similarity": 0.90,
            "n_mols_retrieved": top_n,
        }
        model_parameters = {
            "rm_top": 0,
            "min_cf": 0,
            "min_df": 3,
            "alpha": 0.6,
            "eta": 0.01,
            "seed": 42,
        }
        train_parameters = {
            "parallel": 1,
            "workers": 0,
        }
        dataset_parameters = {
            "acquisition_type": "DDA",
            "charge": 1,
            "name": "ms2lda_dashboard_run",
            "output_folder": "ms2lda_results",
        }
        fingerprint_parameters = {
            "fp_type": "rdkit",
            "threshold": 0.8,
        }
        motif_parameter = 50
        n_iterations = 100

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

        # 1.2) Now load the *trained* tomotopy model from disk, so we can build the LDA dictionary with the EXACT tokens:
        trained_ms2lda = tp.LDAModel.load(os.path.join(dataset_parameters["output_folder"], "ms2lda.bin"))

        # 1.3) Reconstruct the "documents" from the actual model's doc words:
        #      (We do NOT re-run features_to_words ourselves; we just read from model docs.)
        documents = []
        for doc in trained_ms2lda.docs:
            tokens = [trained_ms2lda.vocabs[word_id] for word_id in doc.words]
            documents.append(tokens)

        # 1.4) We also need doc_metadata. By default, the code sets them as "spec_0", "spec_1", etc.
        #      If you want to preserve actual metadata, you might do so separately. Here we do a simple approach:
        doc_metadata = {}
        for i, doc in enumerate(trained_ms2lda.docs):
            doc_name = f"spec_{i}"
            doc_metadata[doc_name] = {"placeholder": f"Doc {i}"}

        # 1.5) We'll pass "spectra=None" to generate_corpusjson_from_tomotopy, letting it handle doc naming from doc_metadata keys:
        from MS2LDA.Visualisation.ldadict import generate_corpusjson_from_tomotopy
        lda_dict = generate_corpusjson_from_tomotopy(
            model=trained_ms2lda,
            documents=documents,
            spectra=None,  # we skip re-cleaning for token consistency
            doc_metadata=doc_metadata,
            filename=None,
        )

        # 1.6) But we *do* want a store of the final cleaned spectra for the network tab (images, etc.).
        #      We'll do that once. It's OK if it doesn't match the tokenization exactly, because we won't feed them into the LDA again.
        loaded_spectra = filetype_check(tmp_file_path)
        cleaned_spectra = clean_spectra(loaded_spectra, preprocessing_parameters=preprocessing_parameters)

        # Turn `optimized_motifs` into Python dict form for dash store:
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

        # Convert optimized motifs:
        optimized_motifs_data = [spectrum_to_dict(m) for m in optimized_motifs]
        # Convert the real "cleaned_spectra" so we can show them in the network:
        spectra_data = [spectrum_to_dict(s) for s in cleaned_spectra]

        # For "clustered_smiles_data", we gather the short_annotation from each motif:
        clustered_smiles_data = []
        for mot in optimized_motifs:
            ann = mot.get("short_annotation")
            if isinstance(ann, list):
                clustered_smiles_data.append(ann)
            elif ann is None:
                clustered_smiles_data.append([])
            else:
                # if single string
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

    # 2) -------------------- If LOAD-RESULTS-BUTTON was clicked --------------------
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

        # Extract from JSON
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
        # Add loss to metadata to link losses to fragments during visualization
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
        # Hide the Cytoscape component when not on the results tab or no data
        return ""

    # Reconstruct spectra from stored data
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
        style={"width": "100%", "height": "100%"},  # Full height of the column
        layout={"name": "cose", "animate": False},  # Set animate to False for faster rendering
        stylesheet=[
            # Motif Nodes
            {
                "selector": 'node[type="motif"]',
                "style": {
                    "background-color": "#00008B",  # Dark Blue
                    "label": "data(label)",
                    "text-background-color": "white",
                    "text-background-opacity": 0.7,
                    "text-background-padding": "3px",
                    "text-background-shape": "roundrectangle",
                    "text-border-color": "black",
                    "text-border-width": 1,
                    "text-valign": "top",
                    "text-halign": "center",
                    "color": "black",  # Text color
                    "font-size": "10px",
                },
            },
            # Fragment and Loss Nodes
            {
                "selector": 'node[type="fragment"]',
                "style": {
                    "background-color": "#008000",  # Green
                    "label": "data(label)",
                    "text-background-color": "white",
                    "text-background-opacity": 0.7,
                    "text-background-padding": "3px",
                    "text-background-shape": "roundrectangle",
                    "text-border-color": "black",
                    "text-border-width": 1,
                    "text-valign": "top",
                    "text-halign": "center",
                    "color": "black",  # Text color
                    "font-size": "8px",
                },
            },
            # Edges
            {
                "selector": "edge",
                "style": {
                    "line-color": "red",
                    "opacity": 0.5,
                    "width": "mapData(weight, 0, 1, 1, 10)",  # Adjust based on actual weight range
                    "target-arrow-shape": "none",
                    "curve-style": "bezier",
                },
            },
            # General Node Style (if needed)
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

            # Create molecules, making sure to filter out None results
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
                # Create grid image with legends
                legends = [f"Match {i + 1}" for i in range(len(mols))]
                from rdkit.Chem import Draw
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=1,  # Set to 1 for vertical stacking
                    subImgSize=(200, 200),
                    legends=legends,
                    returnPNG=True  # This is important!
                )

                # Image is already in PNG format, just need to encode
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

    return ""  # Return empty for non-motif nodes

def create_cytoscape_elements(spectra, smiles_clusters):
    elements = []
    colors = [
        "#FF5733",
        "#33FF57",
        "#3357FF",
        "#F333FF",
        "#FF33A8",
        "#33FFF5",
        "#F5FF33",
        "#A833FF",
        "#FF8633",
        "#33FF86",
    ]  # Add more colors if needed

    # Sets to keep track of created fragment and loss nodes
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

        # Add fragment nodes and edges
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
                        "weight": intensity,  # Use intensity as weight
                    }
                }
            )

        # Add loss nodes and edges
        if spectrum.losses is not None:
            precursor_mz = float(spectrum.metadata.get('precursor_mz', 0))
            for loss_data in spectrum.metadata.get("losses", []):
                loss_mz = loss_data["loss_mz"]
                loss_intensity = loss_data["loss_intensity"]
                corresponding_frag_mz = precursor_mz - loss_mz
                rounded_frag_mz = round(corresponding_frag_mz, 2)
                frag_node = f"frag_{rounded_frag_mz}"

                # Ensure the corresponding fragment node exists
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
                            "weight": loss_intensity,  # Use loss intensity as weight
                        }
                    }
                )
                # Optionally, connect loss to corresponding fragment
                elements.append(
                    {
                        "data": {
                            "source": loss_node,
                            "target": frag_node,
                            "weight": loss_intensity,  # Use loss intensity as weight
                        }
                    }
                )

    return elements



# Helper function to compute motif degrees
def compute_motif_degrees(lda_dict, p_thresh, o_thresh):
    """
    Computes the degree, average document-to-topic probability, and average overlap score for each motif.

    Parameters:
    - lda_dict (dict): The LDA dictionary containing 'beta', 'theta', and 'overlap_scores'.
    - p_thresh (float): The probability threshold (lower bound).
    - o_thresh (float): The overlap threshold (lower bound).

    Returns:
    - list of tuples: Each tuple contains (motif, degree, average_probability, average_overlap).
    """
    motifs = lda_dict["beta"].keys()
    motif_degrees = {m: 0 for m in motifs}
    motif_probabilities = {m: [] for m in motifs}  # Store probabilities for averaging
    motif_overlap_scores = {m: [] for m in motifs}
    docs = lda_dict["theta"].keys()

    for doc in docs:
        for motif, p in lda_dict["theta"][doc].items():
            if p >= p_thresh:  # Apply probability threshold
                o = lda_dict["overlap_scores"][doc].get(motif, 0.0)
                if o >= o_thresh:  # Apply overlap threshold
                    motif_degrees[motif] += 1
                    motif_probabilities[motif].append(p)  # Store probability
                    motif_overlap_scores[motif].append(o)

    md = []
    for motif in motifs:
        avg_probability = np.mean(motif_probabilities[motif]) if motif_probabilities[motif] else 0  # Calculate average probability
        avg_overlap = np.mean(motif_overlap_scores[motif]) if motif_overlap_scores[motif] else 0  # Calculate average overlap
        md.append((motif, motif_degrees[motif], avg_probability, avg_overlap))  # Add avg_probability to the tuple

    md.sort(key=lambda x: x[1], reverse=True)  # Sorting to show the most relevant motifs at the top
    return md

# Callback to update the Motif Rankings TABLE ONLY (Outputs to different container)
@app.callback(
    Output('motif-rankings-table-container', 'children'),  # Output to a container for the table
    Input('lda-dict-store', 'data'),
    Input('probability-thresh', 'value'),
    Input('overlap-thresh', 'value'),
    Input('tabs', 'value'),
)
def update_motif_rankings_table(lda_dict_data, probability_thresh, overlap_thresh, active_tab):
    """
    Updates the Motif Rankings table based on the provided thresholds and active tab.

    Parameters:
    - lda_dict_data (dict): The stored LDA dictionary data.
    - probability_thresh (list or float): The selected probability threshold range.
    - overlap_thresh (list or float): The selected overlap threshold range.
    - active_tab (str): The currently active tab.

    Returns:
    - Dash DataTable component or an empty string.
    """
    if active_tab != 'motif-rankings-tab' or not lda_dict_data:
        return ""

    # Correctly extract the lower bound from RangeSliders to use as thresholds
    p_thresh = probability_thresh[0] if isinstance(probability_thresh, list) else probability_thresh
    o_thresh = overlap_thresh[0] if isinstance(overlap_thresh, list) else overlap_thresh

    # Compute motif degrees with the provided thresholds
    motif_degree_list = compute_motif_degrees(lda_dict_data, p_thresh, o_thresh)

    # Prepare DataFrame
    df = pd.DataFrame(motif_degree_list, columns=[
        'Motif',
        'Degree',
        'Average Doc-Topic Probability',
        'Average Overlap Score'
    ])

    # *** Remove the problematic filter ***
    # df = df[df['Degree'] > 0]  # This line has been removed to include all motifs

    # Add motif annotations if available
    motif_annotations = {}
    if 'topic_metadata' in lda_dict_data:
        for motif, metadata in lda_dict_data['topic_metadata'].items():
            motif_annotations[motif] = metadata.get('annotation', '')

    df['Annotation'] = df['Motif'].map(motif_annotations)

    # Style to make the 'Motif' column look clickable
    style_data_conditional = [
        {
            'if': {'column_id': 'Motif'},
            'cursor': 'pointer',
            'textDecoration': 'underline',
            'color': 'blue',
        },
    ]

    # Create DataTable with the filtered and sorted data
    table = dash_table.DataTable(
        id='motif-rankings-table',
        data=df.to_dict('records'),  # Use the entire DataFrame without filtering out zero-degree motifs
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
    Input('probability-filter', 'value'),  # Input for probability filter
    State('lda-dict-store', 'data'),
    State('clustered-smiles-store', 'data'),
    State('spectra-store', 'data'),
    prevent_initial_call=True,
)
def update_motif_details(selected_motif, probability_range, lda_dict_data, clustered_smiles_data, spectra_data):
    if not selected_motif or not lda_dict_data:
        # Return empty placeholders
        return '', [], [], [], ''  # Removed 'probability-filter-display' from outputs

    motif_name = selected_motif
    motif_data = lda_dict_data['beta'].get(motif_name, {})

    # Apply probability filter to motif_data
    filtered_motif_data = {
        feature: prob for feature, prob in motif_data.items()
        if probability_range[0] <= prob <= probability_range[1]
    }

    # Display Probability Filter range is handled by its own callback

    total_prob = sum(filtered_motif_data.values())  # Calculate total probability after filtering
    content = []

    # Features table (updated to use filtered data)
    feature_table = pd.DataFrame({
        'Feature': filtered_motif_data.keys(),  # Filtered features
        'Probability': filtered_motif_data.values(),  # Filtered probabilities
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
    content.append(html.P(f'Total Probability (Filtered): {total_prob:.4f}'))  # Updated label

    # Prepare spectra data
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

    # Prepare data and columns for spectra-table
    spectra_table_data = spectra_df.to_dict('records')
    spectra_table_columns = [
        {'name': 'Spectrum', 'id': 'Spectrum'},
        {'name': 'Probability', 'id': 'Probability', 'type': 'numeric', 'format': {'specifier': '.4f'}},
        {'name': 'Overlap Score', 'id': 'Overlap Score', 'type': 'numeric', 'format': {'specifier': '.4f'}},
    ]

    # Compute data for bar plots
    features_in_motif = list(filtered_motif_data.keys())
    total_feature_probs = {feature: 0.0 for feature in features_in_motif}
    for motif, feature_probs in lda_dict_data['beta'].items():
        for feature, prob in feature_probs.items():
            if feature in total_feature_probs:
                total_feature_probs[feature] += prob

    barplot1_df = pd.DataFrame({
        'Feature': features_in_motif,
        'Probability in Motif': [filtered_motif_data[feature] for feature in features_in_motif],
        'Total Probability': [total_feature_probs[feature] for feature in features_in_motif],
    })

    # Limit to top 10 items based on 'Probability in Motif'
    barplot1_df = barplot1_df.sort_values(by='Probability in Motif', ascending=False).head(10)

    # Melt the dataframe to long format for grouped bar chart
    barplot1_df_long = pd.melt(
        barplot1_df,
        id_vars='Feature',
        value_vars=['Total Probability', 'Probability in Motif'],
        var_name='Type',
        value_name='Probability'
    )

    # Create horizontal bar plot
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

    # Second bar plot data
    feature_counts = {feature: 0 for feature in features_in_motif}
    for doc in spectra_df['Spectrum']:
        # Assuming 'corpus' is a dictionary mapping doc to features
        # Replace 'corpus' with the actual key if different
        doc_features = lda_dict_data['corpus'].get(doc, {}).keys()
        for feature in features_in_motif:
            if feature in doc_features:
                feature_counts[feature] += 1

    barplot2_df = pd.DataFrame({
        'Feature': features_in_motif,
        'Count': [feature_counts[feature] for feature in features_in_motif],
    })

    # Limit to top 10 items based on 'Count'
    barplot2_df = barplot2_df.sort_values(by='Count', ascending=False).head(10)

    # Create horizontal bar plot
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

    # Spec2Vec Matching Results
    motif_index = int(motif_name.replace('motif_', ''))
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
                legends = [f"Match {i + 1}" for i in range(len(mols))]
                from rdkit.Chem import Draw
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
    # Get spectra IDs
    spectra_ids = spectra_df['Spectrum'].tolist()
    return f"Motif Details: {motif_name}", content, spectra_ids, spectra_table_data, spectra_table_columns

# **FIX FOR INITIAL SPECTRUM DISPLAY AND Syncing Selected Rows with Spectrum Index**
# Combine updating 'selected-spectrum-index' and 'spectra-table.selected_rows' into a single callback to avoid dependency cycles
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
            return current_index, dash.no_update  # Already at last spectrum

    elif triggered_id == 'prev-spectrum':
        if motif_spectra_ids and current_index > 0:
            new_index = current_index - 1
            return new_index, [new_index]
        else:
            return current_index, dash.no_update  # Already at first spectrum

    elif triggered_id == 'selected-motif-store' or triggered_id == 'motif-spectra-ids-store':
        # Reset index when motif changes or spectra IDs are updated
        return 0, [0]

    else:
        return current_index, dash.no_update

# Callback to update the spectrum plot based on selected-spectrum-index and probability filter
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
        # Validate selected_index
        if selected_index is None or selected_index < 0 or selected_index >= len(spectra_ids):
            return html.Div("Selected spectrum index is out of range.")

        spectrum_id = spectra_ids[selected_index]

        # Retrieve the spectrum data based on spectrum_id
        spectrum_dict = next((s for s in spectra_data if s['metadata']['id'] == spectrum_id), None)
        if not spectrum_dict:
            return html.Div("Spectrum data not found.")

        # Reconstruct the Spectrum object
        spectrum = Spectrum(
            mz=np.array(spectrum_dict['mz']),
            intensities=np.array(spectrum_dict['intensities']),
            metadata=spectrum_dict['metadata'],
        )

        # Apply probability filtering to motif features
        motif_data = lda_dict_data['beta'].get(selected_motif, {})
        filtered_motif_data = {
            feature: prob for feature, prob in motif_data.items()
            if probability_range[0] <= prob <= probability_range[1]
        }

        # Extract motif-associated features (fragments and losses) from filtered data
        motif_mz_values = []
        motif_loss_values = []
        for feature in filtered_motif_data.keys():
            if feature.startswith('frag@'):
                try:
                    mz_value = float(feature.replace('frag@', ''))
                    motif_mz_values.append(mz_value)
                except ValueError:
                    continue
            elif feature.startswith('loss@'):
                try:
                    loss_value = float(feature.replace('loss@', ''))
                    motif_loss_values.append(loss_value)
                except ValueError:
                    continue

        # Create DataFrame for plotting
        spectrum_df = pd.DataFrame({
            'mz': spectrum.peaks.mz,
            'intensity': spectrum.peaks.intensities,
        })

        # Identify motif peaks within a tolerance only if they appear in the filtered motif data
        tolerance = 0.1
        spectrum_df['is_motif'] = False
        for mz in motif_mz_values:
            mask = np.abs(spectrum_df['mz'] - mz) <= tolerance
            # Set is_motif = True only for these peaks
            spectrum_df.loc[mask, 'is_motif'] = True

        # Define colors based on whether the peak is a motif peak
        colors = ['#DC143C' if is_motif else '#B0B0B0' for is_motif in spectrum_df['is_motif']]

        # Prepare parent ion information
        parent_ion_present = False
        parent_ion_mz = None
        parent_ion_intensity = None
        if 'precursor_mz' in spectrum.metadata:
            try:
                parent_ion_mz = float(spectrum.metadata['precursor_mz'])
                # Retrieve parent ion intensity if available, else use max intensity
                if 'parent_intensity' in spectrum.metadata:
                    parent_ion_intensity = float(spectrum.metadata['parent_intensity'])
                else:
                    parent_ion_intensity = spectrum.intensities.max() if spectrum.intensities.size > 0 else 0
                parent_ion_present = True
            except (ValueError, TypeError):
                parent_ion_present = False

        # Create Plotly figure
        fig = go.Figure()

        # Add peaks trace
        fig.add_trace(go.Bar(
            x=spectrum_df['mz'],
            y=spectrum_df['intensity'],
            marker=dict(
                color=colors,
                line=dict(color='white', width=0),
            ),
            width=0.2,
            name='Peaks',
            hoverinfo='text',
            hovertext=[
                f"Motif Peak: {is_motif}<br>m/z: {mz:.2f}<br>Intensity: {intensity}"
                for mz, intensity, is_motif in zip(spectrum_df['mz'], spectrum_df['intensity'], spectrum_df['is_motif'])
            ],
            opacity=0.9,
        ))

        # Add parent ion if present
        if parent_ion_present and parent_ion_mz is not None and parent_ion_intensity is not None:
            fig.add_trace(go.Bar(
                x=[parent_ion_mz],
                y=[parent_ion_intensity],
                marker=dict(
                    color='#0000FF',  # Parent Ion Blue
                    line=dict(color='white', width=0),
                ),
                width=0.4,
                name='Parent Ion',
                hoverinfo='text',
                hovertext=[f"Parent Ion<br>m/z: {parent_ion_mz:.2f}<br>Intensity: {parent_ion_intensity}"],
                opacity=1.0,
            ))

        # Draw loss annotations only if they are in filtered_motif_data
        if "losses" in spectrum.metadata and parent_ion_present:
            precursor_mz = parent_ion_mz
            for loss_data in spectrum.metadata["losses"]:
                loss_mz = loss_data["loss_mz"]
                loss_intensity = loss_data["loss_intensity"]
                loss_feature = f"loss@{loss_mz:.4f}"  # Create a string like loss@XXX to match format

                # Check if this loss feature is in filtered_motif_data (and thus above threshold)
                # We'll allow some tolerance for floating point formatting
                # Instead of strict match, check by absolute difference
                # Motif_loss_values are floats from filtered_motif_data, so let's just compare difference:
                if not any(abs(loss_mz - val) <= tolerance for val in motif_loss_values):
                    continue

                # Calculate corresponding fragment m/z
                frag_mz = precursor_mz - loss_mz
                frag_mask = np.abs(spectrum.peaks.mz - frag_mz) <= tolerance
                if not np.any(frag_mask):
                    continue

                closest_frag_mz = spectrum.peaks.mz[frag_mask][0]
                closest_frag_intensity = spectrum.peaks.intensities[frag_mask][0]

                # Add horizontal dash line
                fig.add_shape(
                    type="line",
                    x0=closest_frag_mz,
                    y0=closest_frag_intensity,
                    x1=precursor_mz,
                    y1=closest_frag_intensity,
                    line=dict(
                        color="green",
                        width=2,
                        dash="dash",
                    ),
                )
                fig.add_annotation(
                    x=(closest_frag_mz + precursor_mz) / 2,
                    y=closest_frag_intensity,
                    text=f"-{loss_mz:.2f}",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="green"
                    ),
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

        graph_component = dcc.Graph(
            figure=fig,
            style={
                'width': '100%',
                'height': '600px',
                'margin': 'auto'
            }
        )

        return graph_component


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)