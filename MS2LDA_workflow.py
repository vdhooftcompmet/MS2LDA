import base64
import json
import os
import tempfile
from itertools import chain

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import numpy as np
import pandas as pd
from dash import html, dcc, Input, Output, State
from dash import dash_table
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme
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
from MS2LDA.running import generate_motifs
from MS2LDA.Visualisation.ldadict import generate_corpusjson_from_tomotopy
from MS2LDA.Preprocessing.generate_corpus import features_to_words, combine_features

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
        dbc.Tabs(
            [
                dbc.Tab(label="Run Analysis", tab_id="run-analysis-tab"),
                dbc.Tab(label="Load Results", tab_id="load-results-tab"),
                dbc.Tab(label="View Results", tab_id="results-tab"),
                dbc.Tab(
                    label="All Fragmentation Spectra and Mass2Motifs Matching Details",
                    tab_id="document-motif-table-tab",
                ),
                dbc.Tab(label="Spectrum Details", tab_id="spectrum-details-tab"),
                dbc.Tab(label="Motif Details", tab_id="motif-details-tab"),
            ],
            id="tabs",
            active_tab="run-analysis-tab",
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
                                html.Div(
                                    id="run-status", style={"marginTop": "20px"}
                                ),
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
                                html.Div(
                                    id="save-status", style={"marginTop": "20px"}
                                ),
                            ],
                            width=6,
                        )
                    ],
                    justify="center",
                )
            ],
            style={"display": "block"},  # Initially visible
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
            style={"display": "none"},  # Initially hidden
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
                                        "height": "600px",  # Fixed height for the network
                                    },
                                )
                            ],
                            width=8,  # Wider column for the network
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    id="molecule-images",
                                    style={
                                        "textAlign": "center",
                                        "marginTop": "20px",
                                        "overflowY": "auto",  # Scrollbar if content overflows
                                        "height": "600px",  # Match network height
                                        "padding": "10px",
                                        "backgroundColor": "#f8f9fa",  # Light background
                                        "borderRadius": "5px",
                                    },
                                ),
                            ],
                            width=4,  # Narrower column for chem diagrams
                        ),
                    ],
                    align="start",  # Align items to the top
                    className="g-3",  # Gap between columns
                )
            ],
            style={"display": "none"},  # Initially hidden
        ),
        html.Div(
            id="document-motif-table-tab-content",
            children=[
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H3(
                                            "All Fragmentation Spectra and Mass2Motifs Matching Details"
                                        ),
                                        html.P(
                                            "The following table lists all matchings of Fragmentation Spectra and Mass2Motifs extracted from the dataset."
                                        ),
                                        dash_table.DataTable(
                                            id="document-motif-table",
                                            columns=[],
                                            data=[],
                                            sort_action="native",
                                            filter_action="native",
                                            page_size=20,
                                            sort_by=[
                                                {
                                                    "column_id": "Probability",
                                                    "direction": "desc",
                                                }
                                            ],  # Default sort
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "minWidth": "100px",
                                                "width": "150px",
                                                "maxWidth": "200px",
                                                "whiteSpace": "normal",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {"column_id": "Document"},
                                                    "color": "blue",
                                                    "textDecoration": "underline",
                                                    "cursor": "pointer",
                                                },
                                                {
                                                    "if": {"column_id": "Motif"},
                                                    "color": "blue",
                                                    "textDecoration": "underline",
                                                    "cursor": "pointer",
                                                },
                                            ],
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            style={"display": "none"},  # Initially hidden
        ),
        html.Div(
            id="spectrum-details-tab-content",
            children=[
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H3("Spectrum Details"),
                                        html.Div(
                                            id="spectrum-details-content",
                                            children=[
                                                html.P(
                                                    "Spectrum details will be displayed here."
                                                )
                                            ],
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            style={"display": "none"},  # Initially hidden
        ),
        html.Div(
            id="motif-details-tab-content",
            children=[
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H3("Motif Details"),
                                        html.Div(
                                            id="motif-details-content",
                                            children=[
                                                html.P(
                                                    "Motif details will be displayed here."
                                                )
                                            ],
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            style={"display": "none"},  # Initially hidden
        ),
        # Hidden storage for data to be accessed by callbacks
        dcc.Store(id="clustered-smiles-store"),
        dcc.Store(id="optimized-motifs-store"),
        dcc.Store(id="lda-dict-store"),
        dcc.Store(id="doc-topic-dists-store"),
        dcc.Store(id="document-motif-data-store"),
        dcc.Store(id="clicked-spectrum-store"),  # New store for clicked spectrum
        dcc.Store(id="clicked-motif-store"),  # New store for clicked motif
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
    Output("document-motif-table-tab-content", "style"),
    Output("spectrum-details-tab-content", "style"),
    Output("motif-details-tab-content", "style"),
    Input("tabs", "active_tab"),
)
def toggle_tab_content(active_tab):
    run_style = {"display": "none"}
    load_style = {"display": "none"}
    results_style = {"display": "none"}
    document_motif_style = {"display": "none"}
    spectrum_details_style = {"display": "none"}
    motif_details_style = {"display": "none"}

    if active_tab == "run-analysis-tab":
        run_style = {"display": "block"}
    elif active_tab == "load-results-tab":
        load_style = {"display": "block"}
    elif active_tab == "results-tab":
        results_style = {"display": "block"}
    elif active_tab == "document-motif-table-tab":
        document_motif_style = {"display": "block"}
    elif active_tab == "spectrum-details-tab":
        spectrum_details_style = {"display": "block"}
    elif active_tab == "motif-details-tab":
        motif_details_style = {"display": "block"}

    return (
        run_style,
        load_style,
        results_style,
        document_motif_style,
        spectrum_details_style,
        motif_details_style,
    )


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


# Callback to handle Run Analysis and Load Results
@app.callback(
    Output("run-status", "children"),
    Output("load-status", "children"),
    Output("clustered-smiles-store", "data"),
    Output("optimized-motifs-store", "data"),
    Output("lda-dict-store", "data"),
    Output("doc-topic-dists-store", "data"),
    Output("document-motif-data-store", "data"),
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
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    else:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Initialize outputs
    run_status = dash.no_update
    load_status = dash.no_update
    clustered_smiles_data = dash.no_update
    optimized_motifs_data = dash.no_update
    lda_dict_data = dash.no_update
    doc_topic_dists_data = dash.no_update
    document_motif_data = dash.no_update  # New data

    if triggered_id == "run-button":
        # Handle Run Analysis
        if not data_contents:
            run_status = dbc.Alert(
                "Please upload a mass spectrometry data file.", color="danger"
            )
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

        # Decode the uploaded file
        try:
            content_type, content_string = data_contents.split(",")
            decoded = base64.b64decode(content_string)
        except Exception as e:
            run_status = dbc.Alert(
                f"Error decoding the uploaded file: {str(e)}", color="danger"
            )
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

        # Save the uploaded file to a temporary file
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(data_filename)[1]
            ) as tmp_file:
                tmp_file.write(decoded)
                tmp_file_path = tmp_file.name
        except Exception as e:
            run_status = dbc.Alert(
                f"Error saving the uploaded file: {str(e)}", color="danger"
            )
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

        try:
            # Generate motifs
            # Adjust the function call to get all necessary outputs
            (
                motif_spectra,
                convergence_curve,
                trained_ms2lda,
                feature_words,
                cleaned_spectra,
            ) = generate_motifs(tmp_file_path, n_motifs=n_motifs, iterations=100)

            # Generate lda_dict using the tomotopy model
            # Construct doc_metadata
            doc_metadata = {}
            for spectrum in cleaned_spectra:
                doc_name = spectrum.get("id")
                metadata = spectrum.metadata.copy()
                doc_metadata[doc_name] = metadata

            # Generate lda_dict
            lda_dict = generate_corpusjson_from_tomotopy(
                model=trained_ms2lda,
                documents=feature_words,
                spectra=cleaned_spectra,
                doc_metadata=doc_metadata,
                filename=None,  # Not saving to file here
            )

            # Load Spec2Vec model and library based on polarity
            if polarity == "positive":
                path_model = "MS2LDA/Add_On/Spec2Vec/model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model"
                path_library = (
                    "MS2LDA/Add_On/Spec2Vec/model_positive_mode/positive_s2v_library.pkl"
                )
            else:
                path_model = "MS2LDA/Add_On/Spec2Vec/model_negative_mode/150724_Spec2Vec_neg_CleanedLibraries.model"
                path_library = (
                    "MS2LDA/Add_On/Spec2Vec/model_negative_mode/negative_s2v_library.pkl"
                )

            # Annotate motifs
            s2v_similarity, library = load_s2v_and_library(path_model, path_library)

            # Calculate embeddings and similarity matrix
            motif_embeddings = calc_embeddings(s2v_similarity, motif_spectra)
            similarity_matrix = calc_similarity(motif_embeddings, library.embeddings)

            matching_settings = {
                "similarity_matrix": similarity_matrix,
                "library": library,
                "top_n": top_n,
                "unique_mols": unique_mols,
            }

            library_matches = get_library_matches(matching_settings)

            # Refine Annotation
            (
                clustered_spectra,
                clustered_smiles,
                clustered_scores,
            ) = hit_clustering(
                s2v_similarity,
                motif_spectra,
                library_matches,
                criterium="best",
            )

            # Optimize motifs
            optimized_motifs = []
            for motif_spec, spec_list, smiles_list in zip(
                motif_spectra, clustered_spectra, clustered_smiles
            ):
                opt_motif = optimize_motif_spectrum(
                    motif_spec, spec_list, smiles_list
                )
                optimized_motifs.append(opt_motif)

            # Store data in dcc.Store components
            clustered_smiles_data = clustered_smiles  # list of lists
            optimized_motifs_data = [spectrum_to_dict(s) for s in optimized_motifs]
            lda_dict_data = lda_dict  # Store lda_dict

            # Get document-topic distributions
            doc_topic_dists = []
            document_motif_data = []  # New list to store data for the table
            for idx, doc in enumerate(trained_ms2lda.docs):
                doc_name = cleaned_spectra[idx].get("id")  # Get the spectrum ID
                topic_dist = doc.get_topic_dist()
                doc_topic_dists.append(
                    {"doc_id": doc_name, "topic_dist": topic_dist.tolist(),}
                )

                # Get word-topic assignments
                words = doc.words
                topics = doc.topics
                word_topic_counts = (
                    {}
                )  # Dictionary to count words assigned to each topic
                for word_id, topic_id in zip(words, topics):
                    if topic_id not in word_topic_counts:
                        word_topic_counts[topic_id] = 0
                    word_topic_counts[topic_id] += 1

                total_words = len(words)
                # For each motif (topic), compute overlap score and get probability
                for topic_id in range(trained_ms2lda.k):
                    probability = topic_dist[topic_id]
                    if probability > 0:
                        words_in_topic = word_topic_counts.get(topic_id, 0)
                        overlap_score = (
                            words_in_topic / total_words if total_words > 0 else 0
                        )
                        # Get precursor mass, retention time, and annotation from spectrum metadata
                        spectrum = cleaned_spectra[idx]
                        precursor_mass = spectrum.get("precursor_mz", None)
                        retention_time = spectrum.get("retention_time", None)
                        annotation = spectrum.metadata.get("annotation", None)
                        # Append data to list
                        document_motif_data.append(
                            {
                                "Document": doc_name,
                                "Motif": f"motif_{topic_id}",
                                "Probability": probability,
                                "Overlap Score": overlap_score,
                                "Precursor Mass": precursor_mass,
                                "Retention Time": retention_time,
                                "Document Annotation": annotation,
                            }
                        )

            doc_topic_dists_data = doc_topic_dists

            run_status = dbc.Alert(
                "Analysis Completed Successfully! Switch to the 'View Results' or 'All Fragmentation Spectra and Mass2Motifs Matching Details' tab to view.",
                color="success",
            )

            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

        except Exception as e:
            run_status = dbc.Alert(
                f"An error occurred during analysis: {str(e)}", color="danger"
            )
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

    elif triggered_id == "load-results-button":
        # Handle Load Results
        if not results_contents:
            load_status = dbc.Alert(
                "Please upload a results JSON file.", color="danger"
            )
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

        # Decode the uploaded file
        try:
            content_type, content_string = results_contents.split(",")
            decoded = base64.b64decode(content_string)
            data = json.loads(decoded)
        except Exception as e:
            load_status = dbc.Alert(
                f"Error decoding or parsing the uploaded file: {str(e)}",
                color="danger",
            )
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

        # Validate the presence of required keys
        if (
            "clustered_smiles_data" not in data
            or "optimized_motifs_data" not in data
            or "lda_dict" not in data
            or "doc_topic_dists_data" not in data
            or "document_motif_data" not in data
        ):
            load_status = dbc.Alert(
                "Invalid file format. Missing required data.", color="danger"
            )
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

        try:
            clustered_smiles_data = data["clustered_smiles_data"]
            optimized_motifs_data = data["optimized_motifs_data"]
            lda_dict_data = data["lda_dict"]
            doc_topic_dists_data = data["doc_topic_dists_data"]
            document_motif_data = data["document_motif_data"]
        except Exception as e:
            load_status = dbc.Alert(
                f"Error extracting data from the file: {str(e)}", color="danger"
            )
            return (
                run_status,
                load_status,
                clustered_smiles_data,
                optimized_motifs_data,
                lda_dict_data,
                doc_topic_dists_data,
                document_motif_data,
            )

        load_status = dbc.Alert(
            "Results loaded successfully! Switch to the 'View Results' or 'All Fragmentation Spectra and Mass2Motifs Matching Details' tab to view.",
            color="success",
        )

        return (
            run_status,
            load_status,
            clustered_smiles_data,
            optimized_motifs_data,
            lda_dict_data,
            doc_topic_dists_data,
            document_motif_data,
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
    State("doc-topic-dists-store", "data"),
    State("document-motif-data-store", "data"),
    prevent_initial_call=True,
)
def save_results(
    n_clicks,
    clustered_smiles_data,
    optimized_motifs_data,
    lda_dict,
    doc_topic_dists_data,
    document_motif_data,
):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if (
        not clustered_smiles_data
        or not optimized_motifs_data
        or not lda_dict
        or not doc_topic_dists_data
        or not document_motif_data
    ):
        return (
            dash.no_update,
            dbc.Alert(
                "No analysis results to save. Please run an analysis first.",
                color="warning",
            ),
        )

    try:
        data = {
            "clustered_smiles_data": clustered_smiles_data,
            "optimized_motifs_data": optimized_motifs_data,
            "lda_dict": lda_dict,
            "doc_topic_dists_data": doc_topic_dists_data,
            "document_motif_data": document_motif_data,
        }
        json_data = json.dumps(data)
        return (
            dcc.send_string(json_data, filename="ms2lda_results.json"),
            dbc.Alert("Results saved successfully!", color="success"),
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
    return {
        "metadata": spectrum.metadata,
        "mz": [float(m) for m in spectrum.peaks.mz.tolist()],
        "intensities": [float(i) for i in spectrum.peaks.intensities.tolist()],
        "losses_mz": [float(m) for m in spectrum.losses.mz.tolist()]
        if spectrum.losses
        else [],
        "losses_intensities": [float(i) for i in spectrum.losses.intensities.tolist()]
        if spectrum.losses
        else [],
    }


# Updated Callback to create Cytoscape elements
@app.callback(
    Output("cytoscape-network-container", "children"),
    Input("optimized-motifs-store", "data"),
    Input("clustered-smiles-store", "data"),
    Input("tabs", "active_tab"),
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
        if s["losses_mz"]:
            spectrum.losses = Fragments(
                mz=np.array(s["losses_mz"], dtype=float),
                intensities=np.array(s["losses_intensities"], dtype=float),
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
                    "No valid molecules could be created from SMILES.", color="warning"
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
                    returnPNG=True,  # This is important!
                )

                # Image is already in PNG format, just need to encode
                encoded = base64.b64encode(img).decode("utf-8")

                return html.Div(
                    [
                        html.H5(f"Molecules for Motif {motif_number}"),
                        html.Img(
                            src=f"data:image/png;base64,{encoded}",
                            style={"margin": "10px"},
                        ),
                    ]
                )

            except Exception as e:
                print(f"Error creating grid image: {str(e)}")
                return dbc.Alert(
                    f"Error creating molecular grid image: {str(e)}", color="danger"
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
            for mz, intensity in zip(spectrum.losses.mz, spectrum.losses.intensities):
                rounded_mz = round(mz, 2)
                loss_node = f"loss_{rounded_mz}"
                if loss_node not in created_losses:
                    elements.append(
                        {
                            "data": {
                                "id": loss_node,
                                "label": str(rounded_mz),
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
                            "weight": intensity,  # Use intensity as weight
                        }
                    }
                )

    return elements


# Updated Callback to update the Document-Motif Table
@app.callback(
    Output("document-motif-table", "columns"),
    Output("document-motif-table", "data"),
    Input("document-motif-data-store", "data"),
    Input("tabs", "active_tab"),
)
def update_document_motif_table(document_motif_data, active_tab):
    if active_tab != "document-motif-table-tab" or not document_motif_data:
        return [], []
    else:
        # Prepare the columns
        columns = [
            {"name": "Document", "id": "Document"},
            {"name": "Motif", "id": "Motif"},
            {
                "name": "Probability",
                "id": "Probability",
                "type": "numeric",
                "format": Format(precision=4, scheme=Scheme.fixed),
            },
            {
                "name": "Overlap Score",
                "id": "Overlap Score",
                "type": "numeric",
                "format": Format(precision=4, scheme=Scheme.fixed),
            },
            {
                "name": "Precursor Mass",
                "id": "Precursor Mass",
                "type": "numeric",
                "format": Format(precision=4, scheme=Scheme.fixed),
            },
            {
                "name": "Retention Time",
                "id": "Retention Time",
                "type": "numeric",
                "format": Format(precision=4, scheme=Scheme.fixed),
            },
            {"name": "Document Annotation", "id": "Document Annotation"},
        ]
        data = document_motif_data

        # Sort data by Probability in descending order
        data = sorted(data, key=lambda x: x["Probability"], reverse=True)

        # No need to convert 'Document' and 'Motif' to markdown links
        # We will style them to look like links using CSS

        return columns, data


# Updated Callback to handle clicks on the Document-Motif table
@app.callback(
    Output("tabs", "active_tab"),
    Output("clicked-spectrum-store", "data"),
    Output("clicked-motif-store", "data"),
    Input("document-motif-table", "active_cell"),
    State("document-motif-table", "data"),
    prevent_initial_call=True,
)
def on_table_cell_click(active_cell, table_data):
    if active_cell:
        row = active_cell["row"]
        col = active_cell["column_id"]
        clicked_data = table_data[row]
        if col == "Document":
            spectrum_id = clicked_data["Document"]
            return "spectrum-details-tab", spectrum_id, dash.no_update
        elif col == "Motif":
            motif_id = clicked_data["Motif"]
            return "motif-details-tab", dash.no_update, motif_id
    raise dash.exceptions.PreventUpdate


# Callback to display Spectrum Details
@app.callback(
    Output("spectrum-details-content", "children"),
    Input("clicked-spectrum-store", "data"),
    State("doc-topic-dists-store", "data"),
    prevent_initial_call=True,
)
def display_spectrum_details(spectrum_id, doc_topic_dists_data):
    if spectrum_id and doc_topic_dists_data:
        # Placeholder content
        return html.Div(
            [
                html.H4(f"Details for Spectrum: {spectrum_id}"),
                html.P("This is where spectrum details will be displayed.")
                # You can add more content here based on the spectrum_id
            ]
        )
    else:
        return html.P("No spectrum selected.")


# Callback to display Motif Details
@app.callback(
    Output("motif-details-content", "children"),
    Input("clicked-motif-store", "data"),
    State("optimized-motifs-store", "data"),
    prevent_initial_call=True,
)
def display_motif_details(motif_id, optimized_motifs_data):
    if motif_id and optimized_motifs_data:
        # Placeholder content
        return html.Div(
            [
                html.H4(f"Details for Motif: {motif_id}"),
                html.P("This is where motif details will be displayed.")
                # You can add more content here based on the motif_id
            ]
        )
    else:
        return html.P("No motif selected.")


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
