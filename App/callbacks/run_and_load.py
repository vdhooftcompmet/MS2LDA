# App/callbacks/run_and_load.py
"""Callbacks for Run Analysis and Load Results tabs."""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import tempfile

import dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Serverside
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from App.app_instance import app
from App.callbacks.common import *  # Import helper functions

# -------------------------------- RUN AND LOAD RESULTS --------------------------------

# -------------------------------- RUN AND LOAD RESULTS --------------------------------


# Callback to display uploaded data file info
@app.callback(
    Output("file-upload-info", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(contents, filename):
    if contents:
        return html.Div([
            html.I(className="fas fa-file-alt me-2"),
            f"Selected file: {filename}",
        ], style={"color": "#007bff", "fontWeight": "bold"})
    return html.Div([])


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


# Show/hide motif rankings explanation
@app.callback(
    Output("motif-rankings-explanation-collapse", "is_open"),
    Input("motif-rankings-explanation-button", "n_clicks"),
    State("motif-rankings-explanation-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_motif_rankings_explanation(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# Show/hide spectra search explanation
@app.callback(
    Output("spectra-search-explanation-collapse", "is_open"),
    Input("spectra-search-explanation-button", "n_clicks"),
    State("spectra-search-explanation-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_spectra_search_explanation(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# Show/hide filter controls in motif rankings
@app.callback(
    [Output("filter-controls-collapse", "is_open"),
     Output("filter-controls-toggle-button", "children")],
    Input("filter-controls-toggle-button", "n_clicks"),
    State("filter-controls-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_filter_controls(n_clicks, is_open):
    if n_clicks:
        new_is_open = not is_open
        button_text = "🔍 Hide" if new_is_open else "🔍 Show"
        return new_is_open, button_text
    return is_open, "🔍 Hide" if is_open else "🔍 Show"


# Show/hide search controls in spectra search
@app.callback(
    [Output("search-controls-collapse", "is_open"),
     Output("search-controls-toggle-button", "children")],
    Input("search-controls-toggle-button", "n_clicks"),
    State("search-controls-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_search_controls(n_clicks, is_open):
    if n_clicks:
        new_is_open = not is_open
        button_text = "🔎 Hide" if new_is_open else "🔎 Show"
        return new_is_open, button_text
    return is_open, "🔎 Hide" if is_open else "🔎 Show"


# Show/hide network controls in network visualization
@app.callback(
    [Output("network-controls-collapse", "is_open"),
     Output("network-controls-toggle-button", "children")],
    Input("network-controls-toggle-button", "n_clicks"),
    State("network-controls-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_network_controls(n_clicks, is_open):
    if n_clicks:
        new_is_open = not is_open
        button_text = "🔍 Hide" if new_is_open else "🔍 Show"
        return new_is_open, button_text
    return is_open, "🔍 Hide" if is_open else "🔍 Show"


@app.callback(
    Output("run-status", "children"),
    Output("load-status", "children"),
    Output("clustered-smiles-store", "data"),
    Output("optimized-motifs-store", "data"),
    Output("lda-dict-store", "data"),
    Output("spectra-store", "data"),
    Output("upload-spinner", "spinner_style"),
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
    State("s2v-library-embeddings", "value"),
    State("s2v-library-db", "value"),
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
    s2v_library_embeddings,
    s2v_library_db,
):
    """
    This callback either (1) runs MS2LDA from scratch on the uploaded data (when Run Analysis clicked),
    or (2) loads precomputed results from a JSON file (when Load Results clicked).
    """

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    run_status = no_update
    load_status = no_update
    clustered_smiles_data = no_update
    optimized_motifs_data = no_update
    lda_dict_data = no_update
    spectra_data = no_update
    # Default spinner style - hidden
    spinner_style = {"display": "none"}

    # 1) If RUN-BUTTON was clicked
    if triggered_id == "run-button":
        if not data_contents:
            run_status = dbc.Alert(
                "Please upload a mass spec data file first!", color="danger",
            )
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
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(data_filename)[1],
            ) as tmp_file:
                tmp_file.write(decoded)
                tmp_file_path = tmp_file.name
        except Exception as e:
            run_status = dbc.Alert(
                f"Error handling the uploaded file: {e!s}", color="danger",
            )
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
            "s2v_library_embeddings": s2v_library_embeddings,
            "s2v_library_db": s2v_library_db,
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

        motif_spectra, optimized_motifs, motif_fps = ms2lda_run(
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

        trained_ms2lda = tp.LDAModel.load(
            os.path.join(dataset_parameters["output_folder"], "ms2lda.bin"),
        )

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
        cleaned_spectra = clean_spectra(
            loaded_spectra, preprocessing_parameters=preprocessing_parameters,
        )

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
            ann = mot.get("auto_annotation")
            if isinstance(ann, list):
                clustered_smiles_data.append(ann)
            elif ann is None:
                clustered_smiles_data.append([])
            else:
                clustered_smiles_data.append([ann])

        run_status = dbc.Alert("MS2LDA run completed successfully!", color="success")
        return (
            run_status,
            load_status,
            Serverside(clustered_smiles_data),
            Serverside(optimized_motifs_data),
            Serverside(lda_dict),
            Serverside(spectra_data),
        )

    # 2) If LOAD-RESULTS-BUTTON was clicked
    if triggered_id == "load-results-button":
        if not results_contents:
            load_status = dbc.Alert(
                "Please upload a results JSON file.", color="danger",
            )
            return (
                run_status,
                load_status,
                Serverside(clustered_smiles_data),
                Serverside(optimized_motifs_data),
                Serverside(lda_dict_data),
                Serverside(spectra_data),
                spinner_style,
            )

        # Show spinner while loading
        spinner_style = {"width": "3rem", "height": "3rem"}
        try:
            data = parse_ms2lda_viz_file(results_contents)
        except ValueError as e:
            load_status = dbc.Alert(f"Error parsing the file: {e!s}", color="danger")
            # Hide spinner on error
            spinner_style = {"display": "none"}
            return (
                run_status,
                load_status,
                Serverside(clustered_smiles_data),
                Serverside(optimized_motifs_data),
                Serverside(lda_dict_data),
                Serverside(spectra_data),
                spinner_style,
            )

        required_keys = {
            "clustered_smiles_data",
            "optimized_motifs_data",
            "lda_dict",
            "spectra_data",
        }
        if not required_keys.issubset(data.keys()):
            load_status = dbc.Alert(
                "Invalid results file. Missing required data keys.", color="danger",
            )
            # Hide spinner on error
            spinner_style = {"display": "none"}
            return (
                run_status,
                load_status,
                Serverside(clustered_smiles_data),
                Serverside(optimized_motifs_data),
                Serverside(lda_dict_data),
                Serverside(spectra_data),
                spinner_style,
            )

        try:
            clustered_smiles_data = data["clustered_smiles_data"]
            optimized_motifs_data = data["optimized_motifs_data"]
            lda_dict_data = data["lda_dict"]
            spectra_data = data["spectra_data"]
        except Exception as e:
            load_status = dbc.Alert(
                f"Error reading data from file: {e!s}", color="danger",
            )
            # Hide spinner on error
            spinner_style = {"display": "none"}
            return (
                run_status,
                load_status,
                Serverside(clustered_smiles_data),
                Serverside(optimized_motifs_data),
                Serverside(lda_dict_data),
                Serverside(spectra_data),
                spinner_style,
            )

        load_status = dbc.Alert(
            f"Selected Results File: {results_filename}\nResults loaded successfully! You can now explore the data by clicking on the other tabs: 'Motif Rankings', 'Motif Details', 'Spectra Search', 'View Network', or 'Motif Search'.",
            color="success",
        )

        # Hide spinner on success
        spinner_style = {"display": "none"}

        return (
            run_status,
            load_status,
            Serverside(clustered_smiles_data),
            Serverside(optimized_motifs_data),
            Serverside(lda_dict_data),
            Serverside(spectra_data),
            spinner_style,
        )

    raise dash.exceptions.PreventUpdate


@app.callback(
    [Output("selected-file-info", "children"),
     Output("upload-results", "style"),
     Output("load-status", "children", allow_duplicate=True)],
    Input("upload-results", "filename"),
    State("upload-results", "style"),
    prevent_initial_call=True,
)
def update_selected_file_info(filename, current_style):
    """
    Update the UI to show the selected filename when a file is uploaded
    but before the Load Results button is clicked.
    Also update the style of the upload component to indicate a file is selected.
    And clear any previous load status messages.
    """
    if not current_style:
        current_style = {}

    if filename:
        # Create a copy of the current style to avoid modifying the original
        updated_style = dict(current_style)
        # Update the style to indicate a file is selected
        updated_style.update({
            "borderColor": "#007bff",
            "borderWidth": "2px",
            "backgroundColor": "#f8f9fa"
        })

        return (
            html.Div([
                html.I(className="fas fa-file-alt me-2"),
                f"Selected file: {filename}",
            ], style={"color": "#007bff", "fontWeight": "bold"}),
            updated_style,
            # Clear any previous load status messages
            ""
        )

    # Reset to default style if no file is selected
    default_style = {
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
    }

    return "", default_style, ""


def parse_ms2lda_viz_file(base64_contents: str) -> dict:
    """
    Decode the given base64-encoded MS2LDA results file, which might be
    gzipped JSON (.json.gz) or plain JSON (.json), and return the loaded dict.
    Raises ValueError if decoding/parsing fails.
    """
    try:
        # Split out the "data:application/json;base64," prefix
        content_type, content_string = base64_contents.split(",")
        # Decode from base64 -> raw bytes
        decoded = base64.b64decode(content_string)

        # Try reading as gzipped JSON
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(decoded)) as gz:
                data = json.loads(gz.read().decode("utf-8"))
        except OSError:
            # Not gzipped, parse as normal JSON
            data = json.loads(decoded)

        return data

    except Exception as e:
        msg = f"Error decoding or parsing MS2LDA viz file: {e!s}"
        raise ValueError(msg)


@app.callback(
    Output("demo-load-status", "children"),
    Output("clustered-smiles-store", "data", allow_duplicate=True),
    Output("optimized-motifs-store", "data", allow_duplicate=True),
    Output("lda-dict-store", "data", allow_duplicate=True),
    Output("spectra-store", "data", allow_duplicate=True),
    Output("demo-spinner", "spinner_style"),
    Input("load-mushroom-demo-button", "n_clicks"),
    Input("load-pesticides-demo-button", "n_clicks"),
    Input("load-summer-school-demo-button", "n_clicks"),
    prevent_initial_call=True,
)
def load_demo_data(mushroom_clicks, pesticides_clicks, summer_school_clicks):
    """
    Load demo data when one of the demo buttons is clicked.
    Downloads the data from Zenodo and loads it into the app.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Set the URL based on which button was clicked
    if triggered_id == "load-mushroom-demo-button":
        demo_url = "https://zenodo.org/records/15857387/files/CaseStudy_Results_Mushroom_200_ms2lda_viz.json.gz?download=1"
        demo_name = "Mushroom Demo"
    elif triggered_id == "load-pesticides-demo-button":
        demo_url = "https://zenodo.org/records/15857387/files/CaseStudy_Results_Pesticides250_ms2lda_viz.json.gz?download=1"
        demo_name = "Pesticides Demo"
    elif triggered_id == "load-summer-school-demo-button":
        demo_url = "https://raw.githubusercontent.com/joewandy/MS2LDA_example_data/main/SummerSchool_100M2M.json.gz"
        demo_name = "Summer School Example"
    else:
        raise PreventUpdate

    try:

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "demo_data")

            # Download the file
            response = requests.get(demo_url, stream=True)
            if response.status_code != 200:
                # Hide spinner on error
                spinner_style = {"display": "none"}
                return dbc.Alert(f"Error downloading demo data: HTTP {response.status_code}", color="danger"), no_update, no_update, no_update, no_update, spinner_style

            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            with gzip.open(temp_file, 'rb') as f:
                data = json.loads(f.read().decode('utf-8'))

            # Extract the data
            clustered_smiles_data = data["clustered_smiles_data"]
            optimized_motifs_data = data["optimized_motifs_data"]
            lda_dict_data = data["lda_dict"]
            spectra_data = data["spectra_data"]

            # Return success message and the data
            status = dbc.Alert(f"{demo_name} loaded successfully! You can now explore the data by clicking on the other tabs: 'Motif Rankings', 'Motif Details', 'Spectra Search', 'View Network', or 'Motif Search'.", color="success")
            # Hide spinner on success
            spinner_style = {"display": "none"}
            return status, Serverside(clustered_smiles_data), Serverside(optimized_motifs_data), Serverside(lda_dict_data), Serverside(spectra_data), spinner_style

    except Exception as e:
        # Hide spinner on error
        spinner_style = {"display": "none"}
        return dbc.Alert(f"Error loading {demo_name}: {str(e)}", color="danger"), no_update, no_update, no_update, no_update, spinner_style


