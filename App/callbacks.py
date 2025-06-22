from __future__ import annotations

import base64
import contextlib
import gzip
import io
import json
import os
import re
import tempfile

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash_extensions.enrich import Serverside

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import tomotopy as tp
from dash import ALL, Input, Output, State, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate
from massql4motifs import msql_engine
from matchms import Fragments, Spectrum
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

import MS2LDA
from App.app_instance import MOTIFDB_DIR, app
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import (
    load_motifDB,
    motifDB2motifs,
    motifs2motifDB,
)
from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings
from MS2LDA.Add_On.Spec2Vec.annotation_refined import calc_similarity
from MS2LDA.Mass2Motif import Mass2Motif
from MS2LDA.Preprocessing.load_and_clean import clean_spectra
from MS2LDA.run import filetype_check, load_s2v_model
from MS2LDA.utils import create_spectrum, download_model_and_data
from MS2LDA.Visualisation.ldadict import generate_corpusjson_from_tomotopy

# Cache for consistent motif colors
MOTIF_COLOR_CACHE = {}


def calculate_motif_shares(spec_dict, lda_dict_data, tolerance=0.02):
    """
    For one spectrum, compute each motif's share of every peak.

    Args:
        spec_dict (dict): The spectrum dictionary
        lda_dict_data (dict): The LDA model data
        tolerance (float): m/z tolerance for matching (default: 0.02 Da)

    Returns:
        list: A list the same length as spec_dict["mz"]; each element is either
              None (no motif matches) or a dict mapping motif_id -> share (floats summing to 1)
    """
    meta = spec_dict.get("metadata", {})
    doc_id = meta.get("id", f"spec_{meta.get('index', 0)}")
    parent_mz = meta.get("precursor_mz")

    # Get the theta vector for this document
    theta = lda_dict_data.get("theta", {}).get(doc_id, {})
    if not theta:
        return [None] * len(spec_dict["mz"])

    # Extract all motifs with non-zero theta for this document
    relevant_motifs = {m: t for m, t in theta.items() if t > 0}
    if not relevant_motifs:
        return [None] * len(spec_dict["mz"])

    # Build lookup dictionaries for each motif's fragments and losses
    motif_fragments = {}
    motif_losses = {}

    for motif_id in relevant_motifs:
        if motif_id not in lda_dict_data.get("beta", {}):
            continue

        beta = lda_dict_data["beta"][motif_id]
        frag_dict = {}
        loss_dict = {}

        for ft, prob in beta.items():
            if ft.startswith("frag@"):
                try:
                    mz = float(ft.replace("frag@", ""))
                    frag_dict[mz] = prob
                except ValueError:
                    pass
            elif ft.startswith("loss@"):
                try:
                    mz = float(ft.replace("loss@", ""))
                    loss_dict[mz] = prob
                except ValueError:
                    pass

        motif_fragments[motif_id] = frag_dict
        motif_losses[motif_id] = loss_dict

    # Calculate motif shares for each peak
    mz_arr = np.array(spec_dict["mz"], dtype=float)
    shares = []

    for _i, mz_val in enumerate(mz_arr):
        # Find matching motifs for this peak (fragment or loss)
        matches = {}

        # Check fragment matches
        for motif_id, frags in motif_fragments.items():
            for frag_mz, beta_val in frags.items():
                if abs(mz_val - frag_mz) <= tolerance:
                    # Responsibility = theta[m] * beta[m,f]
                    matches[motif_id] = matches.get(motif_id, 0) + (
                        theta[motif_id] * beta_val
                    )

        # Check loss matches if parent_mz is available
        if parent_mz is not None:
            for motif_id, losses in motif_losses.items():
                for loss_mz, beta_val in losses.items():
                    # Check if this peak corresponds to a loss
                    if abs((parent_mz - loss_mz) - mz_val) <= tolerance:
                        # Responsibility = theta[m] * beta[m,l]
                        matches[motif_id] = matches.get(motif_id, 0) + (
                            theta[motif_id] * beta_val
                        )

        # If no matches, this peak isn't claimed by any motif
        if not matches:
            shares.append(None)
            continue

        # Normalize the responsibilities to sum to 1
        total = sum(matches.values())
        if total > 0:
            normalized = {m: v / total for m, v in matches.items()}
            shares.append(normalized)
        else:
            shares.append(None)

    return shares


def make_spectrum_plot(
    spec_dict,
    motif_id,
    lda_dict_data,
    probability_range=(0, 1),
    tolerance=0.02,
    mode="both",
    highlight_mode="single",
    show_parent_ion=True,
):
    """
    Return a Plotly Figure for one MS/MS spectrum.

    Args:
        spec_dict (dict): The spectrum dictionary
        motif_id (str): The motif ID to highlight (used when highlight_mode='single')
        lda_dict_data (dict): The LDA model data
        probability_range (tuple): Range of probabilities to filter motif features
        tolerance (float): m/z tolerance for matching
        mode (str): 'both', 'fragments', or 'losses'
        highlight_mode (str): 'single', 'all', or 'none'
        show_parent_ion (bool): Whether to display the parent ion marker

    Returns:
        go.Figure: A Plotly figure object
    """
    meta = spec_dict.get("metadata", {})
    parent_mz = meta.get("precursor_mz")
    spectrum_id = meta.get("id", "Unknown")

    mz_arr = np.array(spec_dict["mz"], dtype=float)
    int_arr = np.array(spec_dict["intensities"], dtype=float)

    fig = go.Figure()

    # Fast path for 'none' mode - everything is grey
    if highlight_mode == "none":
        fig.add_trace(
            go.Bar(
                x=mz_arr,
                y=int_arr,
                marker={"color": "#7f7f7f", "line": {"color": "white", "width": 0}},
                width=0.3,
                hoverinfo="text",
                hovertext=[
                    f"m/z: {mz_val:.4f}<br>Intensity: {inten:.3g}"
                    for mz_val, inten in zip(mz_arr, int_arr)
                ],
                opacity=0.9,
                name="Peaks",
            ),
        )

    # Single motif highlighting mode (original behavior)
    elif highlight_mode == "single":
        frag_mzs = []
        loss_mzs = []
        if motif_id and lda_dict_data and motif_id in lda_dict_data.get("beta", {}):
            motif_feats = {
                k: v
                for k, v in lda_dict_data["beta"][motif_id].items()
                if probability_range[0] <= v <= probability_range[1]
            }
            for ft in motif_feats:
                if ft.startswith("frag@"):
                    with contextlib.suppress(ValueError):
                        frag_mzs.append(float(ft.replace("frag@", "")))
                elif ft.startswith("loss@"):
                    with contextlib.suppress(ValueError):
                        loss_mzs.append(float(ft.replace("loss@", "")))

        frag_match_idx = set()
        loss_match_idx = set()
        for i, mz_val in enumerate(mz_arr):
            if mode in ("both", "fragments") and any(
                abs(mz_val - t) <= tolerance for t in frag_mzs
            ):
                frag_match_idx.add(i)
            if (
                mode in ("both", "losses")
                and parent_mz is not None
                and any(abs((parent_mz - l) - mz_val) <= tolerance for l in loss_mzs)
            ):
                loss_match_idx.add(i)

        bar_colors = [
            "#FF0000" if i in frag_match_idx or i in loss_match_idx else "#7f7f7f"
            for i in range(len(mz_arr))
        ]

        fig.add_trace(
            go.Bar(
                x=mz_arr,
                y=int_arr,
                marker={"color": bar_colors, "line": {"color": "white", "width": 0}},
                width=0.3,
                hoverinfo="text",
                hovertext=[
                    f"m/z: {mz_val:.4f}<br>Intensity: {inten:.3g}"
                    for mz_val, inten in zip(mz_arr, int_arr)
                ],
                opacity=0.9,
                name="Peaks",
            ),
        )

        # visualise neutral-loss for single motif
        if parent_mz is not None and loss_mzs and mode in ("both", "losses"):
            for loss_val in loss_mzs:
                frag_mz = parent_mz - loss_val
                # find closest plotted peak within the tolerance
                idx = np.argmin(np.abs(mz_arr - frag_mz))
                if abs(mz_arr[idx] - frag_mz) <= tolerance:
                    y_val = int_arr[idx]
                    fig.add_shape(
                        type="line",
                        x0=mz_arr[idx],
                        y0=y_val,
                        x1=parent_mz,
                        y1=y_val,
                        line={"color": "rgba(0,128,0,0.4)", "dash": "dash", "width": 1},
                    )
                    fig.add_annotation(
                        x=(mz_arr[idx] + parent_mz) / 2,
                        y=y_val,
                        text=f"-{loss_val:.2f}",
                        showarrow=False,
                        font={"size": 10, "color": "green"},
                        bgcolor="rgba(255,255,255,0.7)",
                        xanchor="center",
                    )

    # All motifs highlighting mode (stacked bars)
    elif highlight_mode == "all":
        # Calculate motif shares for each peak
        peak_shares = calculate_motif_shares(spec_dict, lda_dict_data, tolerance)

        # Get all unique motifs that appear in any peak
        all_motifs = set()
        for shares in peak_shares:
            if shares:
                all_motifs.update(shares.keys())

        # Sort motifs for consistent ordering
        sorted_motifs = sorted(all_motifs)

        # Create a color for each motif (cached for consistency)
        import plotly.colors

        for m in sorted_motifs:
            if m not in MOTIF_COLOR_CACHE:
                # Get the next color from Plotly's qualitative palette (cycling if needed)
                palette = plotly.colors.qualitative.Plotly
                MOTIF_COLOR_CACHE[m] = palette[hash(m) % len(palette)]

        # Create a trace for unassigned peaks
        unassigned_y = np.zeros_like(int_arr)
        for i, shares in enumerate(peak_shares):
            if shares is None:
                unassigned_y[i] = int_arr[i]

        if np.any(unassigned_y > 0):
            fig.add_trace(
                go.Bar(
                    x=mz_arr,
                    y=unassigned_y,
                    marker={"color": "#7f7f7f", "line": {"color": "white", "width": 0}},
                    width=0.3,
                    hoverinfo="text",
                    hovertext=[
                        f"m/z: {mz:.4f}<br>Intensity: {inten:.3g}<br>Unassigned"
                        for mz, inten in zip(mz_arr, unassigned_y)
                        if inten > 0
                    ],
                    opacity=0.9,
                    name="Unassigned",
                ),
            )

        # Create a trace for each motif
        for motif in sorted_motifs:
            motif_y = np.zeros_like(int_arr)

            for i, shares in enumerate(peak_shares):
                if shares and motif in shares:
                    motif_y[i] = int_arr[i] * shares[motif]

            if np.any(motif_y > 0):
                fig.add_trace(
                    go.Bar(
                        x=mz_arr,
                        y=motif_y,
                        marker={
                            "color": MOTIF_COLOR_CACHE[motif],
                            "line": {"color": "white", "width": 0},
                        },
                        width=0.3,
                        hoverinfo="text",
                        hovertext=[
                            (
                                f"m/z: {mz:.4f}<br>Intensity: {inten:.3g}<br>Motif: {motif}<br>Share: {shares[motif]:.2f}"
                                if shares and motif in shares
                                else ""
                            )
                            for mz, inten, shares in zip(mz_arr, int_arr, peak_shares)
                            if inten > 0
                        ],
                        opacity=0.9,
                        name=f"{motif}",
                    ),
                )

        # Set barmode to stack
        fig.update_layout(barmode="stack")

        # Add neutral loss connectors for peaks with motif matches
        if parent_mz is not None and mode in ("both", "losses"):
            # Get all loss m/z values from all motifs
            all_losses = set()
            for m in sorted_motifs:
                if m in lda_dict_data.get("beta", {}):
                    for ft, prob in lda_dict_data["beta"][m].items():
                        if (
                            ft.startswith("loss@")
                            and probability_range[0] <= prob <= probability_range[1]
                        ):
                            with contextlib.suppress(ValueError):
                                all_losses.add(float(ft.replace("loss@", "")))

            for loss_val in all_losses:
                frag_mz = parent_mz - loss_val
                # find closest plotted peak within the tolerance
                idx = np.argmin(np.abs(mz_arr - frag_mz))
                if (
                    abs(mz_arr[idx] - frag_mz) <= tolerance
                    and peak_shares[idx] is not None
                ):
                    # Find the dominant motif for this loss
                    max(peak_shares[idx].items(), key=lambda x: x[1])[
                        0
                    ]
                    y_val = int_arr[idx]

                    fig.add_shape(
                        type="line",
                        x0=mz_arr[idx],
                        y0=y_val,
                        x1=parent_mz,
                        y1=y_val,
                        line={"color": "rgba(0,128,0,0.4)", "dash": "dash", "width": 1},
                    )
                    fig.add_annotation(
                        x=(mz_arr[idx] + parent_mz) / 2,
                        y=y_val,
                        text=f"-{loss_val:.2f}",
                        showarrow=False,
                        font={"size": 10, "color": "green"},
                        bgcolor="rgba(255,255,255,0.7)",
                        xanchor="center",
                    )

    # Add precursor ion marker for all modes if show_parent_ion is True
    if parent_mz is not None and show_parent_ion:
        fig.add_shape(
            type="line",
            x0=parent_mz,
            x1=parent_mz,
            y0=0,
            y1=int_arr.max() * 1.05,
            line={"color": "blue", "dash": "dash", "width": 2},
        )
        fig.add_annotation(
            x=parent_mz,
            y=int_arr.max() * 1.06,
            text=f"Parent Ion {parent_mz:.2f}",
            showarrow=False,
            font={"size": 10, "color": "blue"},
            xanchor="center",
        )

    # Set title based on highlight mode
    if highlight_mode == "single":
        title = f"Spectrum: {spectrum_id} — Highlighted Motif: {motif_id or 'None'}"
    elif highlight_mode == "all":
        title = f"Spectrum: {spectrum_id} — All Motifs"
    else:  # 'none'
        title = f"Spectrum: {spectrum_id} — No Highlighting"

    fig.update_layout(
        title=title,
        xaxis_title="m/z (Da)",
        hovermode="closest",
    )
    apply_common_layout(fig, ytitle="Intensity")
    return fig


def apply_common_layout(fig: go.Figure, ytitle: str | None = None) -> None:
    """Apply the shared template, font, margins and bargap to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        font={"size": 12},
        margin={"l": 40, "r": 30, "t": 30, "b": 40},
        bargap=0.35,
    )
    if ytitle is not None:
        fig.update_yaxes(title_text=ytitle)


# Dynamic path – works inside site-packages too
MOTIFDB_FOLDER = str(MOTIFDB_DIR)


def load_motifset_file(json_path):
    """
    Loads a single JSON motifset file.
    Returns a list of motifs in the file as matchms Spectra.
    """
    ms1_df, ms2_df = load_motifDB(json_path)
    return motifDB2motifs(ms2_df)


# Callback to show/hide tab contents based on active tab
@app.callback(
    Output("run-analysis-tab-content", "style"),
    Output("load-results-tab-content", "style"),
    Output("results-tab-content", "style"),
    Output("motif-rankings-tab-content", "style"),
    Output("motif-details-tab-content", "style"),
    Output("screening-tab-content", "style"),
    Output("search-spectra-tab-content", "style"),
    Input("tabs", "value"),
)
def toggle_tab_content(active_tab):
    run_style = {"display": "none"}
    load_style = {"display": "none"}
    results_style = {"display": "none"}
    motif_rankings_style = {"display": "none"}
    motif_details_style = {"display": "none"}
    screening_style = {"display": "none"}
    search_spectra_style = {"display": "none"}

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
    elif active_tab == "screening-tab":
        screening_style = {"display": "block"}
    elif active_tab == "search-spectra-tab":
        search_spectra_style = {"display": "block"}

    return (
        run_style,
        load_style,
        results_style,
        motif_rankings_style,
        motif_details_style,
        screening_style,
        search_spectra_style,
    )


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


@app.callback(
    Output("run-status", "children"),
    Output("load-status", "children"),
    Output("clustered-smiles-store", "data"),
    Output("optimized-motifs-store", "data"),
    Output("lda-dict-store", "data"),
    Output("spectra-store", "data"),
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

        run_status = dbc.Alert("MS2LDA.run completed successfully!", color="success")
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
            )
        try:
            data = parse_ms2lda_viz_file(results_contents)
        except ValueError as e:
            load_status = dbc.Alert(f"Error parsing the file: {e!s}", color="danger")
            return (
                run_status,
                load_status,
                Serverside(clustered_smiles_data),
                Serverside(optimized_motifs_data),
                Serverside(lda_dict_data),
                Serverside(spectra_data),
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
            return (
                run_status,
                load_status,
                Serverside(clustered_smiles_data),
                Serverside(optimized_motifs_data),
                Serverside(lda_dict_data),
                Serverside(spectra_data),
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
            return (
                run_status,
                load_status,
                Serverside(clustered_smiles_data),
                Serverside(optimized_motifs_data),
                Serverside(lda_dict_data),
                Serverside(spectra_data),
            )

        load_status = dbc.Alert(
            f"Selected Results File: {results_filename}\nResults loaded successfully!",
            color="success",
        )

        return (
            run_status,
            load_status,
            Serverside(clustered_smiles_data),
            Serverside(optimized_motifs_data),
            Serverside(lda_dict_data),
            Serverside(spectra_data),
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


# -------------------------------- CYTOSCAPE NETWORK --------------------------------


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
def update_cytoscape(
    optimized_motifs_data,
    clustered_smiles_data,
    active_tab,
    edge_intensity_threshold,
    toggle_loss_edge,
    layout_choice,
):
    if active_tab != "results-tab" or not optimized_motifs_data:
        raise PreventUpdate

    spectra = []
    for s in optimized_motifs_data:
        # Prepare the losses, if any, from s["metadata"]
        if "losses" in s["metadata"]:
            losses_list = s["metadata"]["losses"]
            loss_mz = [loss["loss_mz"] for loss in losses_list]
            loss_intensities = [loss["loss_intensity"] for loss in losses_list]
        else:
            loss_mz = []
            loss_intensities = []

        # Create Mass2Motif object with both fragments and losses
        spectrum = Mass2Motif(
            frag_mz=np.array(s["mz"], dtype=float),
            frag_intensities=np.array(s["intensities"], dtype=float),
            loss_mz=np.array(loss_mz, dtype=float),
            loss_intensities=np.array(loss_intensities, dtype=float),
            metadata=s["metadata"],
        )

        spectra.append(spectrum)

    smiles_clusters = clustered_smiles_data

    # Convert the checkbox list into a boolean
    show_loss_edge = "show_loss_edge" in toggle_loss_edge

    elements = create_cytoscape_elements(
        spectra,
        smiles_clusters,
        intensity_threshold=edge_intensity_threshold,
        show_loss_edge=show_loss_edge,
    )

    # Use the selected layout from the dropdown.
    return cyto.Cytoscape(
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



def create_cytoscape_elements(
    spectra, smiles_clusters, intensity_threshold=0.05, show_loss_edge=False,
):
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
                },
            },
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
                        },
                    },
                )
                created_fragments.add(frag_node)
            elements.append(
                {
                    "data": {
                        "source": motif_node,
                        "target": frag_node,
                        "weight": intensity,
                    },
                },
            )
        if spectrum.losses is not None:
            precursor_mz = float(spectrum.metadata.get("precursor_mz", 0))
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
                            },
                        },
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
                            },
                        },
                    )
                    created_losses.add(loss_node)
                elements.append(
                    {
                        "data": {
                            "source": motif_node,
                            "target": loss_node,
                            "weight": loss_intensity,
                        },
                    },
                )
                # Conditionally re-add the line from loss node to fragment node if user wants it
                if show_loss_edge:
                    elements.append(
                        {
                            "data": {
                                "source": loss_node,
                                "target": frag_node,
                                "weight": loss_intensity,
                            },
                        },
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
            return html.Div("No SMILES found for this motif.")

        smiles_list = clustered_smiles_data[motif_index]
        if not smiles_list:
            return html.Div("This motif has no associated SMILES.")

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
            returnPNG=True,
        )
        encoded = base64.b64encode(grid_img).decode("utf-8")

        # Return an <img> with the PNG
        return html.Img(
            src="data:image/png;base64," + encoded, style={"margin": "10px"},
        )

    # Otherwise (e.g. if user clicks a fragment/loss), do nothing special
    raise PreventUpdate


# -------------------------------- RANKINGS & DETAILS --------------------------------


def compute_motif_degrees(lda_dict, p_low, p_high, o_low, o_high):
    motifs = lda_dict["beta"].keys()
    motif_degrees = dict.fromkeys(motifs, 0)
    motif_probabilities = {m: [] for m in motifs}
    motif_overlap_scores = {m: [] for m in motifs}
    docs = lda_dict["theta"].keys()

    # For each document, check if the motif's doc-topic prob p
    # is within [p_low, p_high], and overlap is within [o_low, o_high].
    for doc in docs:
        for motif, p in lda_dict["theta"][doc].items():
            if p_low <= p <= p_high:
                o = lda_dict["overlap_scores"][doc].get(motif, 0.0)
                if o_low <= o <= o_high:
                    motif_degrees[motif] += 1
                    motif_probabilities[motif].append(p)
                    motif_overlap_scores[motif].append(o)

    md = []
    for motif in motifs:
        if motif_probabilities[motif]:
            avg_probability = np.mean(motif_probabilities[motif])
            avg_overlap = np.mean(motif_overlap_scores[motif])
        else:
            avg_probability = 0
            avg_overlap = 0

        md.append((motif, motif_degrees[motif], avg_probability, avg_overlap))

    md.sort(key=lambda x: x[1], reverse=True)
    return md


@app.callback(
    Output("motif-ranking-massql-matches", "data"),
    Input("motif-ranking-massql-btn", "n_clicks"),
    State("motif-ranking-massql-input", "value"),
    State("optimized-motifs-store", "data"),
)
def run_massql_query(n_clicks, query, motifs_data):
    if not n_clicks or not query or not motifs_data:
        raise PreventUpdate
    specs = [make_spectrum_from_dict(d) for d in motifs_data]
    ms1_df, ms2_df = motifs2motifDB(specs)
    matches = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)

    # If no results
    if matches.empty or "motif_id" not in matches.columns:
        return []
    return matches["motif_id"].unique().tolist()


@app.callback(
    Output("motif-rankings-table", "data"),
    Output("motif-rankings-table", "columns"),
    Output("motif-rankings-count", "children"),
    Input("lda-dict-store", "data"),
    Input("probability-thresh", "value"),
    Input("overlap-thresh", "value"),
    Input("tabs", "value"),
    Input("motif-ranking-massql-matches", "data"),
    State("screening-fullresults-store", "data"),
    State("optimized-motifs-store", "data"),
)
def update_motif_rankings_table(
    lda_dict_data,
    probability_thresh,
    overlap_thresh,
    active_tab,
    massql_matches,
    screening_data,
    optimized_motifs_data,
):
    if active_tab != "motif-rankings-tab" or not lda_dict_data:
        return [], [], ""

    p_low, p_high = probability_thresh
    o_low, o_high = overlap_thresh

    motif_degree_list = compute_motif_degrees(
        lda_dict_data, p_low, p_high, o_low, o_high,
    )
    df = pd.DataFrame(
        motif_degree_list,
        columns=[
            "Motif",
            "Degree",
            "Average Doc-Topic Probability",
            "Average Overlap Score",
        ],
    )

    if massql_matches is not None:
        df = df[df["Motif"].isin(massql_matches)]

    # 1) topic_metadata from LDA
    motif_annotations = {}
    if "topic_metadata" in lda_dict_data:
        for motif, metadata in lda_dict_data["topic_metadata"].items():
            motif_annotations[motif] = metadata.get("annotation", "")

    # 2) short_annotation from optimized_motifs_store
    combined_annotations = []
    for motif_name in df["Motif"]:
        existing_lda_anno = motif_annotations.get(motif_name, "")
        short_anno_str = ""

        # parse motif index
        motif_idx = None
        if motif_name.startswith("motif_"):
            with contextlib.suppress(ValueError):
                motif_idx = int(motif_name.replace("motif_", ""))

        if (
            optimized_motifs_data
            and motif_idx is not None
            and 0 <= motif_idx < len(optimized_motifs_data)
        ):
            # short_annotation might be list of SMILES or None
            short_anno = optimized_motifs_data[motif_idx]["metadata"].get(
                "auto_annotation", "",
            )
            if isinstance(short_anno, list):
                short_anno_str = ", ".join(short_anno)
            elif isinstance(short_anno, str):
                short_anno_str = short_anno

        # combine them
        if existing_lda_anno and short_anno_str:
            combined = f"{existing_lda_anno} / {short_anno_str}"
        elif short_anno_str:
            combined = short_anno_str
        else:
            combined = existing_lda_anno

        combined_annotations.append(combined)

    df["Annotation"] = combined_annotations

    # 3) Screening references in new column
    screening_hits = []
    if screening_data:
        try:
            scdf = pd.read_json(screening_data, orient="records")
            for motif in df["Motif"]:
                # Filter this motif’s hits
                hits_for_motif = scdf[scdf["user_motif_id"] == motif].sort_values(
                    "score", ascending=False,
                )
                if hits_for_motif.empty:
                    screening_hits.append("")
                else:
                    # Collect up to 3 references in the format: "ref_motifset|ref_motif_id(score)"
                    top3 = hits_for_motif.head(3)
                    combined = []
                    for _, row in top3.iterrows():
                        combined.append(
                            f"{row['ref_motifset']}|{row['ref_motif_id']}({row['score']:.2f})",
                        )
                    screening_hits.append("; ".join(combined))
        except Exception:
            # If there's any JSON/parsing error, fallback
            screening_hits = ["" for _ in range(len(df))]
    else:
        screening_hits = ["" for _ in range(len(df))]
    df["Matching Hits"] = screening_hits

    # Filter out motifs that have no docs passing, i.e. degree=0
    df = df[df["Degree"] > 0].copy()

    table_data = df.to_dict("records")
    table_columns = [
        {
            "name": "Motif",
            "id": "Motif",
        },
        {
            "name": "Degree",
            "id": "Degree",
            "type": "numeric",
        },
        {
            "name": "Average Doc-Topic Probability",
            "id": "Average Doc-Topic Probability",
            "type": "numeric",
            "format": {"specifier": ".4f"},
        },
        {
            "name": "Average Overlap Score",
            "id": "Average Overlap Score",
            "type": "numeric",
            "format": {"specifier": ".4f"},
        },
        {
            "name": "Annotation",
            "id": "Annotation",
        },
        {
            "name": "Matching Hits",
            "id": "Matching Hits",
        },
    ]

    row_count_message = f"{len(df)} motif(s) pass the filter"
    return table_data, table_columns, row_count_message


@app.callback(
    Output("tabs", "value", allow_duplicate=True),
    Input("selected-motif-store", "data"),
    State("tabs", "value"),  # New State
    prevent_initial_call=True,
)
def activate_motif_details_tab(selected_motif, current_active_tab):
    if selected_motif:
        if current_active_tab == "search-spectra-tab":
            return dash.no_update  # Stay on search tab
        return "motif-details-tab"  # Go to motif details
    return dash.no_update


@app.callback(
    Output("probability-thresh-display", "children"),
    Input("probability-thresh", "value"),
)
def display_probability_thresh(prob_thresh_range):
    return f"Selected Probability Range: {prob_thresh_range[0]:.2f} - {prob_thresh_range[1]:.2f}"


@app.callback(
    Output("overlap-thresh-display", "children"), Input("overlap-thresh", "value"),
)
def display_overlap_thresh(overlap_thresh_range):
    return f"Selected Overlap Range: {overlap_thresh_range[0]:.2f} - {overlap_thresh_range[1]:.2f}"


@app.callback(
    Output("probability-filter-display", "children"),
    Input("probability-filter", "value"),
)
def display_prob_filter(prob_filter_range):
    return f"Showing features with probability between {prob_filter_range[0]:.2f} and {prob_filter_range[1]:.2f}"


@app.callback(
    Output("doc-topic-filter-display", "children"), Input("doc-topic-filter", "value"),
)
def display_doc_topic_filter(value_range):
    return f"Filtering docs with motif probability between {value_range[0]:.2f} and {value_range[1]:.2f}"


@app.callback(
    Output("overlap-filter-display", "children"), Input("overlap-filter", "value"),
)
def display_overlap_filter(overlap_range):
    return f"Filtering docs with overlap score between {overlap_range[0]:.2f} and {overlap_range[1]:.2f}"


@app.callback(
    Output("motif-details-title", "children"),
    Output("motif-spec2vec-container", "children"),
    Output("motif-features-container", "children"),
    Output("motif-documents-container", "children"),
    Output("motif-spectra-ids-store", "data"),
    Output("spectra-table", "data"),
    Output("spectra-table", "columns"),
    Output("motif-dual-spectrum-container", "children"),
    Input("selected-motif-store", "data"),
    Input("probability-filter", "value"),
    Input("doc-topic-filter", "value"),
    Input("overlap-filter", "value"),
    Input("optimised-motif-fragloss-toggle", "value"),
    Input("dual-plot-bar-width-slider", "value"),
    State("lda-dict-store", "data"),
    State("clustered-smiles-store", "data"),
    State("spectra-store", "data"),
    State("optimized-motifs-store", "data"),
    State("screening-fullresults-store", "data"),
    prevent_initial_call=True,
)
def update_motif_details(
    selected_motif,
    beta_range,
    theta_range,
    overlap_range,
    optimised_fragloss_toggle,
    bar_width,
    lda_dict_data,
    clustered_smiles_data,
    spectra_data,
    optimized_motifs_data,
    screening_data,
):
    if not selected_motif or not lda_dict_data:
        raise PreventUpdate

    motif_name = selected_motif
    motif_title = f"Motif Details: {motif_name}"

    # 1) Raw motif (LDA) – first cut by feature-probability slider
    motif_data = lda_dict_data["beta"].get(motif_name, {})
    filtered_motif_data = {
        f: p for f, p in motif_data.items() if beta_range[0] <= p <= beta_range[1]
    }

    # 2) Counts of features in filtered documents
    feature_counts = dict.fromkeys(filtered_motif_data.keys(), 0)

    # Collect docs that pass the current doc-topic probability + overlap filters
    docs_for_this_motif = []
    for doc_name, topic_probs in lda_dict_data["theta"].items():
        doc_topic_prob = topic_probs.get(motif_name, 0.0)
        if doc_topic_prob <= 0:
            continue
        overlap_score = lda_dict_data["overlap_scores"][doc_name].get(motif_name, 0.0)
        if (theta_range[0] <= doc_topic_prob <= theta_range[1]) and (
            overlap_range[0] <= overlap_score <= overlap_range[1]
        ):
            docs_for_this_motif.append(doc_name)

    # Sum up the occurrences of each feature within these filtered docs only
    for doc_name in docs_for_this_motif:
        w_counts = lda_dict_data["corpus"].get(doc_name, {})
        for ft in filtered_motif_data:
            if ft in w_counts:
                feature_counts[ft] += 1

    # SECOND cut → keep a feature only if it appears in ≥1 passing document
    filtered_motif_data = {
        f: p for f, p in filtered_motif_data.items() if feature_counts.get(f, 0) > 0
    }

    # rebuild summary table
    total_prob = sum(filtered_motif_data.values())

    feature_table = pd.DataFrame(
        {
            "Feature": filtered_motif_data.keys(),
            "Probability": filtered_motif_data.values(),
        },
    ).sort_values(by="Probability", ascending=False)

    feature_table_component = dash_table.DataTable(
        data=feature_table.to_dict("records"),
        columns=[
            {"name": "Feature", "id": "Feature"},
            {
                "name": "Probability",
                "id": "Probability",
                "type": "numeric",
                "format": {"specifier": ".4f"},
            },
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"},
        page_size=10,
    )

    # rebuild bar-plot dataframe and drop zero-count rows
    barplot2_df = pd.DataFrame(
        {
            "Feature": list(feature_counts.keys()),
            "Count": list(feature_counts.values()),
        },
    )
    barplot2_df = barplot2_df[barplot2_df["Count"] > 0].sort_values(
        by="Count", ascending=False,
    )
    # Vertical bar chart
    barplot2_fig = px.bar(
        barplot2_df,
        x="Feature",
        y="Count",
    )
    barplot2_fig.update_layout(
        title=None,
        xaxis_title="Feature",
        yaxis_title="Count within Filtered Motif Documents",
        bargap=0.3,
        font={"size": 12},
    )

    # 3) Spec2Vec matching results
    motif_idx = None
    if motif_name.startswith("motif_"):
        with contextlib.suppress(Exception):
            motif_idx = int(motif_name.replace("motif_", ""))

    spec2vec_container = []
    auto_anno_text = ""
    if (
        optimized_motifs_data
        and motif_idx is not None
        and 0 <= motif_idx < len(optimized_motifs_data)
    ):
        meta_anno = optimized_motifs_data[motif_idx]["metadata"].get(
            "auto_annotation", "",
        )
        if meta_anno:
            auto_anno_text = f"Auto Annotations: {meta_anno}"

    if (
        clustered_smiles_data
        and motif_idx is not None
        and motif_idx < len(clustered_smiles_data)
    ):
        smiles_list = clustered_smiles_data[motif_idx]
        if smiles_list:
            spec2vec_container.append(html.H5("Spec2Vec Matching Results"))
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
                    returnPNG=True,
                )
                encoded = base64.b64encode(grid_img).decode("utf-8")
                spec2vec_container.append(
                    html.Img(
                        src="data:image/png;base64," + encoded, style={"margin": "10px"},
                    ),
                )
    if auto_anno_text:
        spec2vec_container.append(html.Div(auto_anno_text, style={"marginTop": "10px"}))

    # 4) Documents table
    doc2spec_index = lda_dict_data.get("doc_to_spec_index", {})
    docs_for_this_motif_records = []
    for doc_name, topic_probs in lda_dict_data["theta"].items():
        doc_topic_prob = topic_probs.get(motif_name, 0.0)
        if doc_topic_prob <= 0:
            continue
        overlap_score = lda_dict_data["overlap_scores"][doc_name].get(motif_name, 0.0)
        if (theta_range[0] <= doc_topic_prob <= theta_range[1]) and (
            overlap_range[0] <= overlap_score <= overlap_range[1]
        ):
            real_idx = -1
            doc_idx_str = doc_name.replace("spec_", "")
            if doc_idx_str in doc2spec_index:
                real_idx = doc2spec_index[doc_idx_str]
            precursor_mz = None
            retention_time = None
            feature_id = None
            collision_energy = None
            ionmode = None
            ms_level = None
            scans = None
            if real_idx != -1 and real_idx < len(spectra_data):
                sp_meta = spectra_data[real_idx]["metadata"]
                precursor_mz = sp_meta.get("precursor_mz")
                retention_time = sp_meta.get("retention_time")
                feature_id = sp_meta.get("feature_id")
                collision_energy = sp_meta.get("collision_energy")
                ionmode = sp_meta.get("ionmode")
                ms_level = sp_meta.get("ms_level")
                scans = sp_meta.get("scans")

            docs_for_this_motif_records.append(
                {
                    "DocName": doc_name,
                    "SpecIndex": real_idx,
                    "FeatureID": feature_id,
                    "Scans": scans,
                    "PrecursorMz": precursor_mz,
                    "RetentionTime": retention_time,
                    "CollisionEnergy": collision_energy,
                    "IonMode": ionmode,
                    "MsLevel": ms_level,
                    "Doc-Topic Probability": doc_topic_prob,
                    "Overlap Score": overlap_score,
                },
            )

    doc_cols = [
        "DocName",
        "SpecIndex",
        "FeatureID",
        "Scans",
        "PrecursorMz",
        "RetentionTime",
        "CollisionEnergy",
        "IonMode",
        "MsLevel",
        "Doc-Topic Probability",
        "Overlap Score",
    ]
    docs_df = pd.DataFrame(docs_for_this_motif_records, columns=doc_cols)
    if not docs_df.empty:
        docs_df = docs_df.sort_values(by="Doc-Topic Probability", ascending=False)

    table_data = docs_df.to_dict("records")
    table_columns = [
        {"name": "DocName", "id": "DocName"},
        {"name": "SpecIndex", "id": "SpecIndex", "type": "numeric"},
        {"name": "FeatureID", "id": "FeatureID"},
        {"name": "Scans", "id": "Scans"},
        {
            "name": "PrecursorMz",
            "id": "PrecursorMz",
            "type": "numeric",
            "format": {"specifier": ".4f"},
        },
        {
            "name": "RetentionTime",
            "id": "RetentionTime",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {"name": "CollisionEnergy", "id": "CollisionEnergy"},
        {"name": "IonMode", "id": "IonMode"},
        {"name": "MsLevel", "id": "MsLevel"},
        {
            "name": "Doc-Topic Probability",
            "id": "Doc-Topic Probability",
            "type": "numeric",
            "format": {"specifier": ".4f"},
        },
        {
            "name": "Overlap Score",
            "id": "Overlap Score",
            "type": "numeric",
            "format": {"specifier": ".4f"},
        },
    ]
    spectra_ids = docs_df["SpecIndex"].tolist() if not docs_df.empty else []

    # 5) Add screening info
    screening_box = ""
    if screening_data:
        try:
            scdf = pd.read_json(screening_data, orient="records")
            if "user_auto_annotation" in scdf.columns:
                scdf["user_auto_annotation"] = scdf["user_auto_annotation"].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x),
                )
            hits_df = scdf[scdf["user_motif_id"] == motif_name].copy()
            hits_df = hits_df.sort_values("score", ascending=False)
            if not hits_df.empty:
                screening_table = dash_table.DataTable(
                    columns=[
                        {"name": "Ref Motif ID", "id": "ref_motif_id"},
                        {"name": "Ref ShortAnno", "id": "ref_short_annotation"},
                        {"name": "Ref MotifSet", "id": "ref_motifset"},
                        {
                            "name": "Score",
                            "id": "score",
                            "type": "numeric",
                            "format": {"specifier": ".4f"},
                        },
                    ],
                    data=hits_df.to_dict("records"),
                    page_size=5,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left"},
                )
                screening_box = html.Div(
                    [html.H5("Screening Annotations"), screening_table],
                    style={
                        "border": "1px dashed #999",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "margin-bottom": "5px",
                    },
                )
        except:
            pass
    spec2vec_div = html.Div([*spec2vec_container, screening_box])

    # 6) "Features" container
    features_div = html.Div(
        [
            html.Div(
                [
                    html.H5("Motif Features Table"),
                    feature_table_component,
                    html.P(f"Total Probability (Filtered): {total_prob:.4f}"),
                ],
            ),
            html.H5("Counts of Features within Filtered Motif Documents"),
            dcc.Graph(figure=barplot2_fig),
        ],
    )

    docs_div = html.Div(
        [
            html.P(
                "Below is the table of MS2 documents for the motif, subject to doc-topic probability & overlap score.",
            ),
        ],
    )

    # 7) Build the dual motif pseudo-spectrum plot
    motif_idx_opt = motif_idx

    # Prepare data for optimized motif
    raw_frag_mz = []
    raw_frag_int = []
    raw_loss_mz = []
    raw_loss_int = []

    if motif_idx_opt is not None and 0 <= motif_idx_opt < len(optimized_motifs_data):
        om = optimized_motifs_data[motif_idx_opt]
        raw_frag_mz = om.get("mz", [])
        raw_frag_int = om.get("intensities", [])
        if "metadata" in om and "losses" in om["metadata"]:
            for item in om["metadata"]["losses"]:
                raw_loss_mz.append(item["loss_mz"])
                raw_loss_int.append(item["loss_intensity"])

    # Prepare data for raw LDA motif
    raw_lda_frag_mz = []
    raw_lda_frag_int = []
    raw_lda_loss_mz = []
    raw_lda_loss_int = []

    for ft, val in filtered_motif_data.items():
        if ft.startswith("frag@"):
            try:
                raw_lda_frag_mz.append(float(ft.replace("frag@", "")))
                raw_lda_frag_int.append(val)
            except:
                pass
        elif ft.startswith("loss@"):
            try:
                raw_lda_loss_mz.append(float(ft.replace("loss@", "")))
                raw_lda_loss_int.append(val)
            except:
                pass

    # Calculate common x-axis range
    all_x_vals = raw_frag_mz + raw_loss_mz + raw_lda_frag_mz + raw_lda_loss_mz
    common_max = max(all_x_vals) if all_x_vals else 0
    fig_combined = make_subplots(
        rows=2, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.04,
    )

    # Optimised motif
    if optimised_fragloss_toggle == "both":
        if raw_frag_mz:
            fig_combined.add_bar(
                row=1,
                col=1,
                x=raw_frag_mz,
                y=raw_frag_int,
                marker_color="#1f77b4",
                width=bar_width,
                name="Optimised Fragments",
            )
        if raw_loss_mz:
            fig_combined.add_bar(
                row=1,
                col=1,
                x=raw_loss_mz,
                y=raw_loss_int,
                marker_color="#ff7f0e",
                width=bar_width,
                name="Optimised Losses",
            )
    elif optimised_fragloss_toggle == "fragments" and raw_frag_mz:
        fig_combined.add_bar(
            row=1,
            col=1,
            x=raw_frag_mz,
            y=raw_frag_int,
            marker_color="#1f77b4",
            width=bar_width,
            name="Optimised Fragments",
        )
    elif optimised_fragloss_toggle == "losses" and raw_loss_mz:
        fig_combined.add_bar(
            row=1,
            col=1,
            x=raw_loss_mz,
            y=raw_loss_int,
            marker_color="#ff7f0e",
            width=bar_width,
            name="Optimised Losses",
        )

    # Raw LDA motif
    if optimised_fragloss_toggle in ("both", "fragments") and raw_lda_frag_mz:
        fig_combined.add_bar(
            row=2,
            col=1,
            x=raw_lda_frag_mz,
            y=raw_lda_frag_int,
            marker_color="#1f77b4",
            width=bar_width,
            name="Raw Fragments",
        )

    if optimised_fragloss_toggle in ("both", "losses") and raw_lda_loss_mz:
        fig_combined.add_bar(
            row=2,
            col=1,
            x=raw_lda_loss_mz,
            y=raw_lda_loss_int,
            marker_color="#ff7f0e",
            width=bar_width,
            name="Raw Losses",
        )

    # Shared x-range:
    fig_combined.update_xaxes(range=[0, common_max * 1.1])

    # Row-specific y-axis titles:
    fig_combined.update_yaxes(title_text="Rel. Intensity", row=1, col=1)
    fig_combined.update_yaxes(
        title_text="Probability", autorange="reversed", row=2, col=1,
    )

    # Add title to the plot
    fig_combined.update_layout(
        title_text=f"Dual Motif Pseudo-Spectrum for {motif_name}",
    )

    apply_common_layout(fig_combined)  # new helper

    dual_plot = dcc.Graph(figure=fig_combined)

    return (
        motif_title,
        spec2vec_div,
        features_div,
        docs_div,
        spectra_ids,
        table_data,
        table_columns,
        dual_plot,
    )


@app.callback(
    Output("selected-spectrum-index", "data"),
    Output("spectra-table", "selected_rows"),
    Input("spectra-table", "selected_rows"),
    Input("next-spectrum", "n_clicks"),
    Input("prev-spectrum", "n_clicks"),
    Input("selected-motif-store", "data"),
    Input("motif-spectra-ids-store", "data"),
    State("selected-spectrum-index", "data"),
    prevent_initial_call=True,
)
def update_selected_spectrum(
    selected_rows,
    next_clicks,
    prev_clicks,
    selected_motif,
    motif_spectra_ids,
    current_index,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "spectra-table":
        if selected_rows:
            new_index = selected_rows[0]
            return new_index, selected_rows
        return current_index, dash.no_update

    if triggered_id == "next-spectrum":
        if motif_spectra_ids and current_index < len(motif_spectra_ids) - 1:
            new_index = current_index + 1
            return new_index, [new_index]
        return current_index, dash.no_update

    if triggered_id == "prev-spectrum":
        if motif_spectra_ids and current_index > 0:
            new_index = current_index - 1
            return new_index, [new_index]
        return current_index, dash.no_update

    if triggered_id in ["selected-motif-store", "motif-spectra-ids-store"]:
        return 0, [0]

    return current_index, dash.no_update


@app.callback(
    Output("spectrum-plot", "children"),
    Input("selected-spectrum-index", "data"),
    Input("probability-filter", "value"),
    Input("spectrum-fragloss-toggle", "value"),
    Input("spectrum-highlight-mode", "data"),
    Input("spectrum-show-parent-ion", "value"),
    Input("motif-spectra-ids-store", "data"),
    State("spectra-store", "data"),
    State("lda-dict-store", "data"),
    State("selected-motif-store", "data"),
)
def update_spectrum_plot(
    selected_index,
    probability_range,
    fragloss_mode,
    highlight_mode,
    show_parent_ion,
    spectra_ids,
    spectra_data,
    lda_dict_data,
    selected_motif,
):
    if spectra_ids and spectra_data and lda_dict_data and selected_motif:
        if (
            selected_index is None
            or selected_index < 0
            or selected_index >= len(spectra_ids)
        ):
            return html.Div("Selected spectrum index is out of range.")

        spectrum_id = spectra_ids[selected_index]
        spectrum_dict = spectra_data[spectrum_id]

        # Only pass the selected motif when in 'single' mode
        motif_to_highlight = selected_motif if highlight_mode == "single" else None

        fig = make_spectrum_plot(
            spectrum_dict,
            motif_to_highlight,
            lda_dict_data,
            probability_range=probability_range,
            mode=fragloss_mode,
            highlight_mode=highlight_mode,
            show_parent_ion=show_parent_ion,
        )
        return dcc.Graph(
            figure=fig, style={"width": "100%", "height": "600px", "margin": "auto"},
        )

    return ""


@app.callback(
    Output("motif-details-associated-motifs-list", "children"),
    Input("selected-spectrum-index", "data"),
    Input("motif-spectra-ids-store", "data"),
    State("spectra-store", "data"),
    State("lda-dict-store", "data"),
    State("selected-motif-store", "data"),
    prevent_initial_call=True,
)
def show_motif_details_associated_motifs(
    selected_index, spectra_ids, spectra_data, lda_dict_data, selected_motif,
):
    if (
        not spectra_ids
        or not spectra_data
        or not lda_dict_data
        or selected_motif is None
    ):
        return "No motifs to display."

    if (
        selected_index is None
        or selected_index < 0
        or selected_index >= len(spectra_ids)
    ):
        return "Selected spectrum index is out of range."

    spectrum_id = spectra_ids[selected_index]

    doc_theta = lda_dict_data["theta"].get(spectrum_id, {})
    if not doc_theta:
        return "No motifs found for this spectrum."

    # Sort by descending probability
    motif_probs = sorted(doc_theta.items(), key=lambda x: x[1], reverse=True)

    if not motif_probs:
        return "No motifs (above threshold) for this spectrum."

    # Build a list of clickable motif buttons
    layout_items = []
    for motif_id, prob in motif_probs:
        if prob < 0.01:
            continue

        # Check if this is the currently selected motif
        is_selected = motif_id == selected_motif

        motif_label = f"{motif_id} (Prob: {prob:.3f})"
        layout_items.append(
            html.Div(
                [
                    dbc.Button(
                        motif_label,
                        id={"type": "motif-details-motif-link", "index": motif_id},
                        color="primary" if is_selected else "secondary",
                        size="sm",
                        className="me-2",
                    ),
                ],
                className="d-flex align-items-center mb-1",
            ),
        )

    if not layout_items:
        return "No motifs (above threshold) for this spectrum."

    return layout_items


@app.callback(
    Output("spectrum-highlight-mode", "data", allow_duplicate=True),
    Output("selected-motif-store", "data", allow_duplicate=True),
    Input({"type": "motif-details-motif-link", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_motif_details_selected_motif_for_plot(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    import json

    motif_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    return "single", motif_id


@app.callback(
    Output("spectrum-highlight-mode", "data"),
    Output("spectrum-highlight-all-btn", "active"),
    Output("spectrum-highlight-none-btn", "active"),
    Input("spectrum-highlight-all-btn", "n_clicks"),
    Input("spectrum-highlight-none-btn", "n_clicks"),
    State("spectrum-highlight-mode", "data"),
    prevent_initial_call=True,
)
def update_motif_details_highlight_mode(all_clicks, none_clicks, current_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_mode, current_mode == "all", current_mode == "none"

    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "spectrum-highlight-all-btn":
        return "all", True, False
    if btn == "spectrum-highlight-none-btn":
        return "none", False, True

    return current_mode, current_mode == "all", current_mode == "none"


# -------------------------------- SCREENING --------------------------------


@app.callback(
    Output("m2m-folders-checklist", "options"),
    Output("m2m-subfolders-store", "data"),
    Input("tabs", "value"),
)
def auto_scan_m2m_subfolders(tab_value):
    if tab_value != "screening-tab":
        raise dash.exceptions.PreventUpdate

    if not os.path.exists(MOTIFDB_FOLDER):
        return ([], {})

    folder_options = []
    subfolder_data = {}
    for root, _dirs, files in os.walk(MOTIFDB_FOLDER):
        json_files = [
            f for f in files if f.startswith("Motifset") and f.endswith(".json")
        ]
        for jsonf in json_files:
            fullpath = os.path.join(root, jsonf)
            label = jsonf
            ms1_df, ms2_df = load_motifDB(fullpath)
            count_m2m = len(ms2_df["scan"].unique())
            folder_options.append(
                {"label": f"{label} ({count_m2m} motifs)", "value": fullpath},
            )
            subfolder_data[fullpath] = {"folder_label": label, "count_m2m": count_m2m}

    folder_options = sorted(folder_options, key=lambda x: x["label"].lower())
    return folder_options, subfolder_data


def make_spectrum_from_dict(spectrum_dict):
    """
    Creates a Mass2Motif object from a dictionary.
    This is used to reconstruct motifs from the stored data for MassQL queries.
    """
    # Extract fragments from the dictionary
    frag_mz = np.array(spectrum_dict.get("mz", []), dtype=float)
    frag_intensities = np.array(spectrum_dict.get("intensities", []), dtype=float)

    meta_ = spectrum_dict.get("metadata", {})

    # Extract losses from the metadata dictionary
    loss_mz = []
    loss_intensities = []
    if "losses" in meta_:
        for loss_item in meta_.get("losses", []):
            loss_mz.append(loss_item.get("loss_mz"))
            loss_intensities.append(loss_item.get("loss_intensity"))

    sp = Mass2Motif(
        frag_mz=frag_mz,
        frag_intensities=frag_intensities,
        loss_mz=np.array(loss_mz, dtype=float),
        loss_intensities=np.array(loss_intensities, dtype=float),
        metadata=meta_,
    )

    return sp


def filter_and_normalize_spectra(spectrum_list):
    """
    Filters out invalid spectra and normalizes the intensities of valid spectra.
    Fragments and losses are normalized together against the single highest peak, similar to
    the logic in create_spectrum method.
    """

    def trunc_annotation(val, max_len=40):
        """Truncate any string over max_len for readability."""
        if isinstance(val, str) and len(val) > max_len:
            return val[:max_len] + "..."
        return val

    valid = []
    for sp in spectrum_list:
        if not sp.peaks or len(sp.peaks.mz) == 0:
            continue

        # Clone so as not to modify the original object
        current_sp = sp.clone()

        # Normalize fragments and losses together
        frag_intensities = current_sp.peaks.intensities
        loss_intensities = np.array([])
        if current_sp.losses and len(current_sp.losses.mz) > 0:
            loss_intensities = current_sp.losses.intensities

        all_intensities = np.concatenate((frag_intensities, loss_intensities))
        if all_intensities.size == 0:
            continue

        max_intensity = np.max(all_intensities)
        if max_intensity <= 0:
            continue

        # Normalize if the max intensity is greater than 1.0
        normalized_frag_intensities = frag_intensities
        normalized_loss_intensities = loss_intensities
        if max_intensity > 1.0:
            normalized_frag_intensities = frag_intensities / max_intensity
            if loss_intensities.size > 0:
                normalized_loss_intensities = loss_intensities / max_intensity

        # Re-create Mass2Motif object with normalized values
        reconstructed_sp = Mass2Motif(
            frag_mz=current_sp.peaks.mz,
            frag_intensities=normalized_frag_intensities,
            loss_mz=current_sp.losses.mz if current_sp.losses else np.array([]),
            loss_intensities=normalized_loss_intensities,
            metadata=current_sp.metadata
        )

        # Handle annotation on the new object
        ann = reconstructed_sp.get("short_annotation", "")
        if isinstance(ann, list):
            joined = ", ".join(map(str, ann))
            joined = trunc_annotation(joined, 60)
            reconstructed_sp.set("short_annotation", joined)
        elif isinstance(ann, str):
            reconstructed_sp.set("short_annotation", trunc_annotation(ann, 60))

        valid.append(reconstructed_sp)

    return valid


@app.callback(
    Output("screening-fullresults-store", "data"),
    Output("compute-screening-status", "children"),
    Output("screening-progress", "value"),
    Output("compute-screening-button", "disabled"),
    Input("compute-screening-button", "n_clicks"),
    State("m2m-folders-checklist", "value"),
    State("lda-dict-store", "data"),
    State("s2v-model-path", "value"),
    State("optimized-motifs-store", "data"),
    prevent_initial_call=True,
)
def compute_spec2vec_screening(
    n_clicks, selected_folders, lda_dict_data, path_model, optimized_motifs_data,
):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    dbc.Alert(
        "Computing similarities... please wait (this may take a while).", color="info",
    )
    progress_val = 0
    button_disabled = True

    # check input
    if not selected_folders:
        return (
            None,
            dbc.Alert("No reference MotifDB set selected!", color="warning"),
            100,
            False,
        )
    if not lda_dict_data:
        return (
            None,
            dbc.Alert(
                "No LDA results found! Please run MS2LDA or load your data first.",
                color="warning",
            ),
            100,
            False,
        )

    # 1) Convert raw motifs from lda_dict_data['beta']
    user_motifs = []
    beta = lda_dict_data.get("beta", {})
    if not beta:
        return (
            None,
            dbc.Alert("lda_dict_data['beta'] is empty or missing!", color="warning"),
            100,
            False,
        )

    # Get run parameters used for this ms2lda anlaysis
    run_params = lda_dict_data.get("run_parameters", {})
    charge_to_use = run_params.get("dataset_parameters", {}).get("charge", 1)
    sig_digits_to_use = run_params.get("dataset_parameters", {}).get(
        "significant_digits", 2,
    )

    for motif_name, feature_probs_dict in beta.items():
        k = -1
        if motif_name.startswith("motif_"):
            with contextlib.suppress(ValueError):
                k = int(motif_name.replace("motif_", ""))

        motif_features_list = list(feature_probs_dict.items())
        raw_motif_spectrum = create_spectrum(
            motif_features_list,
            max(k, 0),
            frag_tag="frag@",
            loss_tag="loss@",
            significant_digits=sig_digits_to_use,
            charge=charge_to_use,
            motifset=motif_name,
        )
        user_motifs.append(raw_motif_spectrum)

    # 2) Filter & normalize the user_motifs
    user_motifs = filter_and_normalize_spectra(user_motifs)
    if not user_motifs:
        return (
            None,
            dbc.Alert("No valid user motifs after normalization!", color="warning"),
            100,
            False,
        )

    progress_val = 25

    # 3) Gather reference sets from selected_folders
    all_refs = []
    for json_file_path in selected_folders:
        these_refs = load_motifset_file(json_file_path)
        for r in these_refs:
            r.set("source_folder", json_file_path)
        all_refs.extend(these_refs)

    all_refs = filter_and_normalize_spectra(all_refs)
    if not all_refs:
        return (
            None,
            dbc.Alert(
                "No valid references found in the selected file(s)!", color="warning",
            ),
            100,
            False,
        )
    progress_val = 40

    # 4) Load Spec2Vec model
    s2v_sim = load_s2v_model(path_model=path_model)
    progress_val = 60

    # 5) Embeddings
    user_emb = calc_embeddings(s2v_sim, user_motifs)
    ref_emb = calc_embeddings(s2v_sim, all_refs)
    progress_val = 80

    # 6) Similarity
    sim_matrix = calc_similarity(user_emb, ref_emb)
    progress_val = 90

    # 7) Build an optimized_anno_map
    optimized_anno_map = {}
    if optimized_motifs_data:
        for om_entry in optimized_motifs_data:
            om_meta = om_entry.get("metadata", {})
            om_id = om_meta.get("id")  # e.g. "motif_0"
            om_anno = om_meta.get("auto_annotation", "")
            if om_id:
                optimized_anno_map[om_id] = om_anno

    # 8) Build results
    results = []
    for user_i, user_sp in enumerate(user_motifs):
        user_id = user_sp.get("id", "")
        # Use annotation from optimized_anno_map if present
        user_anno = optimized_anno_map.get(user_id, "")
        for ref_j, ref_sp in enumerate(all_refs):
            score = sim_matrix.iloc[ref_j, user_i]
            ref_id = ref_sp.get("id", "")
            ref_anno = ref_sp.get("short_annotation", "")
            ref_motifset = ref_sp.get("motifset", "")

            results.append(
                {
                    "user_motif_id": user_id,
                    "user_auto_annotation": user_anno,
                    "ref_motif_id": ref_id,
                    "ref_short_annotation": ref_anno,
                    "ref_motifset": ref_motifset,
                    "score": round(float(score), 4),
                },
            )

    if not results:
        return (
            None,
            dbc.Alert("No results after similarity!", color="warning"),
            100,
            False,
        )

    # Sort descending
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    json_data = df.to_json(orient="records")

    progress_val = 100
    msg = dbc.Alert(
        f"Computed {len(df)} total matches from {len(user_motifs)} user motifs and {len(all_refs)} references.",
        color="success",
    )
    button_disabled = False

    return Serverside(df.to_json(orient="records")), msg, progress_val, button_disabled


@app.callback(
    Output("screening-results-table", "data"),
    Output("screening-threshold-value", "children"),
    Input("screening-fullresults-store", "data"),
    Input("screening-threshold-slider", "value"),
)
def filter_screening_results(fullresults_json, threshold):
    if not fullresults_json:
        return [], ""

    # The "FutureWarning" can appear for read_json on raw strings.
    # We can ignore or wrap with io.StringIO. For now, ignoring is fine.
    df = pd.read_json(fullresults_json, orient="records")

    # Convert any list in user_auto_annotation to a comma-joined string
    if "user_auto_annotation" in df.columns:
        df["user_auto_annotation"] = df["user_auto_annotation"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x),
        )

    filtered = df[df["score"] >= threshold].copy()
    filtered = filtered.sort_values("score", ascending=False)
    table_data = filtered.to_dict("records")
    label = f"Minimum Similarity: {threshold:.2f} — {len(filtered)}/{len(df)} results"
    return table_data, label


@app.callback(
    Output("download-screening-csv", "data"),
    Output("download-screening-json", "data"),
    Input("save-screening-csv", "n_clicks"),
    Input("save-screening-json", "n_clicks"),
    State("screening-results-table", "data"),
    prevent_initial_call=True,
)
def save_screening_results(csv_click, json_click, table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if not table_data:
        return no_update, no_update

    df = pd.DataFrame(table_data)
    if button_id == "save-screening-csv":
        return (
            dcc.send_data_frame(df.to_csv, "screening_results.csv", index=False),
            no_update,
        )
    if button_id == "save-screening-json":
        out_str = df.to_json(orient="records")
        return no_update, {"content": out_str, "filename": "screening_results.json"}
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("download-motifranking-csv", "data"),
    Output("download-motifranking-json", "data"),
    Input("save-motifranking-csv", "n_clicks"),
    Input("save-motifranking-json", "n_clicks"),
    State("motif-rankings-table", "data"),
    prevent_initial_call=True,
)
def save_motifranking_results(csv_click, json_click, table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if not table_data:
        return no_update, no_update

    df = pd.DataFrame(table_data)
    if button_id == "save-motifranking-csv":
        return (
            dcc.send_data_frame(df.to_csv, "motifranking_results.csv", index=False),
            no_update,
        )
    if button_id == "save-motifranking-json":
        out_str = df.to_json(orient="records")
        return no_update, {"content": out_str, "filename": "motifranking_results.json"}
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("selected-motif-store", "data", allow_duplicate=True),
    [
        Input("motif-rankings-table", "active_cell"),
        Input("screening-results-table", "active_cell"),
    ],
    [
        State("motif-rankings-table", "data"),
        State("screening-results-table", "data"),
        State("motif-rankings-table", "derived_viewport_data"),
    ],
    prevent_initial_call=True,
)
def on_motif_click(
    ranking_active_cell,
    screening_active_cell,
    ranking_data,
    screening_data,
    ranking_dv_data,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "motif-rankings-table":
        if ranking_active_cell and ranking_dv_data:
            col_id = ranking_active_cell["column_id"]
            row_id = ranking_active_cell["row"]
            if col_id == "Motif":
                return ranking_dv_data[row_id]["Motif"]
        raise dash.exceptions.PreventUpdate

    if triggered_id == "screening-results-table":
        if screening_active_cell and screening_data:
            col_id = screening_active_cell["column_id"]
            row_id = screening_active_cell["row"]
            if col_id == "user_motif_id":
                return screening_data[row_id]["user_motif_id"]
        raise dash.exceptions.PreventUpdate

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("download-s2v-status", "children"),
    Output("s2v-download-complete", "data"),
    Input("download-s2v-button", "n_clicks"),
    prevent_initial_call=True,
)
def unlock_run_after_download(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    try:
        msg = download_model_and_data()
        return (msg, "Spec2Vec model + data download complete.")
    except Exception as e:
        return f"Download failed: {e!s}", ""


@app.callback(
    Output("spectra-search-parentmass-slider-display", "children"),
    Input("spectra-search-parentmass-slider", "value"),
)
def display_parentmass_range(value):
    if value:
        return f"Selected Parent Mass Range: {value[0]:.2f} - {value[1]:.2f}"
    return "Selected Parent Mass Range: N/A"


@app.callback(
    Output("search-tab-selected-spectrum-details-store", "data"),
    Output("search-tab-spectrum-details-container", "style"),
    Output("search-tab-selected-motif-id-for-plot-store", "data", allow_duplicate=True),
    Input("spectra-search-results-table", "active_cell"),
    State("spectra-search-results-table", "data"),
    prevent_initial_call=True,
)
def handle_spectrum_selection(active_cell, table_data):
    if not active_cell or not table_data:
        return None, {"marginTop": "20px", "display": "none"}, None

    if active_cell.get("column_id") != "spec_id":
        # Clicked some other column – ignore
        raise dash.exceptions.PreventUpdate

    row_idx = active_cell.get("row", -1)
    if row_idx < 0 or row_idx >= len(table_data):
        return None, {"marginTop": "20px", "display": "none"}, None

    selected_spectrum = table_data[row_idx]

    container_style = {"marginTop": "20px", "display": "block"}
    return selected_spectrum, container_style, None


@app.callback(
    Output("search-highlight-mode", "data"),
    Output("search-highlight-all-btn", "active"),
    Output("search-highlight-none-btn", "active"),
    Input("search-highlight-all-btn", "n_clicks"),
    Input("search-highlight-none-btn", "n_clicks"),
    State("search-highlight-mode", "data"),
    prevent_initial_call=True,
)
def update_highlight_mode(all_clicks, none_clicks, current_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_mode, current_mode == "all", current_mode == "none"

    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "search-highlight-all-btn":
        return "all", True, False
    if btn == "search-highlight-none-btn":
        return "none", False, True

    return current_mode, current_mode == "all", current_mode == "none"


@app.callback(
    Output("search-tab-spectrum-title", "children"),
    Input("search-tab-selected-spectrum-details-store", "data"),
    prevent_initial_call=True,
)
def update_spectrum_title(spectrum_info):
    if not spectrum_info:
        return "Spectrum Details (No Spectrum Selected)"

    spec_id = spectrum_info.get("spec_id", "Unknown")
    return f"Spectrum Details: {spec_id}"


@app.callback(
    Output("search-tab-associated-motifs-list", "children"),
    Input("search-tab-selected-spectrum-details-store", "data"),
    State("lda-dict-store", "data"),
    prevent_initial_call=True,
)
def show_associated_motifs(spectrum_info, lda_dict_data):
    if not spectrum_info:
        raise PreventUpdate

    spec_id = spectrum_info.get("spec_id")
    if not spec_id or not lda_dict_data:
        return "No motifs to display."

    doc_theta = lda_dict_data["theta"].get(spec_id, {})
    if not doc_theta:
        return "No motifs found for this spectrum."

    # Sort by descending probability
    motif_probs = sorted(doc_theta.items(), key=lambda x: x[1], reverse=True)

    if not motif_probs:
        return "No motifs (above threshold) for this spectrum."

    # Build a list of clickable motif name + "Details" button
    layout_items = []
    for motif_id, prob in motif_probs:
        if prob < 0.01:
            continue
        motif_label = f"{motif_id} (Prob: {prob:.3f})"
        layout_items.append(
            html.Div(
                [
                    dbc.Button(
                        motif_label,
                        id={"type": "search-tab-motif-link", "index": motif_id},
                        color="secondary",
                        size="sm",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Motif Details ↗",
                        id={"type": "search-tab-motif-details-btn", "index": motif_id},
                        size="sm",
                        outline=True,
                        color="info",
                        className="me-2",
                    ),
                ],
                className="d-flex align-items-center mb-1",
            ),
        )

    if not layout_items:
        return "No motifs (above threshold) for this spectrum."

    return layout_items


@app.callback(
    Output("selected-motif-store", "data", allow_duplicate=True),
    Output("tabs", "value", allow_duplicate=True),
    Input({"type": "search-tab-motif-details-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def jump_to_motif_details(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    actual_click_detected = False
    motif_id_from_click = None

    for _i, n_clicks_val in enumerate(n_clicks_list):
        if n_clicks_val is not None and n_clicks_val > 0:
            # This indicates a genuine click on one of the buttons.
            # Get the ID of the exact button that was clicked.
            # The triggered_id_str will be like: '{"index":"motif_X","type":"search-tab-motif-details-btn"}.n_clicks'
            triggered_id_str = ctx.triggered[0]["prop_id"].split(".")[0]
            try:
                pmid = json.loads(triggered_id_str)  # pmid = parsed motif id (dict)
                motif_id_from_click = pmid["index"]
                actual_click_detected = True
                break  # Process the first genuine click
            except Exception:
                # Log error or handle appropriately
                raise PreventUpdate

    if actual_click_detected and motif_id_from_click:
        return motif_id_from_click, "motif-details-tab"
    # No genuine click was found (e.g., callback triggered by component creation/update)
    raise PreventUpdate


@app.callback(
    Output("search-tab-selected-motif-id-for-plot-store", "data", allow_duplicate=True),
    Output("search-highlight-mode", "data", allow_duplicate=True),
    Input({"type": "search-tab-motif-link", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_selected_motif_for_plot(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    import json

    motif_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    return motif_id, "single"


@app.callback(
    Output("tabs", "value", allow_duplicate=True),
    Output("search-tab-selected-spectrum-details-store", "data", allow_duplicate=True),
    Output("search-tab-selected-motif-id-for-plot-store", "data"),
    Output("search-tab-spectrum-details-container", "style", allow_duplicate=True),
    Input("jump-to-search-btn", "n_clicks"),
    State("selected-spectrum-index", "data"),
    State("motif-spectra-ids-store", "data"),
    State("spectra-store", "data"),
    State("selected-motif-store", "data"),
    prevent_initial_call=True,
)
def jump_to_search_tab(
    n_clicks, selected_spectrum_index, motif_spectra_ids, spectra_store, selected_motif,
):
    """
    Handles the "Spectrum Details ↗" button click in the Motif Details tab.
    Switches to the Search Spectra tab with the same spectrum selected and the same motif highlighted.
    """
    if not n_clicks:
        raise PreventUpdate

    # Defensive bounds-check
    if (
        selected_spectrum_index is None
        or selected_spectrum_index < 0
        or selected_spectrum_index >= len(motif_spectra_ids)
    ):
        raise PreventUpdate

    # Get the real spectrum index
    spec_idx = motif_spectra_ids[selected_spectrum_index]

    # Shallow copy to avoid mutating the cache
    spectrum_info = spectra_store[spec_idx].copy()

    # Add the extra keys the search tab expects if they're not already there
    if "spec_id" not in spectrum_info:
        if "metadata" in spectrum_info and "id" in spectrum_info["metadata"]:
            spectrum_info["spec_id"] = spectrum_info["metadata"]["id"]
        else:
            spectrum_info["spec_id"] = f"spec_{spec_idx}"

    if "original_spec_index" not in spectrum_info:
        spectrum_info["original_spec_index"] = spec_idx

    container_style = {"marginTop": "20px", "display": "block"}
    return "search-spectra-tab", spectrum_info, selected_motif, container_style


@app.callback(
    Output("search-tab-spectrum-plot-container", "children"),
    Input("search-tab-selected-spectrum-details-store", "data"),
    Input("search-tab-selected-motif-id-for-plot-store", "data"),
    Input("search-fragloss-toggle", "value"),
    Input("search-highlight-mode", "data"),
    Input("search-show-parent-ion", "value"),
    State("spectra-store", "data"),
    State("lda-dict-store", "data"),
    prevent_initial_call=True,
)
def update_search_tab_spectrum_plot(
    spectrum_info,
    motif_for_plot,
    fragloss_mode,
    highlight_mode,
    show_parent_ion,
    all_spectra_data,
    lda_dict_data,
):
    if not spectrum_info:
        raise PreventUpdate

    idx = spectrum_info.get("original_spec_index", -1)
    if idx < 0 or idx >= len(all_spectra_data):
        return html.Div("Invalid spectrum index.", style={"color": "red"})

    spec_dict = all_spectra_data[idx]
    motif_to_highlight = motif_for_plot if highlight_mode == "single" else None

    fig = make_spectrum_plot(
        spec_dict,
        motif_to_highlight,
        lda_dict_data,
        mode=fragloss_mode,
        highlight_mode=highlight_mode,
        show_parent_ion=show_parent_ion,
    )
    return dcc.Graph(figure=fig)


@app.callback(
    Output("spectra-search-results-table", "data"),
    Output("spectra-search-status-message", "children"),  # NEW OUTPUT
    Input("spectra-store", "data"),
    Input("spectra-search-fragloss-input", "value"),
    Input("spectra-search-parentmass-slider", "value"),
    prevent_initial_call=True,
)
def update_spectra_search_table(spectra_data, search_text, parent_mass_range):
    if not spectra_data:
        raise PreventUpdate

    # Prepare query and parent mass bounds
    query = (search_text or "").strip().lower()
    frag_query = None
    loss_query = None
    if query:
        frag_match = re.search(r"frag@(\d+(?:\.\d+)?)", query)
        loss_match = re.search(r"loss@(\d+(?:\.\d+)?)", query)
        if frag_match:
            try:
                frag_query = float(frag_match.group(1))
            except ValueError:
                frag_query = None
        if loss_match:
            try:
                loss_query = float(loss_match.group(1))
            except ValueError:
                loss_query = None
    pmass_low, pmass_high = parent_mass_range

    filtered_rows = []
    for i, spec_dict in enumerate(spectra_data):
        meta = spec_dict.get("metadata", {})
        pmass = meta.get("precursor_mz", None)
        if pmass is None:
            continue
        if not (pmass_low <= pmass <= pmass_high):
            continue

        frag_list = [f"frag@{mzval:.4g}" for mzval in spec_dict["mz"]]
        frag_vals = [float(mzval) for mzval in spec_dict["mz"]]
        loss_list = []
        loss_vals = []
        if "losses" in meta:
            for loss_item in meta["losses"]:
                loss_list.append(f"loss@{loss_item['loss_mz']:.4g}")
                with contextlib.suppress(KeyError, ValueError):
                    loss_vals.append(float(loss_item["loss_mz"]))

        if query:
            if frag_query is not None:
                if not any(abs(mz - frag_query) <= 0.01 for mz in frag_vals):
                    continue
            elif loss_query is not None:
                if not any(abs(mz - loss_query) <= 0.01 for mz in loss_vals):
                    continue
            else:
                combined = frag_list + loss_list
                if not any(query in x.lower() for x in combined):
                    continue

        filtered_rows.append(
            {
                "spec_id": meta.get("id", ""),
                "parent_mass": pmass,
                "feature_id": meta.get("feature_id", ""),
                "fragments": ", ".join(frag_list),
                "losses": ", ".join(loss_list),
                "original_spec_index": i,  # Store the original index for later use
            },
        )

    # Build a simple status message
    status_msg = ""
    default_range = (0, 2000)
    # Check if user has any actual filter
    is_filtered = (query != "") or (
        pmass_low != default_range[0] or pmass_high != default_range[1]
    )

    if filtered_rows:
        status_msg = f"Showing {len(filtered_rows)} matching spectra."
    elif is_filtered:
        status_msg = "No spectra found matching your criteria."
    else:
        status_msg = ""  # No filters active yet

    return filtered_rows, status_msg


app.clientside_callback(
    """
    function(style) {
        if (style && style.display === "block") {
            const el = document.getElementById('search-tab-spectrum-details-container');
            if (el) {
                // slight delay so the layout has finished updating
                setTimeout(() => el.scrollIntoView({behavior: 'smooth', block: 'start'}), 50);
            }
        }
        return '';
    }
    """,
    Output("search-scroll-dummy", "children"),
    Input("search-tab-spectrum-details-container", "style"),
)

app.clientside_callback(
    """
    function(style) {
        if (style && style.display === "block") {
            // give Dash a tick to render, then scroll
            setTimeout(() => window.scrollTo({top: 0, behavior: 'smooth'}), 50);
        }
        return '';
    }
    """,
    Output("motif-details-scroll-dummy", "children"),
    Input("motif-details-tab-content", "style"),
)
