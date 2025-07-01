from __future__ import annotations

import base64
import contextlib
import gzip
import io
import json
import os
import re
import tempfile
import requests
import tempfile
import os
import gzip
import json
import pickle

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
from lark.exceptions import UnexpectedCharacters
from matchms import Fragments, Spectrum
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

import ms2lda
from App.app_instance import MOTIFDB_DIR, app
from ms2lda.Add_On.MassQL.MassQL4MotifDB import (
    load_motifDB,
    motifDB2motifs,
    motifs2motifDB,
)
from ms2lda.Add_On.Spec2Vec.annotation import calc_embeddings
from ms2lda.Add_On.Spec2Vec.annotation_refined import calc_similarity
from ms2lda.Mass2Motif import Mass2Motif
from ms2lda.Preprocessing.load_and_clean import clean_spectra
from ms2lda.run import filetype_check, load_s2v_model
from ms2lda.utils import create_spectrum, download_model_and_data
from ms2lda.Visualisation.ldadict import generate_corpusjson_from_tomotopy

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


# -------------------------------- COMMON CALLBACKS --------------------------------

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


