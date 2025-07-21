# App/callbacks/screening.py
"""Callbacks for Motif Search (Screening) and Spectra Search tabs."""

import json
import dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Serverside
from dash import Input, Output, State, ALL, html, no_update
from dash.exceptions import PreventUpdate

from App.app_instance import app
from App.callbacks.common import *  # Import helper functions


# -------------------------------- SCREENING --------------------------------# -------------------------------- SCREENING --------------------------------


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
        State("screening-results-table", "derived_viewport_data"),
    ],
    prevent_initial_call=True,
)
def on_motif_click(
    ranking_active_cell,
    screening_active_cell,
    ranking_data,
    screening_data,
    ranking_dv_data,
    screening_dv_data,
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
        if screening_active_cell:
            col_id = screening_active_cell["column_id"]
            row_id = screening_active_cell["row"]
            if col_id == "user_motif_id":
                # Use derived_viewport_data if available (for pagination support)
                if screening_dv_data:
                    return screening_dv_data[row_id]["user_motif_id"]
                else:
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
    Output("spectra-search-parentmass-slider", "min"),
    Output("spectra-search-parentmass-slider", "max"),
    Output("spectra-search-parentmass-slider", "value"),
    Output("spectra-search-parentmass-slider", "marks"),
    Input("spectra-store", "data"),
    prevent_initial_call=True,
)
def update_parentmass_slider_range(spectra_data):
    if not spectra_data:
        # Default values if no data is available
        return 0, 2000, [0, 2000], {0: "0", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"}

    # Extract parent masses from spectra data
    parent_masses = []
    for spec_dict in spectra_data:
        meta = spec_dict.get("metadata", {})
        pmass = meta.get("precursor_mz", None)
        if pmass is not None:
            parent_masses.append(pmass)

    if not parent_masses:
        # No valid parent masses found
        return 0, 2000, [0, 2000], {0: "0", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"}

    # Calculate min and max with some margin
    min_mass = max(0, min(parent_masses) - 50)  # Add 50 Da margin, but not below 0
    max_mass = max(parent_masses) + 50  # Add 50 Da margin

    # Round to nice values
    min_mass = int(min_mass // 10) * 10  # Round down to nearest 10
    max_mass = int((max_mass + 9) // 10) * 10  # Round up to nearest 10

    # Create marks at reasonable intervals
    range_size = max_mass - min_mass
    if range_size <= 100:
        step = 20
    elif range_size <= 500:
        step = 100
    elif range_size <= 1000:
        step = 200
    else:
        step = 500

    marks = {}
    for i in range(min_mass, max_mass + 1, step):
        marks[i] = str(i)

    # Always include min and max in marks
    marks[min_mass] = str(min_mass)
    marks[max_mass] = str(max_mass)

    return min_mass, max_mass, [min_mass, max_mass], marks

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
    State("spectra-search-results-table", "derived_viewport_data"),
    prevent_initial_call=True,
)
def handle_spectrum_selection(active_cell, table_data, derived_viewport_data):
    if not active_cell or not table_data:
        return None, {"marginTop": "20px", "display": "none"}, None

    if active_cell.get("column_id") != "spec_id":
        # Clicked some other column – ignore
        raise dash.exceptions.PreventUpdate

    row_idx = active_cell.get("row", -1)
    if row_idx < 0 or (derived_viewport_data and row_idx >= len(derived_viewport_data)) or (not derived_viewport_data and row_idx >= len(table_data)):
        return None, {"marginTop": "20px", "display": "none"}, None

    # Use derived_viewport_data if available (for pagination support)
    if derived_viewport_data:
        selected_spectrum = derived_viewport_data[row_idx]
    else:
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
    Input("spectra-search-fragment-checkbox", "value"),
    Input("spectra-search-loss-checkbox", "value"),
    Input("spectra-search-results-table", "sort_by"),
    prevent_initial_call=True,
)
def update_spectra_search_table(spectra_data, search_text, parent_mass_range, fragment_checked, loss_checked, sort_by):
    if not spectra_data:
        raise PreventUpdate

    # Prepare query and parent mass bounds
    query = (search_text or "").strip().lower()
    query_value = None

    # Try to parse the input as a numeric value directly
    try:
        if query:
            query_value = float(query)
    except ValueError:
        query_value = None
    pmass_low, pmass_high = parent_mass_range

    filtered_rows = []
    for i, spec_dict in enumerate(spectra_data):
        meta = spec_dict.get("metadata", {})
        pmass = meta.get("precursor_mz", None)
        if pmass is None:
            continue
        if not (pmass_low <= pmass <= pmass_high):
            continue

        frag_list = [f"frag@{mzval:.5f}" for mzval in spec_dict["mz"]]
        frag_vals = [float(mzval) for mzval in spec_dict["mz"]]
        loss_list = []
        loss_vals = []
        if "losses" in meta:
            for loss_item in meta["losses"]:
                loss_list.append(f"loss@{loss_item['loss_mz']:.5f}")
                with contextlib.suppress(KeyError, ValueError):
                    loss_vals.append(float(loss_item["loss_mz"]))

        if query:
            # Only use numeric comparison for searching
            skip_spectrum = True

            # If we have a valid numeric query
            if query_value is not None:
                # Check fragments if fragment checkbox is checked
                if fragment_checked:
                    if any(abs(mz - query_value) <= 0.01 for mz in frag_vals):
                        skip_spectrum = False

                # Check losses if loss checkbox is checked
                if loss_checked:
                    if any(abs(mz - query_value) <= 0.01 for mz in loss_vals):
                        skip_spectrum = False

            # If neither fragment nor loss query matched, or if query doesn't match expected format, skip this spectrum
            if skip_spectrum:
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
        status_msg = f"{len(filtered_rows)} spectra pass the filter"
    elif is_filtered:
        status_msg = "No spectra pass the filter"
    else:
        status_msg = ""  # No filters active yet

    # Apply sorting if sort_by is provided
    if sort_by and len(sort_by):
        col_id = sort_by[0]['column_id']
        direction = sort_by[0]['direction']
        ascending = (direction == 'asc')

        # Convert filtered_rows to DataFrame for easier sorting
        df = pd.DataFrame(filtered_rows)

        if col_id == 'spec_id':
            # Implement natural sorting for the 'spec_id' column
            # 1. Extract the number from the 'spec_XXX' string
            # 2. Convert it to an integer so it sorts numerically
            # 3. Sort by this new numeric key
            # 4. Drop the temporary key column
            df['_sort_key'] = df['spec_id'].str.extract(r'(\d+)').astype(int)
            df = df.sort_values('_sort_key', ascending=ascending).drop(columns=['_sort_key'])
        else:
            # For all other columns, use standard pandas sorting
            df = df.sort_values(col_id, ascending=ascending)

        # Convert back to list of dictionaries
        filtered_rows = df.to_dict('records')

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
