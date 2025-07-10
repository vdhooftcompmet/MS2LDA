# App/callbacks/rankings_details.py
"""Callbacks for Motif Rankings and Motif Details tabs."""

import dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Serverside
from dash import Input, Output, State, ALL, no_update, html
from dash.exceptions import PreventUpdate

from App.app_instance import app
from App.callbacks.common import *  # Import helper functions
from App.callbacks.screening import make_spectrum_from_dict


# -------------------------------- RANKINGS & DETAILS --------------------------------# -------------------------------- RANKINGS & DETAILS --------------------------------


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
    Output("motif-ranking-massql-input", "value"),
    Output("motif-ranking-massql-matches", "data"),
    Output("motif-ranking-massql-error", "children"),
    Output("motif-ranking-massql-error", "style"),
    Input("motif-ranking-massql-btn", "n_clicks"),
    Input("motif-ranking-massql-reset-btn", "n_clicks"),
    State("motif-ranking-massql-input", "value"),
    State("optimized-motifs-store", "data"),
    prevent_initial_call=True,
)
def handle_massql_query(run_clicks, reset_clicks, query, motifs_data):
    # Use callback_context to determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Default error style (hidden)
    error_style = {
        "marginTop": "10px",
        "color": "#dc3545",
        "padding": "10px",
        "backgroundColor": "#f8d7da",
        "borderRadius": "5px",
        "display": "none"
    }

    # Handle Reset Query button
    if button_id == "motif-ranking-massql-reset-btn":
        return "", [], "", error_style

    # Handle Run Query button
    if button_id == "motif-ranking-massql-btn":
        if not query or not motifs_data:
            raise PreventUpdate

        specs = [make_spectrum_from_dict(d) for d in motifs_data]
        ms1_df, ms2_df = motifs2motifDB(specs)

        try:
            matches = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)

            # If no results
            if matches.empty or "motif_id" not in matches.columns:
                return no_update, [], "No motifs match the query criteria.", {
                    **error_style,
                    "display": "block"
                }

            return no_update, matches["motif_id"].unique().tolist(), "", error_style

        except UnexpectedCharacters as e:
            # Extract error message and format it for display
            error_message = str(e)
            user_friendly_message = f"Invalid MassQL query: {error_message}"

            # Show the error message
            visible_error_style = {
                **error_style,
                "display": "block"
            }

            return no_update, [], user_friendly_message, visible_error_style

        except Exception as e:
            # Handle any other exceptions
            error_message = str(e)
            user_friendly_message = f"Error processing query: {error_message}"

            # Show the error message
            visible_error_style = {
                **error_style,
                "display": "block"
            }

            return no_update, [], user_friendly_message, visible_error_style

    # Default case (should not happen)
    raise PreventUpdate


@app.callback(
    Output("motif-rankings-table", "data"),
    Output("motif-rankings-table", "columns"),
    Output("motif-rankings-count", "children"),
    Input("lda-dict-store", "data"),
    Input("probability-thresh", "value"),
    Input("overlap-thresh", "value"),
    Input("tabs", "value"),
    Input("motif-ranking-massql-matches", "data"),
    Input("motif-rankings-table", "sort_by"),
    State("screening-fullresults-store", "data"),
    State("optimized-motifs-store", "data"),
)
def update_motif_rankings_table(
    lda_dict_data,
    probability_thresh,
    overlap_thresh,
    active_tab,
    massql_matches,
    sort_by,
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

    # Filter out motifs that have no docs passing, i.e. degree=0
    # Do this before applying MassQL filter to get accurate count of motifs passing probability/overlap thresholds
    df = df[df["Degree"] > 0].copy()

    # Store the count of motifs passing probability/overlap thresholds
    motifs_passing_thresholds = len(df)

    if massql_matches is not None and len(massql_matches) > 0:
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

    if sort_by and len(sort_by):
        col_id = sort_by[0]['column_id']
        direction = sort_by[0]['direction']
        ascending = (direction == 'asc')

        if col_id == 'Motif':
            # Implement natural sorting for the 'Motif' column
            # 1. Extract the number from the 'motif_XXX' string
            # 2. Convert it to an integer so it sorts numerically
            # 3. Sort by this new numeric key
            # 4. Drop the temporary key column
            df['_sort_key'] = df['Motif'].str.extract(r'(\d+)').astype(int)
            df = df.sort_values('_sort_key', ascending=ascending).drop(columns=['_sort_key'])
        else:
            # For all other columns, use standard pandas sorting
            df = df.sort_values(col_id, ascending=ascending)

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

    # Create message showing both the number of motifs passing thresholds and the number displayed after MassQL filtering
    if massql_matches is not None and len(massql_matches) > 0:
        row_count_message = f"{motifs_passing_thresholds} motif(s) pass the filter, {len(df)} displayed after MassQL query"
    else:
        row_count_message = f"{motifs_passing_thresholds} motif(s) pass the filter"
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
        filter_action="native",
        style_data_conditional=[
            {
                'if': {'state': 'active'},
                'backgroundColor': 'transparent',
                'border': 'transparent'
            }
        ],
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


