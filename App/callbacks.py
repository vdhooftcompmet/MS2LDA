import base64
import gzip
import io
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
from dash import Input, Output, State, no_update, dcc
from dash import dash_table
from dash import html
from dash.exceptions import PreventUpdate
from matchms import Fragments
from matchms import Spectrum
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

import MS2LDA
from App.app_instance import app
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import load_motifDB, motifDB2motifs
from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings
from MS2LDA.Add_On.Spec2Vec.annotation_refined import calc_similarity
from MS2LDA.Mass2Motif import Mass2Motif
from MS2LDA.Preprocessing.load_and_clean import clean_spectra
from MS2LDA.Visualisation.ldadict import generate_corpusjson_from_tomotopy
from MS2LDA.run import filetype_check, load_s2v_model
from MS2LDA.utils import download_model_and_data, create_spectrum

# Hardcode the path for .m2m references
MOTIFDB_FOLDER = "./MS2LDA/MotifDB"


def load_motifset_file(json_path):
    """
    Loads a single JSON motifset file.
    Returns a list of motifs in the file as matchms Spectra.
    """
    ms1_df, ms2_df = load_motifDB(json_path)
    motifs = motifDB2motifs(ms2_df)
    return motifs


# Callback to show/hide tab contents based on active tab
@app.callback(
    Output("run-analysis-tab-content", "style"),
    Output("load-results-tab-content", "style"),
    Output("results-tab-content", "style"),
    Output("motif-rankings-tab-content", "style"),
    Output("motif-details-tab-content", "style"),
    Output("screening-tab-content", "style"),
    Input("tabs", "value"),
)
def toggle_tab_content(active_tab):
    run_style = {"display": "none"}
    load_style = {"display": "none"}
    results_style = {"display": "none"}
    motif_rankings_style = {"display": "none"}
    motif_details_style = {"display": "none"}
    screening_style = {"display": "none"}

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

    return (
        run_style,
        load_style,
        results_style,
        motif_rankings_style,
        motif_details_style,
        screening_style,
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
            data = parse_ms2lda_viz_file(results_contents)
        except ValueError as e:
            load_status = dbc.Alert(f"Error parsing the file: {str(e)}", color="danger")
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
        raise ValueError(f"Error decoding or parsing MS2LDA viz file: {str(e)}")


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
def update_cytoscape(optimized_motifs_data, clustered_smiles_data, active_tab, edge_intensity_threshold,
                     toggle_loss_edge, layout_choice):
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

def compute_motif_degrees(lda_dict, p_low, p_high, o_low, o_high):
    motifs = lda_dict["beta"].keys()
    motif_degrees = {m: 0 for m in motifs}
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
    Output("motif-rankings-table", "data"),
    Output("motif-rankings-table", "columns"),
    Output("motif-rankings-count", "children"),
    Input("lda-dict-store", "data"),
    Input("probability-thresh", "value"),
    Input("overlap-thresh", "value"),
    Input("tabs", "value"),
    State("screening-fullresults-store", "data"),
    State("optimized-motifs-store", "data"),
)
def update_motif_rankings_table(lda_dict_data, probability_thresh, overlap_thresh, active_tab,
                                screening_data, optimized_motifs_data):
    if active_tab != 'motif-rankings-tab' or not lda_dict_data:
        return [], [], ""

    p_low, p_high = probability_thresh
    o_low, o_high = overlap_thresh

    motif_degree_list = compute_motif_degrees(lda_dict_data, p_low, p_high, o_low, o_high)
    df = pd.DataFrame(motif_degree_list, columns=[
        'Motif',
        'Degree',
        'Average Doc-Topic Probability',
        'Average Overlap Score'
    ])

    # 1) topic_metadata from LDA
    motif_annotations = {}
    if 'topic_metadata' in lda_dict_data:
        for motif, metadata in lda_dict_data['topic_metadata'].items():
            motif_annotations[motif] = metadata.get('annotation', '')

    # 2) short_annotation from optimized_motifs_store
    combined_annotations = []
    for motif_name in df['Motif']:
        existing_lda_anno = motif_annotations.get(motif_name, "")
        short_anno_str = ""

        # parse motif index
        motif_idx = None
        if motif_name.startswith("motif_"):
            try:
                motif_idx = int(motif_name.replace("motif_", ""))
            except ValueError:
                pass

        if optimized_motifs_data and motif_idx is not None and 0 <= motif_idx < len(optimized_motifs_data):
            # short_annotation might be list of SMILES or None
            short_anno = optimized_motifs_data[motif_idx]["metadata"].get("auto_annotation", "")
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

    df['Annotation'] = combined_annotations

    # 3) Screening references in new column
    screening_hits = []
    if screening_data:
        try:
            scdf = pd.read_json(screening_data, orient="records")
            for motif in df['Motif']:
                # Filter this motif’s hits
                hits_for_motif = scdf[scdf['user_motif_id'] == motif].sort_values('score', ascending=False)
                if hits_for_motif.empty:
                    screening_hits.append("")
                else:
                    # Collect up to 3 references in the format: "ref_motifset|ref_motif_id(score)"
                    top3 = hits_for_motif.head(3)
                    combined = []
                    for _, row in top3.iterrows():
                        combined.append(f"{row['ref_motifset']}|{row['ref_motif_id']}({row['score']:.2f})")
                    screening_hits.append("; ".join(combined))
        except Exception:
            # If there's any JSON/parsing error, fallback
            screening_hits = ["" for _ in range(len(df))]
    else:
        screening_hits = ["" for _ in range(len(df))]
    df['ScreeningHits'] = screening_hits

    # Filter out motifs that have no docs passing, i.e. degree=0
    df = df[df['Degree'] > 0].copy()

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
            "name": "ScreeningHits",
            "id": "ScreeningHits",
        },
    ]

    row_count_message = f"{len(df)} motif(s) pass the filter"
    return table_data, table_columns, row_count_message


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
    Output('motif-optimized-spectrum-container', 'children'),
    Output('motif-raw-spectrum-container', 'children'),
    Input('selected-motif-store', 'data'),
    Input('probability-filter', 'value'),
    Input('doc-topic-filter', 'value'),
    Input('overlap-filter', 'value'),
    Input('optimised-motif-fragloss-toggle', 'value'),
    State('lda-dict-store', 'data'),
    State('clustered-smiles-store', 'data'),
    State('spectra-store', 'data'),
    State('optimized-motifs-store', 'data'),
    State('screening-fullresults-store', 'data'),
    prevent_initial_call=True,
)
def update_motif_details(
        selected_motif,
        beta_range,
        theta_range,
        overlap_range,
        optimised_fragloss_toggle,
        lda_dict_data,
        clustered_smiles_data,
        spectra_data,
        optimized_motifs_data,
        screening_data
):
    if not selected_motif or not lda_dict_data:
        raise PreventUpdate

    motif_name = selected_motif
    motif_title = f"Motif Details: {motif_name}"

    # 1) Raw motif (LDA) using Probability Filter
    motif_data = lda_dict_data['beta'].get(motif_name, {})
    filtered_motif_data = {
        f: p for f, p in motif_data.items()
        if beta_range[0] <= p <= beta_range[1]
    }
    total_prob = sum(filtered_motif_data.values())

    feature_table = pd.DataFrame({
        'Feature': filtered_motif_data.keys(),
        'Probability': filtered_motif_data.values(),
    }).sort_values(by='Probability', ascending=False)

    feature_table_component = dash_table.DataTable(
        data=feature_table.to_dict('records'),
        columns=[
            {'name': 'Feature', 'id': 'Feature'},
            {'name': 'Probability', 'id': 'Probability', 'type': 'numeric',
             'format': {'specifier': '.4f'}},
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10,
    )

    # 2) Counts of features in filtered documents
    feature_counts = {ft: 0 for ft in filtered_motif_data.keys()}

    # Collect docs that pass the current doc-topic probability + overlap filters
    docs_for_this_motif = []
    for doc_name, topic_probs in lda_dict_data['theta'].items():
        doc_topic_prob = topic_probs.get(motif_name, 0.0)
        if doc_topic_prob <= 0:
            continue
        overlap_score = lda_dict_data['overlap_scores'][doc_name].get(motif_name, 0.0)
        if (theta_range[0] <= doc_topic_prob <= theta_range[1]) and (
                overlap_range[0] <= overlap_score <= overlap_range[1]):
            docs_for_this_motif.append(doc_name)

    # Sum up the occurrences of each feature within these filtered docs only
    for doc_name in docs_for_this_motif:
        w_counts = lda_dict_data['corpus'].get(doc_name, {})
        for ft in filtered_motif_data:
            if ft in w_counts:
                feature_counts[ft] += 1

    barplot2_df = pd.DataFrame({
        'Feature': list(feature_counts.keys()),
        'Count': list(feature_counts.values()),
    })
    barplot2_df = barplot2_df.sort_values(by='Count', ascending=False).head(10)
    barplot2_fig = px.bar(
        barplot2_df,
        x='Count',
        y='Feature',
        orientation='h',
    )
    barplot2_fig.update_layout(
        title=None,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Count within Filtered Motif Documents',
        yaxis_title='Feature',
    )

    # 3) Spec2Vec matching results
    motif_idx = None
    if motif_name.startswith('motif_'):
        try:
            motif_idx = int(motif_name.replace('motif_', ''))
        except:
            pass

    spec2vec_container = []
    auto_anno_text = ""
    if (optimized_motifs_data and motif_idx is not None and
            0 <= motif_idx < len(optimized_motifs_data)):
        meta_anno = optimized_motifs_data[motif_idx]['metadata'].get('auto_annotation', "")
        if meta_anno:
            auto_anno_text = f"Auto Annotations: {meta_anno}"

    if clustered_smiles_data and motif_idx is not None and motif_idx < len(clustered_smiles_data):
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
    if auto_anno_text:
        spec2vec_container.append(html.Div(auto_anno_text, style={"marginTop": "10px"}))

    # 4) Documents table
    doc2spec_index = lda_dict_data.get("doc_to_spec_index", {})
    docs_for_this_motif_records = []
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

            docs_for_this_motif_records.append({
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

    doc_cols = [
        'DocName', 'SpecIndex', 'FeatureID', 'Scans', 'PrecursorMz', 'RetentionTime',
        'CollisionEnergy', 'IonMode', 'MsLevel', 'Doc-Topic Probability', 'Overlap Score'
    ]
    docs_df = pd.DataFrame(docs_for_this_motif_records, columns=doc_cols)
    if not docs_df.empty:
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
    spectra_ids = docs_df['SpecIndex'].tolist() if not docs_df.empty else []

    # 5) Add screening info
    screening_box = ""
    if screening_data:
        try:
            scdf = pd.read_json(screening_data, orient="records")
            if "user_auto_annotation" in scdf.columns:
                scdf["user_auto_annotation"] = scdf["user_auto_annotation"].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                )
            hits_df = scdf[scdf['user_motif_id'] == motif_name].copy()
            hits_df = hits_df.sort_values('score', ascending=False)
            if not hits_df.empty:
                screening_table = dash_table.DataTable(
                    columns=[
                        {'name': 'Ref Motif ID', 'id': 'ref_motif_id'},
                        {'name': 'Ref ShortAnno', 'id': 'ref_short_annotation'},
                        {'name': 'Ref MotifSet', 'id': 'ref_motifset'},
                        {'name': 'Score', 'id': 'score', 'type': 'numeric', 'format': {'specifier': '.4f'}}
                    ],
                    data=hits_df.to_dict('records'),
                    page_size=5,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                )
                screening_box = html.Div([
                    html.H5("Screening Annotations"),
                    screening_table
                ],
                    style={
                        "border": "1px dashed #999",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "margin-bottom": "5px"
                    })
        except:
            pass
    spec2vec_div = html.Div(spec2vec_container + [screening_box])

    # 6) "Features" container
    features_div = html.Div([
        html.Div([
            html.H5("Motif Features Table"),
            feature_table_component,
            html.P(f"Total Probability (Filtered): {total_prob:.4f}")
        ]),
        html.H5("Counts of Features within Filtered Motif Documents"),
        dcc.Graph(figure=barplot2_fig),
    ])

    docs_div = html.Div([
        html.P("Below is the table of MS2 documents for the motif, subject to doc-topic probability & overlap score.")
    ])

    # 7) Build the Optimised Motif bar plot with the toggle for "fragments" or "losses"
    om_fig = go.Figure()
    motif_idx_opt = motif_idx
    if motif_idx_opt is not None and 0 <= motif_idx_opt < len(optimized_motifs_data):
        om = optimized_motifs_data[motif_idx_opt]
        raw_frag_mz = om.get("mz", [])
        raw_frag_int = om.get("intensities", [])
        raw_loss_mz = []
        raw_loss_int = []
        if "metadata" in om and "losses" in om["metadata"]:
            for item in om["metadata"]["losses"]:
                raw_loss_mz.append(item["loss_mz"])
                raw_loss_int.append(item["loss_intensity"])

        if optimised_fragloss_toggle == "fragments":
            if raw_frag_mz:
                om_fig.add_trace(go.Bar(
                    x=raw_frag_mz,
                    y=raw_frag_int,
                    marker=dict(color="#1f77b4"),
                    width=0.4,
                    name="Optimised Fragments"
                ))
        else:
            # user wants "losses"
            if raw_loss_mz:
                om_fig.add_trace(go.Bar(
                    x=raw_loss_mz,
                    y=raw_loss_int,
                    marker=dict(color="#ff7f0e"),
                    width=0.4,
                    name="Optimised Losses"
                ))

    om_fig.update_layout(
        title=None,
        xaxis_title="m/z (Da)",
        yaxis_title="Normalized Intensity",
        bargap=0.2
    )
    optim_plot = dcc.Graph(figure=om_fig)

    # 8) Build the RAW LDA Motif bar plot (with Probability Filter)
    raw_fig = go.Figure()
    raw_frag_mz = []
    raw_frag_int = []
    raw_loss_mz = []
    raw_loss_int = []
    for ft, val in filtered_motif_data.items():
        if ft.startswith('frag@'):
            try:
                raw_frag_mz.append(float(ft.replace('frag@', '')))
                raw_frag_int.append(val)
            except:
                pass
        elif ft.startswith('loss@'):
            try:
                raw_loss_mz.append(float(ft.replace('loss@', '')))
                raw_loss_int.append(val)
            except:
                pass

    if raw_frag_mz:
        raw_fig.add_trace(go.Bar(
            x=raw_frag_mz,
            y=raw_frag_int,
            marker=dict(color="#1f77b4"),
            width=0.4,
            name="Raw LDA Fragments"
        ))
    if raw_loss_mz:
        raw_fig.add_trace(go.Bar(
            x=raw_loss_mz,
            y=raw_loss_int,
            marker=dict(color="#ff7f0e"),
            width=0.4,
            name="Raw LDA Losses"
        ))

    raw_fig.update_layout(
        title=None,
        xaxis_title="m/z (Da)",
        yaxis_title="Probability",
        bargap=0.2
    )
    raw_plot = dcc.Graph(figure=raw_fig)

    return (
        motif_title,
        spec2vec_div,
        features_div,
        docs_div,
        spectra_ids,
        table_data,
        table_columns,
        optim_plot,
        raw_plot,
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
    for root, dirs, files in os.walk(MOTIFDB_FOLDER):
        json_files = [f for f in files if f.startswith("Motifset") and f.endswith(".json")]
        for jsonf in json_files:
            fullpath = os.path.join(root, jsonf)
            label = jsonf
            ms1_df, ms2_df = load_motifDB(fullpath)
            count_m2m = len(ms2_df["scan"].unique())
            folder_options.append({"label": f"{label} ({count_m2m} motifs)", "value": fullpath})
            subfolder_data[fullpath] = {"folder_label": label, "count_m2m": count_m2m}

    folder_options = sorted(folder_options, key=lambda x: x["label"].lower())
    return folder_options, subfolder_data


def make_spectrum_from_dict(spectrum_dict):
    try:
        mz_ = np.array(spectrum_dict["mz"], dtype=float)
        ints_ = np.array(spectrum_dict["intensities"], dtype=float)
        meta_ = spectrum_dict["metadata"]
        sp = Spectrum(mz=mz_, intensities=ints_, metadata=meta_)
        if "losses" in meta_:
            loss_mz = []
            loss_int = []
            for x in meta_["losses"]:
                loss_mz.append(x["loss_mz"])
                loss_int.append(x["loss_intensity"])
            sp.losses = Fragments(
                mz=np.array(loss_mz),
                intensities=np.array(loss_int)
            )
        return sp
    except:
        return None


def filter_and_normalize_spectra(spectrum_list):
    def trunc_annotation(val, max_len=40):
        """Truncate any string over max_len for readability."""
        if isinstance(val, str) and len(val) > max_len:
            return val[:max_len] + "..."
        return val

    valid = []
    for sp in spectrum_list:
        if not sp.peaks or len(sp.peaks.mz) == 0:
            continue
        # Normalize peaks
        max_i = sp.peaks.intensities.max()
        if max_i <= 0:
            continue
        if max_i > 1.0:
            sp.peaks = Fragments(
                mz=sp.peaks.mz,
                intensities=sp.peaks.intensities / max_i
            )
        # Normalize losses
        if sp.losses and len(sp.losses.mz) > 0:
            max_l = sp.losses.intensities.max()
            if max_l > 1.0:
                sp.losses = Fragments(
                    mz=sp.losses.mz,
                    intensities=sp.losses.intensities / max_l
                )

        # Possibly convert short_annotation from list->string
        ann = sp.get("short_annotation", "")
        if isinstance(ann, list):
            joined = ", ".join(map(str, ann))
            joined = trunc_annotation(joined, 60)
            sp.set("short_annotation", joined)
        elif isinstance(ann, str):
            sp.set("short_annotation", trunc_annotation(ann, 60))

        valid.append(sp)

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
def compute_spec2vec_screening(n_clicks, selected_folders, lda_dict_data, path_model, optimized_motifs_data):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    message = dbc.Alert("Computing similarities... please wait (this may take a while).", color="info")
    progress_val = 0
    button_disabled = True

    # check input
    if not selected_folders:
        return None, dbc.Alert("No reference MotifDB set selected!", color="warning"), 100, False
    if not lda_dict_data:
        return None, dbc.Alert("No LDA results found! Please run MS2LDA or load your data first.",
                               color="warning"), 100, False

    # 1) Convert raw motifs from lda_dict_data['beta']
    user_motifs = []
    beta = lda_dict_data.get("beta", {})
    if not beta:
        return None, dbc.Alert("lda_dict_data['beta'] is empty or missing!", color="warning"), 100, False

    # Get run parameters used for this ms2lda anlaysis
    run_params = lda_dict_data.get("run_parameters", {})
    charge_to_use = run_params.get("dataset_parameters", {}).get("charge", 1)
    sig_digits_to_use = run_params.get("dataset_parameters", {}).get("significant_digits", 2)

    for motif_name, feature_probs_dict in beta.items():
        k = -1
        if motif_name.startswith("motif_"):
            try:
                k = int(motif_name.replace("motif_", ""))
            except ValueError:
                pass

        motif_features_list = list(feature_probs_dict.items())
        raw_motif_spectrum = create_spectrum(
            motif_features_list,
            k if k >= 0 else 0,
            frag_tag="frag@",
            loss_tag="loss@",
            significant_digits=sig_digits_to_use,
            charge=charge_to_use,
            motifset=motif_name
        )
        user_motifs.append(raw_motif_spectrum)

    # 2) Filter & normalize the user_motifs
    user_motifs = filter_and_normalize_spectra(user_motifs)
    if not user_motifs:
        return None, dbc.Alert("No valid user motifs after normalization!", color="warning"), 100, False

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
        return None, dbc.Alert("No valid references found in the selected file(s)!", color="warning"), 100, False
    progress_val = 40

    # 4) Load Spec2Vec model
    print("loading s2v")
    s2v_sim = load_s2v_model(path_model=path_model)
    print("loaded s2v")
    progress_val = 60

    # 5) Embeddings
    user_emb = calc_embeddings(s2v_sim, user_motifs)
    print("user_emb shape:", user_emb.shape)
    ref_emb = calc_embeddings(s2v_sim, all_refs)
    print("ref_emb shape:", ref_emb.shape)
    progress_val = 80

    # 6) Similarity
    sim_matrix = calc_similarity(user_emb, ref_emb)
    print("sim_matrix shape:", sim_matrix.shape)
    progress_val = 90

    # 7) Build an optimized_anno_map
    optimized_anno_map = {}
    if optimized_motifs_data:
        for om_entry in optimized_motifs_data:
            om_meta = om_entry.get('metadata', {})
            om_id = om_meta.get('id')  # e.g. "motif_0"
            om_anno = om_meta.get('auto_annotation', "")
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

            results.append({
                "user_motif_id": user_id,
                "user_auto_annotation": user_anno,
                "ref_motif_id": ref_id,
                "ref_short_annotation": ref_anno,
                "ref_motifset": ref_motifset,
                "score": round(float(score), 4),
            })

    if not results:
        return None, dbc.Alert("No results after similarity!", color="warning"), 100, False

    # Sort descending
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    json_data = df.to_json(orient="records")

    progress_val = 100
    msg = dbc.Alert(
        f"Computed {len(df)} total matches from {len(user_motifs)} user motifs and {len(all_refs)} references.",
        color="success"
    )
    button_disabled = False

    return json_data, msg, progress_val, button_disabled


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
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
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
        return dcc.send_data_frame(df.to_csv, "screening_results.csv", index=False), no_update
    elif button_id == "save-screening-json":
        out_str = df.to_json(orient="records")
        return no_update, dict(content=out_str, filename="screening_results.json")
    else:
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
        return dcc.send_data_frame(df.to_csv, "motifranking_results.csv", index=False), no_update
    elif button_id == "save-motifranking-json":
        out_str = df.to_json(orient="records")
        return no_update, dict(content=out_str, filename="motifranking_results.json")
    else:
        raise dash.exceptions.PreventUpdate



@app.callback(
    Output("selected-motif-store", "data"),
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
def on_motif_click(ranking_active_cell, screening_active_cell, ranking_data, screening_data, ranking_dv_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "motif-rankings-table":
        if ranking_active_cell and ranking_dv_data:
            col_id = ranking_active_cell["column_id"]
            row_id = ranking_active_cell["row"]
            if col_id == "Motif":
                motif_markdown = ranking_dv_data[row_id]["Motif"]
                return motif_markdown
        raise dash.exceptions.PreventUpdate

    elif triggered_id == "screening-results-table":
        if screening_active_cell and screening_data:
            col_id = screening_active_cell["column_id"]
            row_id = screening_active_cell["row"]
            if col_id == "user_motif_id":
                motif_id = screening_data[row_id]["user_motif_id"]
                return motif_id
        raise dash.exceptions.PreventUpdate

    else:
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
        return f"Download failed: {str(e)}", ""
