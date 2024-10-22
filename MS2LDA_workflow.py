import base64
import os
import tempfile

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import numpy as np
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
# Import your MS2LDA modules
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
        dbc.Tabs(
            [
                dbc.Tab(label="Parameters", tab_id="params-tab"),
                dbc.Tab(label="Results", tab_id="results-tab"),
            ],
            id="tabs",
            active_tab="params-tab",
            className="mt-3",
        ),
        html.Div(id="tab-content"),
        # Include all components in the initial layout
        html.Div(id="cytoscape-network-container", style={"display": "none"}),
        html.Div(id="molecule-images", style={"display": "none"}),
        html.Div(id="run-status", style={"display": "none"}),
        html.Div(id="file-upload-info", style={"display": "none"}),
        # Hidden storage for data to be accessed by callbacks
        dcc.Store(id="clustered-smiles-store"),
        dcc.Store(id="optimized-motifs-store"),
    ],
    fluid=True,
)


# Callback to render tab content
@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab_content(active_tab):
    if active_tab == "params-tab":
        return dbc.Container(
            [
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
                                html.Div(id="run-status", style={"marginTop": "20px"}),
                            ],
                            width=6,
                        )
                    ],
                    justify="center",
                )
            ]
        )
    elif active_tab == "results-tab":
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="cytoscape-network-container",
                                    style={"marginTop": "20px"},
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="molecule-images",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                            ],
                            width=12,
                        )
                    ]
                ),
            ]
        )
    else:
        return html.Div("Unknown tab selected.")


# Callback to display uploaded file info
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


# Callback to run analysis
@app.callback(
    Output("run-status", "children"),
    Output("clustered-smiles-store", "data"),
    Output("optimized-motifs-store", "data"),
    Input("run-button", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("n-motifs", "value"),
    State("top-n", "value"),
    State("unique-mols", "value"),
    State("polarity", "value"),
    prevent_initial_call=True,
)
def run_analysis(
        n_clicks, contents, filename, n_motifs, top_n, unique_mols, polarity
):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if not contents:
        return (
            dbc.Alert(
                "Please upload a mass spectrometry data file.", color="danger"
            ),
            None,
            None,
        )

    # Decode the uploaded file
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(filename)[1]
    ) as tmp_file:
        tmp_file.write(decoded)
        tmp_file_path = tmp_file.name

    try:
        # Generate motifs
        motifs = generate_motifs(tmp_file_path, n_motifs=n_motifs, iterations=100)

        # Load Spec2Vec model and library based on polarity
        if polarity == "positive":
            path_model = (
                "MS2LDA/Add_On/Spec2Vec/model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model"
            )
            path_library = (
                "MS2LDA/Add_On/Spec2Vec/model_positive_mode/positive_s2v_library.pkl"
            )
        else:
            path_model = (
                "MS2LDA/Add_On/Spec2Vec/model_negative_mode/150724_Spec2Vec_neg_CleanedLibraries.model"
            )
            path_library = (
                "MS2LDA/Add_On/Spec2Vec/model_negative_mode/negative_s2v_library.pkl"
            )

        # Annotate motifs
        s2v_similarity, library = load_s2v_and_library(path_model, path_library)

        # Calculate embeddings and similarity matrix
        motif_embeddings = calc_embeddings(s2v_similarity, motifs)
        similarity_matrix = calc_similarity(motif_embeddings, library.embeddings)

        matching_settings = {
            "similarity_matrix": similarity_matrix,
            "library": library,
            "top_n": top_n,
            "unique_mols": unique_mols,
        }

        library_matches = get_library_matches(matching_settings)

        # Refine Annotation
        clustered_spectra, clustered_smiles, clustered_scores = hit_clustering(
            s2v_similarity, motifs, library_matches, criterium="best"
        )

        # Optimize motifs
        optimized_motifs = []
        for motif_spec, spec_list, smiles_list in zip(
                motifs, clustered_spectra, clustered_smiles
        ):
            opt_motif = optimize_motif_spectrum(motif_spec, spec_list, smiles_list)
            optimized_motifs.append(opt_motif)

        # Store data in dcc.Store components
        clustered_smiles_data = clustered_smiles  # list of lists
        optimized_motifs_data = [spectrum_to_dict(s) for s in optimized_motifs]

        status_message = dbc.Alert(
            "Analysis Completed Successfully! Switch to the 'Results' tab to view.",
            color="success",
        )

        return status_message, clustered_smiles_data, optimized_motifs_data

    except Exception as e:
        return (
            dbc.Alert(f"An error occurred: {str(e)}", color="danger"),
            None,
            None,
        )


# Helper function to convert Spectrum to dict (for serialization)
def spectrum_to_dict(spectrum):
    return {
        "metadata": spectrum.metadata,
        "mz": [float(m) for m in spectrum.peaks.mz.tolist()],
        "intensities": [float(i) for i in spectrum.peaks.intensities.tolist()],
        "losses_mz": [float(m) for m in spectrum.losses.mz.tolist()] if spectrum.losses else [],
        "losses_intensities": [float(i) for i in spectrum.losses.intensities.tolist()] if spectrum.losses else [],
    }


# Callback to create Cytoscape elements
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
        style={"width": "100%", "height": "600px"},
        layout={"name": "cose",
                "animate": False},  # Set animate to False for faster rendering
        stylesheet=[
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "width": "mapData(size, 0, 10, 20, 50)",
                    "height": "mapData(size, 0, 10, 20, 50)",
                    "background-color": "data(color)",
                    "font-size": "10px",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "width": 2,
                    "line-color": "data(color)",
                    "target-arrow-color": "data(color)",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                },
            },
        ],
    )

    return cytoscape_component


# Revised create_cytoscape_elements function in Visualisation/visualisation.py

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
        color = colors[i % len(colors)]
        elements.append(
            {
                "data": {
                    "id": motif_node,
                    "label": motif_node,
                    "size": 5,
                    "color": color,
                }
            }
        )

        # Add fragment nodes and edges
        for mz in spectrum.peaks.mz:
            rounded_mz = round(mz, 2)
            frag_node = f"frag_{rounded_mz}"
            if frag_node not in created_fragments:
                elements.append(
                    {
                        "data": {
                            "id": frag_node,
                            "label": str(rounded_mz),
                            "color": "red",
                        }
                    }
                )
                created_fragments.add(frag_node)
            elements.append(
                {
                    "data": {
                        "source": motif_node,
                        "target": frag_node,
                        "color": "red",
                    }
                }
            )

        # Add loss nodes and edges
        if spectrum.losses is not None:
            for mz in spectrum.losses.mz:
                rounded_mz = round(mz, 2)
                loss_node = f"loss_{rounded_mz}"
                if loss_node not in created_losses:
                    elements.append(
                        {
                            "data": {
                                "id": loss_node,
                                "label": str(rounded_mz),
                                "color": "blue",
                            }
                        }
                    )
                    created_losses.add(loss_node)
                elements.append(
                    {
                        "data": {
                            "source": motif_node,
                            "target": loss_node,
                            "color": "blue",
                        }
                    }
                )

    return elements


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
                from rdkit.Chem.Draw import MolDraw2DCairo
                drawer = MolDraw2DCairo(1000, 200)  # Total width x height
                from rdkit.Chem import Draw
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=5,
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


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
