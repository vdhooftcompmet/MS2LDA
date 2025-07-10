# App/callbacks/network.py
"""Callbacks for Network View tab."""

import dash_cytoscape as cyto
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from App.app_instance import app
from App.callbacks.common import *  # Import helper functions


# -------------------------------- CYTOSCAPE NETWORK --------------------------------# -------------------------------- CYTOSCAPE NETWORK --------------------------------


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


