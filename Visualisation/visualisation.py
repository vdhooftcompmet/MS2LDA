from matchms import Spectrum
from matchms.filtering import add_losses
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolsToGridImage
from PIL import Image
import random
import io

def create_network(spectra, significant_figures=2, motif_sizes=None):
    """
    Generates a network for the motifs spectra, where the nodes are the motifs (output of LDA model)
    and the edges are the peaks and losses of the spectra. The size of the nodes can be adjusted with the motif_sizes refined annotation

    ARGS:
        spectra (list): list of matchms.Spectrum.objects; after LDA modelling
        significant_figures (int, optional): number of significant figures to round the mz values
        motif_sizes (list, optional): list of sizes for the `motif_{i}` nodes; should match the length of spectra

    RETURNS:
        network (nx.Graph): network with nodes and edges
    """
    if motif_sizes is not None:
        motif_sizes_filtered = list(map(lambda x: 0.7 if x == '?' else x, motif_sizes)) #filtering ? 

    G = nx.Graph()

    if motif_sizes is not None and len(motif_sizes) != len(spectra):
        raise ValueError("Length of motif_sizes must match the number of spectra")

    for i, spectrum in enumerate(spectra, start=0):
        motif_node = f'motif_{i}'
        G.add_node(motif_node)
        
        peak_list = spectrum.peaks.mz
        rounded_peak_list = [round(x, significant_figures) for x in peak_list]
        loss_list = spectrum.losses.mz
        rounded_loss_list = [round(x, significant_figures) for x in loss_list]
        int_peak_list = spectrum.peaks.intensities
        int_losses_list = spectrum.losses.intensities
        
        for edge, weight in zip(rounded_peak_list, int_peak_list):
            G.add_edge(motif_node, edge, weight=weight, color='red')
        for edge, weight in zip(rounded_loss_list, int_losses_list):
            G.add_edge(motif_node, edge, weight=weight, color='blue')
    
    #Arranging node size - motifs
    node_sizes = {}
    if motif_sizes is None:
        default_size = 1000  
        for i in range(1, len(spectra)):
            node_sizes[f'motif_{i}'] = default_size

    else:
        for i in range(1, len(spectra)):
            node_sizes[f'motif_{i}'] = ((motif_sizes_filtered[i] * 100) ** 2)/2
    
    fig, ax = plt.subplots(figsize=(50, 50))  
    edges = G.edges(data=True)
    weights = [d['weight'] * 10 for (u, v, d) in edges]  
    edge_colors = [d['color'] for (u, v, d) in edges]    
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=weights, edge_color=edge_colors)
    node_size_list = [node_sizes.get(node, 100) for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_size_list, node_color="#210070", alpha=0.9)
    
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(G, pos, font_size=10, bbox=label_options)
    
    #ax.margins(0.1, 0.05)
    #fig.tight_layout()
    #plt.axis("off")
    #plt.show()
    return G



def create_interactive_motif_network(spectra, significant_figures, motif_sizes, smiles_clusters):
    """
    Generates a network for the annotated optimized spectra, after running Spec2Vec annotation, if clicking in a node
    it will shot the spectrum and the molecule associated with it.

    ARGS:
        spectra (list): list of matchms.Spectrum.objects; after Spec2Vec annotation
        significant_figures (int): number of significant figures to round the mz values
        motif_sizes (list, optional): list of sizes for the `motif_{i}` nodes; should match the length of spectra

    RETURNS:
        network (nx.Graph): network with nodes and edges, spectra and structures
    """
    if motif_sizes is not None:
        motif_sizes_filtered = list(map(lambda x: 0.7 if x == '?' else x, motif_sizes))  # filtering '?'

    G = nx.Graph()

    if motif_sizes is not None and len(motif_sizes) != len(spectra):
        raise ValueError("Length of motif_sizes must match the number of spectra")

    for i, spectrum in enumerate(spectra, start=0):
        motif_node = f'motif_{i}'
        G.add_node(motif_node)
        
        peak_list = spectrum.peaks.mz
        rounded_peak_list = [round(x, significant_figures) for x in peak_list]
        loss_list = spectrum.losses.mz
        rounded_loss_list = [round(x, significant_figures) for x in loss_list]
        int_peak_list = spectrum.peaks.intensities
        int_losses_list = spectrum.losses.intensities
        
        for edge, weight in zip(rounded_peak_list, int_peak_list):
            G.add_edge(motif_node, edge, weight=weight, color='red')
        for edge, weight in zip(rounded_loss_list, int_losses_list):
            G.add_edge(motif_node, edge, weight=weight, color='blue')
    
    node_sizes = {}
    if motif_sizes is None:
        default_size = 1000  
        for i in range(1, len(spectra)):
            node_sizes[f'motif_{i}'] = default_size
    else:
        for i in range(1, len(spectra)):
            node_sizes[f'motif_{i}'] = ((motif_sizes_filtered[i] * 100) ** 2) / 2
    
    pos = nx.spring_layout(G)  
    fig, ax = plt.subplots(figsize=(10, 50)) 
    
    edges = G.edges(data=True)
    weights = [d['weight'] * 10 for (u, v, d) in edges]  
    edge_colors = [d['color'] for (u, v, d) in edges]    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=weights, edge_color=edge_colors)
    node_size_list = [node_sizes.get(node, 100) for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_size_list, node_color="#210070", alpha=0.9)
    
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(G, pos, font_size=4, bbox=label_options)
    
    def on_click(event):
        for node, (x, y) in pos.items():
            dist = (x - event.xdata)**2 + (y - event.ydata)**2
            if dist < 0.00025:  
                if isinstance(node, str):  # Check if the node is a string and matches "motif_x"
                    print(f"Node {node} clicked!")
                    node_number = int(node.split('_')[1])
                    first_elements = [sublist[0] for sublist in smiles_clusters]
                    mols = [MolFromSmiles(first_elements[node_number])]
                    img = MolsToGridImage(mols)

                    spectra[node_number].plot()
                    
                    pil_img = Image.open(io.BytesIO(img.data))
                    
                    # Display new window
                    pil_img.show()
                
                break

    # Connect the click event to the on_click function
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    return G



if __name__ == "__main__":
    from matchms import Spectrum
    from matchms.filtering import add_losses
    import numpy as np
    spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                      intensities=np.array([0.7, 0.2, 0.1]),
                      metadata={'id': 'spectrum1',
                                'precursor_mz': 201.})
    spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                        intensities=np.array([0.4, 0.2, 0.1]),
                        metadata={'id': 'spectrum2',
                                  'precursor_mz': 233.})
    spectrum_3 = Spectrum(mz=np.array([110, 140, 195.]),
                        intensities=np.array([0.6, 0.2, 0.1]),
                        metadata={'id': 'spectrum3',
                                  'precursor_mz': 214.})
    spectrum_4 = Spectrum(mz=np.array([100, 150, 200.]),
                        intensities=np.array([0.6, 0.1, 0.6]),
                        metadata={'id': 'spectrum4',
                                  'precursor_mz': 265.})
    
    spectra = [add_losses(spectrum_1), add_losses(spectrum_2), add_losses(spectrum_3), add_losses(spectrum_4)]
    create_network(spectra)