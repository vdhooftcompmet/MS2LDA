from matchms import Spectrum
from matchms.filtering import add_losses
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt


import networkx as nx
import matplotlib.pyplot as plt

def create_network(spectra, motif_sizes=None):
    """
    Generates a network for the LDA output

    ARGS:
        spectra (list): list of matchms.Spectrum.objects; after LDA modelling
        motif_sizes (list, optional): list of sizes for the `motif_{i}` nodes; should match the length of spectra

    RETURNS:
        network (nx.Graph): network with nodes and edges
    """
    G = nx.Graph()

    if motif_sizes is not None and len(motif_sizes) != len(spectra):
        raise ValueError("Length of motif_sizes must match the number of spectra")

    for i, spectrum in enumerate(spectra, start=1):
        motif_node = f'motif_{i}'
        G.add_node(motif_node)
        
        peak_list = spectrum.peaks.mz
        rounded_peak_list = [round(x, 2) for x in peak_list]
        loss_list = spectrum.losses.mz
        rounded_loss_list = [round(x, 2) for x in loss_list]
        int_peak_list = spectrum.peaks.intensities
        rounded_int_peak_list = [round(x, 2) for x in int_peak_list]
        int_losses_list = spectrum.losses.intensities
        rounded_int_losses_list = [round(x, 2) for x in int_losses_list]
        
        for edge, weight in zip(rounded_peak_list, rounded_int_peak_list):
            G.add_edge(motif_node, edge, weight=weight, color='red')
        for edge, weight in zip(rounded_loss_list, rounded_int_losses_list):
            G.add_edge(motif_node, edge, weight=weight, color='blue')
    
    #Arranging node size - motifs
    node_sizes = {}
    if motif_sizes is None:
        default_size = 1000  
        for i in range(1, len(spectra) + 1):
            node_sizes[f'motif_{i}'] = default_size
    else:
        for i in range(1, len(spectra) + 1):
            node_sizes[f'motif_{i}'] = motif_sizes[i-1] * 5000
    
    fig, ax = plt.subplots(figsize=(50, 50))  # Adjust size if needed
    edges = G.edges(data=True)
    weights = [d['weight'] * 10 for (u, v, d) in edges]  
    edge_colors = [d['color'] for (u, v, d) in edges]    
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=weights, edge_color=edge_colors)
    node_size_list = [node_sizes.get(node, 100) for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_size_list, node_color="#210070", alpha=0.9)
    
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(G, pos, font_size=10, bbox=label_options)
    
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.show()


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