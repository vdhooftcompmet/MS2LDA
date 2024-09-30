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
import networkx as nx


def create_network(spectra, significant_figures=2, motif_sizes=None, file_generation=False):
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
    nx.draw_networkx_labels(G, pos, font_size=14, bbox=label_options)
    
    #ax.margins(0.1, 0.05)
    #fig.tight_layout()
    #plt.axis("off")
    #plt.show()
    if file_generation:
        nx.write_graphml(G, "lda_model_output.graphml")
    return G



def create_interactive_motif_network(spectra, significant_figures, motif_sizes, smiles_clusters, spectra_cluster, motif_colors, file_generation=False): #spectra-cluster added motif_colors
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
        default_size = 800  
        for i in range(1, len(spectra)): # error here!?
            node_sizes[f'motif_{i}'] = default_size
    else:
        n_smiles_cluster=[]
        for i in smiles_clusters:
            n_smiles_cluster.append(len(i))
        max_n_smiles_cluster= max(n_smiles_cluster)

        n_frags_cluster=[]
        for i in spectra:
            n_frags_cluster.append(len(i.peaks.mz))
        max_n_frags_cluster = max(n_frags_cluster)

        for i in range(1, len(spectra)):
            node_sizes[f'motif_{i}'] = ((motif_sizes_filtered[i] * 10) **3)/3 + \
                (((n_smiles_cluster[i]/max_n_smiles_cluster)*10)**3)/3 + \
                    (((n_frags_cluster[i]/max_n_frags_cluster)*10)**3)/3
            
    # new; for tox
    node_colors = {}
    if motif_colors is None:
        default_color = "#210070"
        for i in range(1, len(spectra)):
            node_colors[f'motif_{i}'] = default_color
    else:
        for i in range(1, len(spectra)):
            node_colors[f'motif_{i}'] = motif_colors[i]
    #--------------------------

    
    pos = nx.spring_layout(G)  
    fig, ax = plt.subplots(figsize=(10, 50)) 
    
    edges = G.edges(data=True)
    weights = [d['weight'] * 10 for (u, v, d) in edges]  
    edge_colors = [d['color'] for (u, v, d) in edges]    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=weights, edge_color=edge_colors)
    node_size_list = [node_sizes.get(node, 100) for node in G.nodes]
    node_color_list = [node_colors.get(node, "green") for node in G.nodes]
    #node_color_list_flat = [color for sublist in node_color_list for color in (sublist if isinstance(sublist, list) else [sublist])]
    
    #nx.draw_networkx_nodes(G, pos, node_size=node_size_list, node_color="#210070", alpha=0.9) 
    nx.draw_networkx_nodes(G, pos, node_size=node_size_list, node_color=node_color_list, alpha=0.9) 
    
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(G, pos, font_size=6, bbox=label_options)
    
    def on_click(event):
        for node, (x, y) in pos.items():
            dist = (x - event.xdata)**2 + (y - event.ydata)**2
            if dist < 0.00025:  
                if isinstance(node, str):  # Check if the node is a string and matches "motif_x"
                    node_number = int(node.split('_')[1])
                    print(f"Node {node} clicked!\n"
                    f"Cluster similarity: {motif_sizes_filtered[node_number]*100}%\n"
                    f"N of compounds: {(n_smiles_cluster[node_number]/max_n_smiles_cluster)*100}"
                    f"N of features: {(n_frags_cluster[node_number]/max_n_frags_cluster)*100}"
                    f"Fragments: {spectra[node_number].peaks.mz}\n"
                    f"Losses: {spectra[node_number].losses.mz}")
                    mols = [MolFromSmiles(smi) for smi in smiles_clusters[node_number]]
                    img = MolsToGridImage(mols)

                    #spectra[node_number].plot()
                    
                    pil_img = Image.open(io.BytesIO(img.data))
                    
                    # Display new window
                    pil_img.show()

                    for spec in spectra_cluster[node_number]: # also added
                        spectra[node_number].plot_against(spec)
                
                break

    # Connect the click event to the on_click function
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()
    if file_generation:
        nx.write_graphml(G, "lda_model_output.graphml")

    

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