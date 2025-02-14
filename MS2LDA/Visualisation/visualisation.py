import io
import os

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Draw import MolsToGridImage


def _in_jupyter_notebook():
    """Return True if running inside a Jupyter notebook/lab, else False."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        # 'ZMQInteractiveShell' => jupyter notebook or jupyter lab
        return (shell == 'ZMQInteractiveShell')
    except Exception:
        return False

# Try to detect environment
_in_nb = _in_jupyter_notebook()
if not _in_nb:
    # Switch to a non-GUI backend so macOS won't complain about NSWindow in threads
    matplotlib.use('Agg')

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
        int_peak_list = spectrum.peaks.intensities
        for edge, weight in zip(rounded_peak_list, int_peak_list):
            G.add_edge(motif_node, edge, weight=weight, color='red')

        if spectrum.losses:
            loss_list = spectrum.losses.mz
            rounded_loss_list = [round(x, significant_figures) for x in loss_list]
            int_losses_list = spectrum.losses.intensities
            
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

        if spectrum.peaks:
            peak_list = spectrum.peaks.mz
            rounded_peak_list = [round(x, significant_figures) for x in peak_list]
            int_peak_list = spectrum.peaks.intensities

            for edge, weight in zip(rounded_peak_list, int_peak_list):
                G.add_edge(motif_node, edge, weight=weight, color='red')

        if spectrum.losses:
            loss_list = spectrum.losses.mz
            rounded_loss_list = [round(x, significant_figures) for x in loss_list]
            int_losses_list = spectrum.losses.intensities

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
                    #print(f"Node {node} clicked!\n"
                    #f"Cluster similarity: {motif_sizes_filtered[node_number]*100}%\n"
                    #f"N of compounds: {(n_smiles_cluster[node_number]/max_n_smiles_cluster)*100}"
                    #f"N of features: {(n_frags_cluster[node_number]/max_n_frags_cluster)*100}"
                    #f"Fragments: {spectra[node_number].peaks.mz}\n"
                    #f"Losses: {spectra[node_number].losses.mz}")
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



def plot_convergence(convergence_curve):
    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=1, sharey=True, sharex=False)

    # --- Helper function to filter out non-integer ticks ---
    def set_integer_xticks(ax, step_size):
        # Get current x-ticks and filter out non-integer values
        xticks = ax.get_xticks()
        xticks_int = xticks[xticks % 1 == 0].astype(int)  # Only keep whole numbers
        ax.set_xticks(xticks_int)
        ax.set_xticklabels(xticks_int)

        # Set iterations on the secondary x-axis (below the plot)
        ax_x = ax.secondary_xaxis(-0.15)
        ax_x.set_xticks(xticks_int)
        ax_x.set_xticklabels((xticks_int * step_size).astype(int))
        ax_x.set_xlabel("Iterations")

    # --- Plot for the first subplot (left side) ---
    c1_1, = ax.plot(convergence_curve["perplexity_history"], label="Perplexity Score", color="black")
    ax1_2 = ax.twinx()
    c1_2, = ax1_2.plot(convergence_curve["entropy_history_topic"], label="Topic Entropy", color="blue")
    ax1_3 = ax.twinx()
    ax1_3.spines['right'].set_position(('outward', 60))
    c1_3, = ax1_3.plot(convergence_curve["entropy_history_doc"], label="Document Entropy", color="orange")
    ax1_4 = ax.twinx()
    ax1_4.spines['right'].set_position(('outward', 120))
    c1_4, = ax1_4.plot(convergence_curve["log_likelihood_history"], label="Log Likelihood", color="green")
    ax.set_xlabel("Checkpoints")

    # Apply integer ticks
    set_integer_xticks(ax, 50)

    ax.set_xlim(0, len(convergence_curve["perplexity_history"]))

    ax.set_ylabel("Perplexity")
    ax1_2.set_ylabel("Topic Entropy")
    ax1_3.set_ylabel("Document Entropy")
    ax1_4.set_ylabel("Log Likelihood")

    # Coloring the axes to match the lines for the third subplot
    ax1_2.tick_params(axis='y', colors=c1_2.get_color())
    ax1_3.tick_params(axis='y', colors=c1_3.get_color())
    ax1_4.tick_params(axis='y', colors=c1_4.get_color())
    ax1_2.yaxis.label.set_color(c1_2.get_color())
    ax1_3.yaxis.label.set_color(c1_3.get_color())
    ax1_4.yaxis.label.set_color(c1_4.get_color())

    # Adding legends
    #ax.legend(handles=[c1_1, c1_2, c1_3], loc='best')

    # Add a shared header for all three plots
    fig.suptitle('Different Convergence Curves', fontsize=16)
    fig.subplots_adjust(top=0.85, bottom=0.2, right=0.85)
    plt.tight_layout()
    return fig


def show_annotated_motifs(opt_motif_spectra, motif_spectra, clustered_smiles, savefig=None):
    """
    Show side-by-side RDKit molecule images from clustered SMILES,
    and plot motif vs. optimized motif.

    - If in a Jupyter notebook, we'll try the 'Notebook-friendly' style.
    - If not in Jupyter, we'll switch to a headless backend (no GUI windows),
      skip plt.show(), and just close figures if not saving.
    """
    assert len(opt_motif_spectra) == len(motif_spectra), (
        "Lengths of opt_motif_spectra and motif_spectra must match!"
    )

    # Create output folder if needed
    if savefig is not None:
        os.makedirs(savefig, exist_ok=True)

    # We'll pass 'returnPNG=not_in_jupyter' so that in Jupyter we do no `returnPNG`,
    # in Dash we do `returnPNG=True`.
    not_in_jupyter = not _in_nb

    for m in range(len(motif_spectra)):
        mass_to_charge_opt = opt_motif_spectra[m].peaks.mz
        intensities_opt = opt_motif_spectra[m].peaks.intensities
        mass_to_charge = motif_spectra[m].peaks.mz
        intensities = motif_spectra[m].peaks.intensities

        # Convert SMILES -> RDKit mols
        mols = []
        for smi in clustered_smiles[m]:
            mol = MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)

        # If no valid SMILES, fallback to blank
        pil_img = None
        if len(mols) == 0:
            pil_img = Image.new("RGB", (400, 400), "white")
        else:
            # Attempt to get either a PIL object or bytes from MolsToGridImage
            # If in Jupyter => returnPNG=False
            # if not => returnPNG=True
            result = MolsToGridImage(
                mols,
                molsPerRow=len(mols),
                subImgSize=(400, 400),
                returnPNG=not_in_jupyter
            )
            pil_img = _convert_molgrid_result_to_pil(result)
            if pil_img is None:
                # If that fails, fallback blank
                pil_img = Image.new("RGB", (400, 400), "white")

        # Make figure
        fig = plt.figure(figsize=(10, 6), facecolor='none', edgecolor='none')

        # Top subplot: molecule grid
        ax_top = fig.add_subplot(2, 1, 1)
        ax_top.imshow(pil_img)
        ax_top.axis("off")
        top_pos = ax_top.get_position()
        ax_top.set_position([top_pos.x0, top_pos.y0 - 0.1, top_pos.width, top_pos.height])

        # Bottom subplot: motif vs. optimized motif
        ax_bot = fig.add_subplot(2, 1, 2)
        ax_bot.stem(mass_to_charge, intensities,
                    basefmt="k-", markerfmt="", linefmt="black",
                    label=f"motif_{m}")
        if len(mass_to_charge_opt) > 0:
            ax_bot.stem(mass_to_charge_opt, intensities_opt,
                        basefmt="k-", markerfmt="", linefmt="red",
                        label=f"opt motif_{m}")

        ax_bot.set_ylim(0,)
        ax_bot.set_xlabel('m/z', fontsize=12)
        ax_bot.set_ylabel('Intensity', fontsize=12)
        ax_bot.spines['right'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_bot.spines['left'].set_color('black')
        ax_bot.spines['bottom'].set_color('black')
        ax_bot.spines['left'].set_linewidth(1.5)
        ax_bot.spines['bottom'].set_linewidth(1.5)
        ax_bot.tick_params(axis='both', which='major', direction='out',
                           length=6, width=1.5, color='black')
        plt.legend(loc="best")

        # Save or close
        if savefig:
            outfile = os.path.join(savefig, f"motif_{m}.png")
            plt.savefig(outfile, format="png", dpi=400)
            plt.close(fig)
        else:
            # If in Jupyter => show
            if _in_nb:
                plt.show()
            else:
                plt.close(fig)

def _convert_molgrid_result_to_pil(res):
    """
    Attempt to convert the result of MolsToGridImage(...) into a PIL image.
    """
    # If we get a direct PIL image
    if isinstance(res, Image.Image):
        if hasattr(res, "data"):
            try:
                return Image.open(io.BytesIO(res.data))
            except Exception:
                return res
        else:
            return res

    # If it's bytes from returnPNG=True
    if isinstance(res, bytes):
        try:
            return Image.open(io.BytesIO(res))
        except Exception:
            pass

    return None

def compare_annotated_motifs(opt_motif_spectra, motif_spectra, clustered_smiles, valid_spectra, valid_mols, savefig=None):

    for m in range(len(motif_spectra)):
        mass_to_charge_0 = motif_spectra[m].peaks.mz
        intensities_0 = motif_spectra[m].intensities
        mass_to_charge_1 = opt_motif_spectra[m].peaks.mz
        intensities_1 = opt_motif_spectra[m].intensities
        mass_to_charge_2 = valid_spectra[m].peaks.mz
        intensities_2 = valid_spectra[m].intensities

        img_1 = MolToImage(valid_mols[m], size=(400, 400))
        img_2 = MolsToGridImage([MolFromSmiles(smi) for smi in clustered_smiles[m]], molsPerRow=len(clustered_smiles[m]), subImgSize=(400, 400))
        img_2 = Image.open(BytesIO(img_2.data))

        fig = plt.figure(figsize=(10,12), facecolor='none', edgecolor='none')
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.imshow(img_2)
        ax3.axis("off")
        pos = ax3.get_position()  # Get current position
        ax3.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height])  # Move it down by 0.05


        ax2 = fig.add_subplot(3, 1, 2)
        ax2.stem(mass_to_charge_2, intensities_2, basefmt="k-", markerfmt="", linefmt="black", label=f"{valid_spectra[m].get('id')}")

        # Stem plot for the second set (negative intensities for mirror effect)
        ax2.stem(mass_to_charge_0, [-i for i in intensities_0], basefmt="k-", markerfmt="", linefmt="grey", label=f"{opt_motif_spectra[m].get('id')}")
        #if mass_to_charge_1:
        ax2.stem(mass_to_charge_1, [-i for i in intensities_1], basefmt="k-", markerfmt="", linefmt="red", label=f"opt {opt_motif_spectra[m].get('id')}")

        # Set plot limits
        #plt.ylim(-max(intensities_2) - 10, max(intensities_1) + 10)
        ax2.set_yticks([-1, -0.75,-0.5,-0.25, 0,0.25,0.5,0.75, 1], ["1", "0.75","0.5","0.25", "0","0.25","0.5","0.75", "1"])
        # Add labels
        ax2.set_xlabel('m/z', fontsize=12)
        ax2.set_ylabel('Intensity', fontsize=12)

        # Add legend
        ax2.legend(loc='best', fontsize=12)

        # Customize the axes (remove top and right spines)
        ###ax = plt.gca()  # Get current axis
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # Optionally change color and width of remaining spines
        ax2.spines['left'].set_color('black')
        ax2.spines['bottom'].set_color('black')
        ax2.spines['left'].set_linewidth(1.5)
        ax2.spines['bottom'].set_linewidth(1.5)


        # Customize the ticks
        ax2.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, color='black')


        ax1 = fig.add_subplot(3, 1, 1)
        ax1.imshow(img_1)
        ax1.axis("off")
        pos = ax1.get_position()  # Get current position
        ax1.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height])  # Move it down by 0.05


        #plt.tight_layout()
        # Show the plot
        if savefig:
            plt.savefig(f"savefig/motif_{m}_spec_{m}.png", format="png", dpi=400)
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
