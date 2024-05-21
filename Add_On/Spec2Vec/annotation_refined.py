# this script contains masking and hierachical clustering

from matchms import Spectrum, Fragments

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity

import numpy as np

def mask_fragments(spectrum, mask=1.0):
    """masks fragments one by one
    
    ARGS:
        spectrum: matchms spectrum object
        mask (float): mz with which fragments will be replaced with
        
    RETURNS:
        masked_spectra (list): list of matchms spectrum objects
    """
    identifier = spectrum.get("id")
    
    fragments_mz = list(spectrum.peaks.mz)
    fragments_intensities = list(spectrum.peaks.intensities)

    losses_mz = spectrum.losses.mz
    losses_intensities = spectrum.losses.intensities

    masked_spectra = []
    for index in range(len(fragments_mz)):
        masked_fragments_mz = fragments_mz.copy()
        masked_fragments_intensities = fragments_intensities.copy()

        masked_fragments_mz.pop(index)
        masked_fragments_intensities.pop(index)

        retrieved_fragment_intensity = fragments_intensities[index]

        masked_fragments_mz = [mask] + masked_fragments_mz
        masked_fragments_intensities = [retrieved_fragment_intensity] + masked_fragments_intensities

        masked_spectrum = Spectrum(
            mz=np.array(masked_fragments_mz),
            intensities=np.array(masked_fragments_intensities),
            metadata={
                "id": identifier,
                "precursor_mz": max(fragments_mz)
            }
        )

        masked_spectrum.losses = Fragments(
            mz=losses_mz,
            intensities=losses_intensities)

        masked_spectra.append(masked_spectrum)

    return masked_spectra


def mask_losses(spectrum, mask=1.0):
    """masks losses one by one
    
    ARGS:
        spectrum: matchms spectrum object
        mask (float): mz with which losses will be replaced with
        
    RETURNS:
        masked_spectra (list): list of matchms spectrum objects
    """
    identifier = spectrum.get("id")
    
    fragments_mz = spectrum.peaks.mz
    fragments_intensities = spectrum.peaks.intensities

    losses_mz = list(spectrum.losses.mz)
    losses_intensities = list(spectrum.losses.intensities)
   
    masked_spectra = []
    for index in range(len(losses_mz)):
        masked_losses_mz = losses_mz.copy()
        masked_losses_intensities = losses_intensities.copy()

        masked_losses_mz.pop(index)
        masked_losses_intensities.pop(index)

        retrieved_loss_intensity = losses_intensities[index]

        masked_losses_mz = [mask] + masked_losses_mz
        masked_losses_intensities = [retrieved_loss_intensity] + masked_losses_intensities

        masked_spectrum = Spectrum(
            mz=fragments_mz,
            intensities=fragments_intensities,
            metadata={
                "id": identifier,
                "precursor_mz": max(fragments_mz)
            }
        )

        masked_spectrum.losses = Fragments(
            mz=np.array(masked_losses_mz),
            intensities=np.array(masked_losses_intensities))

        masked_spectra.append(masked_spectrum)

    return masked_spectra


def mask_spectra(motif_spectra, masks=[1.0,1.0]):
    """mask the fragments and losses for a list of spectra
    1. mask is for fragments
    2. mask is for losses
    
    ARGS:
        motif_spectra (list): list of matchms spectrum objects
        masks (list): list of float values for fragment and loss mask

    RETURNS:
        masked_motifs_spectra (list): list of lists of matchms spectrum objects; every list is for one motif
    """

    masked_motifs_spectra = []
    for spectrum in motif_spectra:
        masked_fragments_spectra = mask_fragments(spectrum, masks[0])
        masked_losses_spectra = mask_losses(spectrum, masks[1])
        masked_features_spectra = masked_fragments_spectra + masked_losses_spectra

        masked_motifs_spectra.append(masked_features_spectra)

    return masked_motifs_spectra


def hierachical_clustering(s2v_similarity, top_n_spectra, masked_spectra): # threshold adding
    """recursive function to determine to seperate groups in the found motifs
    
    ARGS:
        s2v_similarity: gensim word2vec based similarity model for Spec2Vec
        top_n_spectra (list): list of matchms spectrum objects
        masked_spectra (list): list of matchms spectrum objects
        
    RETURNS:
        top_spectra (list): list of matchms spectrum objects"""

    # 1. Termination condition: TOO FEW SPECTRA
    if len(top_n_spectra) < 3:
        return top_n_spectra
    
    # calcuate embeddings
    embeddings_top_n_spectra = calc_embeddings(s2v_similarity, top_n_spectra)
    embeddings_masked_spectra = calc_embeddings(s2v_similarity, masked_spectra)

    # calculate similarity
    masked_spectra_similarity = calc_similarity(embeddings_top_n_spectra, embeddings_masked_spectra)

    # calculate spearman correlation for pairs
    similarity_matrix = [list() for _ in range(len(top_n_spectra))]
    
    for i1 in range(len(top_n_spectra)):
        for i2 in range(len(top_n_spectra)):
            spearman_correlation = spearmanr(masked_spectra_similarity[i1], masked_spectra_similarity[i2])[0]
            similarity_matrix[i1].append(spearman_correlation)
    
    similarity_matrix = np.array(similarity_matrix)

    # 2. Termination condition: SPECTRA ARE VERY SIMILAR
    print(np.min(similarity_matrix.flatten()))
    if np.min(similarity_matrix.flatten()) > 0.60:
        return top_n_spectra

    # build clusters
    Z = linkage(similarity_matrix, method='complete')
    num_clusters = 2  

    cluster_index = fcluster(Z, num_clusters, criterion='maxclust')
    hierachical_clusters = [list() for _ in range(num_clusters)]
    for spectrum, index in zip(top_n_spectra, cluster_index):
        cluster_num = index - 1
        hierachical_clusters[cluster_num].append(spectrum)
        
    if len(hierachical_clusters[0]) > len(hierachical_clusters[1]):
        index = 0
    elif len(hierachical_clusters[0]) < len(hierachical_clusters[1]):
        index = 1
    else:
        print("clusters have same size")
        index = 0
    print(cluster_index)
    #all_clusters = []
    #for cluster in hierachical_clusters:
    #   all_clusters.extend(hierachical_clustering(s2v_similarity, cluster, masked_spectra))
    
    return hierachical_clustering(s2v_similarity, hierachical_clusters[index], masked_spectra)

if __name__ == "__main__":
    pass