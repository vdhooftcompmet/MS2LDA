# this script contains masking and hierachical clustering

from matchms import Spectrum, Fragments

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster

from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity
#from Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity


from functools import reduce
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
                "precursor_mz": None
            }
        )

        masked_spectrum.losses = Fragments(
            mz=losses_mz,
            intensities=losses_intensities)

        masked_spectra.append(masked_spectrum)

    return masked_spectra


def mask_losses(spectrum, mask=1.0): # manually connecting mask_losses and mask fragments kind of failed when not combining (no frags)
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
                "precursor_mz": None
            }
        )

        masked_spectrum.losses = Fragments(
            mz=np.array(masked_losses_mz),
            intensities=np.array(masked_losses_intensities))

        masked_spectra.append(masked_spectrum)

    return masked_spectra


def mask_spectra(motif_spectra, masks=[1.0,1.0]): #BUG: if there are not fragments it fails!!!
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


def hierachical_clustering(s2v_similarity, top_n_spectra, top_n_scores, masked_spectra, masked_spectra_similarity=None): # threshold adding
    """recursive function to determine to seperate groups in the found motifs
    
    ARGS:
        s2v_similarity: gensim word2vec based similarity model for Spec2Vec
        top_n_spectra (list): list of matchms spectrum objects
        top_n_scores (list): list of float representing Spec2Vec similarity scores between a spectrum and motif
        masked_spectra (list): list of matchms spectrum objects
        
    RETURNS:
        top_spectra (list): list of matchms spectrum objects
    """
    
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

    # 1. Termination condition: TOO FEW SPECTRA
    if len(top_n_spectra) < 2:
        print("One compound cluster!")
        return top_n_spectra, top_n_scores, masked_spectra_similarity, "?"
    

    # 2. Termination condition: SPECTRA ARE VERY SIMILAR
    if np.min(similarity_matrix.flatten()) > 0.70:
        print("Similarity Match: ", np.min(similarity_matrix.flatten()))
        return top_n_spectra, top_n_scores, masked_spectra_similarity, np.min(similarity_matrix.flatten())

    # build clusters
    Z = linkage(similarity_matrix, method='complete')
    num_clusters = 2  

    cluster_index = fcluster(Z, num_clusters, criterion='maxclust')
    #cluster_index = fcluster(Z, num_clusters, criterion='distance')
    hierachical_clusters = [list() for _ in range(num_clusters)]
    hierachical_cluster_scores = [list() for _ in range(num_clusters)]

    for spectrum, score, index in zip(top_n_spectra, top_n_scores, cluster_index):
        cluster_num = index - 1
        hierachical_clusters[cluster_num].append(spectrum)
        hierachical_cluster_scores[cluster_num].append(score)
        
    if not hierachical_cluster_scores[0]:
        print("Only one cluster: ", np.min(similarity_matrix.flatten()))
        return top_n_spectra, top_n_scores, masked_spectra_similarity, np.min(similarity_matrix.flatten())
    elif not hierachical_cluster_scores[1]:
        print("Only one cluster: ", np.min(similarity_matrix.flatten()))
        return top_n_spectra, top_n_scores, masked_spectra_similarity, np.min(similarity_matrix.flatten())
    elif np.max(hierachical_cluster_scores[0]) > np.max(hierachical_cluster_scores[1]):
        index = 0
    elif np.max(hierachical_cluster_scores[0]) < np.max(hierachical_cluster_scores[1]):
        index = 1
    else:
        print("matches have same similarity")
        index = 0
    
    return hierachical_clustering(s2v_similarity, hierachical_clusters[index], hierachical_cluster_scores[index], masked_spectra, masked_spectra_similarity)


def find_important_features(unmasked_spectrum_similarity, masked_spectra_similarity):
    """returns a motif spectrum for based on something
    
    ARGS:
        unmasked_spectrum_similarity (float): Spec2Vec similarity for compound to unmasked motif spectrum
        masked_spectra_similarity (list): list of floats with Spec2Vec similarity for compound to each masked spectrum

    RETURNS:
        important_features (list): list of int values giving the indices of a feature in a motif spectrum
    """
    important_features = []

    for index_feature, masked_spectrum_similarity in enumerate(masked_spectra_similarity):
        feature_importance = masked_spectrum_similarity - unmasked_spectrum_similarity

        if feature_importance < 0: 
            important_features.append(index_feature)


    return important_features


def merge_important_features(unmasked_spectra_similarity, multiple_masked_spectra_similarity):
    """merges important features
    
    ARGS:
        unmasked_spectra_similarity (float): Spec2Vec similarity for compound to unmasked motif spectrum
        multiple_masked_spectra_similarity (pd.DataFrame): dataframe of floats with Spec2Vec similarity for all compounds to each masked spectrum
        
    RETURNS:
        aligned_important_features (list): intersection of feature sets that have when masked no or a positive effect across all sets
    """
    multiple_important_features = []

    for unmasked_spectrum_similarity, (_, masked_spectra_similarity) in zip(unmasked_spectra_similarity, multiple_masked_spectra_similarity.items()):
        important_features = find_important_features(unmasked_spectrum_similarity, masked_spectra_similarity)
        
        multiple_important_features.append(important_features)

    aligned_important_features = list(reduce(np.intersect1d, multiple_important_features))

    return aligned_important_features


def reconstruct_motif_spectrum(motif_spectrum, aligned_important_features, smiles_cluster):
    """something
    
    ARGS:
        motif_spectrum: matchms spectrum object
        aligned_important_features (list): intersection of feature sets that have when masked no or a positive effect across all sets
        
    RETURNS: 
        reconstructed_motif_spectrum: matchms spectrum object
    """
    fragments_mz = motif_spectrum.peaks.mz
    fragments_intensities = motif_spectrum.peaks.intensities
    fragments_range = len(fragments_mz) # to distinguish between fragments and losses

    losses_mz = motif_spectrum.losses.mz
    losses_intensities = motif_spectrum.losses.intensities

    new_fragments_mz = []
    new_fragments_intensities = []

    new_losses_mz = []
    new_losses_intensities = []
    
    for aligned_important_feature in aligned_important_features:
        if aligned_important_feature < fragments_range:
            # it's a fragment
            new_fragments_mz.append(fragments_mz[aligned_important_feature])
            new_fragments_intensities.append(fragments_intensities[aligned_important_feature])

        else:
            # it's a loss
            new_index = fragments_range - aligned_important_feature
            new_losses_mz.append(losses_mz[new_index])
            new_losses_intensities.append(losses_intensities[new_index])


    reconstructed_motif_spectrum = Spectrum(
        mz = np.array(new_fragments_mz),
        intensities = np.array(new_fragments_intensities),
        metadata={
            "short_annotation": smiles_cluster,
            "charge": motif_spectrum.get("charge"),
            "ms2accuracy": motif_spectrum.get("ms2accuracy"),
            "motifset": motif_spectrum.get("motifset"),
            "annotation": None,
            "id": motif_spectrum.get("id")
            }
    )

    if new_losses_mz: # losses are for some reason not in order
        losses_mz_intensities = list(zip(new_losses_mz, new_losses_intensities))
        losses_mz_intensities_sorted = sorted(losses_mz_intensities, key=lambda x: x[0])
        new_losses_mz, new_losses_intensities = zip(*losses_mz_intensities_sorted)

    reconstructed_motif_spectrum.losses = Fragments(
        mz=np.array(new_losses_mz),
        intensities=np.array(new_losses_intensities)
    )

    return reconstructed_motif_spectrum


def refine_annotation(s2v_similarity, library_matches, masked_motifs_spectra, motif_spectra):
    """runs the refined annotation from hierachical clustering to removing features based on it's importance for the found compounds
    
    ARGS:
        s2v_similarity: gensim word2vec based similarity model for Spec2Vec
        library_matches (list): list of lists, where each motif contains of three lists with spectra, SMILES and s2v scores
        masked_motifs_spectra (list): list of lists of matchms spectrum objects; every list is for one motif
        motif_spectra (list): list of matchms spectrum objects
        
    RETURNS:
        optimized_motif_spectra (list): list of matchms spectrum objects which can contain the same or less features than the original one
        optimized_clusters (list): list of lists with the compound spectra found by the annotation and seperated by hierachical clustering
        smiles_clusters (list): list of lists with the compound SMILES found by the annotation and seperated by hierachical clustering
    """

    optimized_motif_spectra = []
    optimized_clusters = []
    smiles_clusters = []
    clusters_similarity = []
    for i in range(len(masked_motifs_spectra)):
        cluster, scores, masked_spectra_similarity, cluster_similarity = hierachical_clustering(s2v_similarity, library_matches[i][1], library_matches[i][2], masked_motifs_spectra[i])
        
        optimized_clusters.append(cluster)
        clusters_similarity.append(cluster_similarity)
        smiles_cluster = []
        for spectrum in cluster:
            smiles = spectrum.get("smiles")
            smiles_cluster.append(smiles)

        smiles_clusters.append(smiles_cluster)

        aligned_important_features = merge_important_features(scores, masked_spectra_similarity)
        optimized_motif_spectrum = reconstruct_motif_spectrum(motif_spectra[i], aligned_important_features, smiles_cluster)

        optimized_motif_spectra.append(optimized_motif_spectrum)

    
    return optimized_motif_spectra, optimized_clusters, smiles_clusters, clusters_similarity
            

if __name__ == "__main__":
    pass