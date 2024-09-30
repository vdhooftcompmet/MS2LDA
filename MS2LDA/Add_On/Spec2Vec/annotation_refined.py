# this script contains masking and hierachical clustering

from matchms import Spectrum, Fragments

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster

from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity
#from Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity


from functools import reduce
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from ordered_set import OrderedSet

#-------------------------------------------------reconstruct motif spectrum---------------------------------------#

def get_mz(spectra, frag_err=2, loss_err=2):
    """extracts fragments and losses from a list of spectra
    
    ARGS:
        spectra (list): list of matchms spectra objects
        frag_err (int; default = 2): number of significant digits to round for fragments
        loss_err (int; default = 2): number of significant digits to round for losses

    RETURNS:
        fragments_mz (list): list of rounded float numbers for fragments
        losses_mz (list): list of rounded float numbers for losses
    """
    fragments_mz = []
    losses_mz = []
    for spectrum in spectra:
        fragments_mz.append(set([round(frag, frag_err) for frag in spectrum.peaks.mz]))
        losses_mz.append(set([round(loss, loss_err) for loss in spectrum.losses.mz]))

    return fragments_mz, losses_mz


def hits_intersection(features): 
    """returns values that a present across all input lists
    
    ARGS:
        features (list): list of either losses or fragments

    RETURNS:
        common_features (list): list of either losses or fragments that are the intersection of the given lists
    """
    common_features = set.intersection(*features)
    return common_features


def motif_intersection_fragments(motif_spectrum, common_fragments, frag_err=2):
    """retrieves mz values and intensities for fragments that are the intersection between the motif spectrum and the common hits fragments

    ARGS:
        motif_spectrum: matchms.spectrum.object
        common_fragments (list): list of float values
        frag_err (int, default = 2): number of significant digits to round for fragments

    RETURNS:
        opt_motif_fragments_mz (list): list of float values representing mz values for an optimized motif
        opt_motif_fragments_intensities (list): list of float values representing intensity values for an optimized motif
    """
    opt_motif_fragments_mz = []
    opt_motif_fragments_intensities = []

    motif_spectrum_fragments_mz = [round(frag, frag_err) for frag in motif_spectrum.peaks.mz]
    for fragment_mz in common_fragments:
        if fragment_mz in motif_spectrum_fragments_mz:
            index = motif_spectrum_fragments_mz.index(fragment_mz)
            fragment_intensity = motif_spectrum.peaks.intensities[index]

            opt_motif_fragments_mz.append(fragment_mz)
            opt_motif_fragments_intensities.append(fragment_intensity)

    return opt_motif_fragments_mz, opt_motif_fragments_intensities


def motif_intersection_losses(motif_spectrum, common_losses, loss_err=2):
    """retrieves mz values and intensities for losses that are the intersection between the motif spectrum and the common hits losses

    ARGS:
        motif_spectrum: matchms.spectrum.object
        common_losses (list): list of float values
        loss_err (int, default = 2): number of significant digits to round for losses

    RETURNS:
        opt_motif_losses_mz (list): list of float values representing mz values for an optimized motif
        opt_motif_losses_intensities (list): list of float values representing intensity values for an optimized motif
    """
    opt_motif_losses_mz = []
    opt_motif_losses_intensities = []
    
    motif_spectrum_losses_mz = [round(loss, loss_err) for loss in motif_spectrum.losses.mz]
    for loss_mz in common_losses:
        if loss_mz in motif_spectrum_losses_mz:
            index = motif_spectrum_losses_mz.index(loss_mz)
            loss_intensity = motif_spectrum.losses.intensities[index]

            opt_motif_losses_mz.append(loss_mz)
            opt_motif_losses_intensities.append(loss_intensity)

    return opt_motif_losses_mz, opt_motif_losses_intensities


def reconstruct_motif_spectrum(opt_motif_fragments_mz, opt_motif_fragments_intensities, opt_motif_losses_mz, opt_motif_losses_intensities):
    """creates a matchms spectrum object based on the optimized features
    
    ARGS:
        opt_motif_fragments_mz (list): list of float values representing mz values for an optimized motif (fragments)
        opt_motif_fragments_intensities (list): list of float values representing intensity values for an optimized motif (fragments)
        opt_motif_losses_mz (list): list of float values representing mz values for an optimized motif (losses)
        opt_motif_losses_intensities (list): list of float values representing intensity values for an optimized motif (losses)

    RETURNS: 
        opt_motif_spectrum: matchms spectrum object
    """
    if opt_motif_fragments_mz:
        sorted_fragments = sorted(zip(opt_motif_fragments_mz, opt_motif_fragments_intensities))
        opt_motif_fragments_mz, opt_motif_fragments_intensities = zip(*sorted_fragments)
    else:
        opt_motif_fragments_mz = []
        opt_motif_losses_intensities = []


    opt_motif_spectrum = Spectrum(
        mz = np.array(opt_motif_fragments_mz),
        intensities = np.array(opt_motif_fragments_intensities),
        metadata={
            #"short_annotation": smiles_cluster, # and add metadata function would be useful
            #"charge": motif_spectrum.get("charge"),
            #"ms2accuracy": motif_spectrum.get("ms2accuracy"),
            #"motifset": motif_spectrum.get("motifset"),
            "annotation": None,
            #"id": motif_spectrum.get("id")
            }
    )

    if opt_motif_losses_mz:
        sorted_losses = sorted(zip(opt_motif_losses_mz, opt_motif_losses_intensities))
        opt_motif_losses_mz, opt_motif_losses_intensities = zip(*sorted_losses)

        opt_motif_spectrum.losses = Fragments(
            mz=np.array(opt_motif_losses_mz),
            intensities=np.array(opt_motif_losses_intensities)
        )

    return opt_motif_spectrum


def optimize_motif_spectrum(motif_spectrum, hit_spectra, frag_err=2, loss_err=2):
    """runs all scripts from extracting features to overlapping them and creating an optimized motif
    
    ARGS:
        motif_spectrum: matchms spectrum object
        hit_spectra (list): list of matchms spectrum objects
        frag_err (int; default = 2): number of significant digits to round for fragments
        loss_err (int; default = 2): number of significant digits to round for losses

    RETURNS:
        opt_motif_spectrum: matchms spectrum object
    """
    fragments_mz, losses_mz = get_mz(hit_spectra)
    
    common_fragments = hits_intersection(fragments_mz)
    opt_motif_fragments_mz, opt_motif_fragments_intensities = motif_intersection_fragments(motif_spectrum, common_fragments, frag_err)
    
    common_losses = hits_intersection(losses_mz)    
    opt_motif_losses_mz, opt_motif_losses_intensities = motif_intersection_losses(motif_spectrum, common_losses, loss_err)
    
    opt_motif_spectrum = reconstruct_motif_spectrum(opt_motif_fragments_mz, opt_motif_fragments_intensities, opt_motif_losses_mz, opt_motif_losses_intensities)

    return opt_motif_spectrum

#-------------------------------------------------cluster motif hits---------------------------------------#

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


def mask_losses(spectrum, mask=0.0): # manually connecting mask_losses and mask fragments kind of failed when not combining (no frags)
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


def calc_similarity_matrix(s2v_similarity, top_n_spectra, masked_spectra):
    """calculates a similarity matrix between top hits 
    """
    # calcuate embeddings
    embeddings_top_n_spectra = calc_embeddings(s2v_similarity, top_n_spectra)
    embeddings_masked_spectra = calc_embeddings(s2v_similarity, masked_spectra)

    # calculate similarity
    masked_spectra_similarity = calc_similarity(embeddings_top_n_spectra, embeddings_masked_spectra)

    return masked_spectra_similarity.T


def agglomerative_clustering(masked_spectra_similarity, cosine_similarity=0.6):
    if masked_spectra_similarity.shape[0] > 1:
        cosine_distance = 1 - cosine_similarity
        cosine_distance_matrix = 1 - masked_spectra_similarity
        clustering = AgglomerativeClustering(
            distance_threshold=cosine_distance,  
            n_clusters= None,
            linkage="complete",
        )

        labels = clustering.fit_predict(cosine_distance_matrix)

    else:
        labels = np.array([0])

    return labels

#-------------------------------------summary functions---------------------#

def hit_clustering(s2v_similarity, motif_spectra, library_matches, criterium="best"):
    masked_spectra = mask_spectra(motif_spectra)

    clustered_spec = []
    clustered_smiles = []
    clustered_scores = []
    for library_match, masked_spec in zip(library_matches, masked_spectra):
        top_n_smiles = library_match[0]
        top_n_spectra = library_match[1]
        top_n_scores = library_match[2]

        s2v_similarity4masked_motifs = calc_similarity_matrix(s2v_similarity, top_n_spectra, masked_spec)
        labels = agglomerative_clustering(s2v_similarity4masked_motifs)

        spectra_same_label = []
        smiles_same_label = []
        scores_same_label = []

        if criterium == "best":
            best_hit_label = labels[0]
            index_same_label = np.argwhere(labels==best_hit_label).flatten()
            
            for index in index_same_label:
                spectra_same_label.append(top_n_spectra[index])
                smiles_same_label.append(top_n_smiles[index])
                scores_same_label.append(top_n_scores[index])
            clustered_spec.append(spectra_same_label)
            clustered_smiles.append(smiles_same_label)
            clustered_scores.append(scores_same_label)

        elif criterium == "biggest":
            counts = np.bincount(labels)
            biggest_label = np.argmax(counts)
            index_same_label = np.argwhere(labels==biggest_label).flatten()
    
            for index in index_same_label:
                spectra_same_label.append(top_n_spectra[index])
                smiles_same_label.append(top_n_smiles[index])
                scores_same_label.append(top_n_scores[index])
            clustered_spec.append(spectra_same_label)
            clustered_smiles.append(smiles_same_label)
            clustered_scores.append(scores_same_label)

    return clustered_spec, clustered_smiles, clustered_scores
            






if __name__ == "__main__":
    from matchms.filtering import add_losses
    spectrum_1 = Spectrum(mz=np.array([100.0, 130.0, 200.0]),
                      intensities=np.array([0.7, 0.2, 0.1]),
                      metadata={'id': 'spectrum1',
                                'precursor_mz': 201.0})
    spectrum_2 = Spectrum(mz=np.array([100.0, 140.0, 200.]),
                        intensities=np.array([0.4, 0.2, 0.1]),
                        metadata={'id': 'spectrum2',
                                  'precursor_mz': 211.0})
    spectrum_3 = Spectrum(mz=np.array([60.0, 100.0, 140.0, 200.]),
                        intensities=np.array([0.8, 0.4, 0.2, 0.1]),
                        metadata={'id': 'spectrum2',
                                  'precursor_mz': 211.0})
    spectrum_4 = Spectrum(mz=np.array([100.0, 120.0, 140.0, 200.]),
                        intensities=np.array([0.4, 0.6, 0.2, 0.1]),
                        metadata={'id': 'spectrum2',
                                  'precursor_mz': 211.0})
    
    motif_spectrum = add_losses(spectrum_1)
    s2 = add_losses(spectrum_2)
    s3 = add_losses(spectrum_3)
    s4 = add_losses(spectrum_4)

    test_spectra = [s2, s3, s4]
    print(motif_spectrum.losses.mz)
    print(s2.losses.mz)
    print(s3.losses.mz)
    print(s4.losses.mz)
    s = new_motif_spectrum(motif_spectrum, test_spectra)
    print(s)
    print(s.peaks.mz)
    print(s.losses.mz)
    from MS2LDA.Add_On.Spec2Vec.annotation import load_s2v_and_library
    path_model = "model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model"
    path_library = "model_positive_mode/positive_s2v_library.pkl"
    s2v_similarity, library = load_s2v_and_library(path_model, path_library)
    print("Model loaded ...")

    masked_spectra = mask_spectra([motif_spectrum])

    masked_spectra_similarity = calc_similarity_matrix(s2v_similarity, test_spectra, masked_spectra[0])
    print(masked_spectra_similarity)
    similarities = calc_cosine_matrix(masked_spectra_similarity)
    print(similarities)
    labels = agglomerative_clustering(similarities)
    print(labels)