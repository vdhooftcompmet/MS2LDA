# this script contains masking and hierachical clustering

from matchms import Spectrum, Fragments
from ms2lda.Mass2Motif import Mass2Motif

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster

from ms2lda.Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity_faiss

# from Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity
from spec2vec.vector_operations import cosine_similarity_matrix

from functools import reduce
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from ms2lda.Mass2Motif import Mass2Motif
import pandas as pd


# -------------------------------------------------reconstruct motif spectrum---------------------------------------#


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

    motif_spectrum_fragments_mz = [
        round(frag, frag_err) for frag in motif_spectrum.peaks.mz
    ]
    for fragment_mz in common_fragments:
        if fragment_mz in motif_spectrum_fragments_mz:
            index = motif_spectrum_fragments_mz.index(fragment_mz)
            fragment_intensity = motif_spectrum.peaks.intensities[index]
            fragment_mz = motif_spectrum.peaks.mz[index]

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

    motif_spectrum_losses_mz = [
        round(loss, loss_err) for loss in motif_spectrum.losses.mz
    ]
    for loss_mz in common_losses:
        if loss_mz in motif_spectrum_losses_mz:
            index = motif_spectrum_losses_mz.index(loss_mz)
            loss_intensity = motif_spectrum.losses.intensities[index]
            loss_mz = motif_spectrum.losses.mz[index]

            opt_motif_losses_mz.append(loss_mz)
            opt_motif_losses_intensities.append(loss_intensity)

    return opt_motif_losses_mz, opt_motif_losses_intensities


def reconstruct_motif_spectrum(
    opt_motif_fragments_mz,
    opt_motif_fragments_intensities,
    opt_motif_losses_mz,
    opt_motif_losses_intensities,
):
    """creates a matchms spectrum object based on the optimized features

    ARGS:
        opt_motif_fragments_mz (list): list of float values representing mz values for an optimized motif (fragments)
        opt_motif_fragments_intensities (list): list of float values representing intensity values for an optimized motif (fragments)
        opt_motif_losses_mz (list): list of float values representing mz values for an optimized motif (losses)
        opt_motif_losses_intensities (list): list of float values representing intensity values for an optimized motif (losses)

    RETURNS:
        opt_motif_spectrum: matchms spectrum object
    """
    if (
        len(opt_motif_fragments_mz) == len(opt_motif_fragments_intensities)
        and len(opt_motif_fragments_mz) > 0
    ):
        sorted_fragments = sorted(
            zip(opt_motif_fragments_mz, opt_motif_fragments_intensities)
        )
        opt_motif_fragments_mz, opt_motif_fragments_intensities = zip(*sorted_fragments)
    else:
        opt_motif_fragments_mz = np.array([])
        opt_motif_losses_intensities = np.array([])

    if (
        len(opt_motif_losses_mz) == len(opt_motif_losses_intensities)
        and len(opt_motif_losses_mz) > 0
    ):  # I once saw that there was a loss mz 0f 1.003 and no intensity!!
        sorted_losses = sorted(zip(opt_motif_losses_mz, opt_motif_losses_intensities))
        opt_motif_losses_mz, opt_motif_losses_intensities = zip(*sorted_losses)

    else:
        opt_motif_losses_mz = np.array([])
        opt_motif_losses_intensities = np.array([])

    # opt_motif_spectrum = Spectrum(
    #    mz = np.array(opt_motif_fragments_mz),
    #    intensities = np.array(opt_motif_fragments_intensities),
    # )

    # if opt_motif_losses_mz and opt_motif_losses_intensities: # for some reasons it can be that losses have mz but no intensity for large numbers of extracted compounds
    #    sorted_losses = sorted(zip(opt_motif_losses_mz, opt_motif_losses_intensities))
    #    opt_motif_losses_mz, opt_motif_losses_intensities = zip(*sorted_losses)

    #    opt_motif_spectrum.losses = Fragments(
    #        mz=np.array(opt_motif_losses_mz),
    #        intensities=np.array(opt_motif_losses_intensities)
    #    )
    opt_motif_spectrum = Mass2Motif(
        frag_mz=np.array(opt_motif_fragments_mz),
        frag_intensities=np.array(opt_motif_fragments_intensities),
        loss_mz=np.array(opt_motif_losses_mz),
        loss_intensities=np.array(opt_motif_losses_intensities),
    )

    return opt_motif_spectrum


def optimize_motif_spectrum(
    motif_spectrum, hit_spectra, smiles_cluster, frag_err=2, loss_err=2
):
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
    opt_motif_fragments_mz, opt_motif_fragments_intensities = (
        motif_intersection_fragments(motif_spectrum, common_fragments, frag_err)
    )

    common_losses = hits_intersection(losses_mz)
    opt_motif_losses_mz, opt_motif_losses_intensities = motif_intersection_losses(
        motif_spectrum, common_losses, loss_err
    )

    opt_motif_spectrum = reconstruct_motif_spectrum(
        opt_motif_fragments_mz,
        opt_motif_fragments_intensities,
        opt_motif_losses_mz,
        opt_motif_losses_intensities,
    )
    opt_motif_spectrum.set("Auto_annotation", smiles_cluster)
    opt_motif_spectrum.set("short_annotation", None)
    opt_motif_spectrum.set("charge", motif_spectrum.get("charge"))
    opt_motif_spectrum.set("ms2accuracy", motif_spectrum.get("ms2accuracy"))
    opt_motif_spectrum.set("motifset", motif_spectrum.get("motifset"))
    opt_motif_spectrum.set("annotation", None)
    opt_motif_spectrum.set("id", motif_spectrum.get("id"))

    return opt_motif_spectrum


# -------------------------------------------------cluster motif hits---------------------------------------#


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
        masked_fragments_intensities = [
            retrieved_fragment_intensity
        ] + masked_fragments_intensities

        masked_spectrum = Mass2Motif(
            frag_mz=np.array(masked_fragments_mz),
            frag_intensities=np.array(masked_fragments_intensities),
            loss_mz=np.array(losses_mz),
            loss_intensities=np.array(losses_intensities),
            metadata={"id": identifier, "precursor_mz": None},
        )

        masked_spectra.append(masked_spectrum)

    return masked_spectra


def mask_losses(
    spectrum, mask=0.0
):  # manually connecting mask_losses and mask fragments kind of failed when not combining (no frags)
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
        masked_losses_intensities = [
            retrieved_loss_intensity
        ] + masked_losses_intensities

        masked_spectrum = Mass2Motif(
            frag_mz=np.array(fragments_mz),
            frag_intensities=np.array(fragments_intensities),
            loss_mz=np.array(masked_losses_mz),
            loss_intensities=np.array(masked_losses_intensities),
            metadata={"id": identifier, "precursor_mz": None},
        )

        masked_spectra.append(masked_spectrum)

    return masked_spectra


def mask_spectra(
    motif_spectra, masks=[-1.0, -1.0]
):  # BUG: if there are not fragments it fails!!!
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


import warnings
import logging


def calc_similarity(embeddings_A, embeddings_B):
    """calculates the cosine similarity of spectral embeddings

    ARGS:
        embeddings_A (np.array): query spectral embeddings
        embeddings_B (np.array): reference spectral embeddings

    RETURNS:
        similarities_scores (list): list of lists with s2v similarity scores
    """
    if type(embeddings_B) == pd.core.series.Series:
        embeddings_B = np.vstack(embeddings_B.to_numpy())

    similarity_vectors = []
    for embedding_A in embeddings_A:
        similarity_scores = cosine_similarity_matrix(
            np.array([embedding_A]), embeddings_B
        )[0]
        similarity_vectors.append(similarity_scores)

    similarities_matrix = pd.DataFrame(
        np.array(similarity_vectors).T, columns=range(len(embeddings_A))
    )

    return similarities_matrix


def calc_similarity_matrix(s2v_similarity, top_n_spectra, masked_spectra):
    """Calculates a similarity matrix between top hits."""

    # Suppress the specific warning from spec2vec
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Spectrum without peaks known by the used model.*"
        )

        # Suppress logging warnings from spec2vec
        spec2vec_logger = logging.getLogger("spec2vec")
        spec2vec_logger.setLevel(
            logging.ERROR
        )  # Set logging level to ERROR to suppress warnings

        # Calculate embeddings
        embeddings_top_n_spectra = calc_embeddings(s2v_similarity, top_n_spectra)
        embeddings_masked_spectra = calc_embeddings(s2v_similarity, masked_spectra)

        # Calculate similarity
        masked_similarities = calc_similarity(
            embeddings_top_n_spectra, embeddings_masked_spectra
        )

    return masked_similarities.T


def agglomerative_clustering(masked_spectra_similarity, cosine_similarity=0.6):
    if masked_spectra_similarity.shape[0] > 1:
        cosine_distance = 1 - cosine_similarity
        cosine_distance_matrix = 1 - masked_spectra_similarity
        clustering = AgglomerativeClustering(
            distance_threshold=cosine_distance,
            n_clusters=None,
            linkage="complete",
        )

        labels = clustering.fit_predict(cosine_distance_matrix)

    else:
        labels = np.array([0])

    return labels


# -------------------------------------summary functions---------------------#


def hit_clustering(
    s2v_similarity,
    motif_spectra,
    library_matches,
    criterium="best",
    cosine_similarity=0.6,
):
    masked_spectra = mask_spectra(motif_spectra)

    clustered_spec = []
    clustered_smiles = []
    clustered_scores = []
    for library_match, masked_spec in zip(library_matches, masked_spectra):
        top_n_smiles = library_match[0]
        top_n_spectra = library_match[1]
        top_n_scores = library_match[2]

        s2v_similarity4masked_motifs = calc_similarity_matrix(
            s2v_similarity, top_n_spectra, masked_spec
        )
        s2v_similarity4masked_motifs_filtered = s2v_similarity4masked_motifs.dropna()
        labels = agglomerative_clustering(
            s2v_similarity4masked_motifs_filtered, cosine_similarity
        )
        spectra_same_label = []
        smiles_same_label = []
        scores_same_label = []

        if criterium == "best":
            best_hit_label = labels[0]
            index_same_label = np.argwhere(labels == best_hit_label).flatten()

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
            index_same_label = np.argwhere(labels == biggest_label).flatten()

            for index in index_same_label:
                spectra_same_label.append(top_n_spectra[index])
                smiles_same_label.append(top_n_smiles[index])
                scores_same_label.append(top_n_scores[index])
            clustered_spec.append(spectra_same_label)
            clustered_smiles.append(smiles_same_label)
            clustered_scores.append(scores_same_label)

    return clustered_spec, clustered_smiles, clustered_scores


# -------------------------------------optimization---------------------#


def motif_optimization(motif_spectra, clustered_spectra, clustered_smiles, loss_err=1):
    # This generates a spectrum object where the fragments/losses present are the ones that are present in all hits
    opt_motif_spectra = []
    for motif_spec, spec, smiles_cluster in zip(
        motif_spectra, clustered_spectra, clustered_smiles
    ):
        opt_motif_spec = optimize_motif_spectrum(
            motif_spec, spec, smiles_cluster, loss_err=1
        )
        opt_motif_spectra.append(opt_motif_spec)

    return opt_motif_spectra


if __name__ == "__main__":
    from matchms.filtering import add_losses

    spectrum_1 = Spectrum(
        mz=np.array([100.0, 130.0, 200.0]),
        intensities=np.array([0.7, 0.2, 0.1]),
        metadata={"id": "spectrum1", "precursor_mz": 201.0},
    )
    spectrum_2 = Spectrum(
        mz=np.array([100.0, 140.0, 200.0]),
        intensities=np.array([0.4, 0.2, 0.1]),
        metadata={"id": "spectrum2", "precursor_mz": 211.0},
    )
    spectrum_3 = Spectrum(
        mz=np.array([60.0, 100.0, 140.0, 200.0]),
        intensities=np.array([0.8, 0.4, 0.2, 0.1]),
        metadata={"id": "spectrum2", "precursor_mz": 211.0},
    )
    spectrum_4 = Spectrum(
        mz=np.array([100.0, 120.0, 140.0, 200.0]),
        intensities=np.array([0.4, 0.6, 0.2, 0.1]),
        metadata={"id": "spectrum2", "precursor_mz": 211.0},
    )

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
    from ms2lda.Add_On.Spec2Vec.annotation import load_s2v_and_library

    path_model = "model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model"
    path_library = "model_positive_mode/positive_s2v_library.pkl"
    s2v_similarity, library = load_s2v_and_library(path_model, path_library)
    print("Model loaded ...")

    masked_spectra = mask_spectra([motif_spectrum])

    masked_spectra_similarity = calc_similarity_matrix(
        s2v_similarity, test_spectra, masked_spectra[0]
    )
    print(masked_spectra_similarity)
    similarities = calc_cosine_matrix(masked_spectra_similarity)
    print(similarities)
    labels = agglomerative_clustering(similarities)
    print(labels)
