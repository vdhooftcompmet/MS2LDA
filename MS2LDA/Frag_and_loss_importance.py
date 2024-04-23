import numpy as np
from matchms import Fragments, Spectrum
from matchms import calculate_scores

from gensim.models import Word2Vec
from spec2vec import Spec2Vec

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr

def mask_fragments(spectrum, mask):
    """masks fragments and losses one by one"""

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



def mask_losses(spectrum, mask):
    """masks fragments and losses one by one"""

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



def masked_motif_similarity(candidate_spectra, masked_motif):
    """similarity with spec2vec with masked spectra"""
    s2v_model = Word2Vec.load(r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\Spec2Struc\Project_SubstructureIdentification\scripts\programming_scripts\models\spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")

    spec2vec_similarity = Spec2Vec(model=s2v_model,
                               intensity_weighting_power=0.5,
                               allowed_missing_percentage=100)
    
    similarity_per_candidates = [list() for _ in range(len(candidate_spectra))]

    for motif_spectrum in masked_motif:
        similarity_per_candidate = calculate_scores(candidate_spectra, [motif_spectrum], spec2vec_similarity)

        for candidate_idx, (_, _, score) in enumerate(similarity_per_candidate):
            score = score[0]
            similarity_per_candidates[candidate_idx].append(score)

    return similarity_per_candidates



def cluster_by_similarity(similarity_per_candidates, n_clusters=2):
    """cluster candidate spectra based on masked motif similarity values"""

    integer_iterator = range(len(similarity_per_candidates))

    similarity_matrix = [list() for _ in integer_iterator]
    for i in integer_iterator:
        for ii in integer_iterator:
            similarity_matrix[i].append(spearmanr(similarity_per_candidates[i], similarity_per_candidates[ii])[0])

    Z = linkage(np.array(similarity_matrix), method="complete")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    return clusters, Z, similarity_matrix
    




if __name__ == "__main__":
    motif = Spectrum()
    candidate_spectra = Spectrum()
    # only for one motif
    masked_fragment_motif = mask_fragments(motif, 1.00)
    masked_loss_motif = mask_losses(motif, 1.00)
    masked_motif = masked_fragment_motif + masked_loss_motif

    similarity_per_candidates = masked_motif_similarity(candidate_spectra, masked_motif)
    clusters = cluster_by_similarity(similarity_per_candidates)
    


