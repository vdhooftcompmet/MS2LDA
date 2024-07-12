import matchms.filtering as msfilters

from scipy.stats import pearsonr as calc_pearson
import numpy as np
import pandas as pd

from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem

from msbuddy import Msbuddy


def match_spectral_overlap(spectrum_1, spectrum_2, margin_fragments=0.005, margin_losses=0.01):
    """calculates the overlap of two spectra considering the fragments and losses
    
    ARGS:
        spectrum_1: matchms spectrum object
        spectrum_2: matchms spectrum object
        margin_fragments (float): margin of error for fragments for a match
        margin_losses (float): margin of error for losses for a match
        
    RETURNS:
        overlaping_fragments (list): list for overlaping mz values of fragments
        overlaping_losses (list): list for overlaping mz values of losses
    """

    spectrum_1_fragments_mz = spectrum_1.peaks.mz
    spectrum_1_fragments_intensities = spectrum_1.peaks.intensities
    spectrum_1_losses_mz = spectrum_1.losses.mz
    spectrum_1_losses_intensities = spectrum_1.losses.intensities

    spectrum_2_fragments_mz = spectrum_2.peaks.mz
    spectrum_2_fragments_intensities = spectrum_2.peaks.intensities
    spectrum_2_losses_mz = spectrum_2.losses.mz
    spectrum_2_losses_intensities = spectrum_2.losses.intensities


    def find_overlap(arr1, arr2, margin):
        """finds the overlap of two arrays
        
        ARGS:
            arr1 (np.array): array of floats
            arr2 (np.array): array of floats
            maring (float): margin of error
        
        RETURNS:
            overlap (list): overlaping float values within the error margin
        """
        i, j = 0, 0
        overlap = []
        idx_spectrum_1_intensities = []
        idx_spectrum_2_intensities = []
        for i in range(len(arr1)):
            for j in range(len(arr2)):
                if abs(arr1[i] - arr2[j]) <= margin:
                    overlap.append(arr1[i]) # because arr1 is the motif spectrum
                    idx_spectrum_1_intensities.append(i)
                    idx_spectrum_2_intensities.append(j)
                    break 
                if arr1[i] < arr2[j]:
                    break

        return overlap, idx_spectrum_1_intensities, idx_spectrum_2_intensities
    
    overlaping_fragments, idx_fragments_1_intensities, idx_fragments_2_intensities = find_overlap(spectrum_1_fragments_mz, spectrum_2_fragments_mz, margin_fragments)
    overlaping_losses, idx_losses_1_intensities, idx_losses_2_intensities = find_overlap(spectrum_1_losses_mz, spectrum_2_losses_intensities, margin_losses)

    intensities_spectrum_1 = list(spectrum_1_fragments_intensities[idx_fragments_1_intensities]) + list(spectrum_1_losses_intensities[idx_losses_1_intensities])
    intensities_spectrum_2 = list(spectrum_2_fragments_intensities[idx_fragments_2_intensities]) + list(spectrum_2_losses_intensities[idx_losses_2_intensities])
    if len(intensities_spectrum_1) >= 2:
        pearson_score = calc_pearson(intensities_spectrum_1, intensities_spectrum_2)[0]
    else:
        pearson_score = 0
    
    return overlaping_fragments, overlaping_losses, pearson_score



def calc_overlaping_stats(motif, spectrum, overlaping_fragments, overlaping_losses):
    """calculates the number of overlaping features and their cumulative intensity
    
    ARGS:
        motif: matchms spectrum object
        overlaping_fragments (list): list of floats of mz values for fragments
        overlaping_losses (list): list of float of mz values for losses

    RETURNS:
        n_overlaping_features (int): number of features that did overlap between motif and query spectrum
        sum_overlaping_features_intensities (float): consensus intensity for all features that overlap between motif and query spectrum
    """
    motif_overlaping_fragments_intensities = []
    spectrum_overlaping_fragments_intensities = []
    for overlaping_fragment in overlaping_fragments:
        motif_fragment_index = np.where(motif.peaks.mz == overlaping_fragment)[0][0]
        motif_fragment_intensity = motif.peaks.intensities[motif_fragment_index]
        motif_overlaping_fragments_intensities.append(motif_fragment_intensity)

        #spectrum_fragment_index = np.where(spectrum.peaks.mz == overlaping_fragment)[0][0]
        #spectrum_fragment_intensity = spectrum.peaks.intensities[spectrum_fragment_index]
        #spectrum_overlaping_fragments_intensities.append(spectrum_fragment_intensity)

    motif_overlaping_losses_intensities = []
    spectrum_overlaping_losses_intensities = []
    for overlaping_loss in overlaping_losses:
        motif_loss_index = np.where(motif.losses.mz == overlaping_loss)[0][0]
        motif_loss_intensity = motif.losses.intensities[motif_loss_index]
        motif_overlaping_losses_intensities.append(motif_loss_intensity)

        #spectrum_loss_index = np.where(spectrum.losses.mz == overlaping_loss)[0][0]
        #spectrum_loss_intensity = spectrum.losses.intensities[spectrum_loss_index]
        #spectrum_overlaping_losses_intensities.append(spectrum_loss_intensity)


    motif_overlaping_features_intensities = motif_overlaping_fragments_intensities + motif_overlaping_losses_intensities
    motif_sum_overlaping_features_intensities= np.sum(motif_overlaping_features_intensities)
    n_overlaping_features = len(motif_overlaping_features_intensities)

    return n_overlaping_features, motif_sum_overlaping_features_intensities



def screen_score(n_overlaping_features, sum_overlaping_features_intensities, pearson_score):
    """based on degree of query spectra-motif spectra overlapp the spectrum gets assigned to a certain level of similarity
    
    ARGS:
        n_overlaping_features (int): number of features that did overlap between motif and query spectrum
        sum_overlaping_features_intensities (float): consensus intensity for all features that overlap between motif and query spectrum
        
    RETURNS:
        (character): A,B,C or D if it fits one of the categories else None
    """

    if n_overlaping_features >= 3 and sum_overlaping_features_intensities >= 0.9 and pearson_score > 0.8:
        return "A"
    elif n_overlaping_features == 1 and sum_overlaping_features_intensities >= 0.9:
        return "B"
    elif n_overlaping_features >= 2 and sum_overlaping_features_intensities >= 0.7:
        return "C"
    elif n_overlaping_features >= 2 and sum_overlaping_features_intensities >= 0.5:
        return "D"
    else:
        return None



def run_screen(motif, spectra):
    """runs the screening for a given set of spectra against a motif spectrum
    
    ARGS:
        motif: matchms spectrum object
        spectra (list): list of matchms spectrum objects
        
    RETURNS:
        A,B,C,D (list): list of matchms spectra objects; A are the best matches D the worst matches
    """

    A,B,C,D = [],[],[],[]
    for spectrum in spectra:
        overlaping_fragments, overlaping_losses, pearson_score = match_spectral_overlap(motif, spectrum)
        n_overlaping_features, sum_overlaping_features_intensities = calc_overlaping_stats(motif, spectrum, overlaping_fragments, overlaping_losses)
        match_category = screen_score(n_overlaping_features, sum_overlaping_features_intensities, pearson_score)
        if match_category == "A":
            A.append(spectrum)
        elif match_category == "B":
            B.append(spectrum)
        elif match_category == "C":
            C.append(spectrum)
        elif match_category == "D":
            D.append(spectrum)

    return A,B,C,D



def save_as_csv(spectra, smiles):
    """saves information about the matching spectra in comparison to the annotated compound/spectrum"""
    precursor_mzs = []
    retention_times = []
    mass_diffs = []
    formula_diffs = []

    engine = Msbuddy()

    for spectrum in spectra:
        precursor_mz = spectrum.get("precursor_mz")
        retention_time = spectrum.get("retention_time")
        

        if smiles:
            precursor_mz_motif_mol = ExactMolWt(Chem.MolFromSmiles(smiles))
            mass_diff = abs(precursor_mz - 1.007276470 - precursor_mz_motif_mol) 

            formula_list = engine.mass_to_formula(mass_diff, 0.01, False)
            if formula_list:
                formula_diff = formula_list[0].formula
            else:
                formula_diff = "NaN"

            mass_diffs.append(mass_diff)
            formula_diffs.append(formula_diff)

        precursor_mzs.append(precursor_mz)
        retention_times.append(retention_time)

    results = pd.DataFrame({
        "precursor_mz": precursor_mzs,
        "mass_difference": mass_diffs,
        "formula_diff": formula_diffs,
        "retention_time": retention_times,
        # ionmode
    })

    results.to_csv("results.csv")

    return results


if __name__ == "__main__":
    run_screen() # not working

        
