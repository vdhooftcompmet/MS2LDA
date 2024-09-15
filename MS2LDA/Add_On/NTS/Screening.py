from scipy.stats import pearsonr as calc_pearson
import numpy as np
import pandas as pd

from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem

from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import MACCSkeys

from msbuddy import Msbuddy

#--- structure annotation
import requests

import pubchempy as pcp
from pubchempy import BadRequestError, PubChemHTTPError
from rdkit.Chem.inchi import MolFromInchi
from rdkit.Chem import MolFromSmiles

import concurrent.futures

import json
import time
import sqlite3


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
        pearson_score (float): pearson score for spectra with more than 1 overlaping feature
    """


    def find_overlap(arr1, arr2, margin):
        """finds the overlap of two arrays
        
        ARGS:
            arr1 (np.array): array of floats
            arr2 (np.array): array of floats
            maring (float): margin of error
        
        RETURNS:
            overlap (list): overlaping float values within the error margin
            idx_spectrum_1_intensities (list): indices of overlaping features in spectrum 1
            idx_spectrum_2_intensities (list): indices of overlaping features in spectrum 2
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
    
    # spectrum 1 fragments and losses
    spectrum_1_fragments_mz = spectrum_1.peaks.mz
    spectrum_1_fragments_intensities = spectrum_1.peaks.intensities
    spectrum_1_losses_mz = spectrum_1.losses.mz
    spectrum_1_losses_intensities = spectrum_1.losses.intensities

    # spectrum 2 fragments and losses
    spectrum_2_fragments_mz = spectrum_2.peaks.mz
    spectrum_2_fragments_intensities = spectrum_2.peaks.intensities
    #spectrum_2_losses_mz = spectrum_2.losses.mz
    #spectrum_2_losses_intensities = spectrum_2.losses.intensities # what if there are no losses !!!!
    spectrum_2_losses_mz = np.array([0])
    spectrum_2_losses_intensities = np.array([0])
    
    # find overlaping features
    overlaping_fragments, idx_fragments_1_intensities, idx_fragments_2_intensities = find_overlap(spectrum_1_fragments_mz, spectrum_2_fragments_mz, margin_fragments)
    overlaping_losses, idx_losses_1_intensities, idx_losses_2_intensities = find_overlap(spectrum_1_losses_mz, spectrum_2_losses_mz, margin_losses)

    # pearson score for intensity trend
    intensities_spectrum_1 = list(spectrum_1_fragments_intensities[idx_fragments_1_intensities]) + list(spectrum_1_losses_intensities[idx_losses_1_intensities])
    intensities_spectrum_2 = list(spectrum_2_fragments_intensities[idx_fragments_2_intensities]) + list(spectrum_2_losses_intensities[idx_losses_2_intensities])

    if len(intensities_spectrum_1) >= 2:
        pearson_score = calc_pearson(intensities_spectrum_1, intensities_spectrum_2)[0]
    else:
        pearson_score = 0
    
    return overlaping_fragments, overlaping_losses, pearson_score



def calc_overlaping_stats(motif, overlaping_fragments, overlaping_losses):
    """calculates the number of overlaping features and their cumulative intensity
    
    ARGS:
        motif: matchms spectrum object
        overlaping_fragments (list): list of floats of mz values for fragments
        overlaping_losses (list): list of float of mz values for losses

    RETURNS:
        n_overlaping_features (int): number of features that did overlap between motif and query spectrum
        sum_overlaping_features_intensities (float): consensus intensity for all features that overlap between motif and query spectrum
    """

    # motif intensities of overlaping fragments
    motif_overlaping_fragments_intensities = []
    for overlaping_fragment in overlaping_fragments:
        motif_fragment_index = np.where(motif.peaks.mz == overlaping_fragment)[0][0]
        motif_fragment_intensity = motif.peaks.intensities[motif_fragment_index]
        motif_overlaping_fragments_intensities.append(motif_fragment_intensity)

    # motif intensities of overlaping losses
    motif_overlaping_losses_intensities = []
    for overlaping_loss in overlaping_losses:
        motif_loss_index = np.where(motif.losses.mz == overlaping_loss)[0][0]
        motif_loss_intensity = motif.losses.intensities[motif_loss_index]
        motif_overlaping_losses_intensities.append(motif_loss_intensity)

    # number of overlaping features, and cumulative intensity of overlaping features
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
        (character): A, B, C or D if it fits one of the categories else None
    """

    if n_overlaping_features >= 3 and sum_overlaping_features_intensities >= 0.9 and pearson_score >= 0.7:
        return "A"
    elif n_overlaping_features >= 3 and sum_overlaping_features_intensities >= 0.9 and pearson_score >= 0.5:
        return "B"
    elif n_overlaping_features >= 2 and sum_overlaping_features_intensities >= 0.5 and pearson_score >= 0.3:
        return "C"
    elif n_overlaping_features >= 2 and sum_overlaping_features_intensities >= 0.8:
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
        n_overlaping_features, sum_overlaping_features_intensities = calc_overlaping_stats(motif, overlaping_fragments, overlaping_losses)
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

    results = pd.DataFrame({
        "id": list(range(len(spectra)))
    })
    
    engine = Msbuddy()

    for i, smi in enumerate(smiles):

        precursor_mzs = []
        retention_times = []
        mass_diffs = []
        formula_diffs = []

        for spectrum in spectra:
            precursor_mz = spectrum.get("precursor_mz")
            retention_time = spectrum.get("retention_time")
        
            if smi:
                precursor_mz_motif_mol = ExactMolWt(Chem.MolFromSmiles(smi))
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

        results[f"mass_diff_{i}"] = mass_diffs
        results[f"formula_diff_{i}"] = formula_diffs

    results["precursor_mz"] = precursor_mzs
    results["retention_time"] = retention_times

    

    results.to_csv("results.csv")

    return results


def mf_finder(mz, session):
    chemcalcURL = 'https://www.chemcalc.org/chemcalc/em'
    options = {'mfRange': 'C0-100H0-202N0-20O0-20',
               'numberOfResultsOnly': False,
               'typedResult': False,
               'useUnsaturation': False,
               'minUnsaturation': 0,
               'maxUnsaturation': 50,
               'jcampBaseURL': 'http://www.chemcalc.org/service/jcamp/',
               'monoisotopicMass': mz-1.007276470,
               'jcampLink': True,
               # The 'jcamplink' returns a link to a file containing isotopic
               # distribution of retrieved molecular structure.
               'integerUnsaturation': False,
               # Ions/Radicals can have non-integer unsaturation
               'referenceVersion': '2013',
               'massRange': 0.5
#              'minMass': -0.5,
#              'maxMass': 0.5,
              }
    response = session.get(chemcalcURL, params=options)
    return response.json()

def mf_finder_local(mz):
    chemcalcURL = 'https://www.chemcalc.org/chemcalc/em'
    options = {'mfRange': 'C0-100H0-202N0-20O0-20S0-4',
               'numberOfResultsOnly': False,
               'typedResult': False,
               'useUnsaturation': False,
               'minUnsaturation': 0,
               'maxUnsaturation': 50,
               'jcampBaseURL': 'http://www.chemcalc.org/service/jcamp/',
               'monoisotopicMass': mz-1.007276470,
               'jcampLink': True,
               # The 'jcamplink' returns a link to a file containing isotopic
               # distribution of retrieved molecular structure.
               'integerUnsaturation': False,
               # Ions/Radicals can have non-integer unsaturation
               'referenceVersion': '2013',
               'massRange': 0.5
#              'minMass': -0.5,
#              'maxMass': 0.5,
              }
    
    return requests.get(chemcalcURL, options).json()

def analog_finder_local(spectrum, motif_fp_array, database):

    prec_mz = spectrum.get("precursor_mz")

    mfs = pd.DataFrame(mf_finder_local(prec_mz)["results"])["mf"][:50].to_list() # why 10; make it variable as and argument
    time.sleep(1) # otherwise the json request will fail
    
    analog_candidates = []

    for mf in mfs:
        analog_candidates_smiles = database.loc[database["molecular_formula"] == mf].canonical_smiles.to_list() # coconut inchi keys are worse then smiles ## often errors
        if analog_candidates_smiles:
            motif_fp = array_to_fingerprint(motif_fp_array)
            for candidate_smiles in analog_candidates_smiles:
                candidate_mol = MolFromSmiles(candidate_smiles)
                candidate_fp = MACCSkeys.GenMACCSKeys(candidate_mol)                    

                tanimoto_score = DataStructs.TanimotoSimilarity(motif_fp, candidate_fp)
                analog_candidates.append((tanimoto_score, candidate_smiles))

    if analog_candidates:
        top_tanimoto_score, top_analog_candidates = sorted(analog_candidates, key=lambda tup: tup[0], reverse=True)[0]
        return top_tanimoto_score, top_analog_candidates
    
    else:
        return 0, "O"




def analog_finder(spectrum, motif_fp_array):

    prec_mz = spectrum.get("precursor_mz")

    mfs = pd.DataFrame(mf_finder_local(prec_mz)["results"])["mf"][:50].to_list() # why 10; make it variable as and argument
    time.sleep(0.5) # otherwise the json request will fail

    with requests.Session() as session:
        mfs = pd.DataFrame(mf_finder(prec_mz, session)["results"])["mf"][:10].to_list() # why 10; make it variable as and argument
    
    analog_candidates = []

    def fetch_and_compare(mf):
        try:
            analog_candidates_inchis = pcp.get_compounds(mf, "formula", as_dataframe=True).inchi.to_list()

            if analog_candidates_inchis:
                motif_fp = array_to_fingerprint(motif_fp_array)
                local_candidates = []
                for candidate_inchi in analog_candidates_inchis:
                    candidate_mol = MolFromInchi(candidate_inchi)
                    candidate_fp = MACCSkeys.GenMACCSKeys(candidate_mol)                    

                    tanimoto_score = DataStructs.TanimotoSimilarity(motif_fp, candidate_fp)
                    local_candidates.append((tanimoto_score, candidate_inchi))
                return local_candidates
            else:
                return []
        except BadRequestError:
            return []
        
        except json.JSONDecodeError:
            print(f"JSONDecodeError: Could not decode the response for {mf}.")
            return []
        except PubChemHTTPError:
            print("HTTP Error")
            return []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_and_compare, mfs))

    for result in results:
        if result:
            analog_candidates.extend(result)
        else:
            print("nothing retrieved")

    if analog_candidates:
        top_tanimoto_score, top_analog_candidates = sorted(analog_candidates, key=lambda tup: tup[0], reverse=True)[0]
        return top_tanimoto_score, top_analog_candidates
    
    else:
        return "No Match Found"


def find_analogs(spectrum, motif_fp_array):
    prec_mz = spectrum.get("precursor_mz")

    mfs = pd.DataFrame(mf_finder_local(prec_mz)["results"])["mf"][:50].to_list() # why 10; make it variable as and argument
    time.sleep(0.5) # otherwise the json request will fail

    analog_candidates = []
    conn = sqlite3.connect("molecules.db")
    for mf in mfs:
        analog_candidates_smiles = query_smiles(mf, conn).SMILES.to_list()
        if analog_candidates_smiles:
            motif_fp = array_to_fingerprint(motif_fp_array)
            for candidate_smiles in analog_candidates_smiles:
                candidate_mol = MolFromSmiles(candidate_smiles)
                candidate_fp = MACCSkeys.GenMACCSKeys(candidate_mol)                    

                tanimoto_score = DataStructs.TanimotoSimilarity(motif_fp, candidate_fp)
                analog_candidates.append((tanimoto_score, candidate_smiles))

    if analog_candidates:
        top_tanimoto_score, top_analog_candidates = sorted(analog_candidates, key=lambda tup: tup[0], reverse=True)[0]
        return top_tanimoto_score, top_analog_candidates
    
    else:
        return 0, "O"


def query_smiles(mf, conn):
    query = "SELECT SMILES FROM molecules WHERE MolFormula = ?"
    result = pd.read_sql_query(query, conn, params=(mf, ))
    return result

def array_to_fingerprint(binary_array):
    fp = ExplicitBitVect(len(binary_array))
    for i, bit in enumerate(binary_array):
        if bit:
            fp.SetBit(i)

    return fp


if __name__ == "__main__":
    pass
    #run_screen()

        
