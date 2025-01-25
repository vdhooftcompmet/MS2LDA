from matchms.importing import load_from_mgf
from matchms.importing import load_from_mzml
from matchms.importing import load_from_msp


import matchms.filtering as msfilters

import numpy as np


def load_mgf(spectra_path):
    """loads spectra from a mgf file

    ARGS:
        spectra_path (str): path to the spectra.mgf file

    RETURNS:
        spectra (generator): matchms generator object with the loaded spectra
    """
    
    spectra = load_from_mgf(spectra_path)
    
    return spectra

def load_mzml(spectra_path):
    """loads spectra from a mzml file

    ARGS:
        spectra_path (str): path to the spectra.mgf file

    RETURNS:
        spectra (generator): matchms generator object with the loaded spectra
    """
    
    spectra = load_from_mzml(spectra_path)
    
    return spectra

def load_msp(spectra_path):
    """loads spectra from a mzml file

    ARGS:
        spectra_path (str): path to the spectra.mgf file

    RETURNS:
        spectra (generator): matchms generator object with the loaded spectra
    """
    
    spectra = load_from_msp(spectra_path)
    
    return spectra


def load_mzml(spectra_path):
    """loads spectra from a mzml file

    ARGS:
        spectra_path (str): path to the spectra.mgf file

    RETURNS:
        spectra (generator): matchms generator object with the loaded spectra
    """
    
    spectra = load_from_mzml(spectra_path)
    
    return spectra


def clean_spectra(spectra, preprocessing_parameters={}):
    """uses matchms to normalize intensities, add information and add losses to the spectra
    
    ARGS:
        spectra (generator): generator object of matchms.Spectrum.objects loaded via matchms in python
        entropy_threshold (float): spectral entropy threshold to sort out noisy spectra (see MoNA spectral entropy)
    
    RETURNS:
        cleaned_spectra (list): list of matchms.Spectrum.objects; spectra that do not fit will be removed
    """
    ensure_key = lambda parameters, key, default: parameters.setdefault(key, default)

    ensure_key(preprocessing_parameters, "min_mz", 0)
    ensure_key(preprocessing_parameters, "max_mz", 1000)
    ensure_key(preprocessing_parameters, "max_frags",  500)
    ensure_key(preprocessing_parameters, "min_frags", 3)
    ensure_key(preprocessing_parameters, "min_intensity", 0.001)
    ensure_key(preprocessing_parameters, "max_intensity", 1)
    
    
    cleaned_spectra = []
    count = 0

    for i, spectrum in enumerate(spectra):
        # metadata filters
        spectrum = msfilters.default_filters(spectrum)
        spectrum = msfilters.add_retention_index(spectrum)
        spectrum = msfilters.add_retention_time(spectrum)
        #spectrum = msfilters.require_precursor_mz(spectrum) # do we need this

        # normalize and filter peaks
        spectrum = msfilters.normalize_intensities(spectrum)
        spectrum = msfilters.select_by_relative_intensity(spectrum, intensity_from=preprocessing_parameters["min_intensity"], intensity_to=preprocessing_parameters["max_intensity"])
        spectrum = msfilters.select_by_mz(spectrum, mz_from=preprocessing_parameters["min_mz"], mz_to=preprocessing_parameters["max_mz"])
        spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=preprocessing_parameters["max_frags"])
        spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=preprocessing_parameters["min_frags"])
        spectrum = msfilters.add_losses(spectrum)


        if spectrum:
            spectrum.set("id", f"spec_{count}")  # reindex
            cleaned_spectra.append(spectrum)
            count += 1

    return cleaned_spectra


if __name__ == "__main__":
    spectra_path = "../test_data/pos_ache_inhibitors_pesticides.mgf"
    spectra = load_mgf(spectra_path)
    cleaned_spectra = clean_spectra(spectra)
    print(cleaned_spectra)