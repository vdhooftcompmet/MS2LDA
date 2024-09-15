from matchms.importing import load_from_mgf
from matchms.importing import load_from_mzml
import matchms.filtering as msfilters

import ms_entropy
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


def clean_spectra(spectra, entropy_threshold=3.):
    """uses matchms to normalize intensities, add information and add losses to the spectra
    
    ARGS:
        spectra (generator): generator object of matchms.Spectrum.objects loaded via matchms in python
        entropy_threshold (float): spectral entropy threshold to sort out noisy spectra (see MoNA spectral entropy)
    
    RETURNS:
        cleaned_spectra (list): list of matchms.Spectrum.objects; spectra that do not fit will be removed
    """
    cleaned_spectra = []

    for spectrum in spectra:
        # metadata filters
        spectrum = msfilters.default_filters(spectrum)
        spectrum = msfilters.add_retention_index(spectrum)
        spectrum = msfilters.add_retention_time(spectrum)
        spectrum = msfilters.require_precursor_mz(spectrum)

        # normalize and filter peaks
        spectrum = msfilters.normalize_intensities(spectrum)
        spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
        spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
        spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
        spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)
        spectrum = msfilters.add_losses(spectrum)


        if spectrum: # why was it without this? This did exclude an important compound!!!
            cleaned_spectra.append(spectrum)
        #    spectral_entropy = ms_entropy.calculate_spectral_entropy(list(zip(spectrum.peaks.mz, spectrum.peaks.intensities)),
        #                                                    clean_spectrum = True,
        #                                                    min_ms2_difference_in_da = 0.05)
        #    if spectral_entropy < entropy_threshold:
        #        cleaned_spectra.append(spectrum)

    return cleaned_spectra


if __name__ == "__main__":
    spectra_path = "../test_data/pos_ache_inhibitors_pesticides.mgf"
    spectra = load_mgf(spectra_path)
    cleaned_spectra = clean_spectra(spectra)
    print(cleaned_spectra)