from ms2query.ms2library import create_library_object_from_one_dir
from ms2query.utils import SettingsRunMS2Query



def ms2query_annotation(processed_spectra, ms2query_library_files_directory, top_n=1):
    """ Convert processed spectra to SMILES using ms2query library.
    Args:
    - processed_spectra: list of processed spectra (matchms.Spectrum objects)
    - ms2query_library_files_directory: directory containing ms2query library files (positive or negative mode)

    Returns:
    - list of SMILES strings

    """
    ms2query_settings = SettingsRunMS2Query(nr_of_top_analogs_to_save=top_n)
    ms2query_library = create_library_object_from_one_dir(ms2query_library_files_directory)
    analogs = ms2query_library.analog_search_yield_df(processed_spectra, progress_bar=True, settings=ms2query_settings)

    analogs_smiles = []
    tanimoto_scores = []
    for analog in analogs:
        analogs_smiles.append(analog.smiles[:]) 
        tanimoto_scores.append(analog.ms2query_model_prediction[:])
        
    return analogs_smiles, tanimoto_scores



# you have to convert a list of str(frags to a spectrum and what about the intensity??? maybe the peaks importance for the motif?)

if __name__ == "__main__":
    from matchms.importing import load_from_mgf
    from matchms.filtering import normalize_intensities

    # Load example spectra
    dummy_spectra_path = r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\Spec2Struc\Project_SimilaritySearch\raw_data\_RAWdata1\dummy_spectra.mgf"
    dummy_spectra_file = load_from_mgf(dummy_spectra_path)

    spectra = []
    for spectrum in dummy_spectra_file:
        #spectrum = default_filters(spectrum)
        spectrum = normalize_intensities(spectrum)
        spectra.append(spectrum)

    
    # Convert spectra to SMILES (LIKE IN NORMAL SCRIPT)
    ms2query_library_files_directory = r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\Spec2Struc\Project_SimilaritySearch\scripts\ms2query_library_files\positive_mode"
    smiles, scores = ms2query_annotation(spectra, ms2query_library_files_directory, top_n=3)
    print(smiles)
    