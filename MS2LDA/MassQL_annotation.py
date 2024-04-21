from massql import msql_engine
from massql import msql_fileloading
from matchms.importing import load_from_mgf

import pickle
# currently I don't know how to add SMILES, precursor mass and retention time
# as well as to run each query by not loading the file again all the time

def add_column(df, spectra, key):
    """add metadata information to filtered MassQL query dataframe
    
    ARGS:
        df (dataframe): resulting dataframe based on MassQL query
        spectra (matchms.spectrum.object): matchms spectra that should be queried
        key (str): metadata key as written in the .mgf file

    RETURNS:
        matched_key (list): list of metadata info based on given key and found scan numbers
        """
    
    matched_scans = df["scan"].to_list()
    matched_key = [spectrum.get(key) for spectrum in spectra if int(spectrum.get("scans")) in matched_scans]
    
    return matched_key

def create_scans2smiles_table(filename):
    """does something"""

    spectra = list(load_from_mgf(filename))

    scans2smiles_table = dict()
    for spectrum in spectra:
        scans = spectrum.get("scans")
        smiles = spectrum.get("smiles")
        scans2smiles_table[scans] = smiles

    with open("Scan2SMILES_Table.pickle", "wb") as outfile:
        pickle.dump(scans2smiles_table, outfile)

    return "Done"


def correct_mgf_scannumber(filename):
    """correct the scan number
    """

    with open(filename, "r") as mgf_file, open(filename[:-4] + "_corrected_scans.mgf", "w") as new_mgf_file:
        mgf_file = mgf_file.readlines()
        counter = 0
        for line in mgf_file:
            if line.startswith("SCANS="):
                counter += 1
                new_mgf_file.write(f"SCANS={counter}\n")
            else:
                new_mgf_file.write(line)
                
    return "Done"



def initialize_massql_DB(filename):
    """creates a massql dataframe based on a given input file

    ARGS:
        filename (str): path to .mgf file

    RETURNS:
        ms1_df (dataframe): What is it?
        ms2_df (dataframe): What is it?

    """
    ms1_df, ms2_df = msql_fileloading.load_data(filename)

    return ms1_df, ms2_df


def search_massql_DB(filename, input_query, ms1_df, ms2_df):
    """The idea is to query all spectra-motif overlaping peaks. based on this information the smiles can be compared and this time a substructure retrieval (mcs) methods can be used"""
    
    results_df = msql_engine.process_query(input_query, filename, ms1_df=ms1_df, ms2_df=ms2_df)
    # error if results_df empty
    with open("Scan2SMILES_Table.pickle", "rb") as infile:
        scans2smiles_table = pickle.load(infile)

    
    matched_scans = results_df.scan.to_numpy()

    matched_smiles = []
    for scan in matched_scans:
        smiles = scans2smiles_table[str(scan)]
        matched_smiles.append(smiles)

    # This could be the normal one, and then there is a function fast with the dictionary
    #spectra = list(load_from_mgf(filename))

    #results_df["smiles"] = add_column(results_df, spectra, "smiles")
    #results_df["precmz"] = add_column(results_df, spectra, "precursor_mz")
    #results_df["rt"] = add_column(results_df, spectra, "retention_time")

    return results_df, matched_smiles


if __name__ == "__main__":
    filename = r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\MEDUSA\notebooks\MS2LDA\PDE5_standards_annotated_pos_unique.mgf"
    input_query = "QUERY scaninfo(MS2DATA) WHERE MS2PROD=99 AND MS2PROD=169"
    ms1_df, ms2_df = initialize_massql_DB(filename)
    
    results_df = search_massql_DB(filename, input_query, ms1_df, ms2_df)
    print(results_df)
