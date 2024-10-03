import hashlib
import numpy as np
import pandas as pd

from matchms import Spectrum, Fragments

def motifs2motifDB(spectra):
    """converts a set of motif spectra into a MassQL dataframe format
    
    ARGS:
        spectra (list): list of matchms spectra objects
        
    RETURNS:
        ms1_df (pd.dataframe): dataframe with ms1 information (not used, but important for massql algo)
        ms2_df (pd.dataframe): dataframe with ms2 (frag and loss) information
    """

    def add_default_info(feature_dict, spectrum):
        """adds information about the motif like charge and its annotation as well as a hashed motif_id
        
        ARGS:
            feature_dict (dict): dictionary with fragment or loss properties
            spectrum: matchms spectrum object for motif
            
        RETURNS:
            feature_dict (dict): modified dictionary with add fragment and loss properties
        """
        
        feature_dict["charge"] = spectrum.get("charge")
        feature_dict["ms2accuracy"] = spectrum.get("ms2accuracy")
        feature_dict["short_annotation"] = spectrum.get("short_annotation") # maybe long annotation would be still useful to have
        feature_dict["annotation"] = spectrum.get("annotation")
        feature_dict["motif_id"] = spectrum.get("id")
        feature_dict["motifset"] = spectrum.get("motifset")
        string_to_hash = spectrum.get("id") + spectrum.get("motifset")
        byte_string = string_to_hash.encode("utf-8")
        hash_id = hashlib.md5(byte_string, usedforsecurity=False).hexdigest()
        feature_dict["scan"] = hash_id
        feature_dict["ms1scan"] = 0

        return feature_dict

    
    ms2mz_list = []
    for spectrum in spectra:

        if spectrum.peaks:    
                
            fragments_mz = list(spectrum.peaks.mz)
            fragments_intensities = list(spectrum.peaks.intensities)
    
            for i in range(len(fragments_mz)):
    
                feature_dict = {}

                feature_dict["frag_mz"] = fragments_mz[i] 
                feature_dict["frag_intens"] = fragments_intensities[i]
                
                feature_dict["loss_mz"] = np.nan
                feature_dict["loss_intens"] = np.nan

                feature_dict = add_default_info(feature_dict, spectrum)
    
                ms2mz_list.append(feature_dict)

        
        if spectrum.losses:
    
            losses_mz = list(spectrum.losses.mz)
            losses_intensities = list(spectrum.losses.intensities)

            for i in range(len(losses_mz)):

                feature_dict = {}

                feature_dict["frag_mz"] = np.nan
                feature_dict["frag_intens"] = np.nan
             
                feature_dict["loss_mz"] = losses_mz[i]
                feature_dict["loss_intens"] = losses_intensities[i] 

                feature_dict = add_default_info(feature_dict, spectrum)
    
                ms2mz_list.append(feature_dict)

    
    #ms1_df = pd.DataFrame([feature_dict])
    ms2_df = pd.DataFrame(ms2mz_list)

    return ms2_df

def motifDB2motifs(motifDB_ms2, result_feature_table):
    """converts a (filtered) MotifDB to motif spectra objects
    
    ARGS:
        motifDB_ms2 (pd.dataframe): MassQL dataframe format for MS2 data
        result_feature_table (pd.dataframe): MassQL dataframe format for query results
        
    RETURNS (list): list of matchms spectra objects
    """

    motif_ids = result_feature_table["scan"].to_list()
    filtered_motifDB_ms2 = motifDB_ms2[motifDB_ms2["scan"].isin(motif_ids)]
    grouped_motifDB_results = filtered_motifDB_ms2.groupby("scan")

    motif_spectra = []
    for motif_id, group in grouped_motifDB_results:
        fragments_mz = group["frag_mz"].dropna().to_numpy()
        fragments_intensities = group["frag_intens"].dropna().to_numpy()
        losses_mz = group["loss_mz"].dropna().to_numpy()
        losses_intensities = group["loss_intens"].dropna().to_numpy()

        name = group["motif_id"].drop_duplicates().to_list()[0]
        charge = group["charge"].drop_duplicates().to_list()[0]
        short_annotation = group["short_annotation"].drop_duplicates().to_list()[0]
        annotation = group["annotation"].drop_duplicates().to_list()[0]
        ms2accuracy = group["ms2accuracy"].drop_duplicates().to_list()[0]
        motifset = group["motifset"].drop_duplicates().to_list()[0]

        motif_spectrum = Spectrum(
            mz=fragments_mz,
            intensities=fragments_intensities,
            metadata = {
                "id": name,
                "charge": charge,
                "short_annotation": short_annotation,
                "annotation": annotation,
                "ms2accuracy": ms2accuracy,
                "motifset": motifset,
                "motif_id": motif_id
            }
        )

        motif_spectrum.losses = Fragments(
            mz=losses_mz,
            intensities=losses_intensities
        )

        motif_spectra.append(motif_spectrum)

    return motif_spectra





