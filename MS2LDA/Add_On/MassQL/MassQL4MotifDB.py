import hashlib
import random
import numpy as np
import pandas as pd
import json

from MS2LDA.Mass2Motif import Mass2Motif

def motifs2motifDB(spectra):
    """converts a set of motif spectra into a MassQL dataframe format

    ARGS:
        spectra (list): list of matchms spectra objects

    RETURNS:
        ms1_df (pd.dataframe): dataframe with ms1 information (not used, but important for massql algo)
        ms2_df (pd.dataframe): dataframe with ms2 (frag and loss) information
    """

    def add_default_info(feature_dict, spectrum, hash_id):
        """adds information about the motif like charge and its annotation as well as a hashed motif_id

        ARGS:
            feature_dict (dict): dictionary with fragment or loss properties
            spectrum: matchms spectrum object for motif

        RETURNS:
            feature_dict (dict): modified dictionary with add fragment and loss properties
        """

        feature_dict["charge"] = spectrum.get("charge")
        feature_dict["ms2accuracy"] = spectrum.get("ms2accuracy")
        feature_dict["short_annotation"] = spectrum.get("short_annotation")
        feature_dict["annotation"] = spectrum.get("annotation")
        feature_dict["auto_annotation"] = spectrum.get("auto_annotation")
        feature_dict["motif_id"] = spectrum.get("id")
        feature_dict["motifset"] = spectrum.get("motifset")
        feature_dict["analysis_massspectrometer"] = spectrum.get("analysis_massspectrometer")
        feature_dict["collision_energy"] = spectrum.get("collision_energy")
        feature_dict["other_information"] = spectrum.get("other_information")
        feature_dict["scientific_name"] = spectrum.get("scientific_name")
        feature_dict["sample_type"] = spectrum.get("sample_type")
        feature_dict["massive_id"] = spectrum.get("massive_id")
        feature_dict["taxon_id"] = spectrum.get("taxon_id")
        feature_dict["analysis_ionizationsource"] = spectrum.get("analysis_ionizationsource")
        feature_dict["analysis_chromatographyandphase"] = spectrum.get("analysis_chromatographyandphase")
        feature_dict["analysis_polarity"] = spectrum.get("analysis_polarity")
        feature_dict["paper_url"] = spectrum.get("paper_url")
        feature_dict["property"] = spectrum.get("property")


        feature_dict["scan"] = hash_id
        feature_dict["ms1scan"] = 0

        return feature_dict  # here jsonschema would be nice

    # jsonschema for submitting to MotifDB
    # add more columns: instrument, dda or dia, author, publication?
    # how to push to a public motifDB, pull request on github?

    ms2mz_list = []
    for spectrum in spectra:
        hash_id = random.getrandbits(128)
        if spectrum.peaks:

            fragments_mz = list(spectrum.peaks.mz)
            fragments_intensities = list(spectrum.peaks.intensities)

            for i in range(len(fragments_mz)):
                feature_dict = {}

                feature_dict["frag_mz"] = fragments_mz[i]
                feature_dict["frag_intens"] = fragments_intensities[i]

                feature_dict["loss_mz"] = np.nan
                feature_dict["loss_intens"] = np.nan

                feature_dict = add_default_info(feature_dict, spectrum, hash_id)

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

                feature_dict = add_default_info(feature_dict, spectrum, hash_id)

                ms2mz_list.append(feature_dict)

    ms1_df = pd.DataFrame([feature_dict])
    ms2_df = pd.DataFrame(ms2mz_list)

    return ms1_df, ms2_df


def motifDB2motifs(motifDB_ms2, filter_table=None):
    """converts a (filtered) MotifDB to motif spectra objects

    ARGS:
        motifDB_ms2 (pd.dataframe): MassQL dataframe format for MS2 data
	filter: output from massql query or None to convert all

    RETURNS (list): list of matchms spectra objects
    """

    if filter_table:
        filtered_motifs = filter_table["scan"].to_list()
        filtered_motifDB = motifDB_ms2[motifDB_ms2["scan"].isin(filtered_motifs)]

    else:
        filtered_motifDB = motifDB_ms2

    ms2_df_grouped = group_ms2(ms2_df=filtered_motifDB)

    motif_spectra = []
    for motif in ms2_df_grouped.itertuples():
        fragments_mz = np.array(motif.frag_mz)
        fragments_mz = fragments_mz[~np.isnan(fragments_mz)]
        fragments_intensities = np.array(motif.frag_intens)
        fragments_intensities = fragments_intensities[~np.isnan(fragments_intensities)]

        losses_mz = np.array(motif.loss_mz)
        losses_mz = losses_mz[~np.isnan(losses_mz)]
        losses_intensities = np.array(motif.loss_intens)
        losses_intensities = losses_intensities[~np.isnan(losses_intensities)]

        name = motif.motif_id
        charge = motif.charge
        short_annotation = motif.short_annotation
        annotation = motif.annotation
        ms2accuracy = motif.ms2accuracy
        motifset = motif.motifset
        motif_id = motif.motif_id
        analysis_massspectrometer = motif.analysis_massspectrometer
        collision_energy = motif.collision_energy
        other_information = motif.other_information
        scientific_name = motif.scientific_name
        sample_type = motif.sample_type
        massive_id = motif.massive_id
        taxon_id = motif.taxon_id
        analysis_ionizationsource = motif.analysis_ionizationsource
        analysis_chromatographyandphase = motif.analysis_chromatographyandphase
        analysis_polarity = motif.analysis_polarity
        paper_url = motif.paper_url
        auto_annotation = motif.auto_annotation
        property = motif.property

        motif_spectrum = Mass2Motif(
            frag_mz=fragments_mz,
            frag_intensities=fragments_intensities,
            loss_mz=losses_mz,
            loss_intensities=losses_intensities,
            metadata = {
                "id": name,
                "charge": charge,
                "short_annotation": short_annotation,
                "annotation": annotation,
                "ms2accuracy": ms2accuracy,
                "motifset": motifset,
                "motif_id": motif_id,
                "analysis_massspectrometer": analysis_massspectrometer,
                "collision_energy": collision_energy,
                "other_information": other_information,
                "scientific_name": scientific_name,
                "sample_type": sample_type,
                "massive_id": massive_id,
                "taxon_id": taxon_id,
                "analysis_ionizationsource": analysis_ionizationsource,
                "analysis_chromatographyandphase": analysis_chromatographyandphase,
                "analysis_polarity": analysis_polarity,
                "paper_url": paper_url,
                "auto_annotation": auto_annotation,
                "property": property,
            }
        )

        motif_spectra.append(motif_spectrum)

    return motif_spectra


def group_ms2(ms2_df):
    ms2_df_grouped = ms2_df.groupby("scan").agg(
        {
            "frag_mz": list,
            "frag_intens": list,
            "loss_mz": list,
            "loss_intens": list,
            "charge": "first",
            "ms2accuracy": "first",
            "short_annotation": "first",
            "annotation": "first",
            "motif_id": "first",
            "motifset": "first",
            "ms1scan": "first",
            "analysis_massspectrometer": "first",
            "collision_energy": "first",
            "other_information": "first",
            "scientific_name": "first",
            "sample_type": "first",
            "massive_id": "first",
            "taxon_id": "first",
            "analysis_ionizationsource": "first",
            "analysis_chromatographyandphase": "first",
            "analysis_polarity": "first",
            "paper_url": "first",
            "auto_annotation": "first",
            "property": "first",
        }
    ).reset_index()

    return ms2_df_grouped


def store_motifDB(ms1_df, ms2_df, name="motifDB.json"):
    # ms1_df["ms_level"] = "ms1"
    # ms2_df["ms_level"] = "ms2"
    ms2_df_grouped = group_ms2(ms2_df)

    motifDB = {
        "ms1": ms1_df.to_dict(orient="records"),
        "ms2": ms2_df_grouped.to_dict(orient="records"),
    }

    with open(name, "w") as outfile:
        json.dump(motifDB, outfile)

    return True


def load_motifDB(motifDB_filename):
    with open(motifDB_filename, "r") as infile:
        motifDB = json.load(infile)

    ms1_df = pd.DataFrame(motifDB["ms1"])
    ms2_df = pd.DataFrame(motifDB["ms2"])

    ms2_df_expanded = ms2_df.explode(["frag_mz", "frag_intens", "loss_mz", "loss_intens"]).reset_index(drop=True)
    ms2_df_expanded["frag_mz"] = ms2_df_expanded["frag_mz"].astype(float)
    ms2_df_expanded["frag_intens"] = ms2_df_expanded["frag_intens"].astype(float)
    ms2_df_expanded["loss_mz"] = ms2_df_expanded["loss_mz"].astype(float)
    ms2_df_expanded["loss_intens"] = ms2_df_expanded["loss_intens"].astype(float)
    ms2_df_expanded["charge"] = ms2_df_expanded["charge"].astype(int)
    ms2_df_expanded["ms2accuracy"] = ms2_df_expanded["ms2accuracy"].astype(float)

    return ms1_df, ms2_df_expanded


def store_motifDB_excel(ms1_df, ms2_df, name="motifDB.xlsx"):
    with pd.ExcelWriter(name) as writer:
        ms1_df.to_excel(writer, sheet_name="ms1")
        ms2_df.to_excel(writer, sheet_name="ms2")

    return True


def load_motifDB_excel(motifDB_filename):
    ms1_df = pd.read_excel(motifDB_filename, sheet_name='ms1')
    ms2_df = pd.read_excel(motifDB_filename, sheet_name='ms2')

    return ms1_df, ms2_df