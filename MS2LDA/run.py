from MS2LDA.Preprocessing.load_and_clean import load_mgf
from MS2LDA.Preprocessing.load_and_clean import load_mzml
from MS2LDA.Preprocessing.load_and_clean import load_msp
from MS2LDA.Preprocessing.load_and_clean import clean_spectra

from MS2LDA.Preprocessing.generate_corpus import features_to_words

from MS2LDA.modeling import define_model
from MS2LDA.modeling import train_model
from MS2LDA.modeling import extract_motifs
from MS2LDA.modeling import create_motif_spectra

from MS2LDA.Add_On.Spec2Vec.annotation import load_s2v_and_library
from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity
from MS2LDA.Add_On.Spec2Vec.annotation import get_library_matches

from MS2LDA.Add_On.Spec2Vec.annotation_refined import hit_clustering
from MS2LDA.Add_On.Spec2Vec.annotation_refined import motif_optimization

from MS2LDA.Add_On.Fingerprints.FP_annotation import annotate_motifs as calc_fingerprints
from MS2LDA.Visualisation.visualisation import create_interactive_motif_network
from MS2LDA.Visualisation.visualisation import create_network
from MS2LDA.Visualisation.visualisation import plot_convergence
from MS2LDA.Visualisation.visualisation import show_annotated_motifs

from MS2LDA.motif_parser import store_m2m_folder
# MotifDB
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import motifDB2motifs
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import load_motifDB
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import load_motifDB_excel

from MS2LDA.Add_On.MassQL.MassQL4MotifDB import motifs2motifDB
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import store_motifDB
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import store_motifDB_excel

from massql import msql_engine

from MS2LDA.Add_On.Fingerprints.FP_annotation import tanimoto_similarity

import matchms
import matplotlib.pyplot as plt
import os
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np

from MS2LDA.Visualisation.ldadict import save_visualization_data


#--------------------------------------------------------main functions------------------------------------------------------------#
def run(dataset, n_motifs, n_iterations,
        dataset_parameters,
        train_parameters,
        model_parameters,
        convergence_parameters,
        annotation_parameters,
        preprocessing_parameters,
        motif_parameter,
        fingerprint_parameters):
    """main function to run MS2LDA workflow in a jupyter notebook"""

    loaded_spectra = filetype_check(dataset=dataset)
    cleaned_spectra = clean_spectra(loaded_spectra, preprocessing_parameters)
    print("Cleaning spectra ...", len(cleaned_spectra), "spectra left")
    feature_words = features_to_words(spectra=cleaned_spectra, significant_figures=2, acquisition_type=dataset_parameters["acquisition_type"])

    # Modeling
    ms2lda = define_model(n_motifs=n_motifs, model_parameters=model_parameters)
    trained_ms2lda, convergence_curve = train_model(ms2lda, feature_words, iterations=n_iterations, train_parameters=train_parameters, convergence_parameters=convergence_parameters)

    # Motif Generation
    motifs = extract_motifs(trained_ms2lda, top_n=motif_parameter)
    motif_spectra = create_motif_spectra(motifs, charge=dataset_parameters["charge"], motifset_name=dataset_parameters["name"]) # output name

    # Motif Annotation and Optimization
    library_matches, s2v_similarity = s2v_annotation(motif_spectra, annotation_parameters)
    clustered_spectra, clustered_smiles, clustered_scores = hit_clustering(s2v_similarity=s2v_similarity, motif_spectra=motif_spectra, library_matches=library_matches, criterium=annotation_parameters["criterium"], cosine_similarity=annotation_parameters["cosine_similarity"])
    motif_spectra = add_annotation(motif_spectra, clustered_smiles)
    optimized_motifs = motif_optimization(motif_spectra, clustered_spectra, clustered_smiles, loss_err=1)
    motif_fps = calc_fingerprints(clustered_smiles, fp_type=fingerprint_parameters["fp_type"], threshold=fingerprint_parameters["threshold"])


    store_results(trained_ms2lda, motif_spectra, optimized_motifs, convergence_curve, clustered_smiles, dataset_parameters["output_folder"])

    # Save additional viz data
    save_visualization_data(
        trained_ms2lda,
        cleaned_spectra,
        optimized_motifs,
        output_folder=dataset_parameters["output_folder"],
        filename="ms2lda_viz.json"
    )

    return motif_spectra, optimized_motifs, motif_fps



def screen_spectra(motifs_stored=None, dataset=None, motif_spectra=None, motifDB=None, motifDB_query=None, s2v_similarity=None, output_folder="MS2LDA_Results", threshold=0.5):

    if not s2v_similarity:
        print("Loading Spec2Vec model ...")
        s2v_similarity, _ = load_s2v()

    if motifDB or motifDB_query:
        if motifDB and motifDB_query:
            if type(motifDB) == str:
                if motifDB.endswith(".xlsx"):
                    ms1_motifDB, ms2_motifDB = load_motifDB_excel(motifDB)
                    motifs_stored = massql_search(ms1_motifDB, ms2_motifDB, motifDB_query)
                else:
                    raise ValueError("This file format is not supported")
            elif type(motifDB) == tuple and len(motifDB) == 2:
                ms1_motifDB = motifDB[0]
                ms2_motifDB = motifDB[1]
                motifs_stored = massql_search(ms1_motifDB, ms2_motifDB, motifDB_query)
            else:
                raise ValueError("This file format is not supported")

        else:
            raise ValueError("A MotifDB dataframe and a query need to be used as an input")

    screening_hits_spectra = []
    screening_hits_motifs = []
    if dataset:
        screening_hits_spectra = spectrum_screening(dataset, motifs_stored, s2v_similarity, threshold=threshold)

    if motif_spectra:
        screening_hits_motifs = motif_screening(motif_spectra, motifs_stored, s2v_similarity, threshold=threshold)

    screening_hits = pd.DataFrame(screening_hits_spectra + screening_hits_motifs)

    curr_dir = os.getcwd()
    os.chdir(output_folder)
    with pd.ExcelWriter("spectra_screening.xlsx") as writer:
        screening_hits.to_excel(writer)

    os.chdir(curr_dir)

    return screening_hits


def screen_structure(motif_fps, motif_spectra, structure_query, fp_type="rdkit", output_folder="MS2LDA_Results", threshold=0.7):
    query_fps = []
    for smiles in structure_query:
        query_fp = calc_fingerprints([[smiles]], fp_type=fp_type)
        query_fps.append(query_fp[0])

    if len(motif_fps[0]) != len(query_fps[0]):
        raise ValueError("Not the same fingerprints used", len(motif_fps[0]), "vs", len(query_fps[0]))

    tanimoto_scores = tanimoto_similarity(query_fps, motif_fps)
    matching_motif_idx = np.argwhere(tanimoto_scores >= threshold)
    output = generate_output(matching_motif_idx, motif_spectra, tanimoto_scores, structure_query)

    curr_dir = os.getcwd()
    os.chdir(output_folder)
    with pd.ExcelWriter("structure_screening.xlsx") as writer:
        output = pd.DataFrame(output)
        output.to_excel(writer)
    os.chdir(curr_dir)

    return output


def generate_output(matching_motif_idx, motif_spectra, tanimoto_scores, structure_query):
    results = {
        "Query_Smiles": [],
        "Motif_ID": [],
        "Tanimoto_Score": [],
        "Motif_Smiles": [],
    }

    for query_idx, motif_idx in matching_motif_idx:

        results["Motif_Smiles"].append(motif_spectra[motif_idx].get("short_annotation"))
        results["Motif_ID"].append(motif_spectra[motif_idx].get("id"))
        results["Tanimoto_Score"].append(tanimoto_scores[query_idx, motif_idx])
        results["Query_Smiles"].append(structure_query[query_idx])

    return results


#---------------------------------------------------------------------------------------------#
def massql_search(ms1_motifDB, ms2_motifDB, motifDB_query):
    motifDB_matches = msql_engine.process_query(motifDB_query, ms1_df=ms1_motifDB, ms2_df=ms2_motifDB)
    if not motifDB_matches.empty:
        motif_matches = motifDB2motifs(ms2_motifDB, motifDB_matches)
    else:
        raise ValueError("No matching motif in MotifDB")
    return motif_matches




#--------------------------------------------------------helper functions------------------------------------------------------------#

def spectrum_screening(dataset, motifs_stored, s2v_similarity, threshold=0.5):
    dataset_spectra = filetype_check(dataset=dataset)
    dataset_spectra = clean_spectra(dataset_spectra)
    dataset_embeddings = calc_embeddings(s2v_similarity, dataset_spectra)
    motif_stored_embeddings = calc_embeddings(s2v_similarity, motifs_stored)
    spectrum_motif_similarities = calc_similarity(dataset_embeddings, motif_stored_embeddings)

    screening_hits = []
    for row_index, row in tqdm(spectrum_motif_similarities.iterrows()):
        for col_index, value in enumerate(row):
            if value >= threshold:
                screening_hit = add_metadata(dataset_spectra[col_index], motifs_stored[row_index], value, "spectrum-motif")
                screening_hits.append(screening_hit)

    return screening_hits



def motif_screening(motifs_stored, motif_spectra, s2v_similarity, threshold=0.7):
    motif_stored_embeddings = calc_embeddings(s2v_similarity, motifs_stored)
    motif_embeddings = calc_embeddings(s2v_similarity, motif_spectra)
    motif_motif_similarities = calc_similarity(motif_embeddings, motif_stored_embeddings)

    screening_hits = []
    for row_index, row in tqdm(motif_motif_similarities.iterrows()):
        for col_index, value in enumerate(row):
            if value >= threshold:
                screening_hit = add_metadata(motif_spectra[col_index], motifs_stored[row_index], value, "motif-motif")
                screening_hits.append(screening_hit)

    return screening_hits



def add_metadata(spectrum, motif_spectrum, value, screen_type):
    return {
        "hit_id": spectrum.get("id"),
        "screen_type": screen_type,
        "score": round(value, 2),
        "ref_motifset": motif_spectrum.get("motifset"),
        "ref_motif_id": motif_spectrum.get("id"),
        "ref_short_annotation": motif_spectrum.get("short_annotation"),
        "ref_annotation": motif_spectrum.get("annotation"),
        "ref_charge": motif_spectrum.get("charge"),
        }


#-------------------------------------------------------------------------------------------------------------------------------------------------#

def add_annotation(motif_spectra, clustered_smiles):

    motif_spectra_ = []
    for motif_spectrum, clustered_smi in zip(motif_spectra, clustered_smiles):
        motif_spectrum.set("short_annotation", clustered_smi)
        motif_spectra_.append(motif_spectrum)

    return motif_spectra_
#-------------------------------------------------------------------------------------------------------------------------------------------------#
def store_results(trained_ms2lda, motif_spectra, optimized_motifs, convergence_curve, clustered_smiles, output_folder="MS2LDA_Results"):
    curr_dir = os.getcwd()
    os.mkdir(output_folder)
    os.chdir(output_folder)
    store_m2m_folder(motif_spectra, "motifs")

    convergence_curve_fig = plot_convergence(convergence_curve)
    convergence_curve_fig.savefig("convergence_curve.png",dpi=300)

    network_fig = create_network(motif_spectra, significant_figures=2)
    nx.write_graphml(network_fig, "network.graphml")

    os.mkdir("motif_figures")
    show_annotated_motifs(optimized_motifs, motif_spectra, clustered_smiles, savefig="motif_figures")

    trained_ms2lda.save("ms2lda.bin")

    ms1_motifDB_opt, ms2_motifDB_opt = motifs2motifDB(optimized_motifs) # of motif_spectra?
    store_motifDB_excel(ms1_motifDB_opt, ms2_motifDB_opt, name="motifDB_optimized.xlsx")
    ms1_motifDB, ms2_motifDB = motifs2motifDB(motif_spectra)
    store_motifDB_excel(ms1_motifDB, ms2_motifDB)

    os.chdir(curr_dir)




def filetype_check(dataset):
    if type(dataset) == str:
        dataset_lowercase = dataset.lower()
        if dataset_lowercase.endswith(".mgf"):
            loaded_spectra = load_mgf(dataset)
        elif dataset_lowercase.endswith(".mzml"):
            loaded_spectra = load_mzml(dataset)
        elif dataset_lowercase.endswith(".msp"):
            loaded_spectra = load_msp(dataset)
        else:
            raise TypeError("File format not supported. Only .mgf, .mzml, and .msp")

    elif type(dataset) == list: # and type(dataset[0]) == matchms.Spectrum.Spectrum:
        loaded_spectra = dataset

    else:
        raise ValueError("No valid dataset found!")

    return loaded_spectra



def s2v_annotation(motif_spectra, annotation_parameters):
    print("Loading Spec2Vec model ...")
    s2v_similarity, library = load_s2v()
    print("Searches for candidates ...")
    motif_embeddings = calc_embeddings(s2v_similarity, motif_spectra)
    similarity_matrix = calc_similarity(motif_embeddings, library.embeddings)

    matching_settings = {
        "similarity_matrix": similarity_matrix,
        "library": library,
        "top_n": annotation_parameters["n_mols_retrieved"],
        "unique_mols": True,
    }

    library_matches = get_library_matches(matching_settings)

    return library_matches, s2v_similarity


def load_s2v(
        path_model = "/Users/rosinatorres/Documents/PhD/WP1/Project/Code/MS2LDA/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model",
        path_library = "/Users/rosinatorres/Documents/PhD/WP1/Project/Code/MS2LDA/MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/positive_s2v_library.pkl"
        ):

    s2v_similarity, library = load_s2v_and_library(path_model, path_library)

    return s2v_similarity, library