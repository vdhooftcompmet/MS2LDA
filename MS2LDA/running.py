from MS2LDA.Preprocessing.load_and_clean import load_mgf
from MS2LDA.Preprocessing.load_and_clean import load_mzml
from MS2LDA.Preprocessing.load_and_clean import load_msp
from MS2LDA.Preprocessing.load_and_clean import clean_spectra

from MS2LDA.Preprocessing.generate_corpus import features_to_words
from MS2LDA.Preprocessing.generate_corpus import combine_features

from MS2LDA.modeling import define_model
from MS2LDA.modeling import train_model
from MS2LDA.modeling import extract_motifs
from MS2LDA.modeling import create_motif_spectra

from MS2LDA.Add_On.Spec2Vec.annotation import load_s2v_and_library
from MS2LDA.Add_On.Spec2Vec.annotation import calc_embeddings, calc_similarity
from MS2LDA.Add_On.Spec2Vec.annotation import get_library_matches

from MS2LDA.Add_On.Spec2Vec.annotation_refined import mask_spectra
#from MS2LDA.Add_On.Spec2Vec.annotation_refined import refine_annotation

from MS2LDA.Add_On.Fingerprints.FP_annotation import annotate_motifs as calc_fingerprints
from MS2LDA.Visualisation.visualisation import create_interactive_motif_network


def generate_motifs(mgf_path, 
                    n_motifs = 50,
                    model_parameters = {
                        "rm_top": 0,
                        "min_cf": 0,
                        "min_df": 0,
                        "alpha": 0.1,
                        "eta": 0.1,
                        "seed": 42,
                    },
                    train_parameters = {
                        "parallel": 1, 
                        "workers": 1
                    }, 
                    motif_parameter = 20,
                    charge=1,
                    motifset_name="unknown"):
    
    """generates the motif spectra based on a given mgf file
    
    ARGS:
        mgf_path (str): path to the mgf file
        model_parameters (dict): model parameters that can be set for a tomotopy LDA model
        train_parameters (dict): train parameters that can be set for a tomotopy training of an LDA model
        motif_parameter (int): number of top n most important features per motif
        
    RETURNS:
        motif_spectra (list): list of matchms spectrum objects (no precursor ion) 
    """
    # Preprocessing
    if mgf_path.endswith(".mgf"):
        loaded_spectra = load_mgf(mgf_path)
    elif mgf_path.endswith(".mzml") or mgf_path.endswith(".mzML"):
        loaded_spectra = load_mzml(mgf_path)
    elif mgf_path.endswith(".msp"):
        loaded_spectra = load_msp(mgf_path)
    cleaned_spectra = clean_spectra(loaded_spectra)
    print(len(cleaned_spectra))

    # Corpus Generation
    fragment_words, loss_words = features_to_words(cleaned_spectra)
    feature_words = combine_features(fragment_words, loss_words)

    # Modeling
    ms2lda = define_model(n_motifs=n_motifs, model_parameters=model_parameters)
    trained_ms2lda = train_model(ms2lda, feature_words, iterations=100, train_parameters=train_parameters)

    # Motif Generation
    motifs = extract_motifs(trained_ms2lda, top_n=motif_parameter)
    motif_spectra = create_motif_spectra(motifs, charge, motifset_name)

    return motif_spectra



def annotate_motifs(motif_spectra, 
                    top_n_matches = 5,
                    unique_mols = True,
                    path_model = "MS2LDA/Add_On/Spec2Vec/model_positive_mode/020724_Spec2Vec_pos_CleanedLibraries.model",
                    path_library = "MS2LDA/Add_On/Spec2Vec/model_positive_mode/positive_s2v_library.pkl"):
    """annotates motif with Spec2Vec
    
    ARGS:
        top_n_matches (int): top n compounds retrieved the database 
        unique_mols (boolean): True if only unique compounds or False duplicates can also be retrieved
        path_model (str): path to Spec2Vec model
        path_library (str): path the pkl library file, which contains embeddings, spectra and smiles
        
    RETURNS:
        optimized_motif_spectra (list): list of matchms motif spectra
        optimized_clusters (list): list of lists of spectra from clustered compounds
        smiles_clusters (list) list of lists of SMILES for clustered compounds
    """

    
    s2v_similarity, library = load_s2v_and_library(path_model, path_library)
    print("Model loaded ...")

    motif_embeddings = calc_embeddings(s2v_similarity, motif_spectra)
    similarity_matrix = calc_similarity(motif_embeddings, library.embeddings)
   
    matching_settings = {
                        "similarity_matrix": similarity_matrix,
                        "library": library,
                        "top_n": 5,
                        "unique_mols": True,
                    }

    
    library_matches = get_library_matches(matching_settings)

    masked_motif_spectra = mask_spectra(motif_spectra)
    optimized_motif_spectra, optimized_clusters, smiles_clusters, clusters_similarity = refine_annotation(s2v_similarity, library_matches, masked_motif_spectra, motif_spectra)

    return optimized_motif_spectra, optimized_clusters, smiles_clusters, clusters_similarity