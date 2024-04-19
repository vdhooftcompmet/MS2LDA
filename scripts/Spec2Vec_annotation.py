from gensim.models import Word2Vec

from spec2vec.vector_operations import cosine_similarity_matrix
from spec2vec import Spec2Vec

import numpy as np
import pandas as pd

from heapq import nlargest
import operator

from MS2LDA_core import load_mgf
from MS2LDA_core import clean_spectra

def create_embeddings(s2v_model_path, spectra_path):
    """create new embeddings
    
    """
    spectra_DB = load_mgf(spectra_path)
    cleaned_spectra_DB = clean_spectra(spectra_DB)

    s2v_model = Word2Vec.load(s2v_model_path)

    spec2vec_similarity = Spec2Vec(
        model=s2v_model,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=100,
        )
    
    embeddings_DB = []
    smiles_DB = []
    for spectrum in cleaned_spectra_DB:
        embedding_DB = spec2vec_similarity._calculate_embedding(spectrum)
        smi_DB = spectrum.get("smiles")

        embeddings_DB.append(embedding_DB)
        smiles_DB.append(smi_DB)


    embeddings_smiles_DB = pd.DataFrame({
        "embeddings": embeddings_DB,
        "smiles": smiles_DB,
        "spectra": cleaned_spectra_DB,
    })

    embeddings_smiles_DB.to_pickle("embeddings_smiles_spectra_DB.pickle")

    return "Done"
    


def load_model_and_data(path=r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\Spec2Struc\Project_SubstructureIdentification\scripts\programming_scripts\models"):
    """load the Spec2Vec model
    
    ARGS:
        path (str): path to Spec2Vec model

    RETURNS:
        model (gensim.word2vec.object): returns the loaded gensim word2vec model object    
    """

    s2v_model = Word2Vec.load(path + "\spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
    embeddings_smiles_DB = pd.read_pickle("embeddings_smiles_spectra_DB.pickle")

    return s2v_model, embeddings_smiles_DB


def calc_similarity(s2v_model, motif_spectra, embeddings_DB):
    """calculates the spec2vec scores between ref and query spectra
    
    ARGS:
        model (gensim.word2vec.object): spec2vec model
        ref_spectra (matchms.spectrum.object): unknown spectra
        query_spectra (matchms.spectrum.object): known spectra (with smiles)
        
    RETURNS:
        scores (list): list of lists with spec2vec scores between ref and query spectra   

    """
    
    spec2vec_similarity = Spec2Vec(
        model=s2v_model,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=100,
        )
    
    motif_embeddings = np.array([spec2vec_similarity._calculate_embedding(motif_spectrum) for motif_spectrum in motif_spectra])

    s2v_scores = []
    for motif_embedding in motif_embeddings:
        s2v_score = cosine_similarity_matrix(np.array([motif_embedding]), np.array(embeddings_DB))[0]
        s2v_scores.append(s2v_score)
    
    return s2v_scores


def retrieve_top_hits(s2v_scores, motif_number, smiles, spectra):
    """retrieves scores and SMILES from top n query spectra hits

    ARGS:
        ref_spectra (matchms.spectrum.object): unknown spectra
        query_spectra (matchms.spectrum.object): known spectra (with smiles)
        motif_id (int): first motif would be 1 and second 2
        top_n (int): number of hits; 1 would be 1 hit

    RETURNS:
        top_scores (list): top n scores
        top_smiles (list): top n SMILES

    """
    top_index, top_scores = list(zip(*nlargest(10, enumerate(s2v_scores[motif_number]), key=operator.itemgetter(1))))
    top_smiles = [smiles[index] for index in top_index]
    top_spectra = [spectra[index] for index in top_index]

    return top_scores, top_smiles, top_spectra


def s2v_annotation(ref_spectra, query_spectra, motif_id, top_n=5):
    model= load_model_and_data()
    scores = calc_similarity(model, ref_spectra, query_spectra)
    top_scores, top_smiles = retrieve_top_hits(ref_spectra, scores, motif_id, top_n)

    return top_scores, top_smiles
 



if __name__ == "__main__":
    path_model = r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\Spec2Struc\Project_SubstructureIdentification\scripts\models\spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
    model, ref_embeddings = load_model()
    scores = calc_similarity(model, ref_embeddings, query_spectra)
