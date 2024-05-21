
from gensim.models import Word2Vec
from spec2vec import Spec2Vec
from spec2vec.vector_operations import cosine_similarity_matrix

import numpy as np
import pandas as pd
from functools import partial




def load_s2v_and_library(path):
    """load spec2vec model and library embeddings, SMILES and spectra
    
    ARGS:
        path (str): path to spec2vec model and embeddings
        
    RETURNS:
        s2v_similarity: gensim word2vec based similarity model for Spec2Vec
        library (pd.dataframe): dataframe with embeddings, SMILES and spectra
    """

    s2v = Word2Vec.load(path + "\spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
    library = pd.read_pickle(path + "\library.pkl")

    s2v_similarity = Spec2Vec(
        model=s2v,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=100
    )

    return s2v_similarity, library


def calc_embeddings(s2v_similarity, spectra):
    """calculates spectral embeddings for a list of spectra
    
    ARGS:
        s2v_similarity: gensim word2vec based similarity model for Spec2Vec
        spectra (list): list of matchms spectrum objects

    RETURNS:
        spectral_embeddings (np.array): array of arrarys with Spec2Vec embeddings
    """
    
    spectral_embeddings = []
    for spectrum in spectra:
        spectral_embedding = np.array(s2v_similarity._calculate_embedding(spectrum))
        spectral_embeddings.append(spectral_embedding)

    return np.array(spectral_embeddings)


def calc_similarity(embeddings_A, embeddings_B):
    """calculates the cosine similarity of spectral embeddings
    
    ARGS:
        embeddings_A (np.array): query spectral embeddings
        embeddings_B (np.array): reference spectral embeddings
        
    RETURNS:
        similarities_scores (list): list of lists with s2v similarity scores
    """
    if type(embeddings_B) == pd.core.series.Series:
        embeddings_B = np.vstack(embeddings_B.to_numpy())

    similarity_vectors = []
    for embedding_A in embeddings_A:
        similarity_scores = cosine_similarity_matrix(np.array([embedding_A]), embeddings_B)[0]
        similarity_vectors.append(similarity_scores)
    
    similarities_matrix = pd.DataFrame( np.array(similarity_vectors).T, columns=range(len(embeddings_A)) )
    
    return similarities_matrix



def get_library_matches_per_motif(similarity_matrix, library, top_n=10, motif_number=0): 
    """returns similarity scores, SMILES and spectra for top n matches for ONE motif
    
    ARGS:
        similarity_matrix (pd.dataframe): dataframe containing all s2v similarities between motifs and library
        library (pd.dataframe): dataframe containing library embeddings, smiles and spectra
        motif_number (int): number of motif 
        top_n (int): number how many top matches should be retrieved

    RETURNS:
        top_smiles (list): list of SMILES strings
        top_spectra (list): list of matchms spectrum objects
        top_scores (list): list of floats (Spec2Vec scores)
    """

    top_n_rows = similarity_matrix.nlargest(top_n, motif_number).index.to_list()
    
    top_smiles = []
    top_spectra = []
    top_scores = []

    for row in top_n_rows:
        score = similarity_matrix.iat[row, motif_number]
        smi = library.iat[row, 1]
        spectrum = library.iat[row, 2]

        top_scores.append(score)
        top_smiles.append(smi)
        top_spectra.append(spectrum)

    return top_smiles, top_spectra, top_scores



def get_library_matches(matching_settings):
    """returns similarity scores, SMILES and spectra for top n matches for ALL motifs
    
    ARGS:
        matching_settings (dict):
                key 'similarity_matrix' -> (pd.dataframe): dataframe containing all s2v similarities between motifs and library
                key 'library' -> (pd.dataframe): dataframe containing library embeddings, smiles and spectra
                key 'top_n' -> (int): number how many top matches should be retrieved
    
    RETURNS:
        library_matches (list): list of lists (per motif) of lists (smiles, spectra, similarity_scores)
    """

    similarity_matrix = matching_settings["similarity_matrix"]
    library = matching_settings["library"]
    top_n = matching_settings["top_n"]

    preset_get_library_matches_per_motif = partial(get_library_matches_per_motif, similarity_matrix=similarity_matrix, library=library, top_n=top_n)
    
    n_motifs = similarity_matrix.shape[1]

    library_matches = []
    for motif_number in range(n_motifs):
        library_match = preset_get_library_matches_per_motif(motif_number=motif_number)
        library_matches.append(library_match)

    return library_matches





if __name__ == "__main__":

    from matchms.filtering import add_losses
    from matchms import Spectrum

    # first, let's generate some dummy spectra
    spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                      intensities=np.array([0.7, 0.2, 0.1]),
                      metadata={'id': 'spectrum1',
                                'precursor_mz': 201.})
    spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                        intensities=np.array([0.4, 0.2, 0.1]),
                        metadata={'id': 'spectrum2',
                                  'precursor_mz': 233.})
    spectrum_3 = Spectrum(mz=np.array([110, 140, 195.]),
                        intensities=np.array([0.6, 0.2, 0.1]),
                        metadata={'id': 'spectrum3',
                                  'precursor_mz': 214.})
    spectrum_4 = Spectrum(mz=np.array([100, 150, 200.]),
                        intensities=np.array([0.6, 0.1, 0.6]),
                        metadata={'id': 'spectrum4',
                                  'precursor_mz': 265.})
    
    spectra = [add_losses(spectrum_1), add_losses(spectrum_2), add_losses(spectrum_3), add_losses(spectrum_4)]
    
    # retrieve top matches from a reference library
    s2v_similarity, library = load_s2v_and_library(".")
    spectral_embeddings = calc_embeddings(s2v_similarity, spectra)
    similarity_matrix = calc_similarity(spectral_embeddings, library.embeddings)

    # get matches for all motifs; in this case the four spectra from above
    matching_settings = {
    "similarity_matrix": similarity_matrix,
    "library": library,
    "top_n": 10
    }

    library_matches = get_library_matches(matching_settings)
    