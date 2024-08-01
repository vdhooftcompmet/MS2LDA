import tomotopy as tp
import numpy as np
#from MS2LDA.utils import create_spectrum
from utils import create_spectrum


def define_model(n_motifs, model_parameters={}):
    """creating a LDA model using the tomotopy library
    
    ARGS:
        n_motifs (int): number of motifs that will be generated
        model_parameters (dict): defines all further parameters that can be set in the tomotopy LDA model (see https://bab2min.github.io/tomotopy/v0.12.6/en/#tomotopy.LDAModel)    

    RETURNS:
        model: tomotopy LDAModel class
    """

    model = tp.LDAModel(k=n_motifs, **model_parameters)
    
    return model


def train_model(model, documents, iterations=100, train_parameters={}):
    """trains the LDA model on the given documents
    
    ARGS:
        model: tomotopy LDAModel class
        documents (list): list of lists with frag@/loss@ strings representing spectral features
        iterations (int): number of iterations in the training
        train_parameters (dict): defines all further parameters that can be set in the tomotopy training function (see https://bab2min.github.io/tomotopy/v0.12.6/en/#tomotopy.LDAModel.train)
        
    RETURNS:
        model: tomotopy LDAModel class
    """

    for doc in documents:
        model.add_doc(doc)

    model.train(iterations, **train_parameters)

    return model


def extract_motifs(model, top_n=3):
    """extract motifs from the trained LDA model
    
    ARGS:
        model: tomotopy LDAModel class
        top_n (int): number of top n features extracted per motif
        
    RETURNS:
        motif_features (list): tuples within a list of lists with spectral features assigned per motif and their given motif importance
    """

    motif_features = []

    for motif_index in range(model.k):
        motif_k_features = model.get_topic_words(motif_index, top_n=top_n)
        motif_features.append(motif_k_features)

    return motif_features


def create_motif_spectra(motif_features, charge=1, motifset_name="unknown"):
    """creates a matchms spectrum object for the found motifs
    
    ARGS:
        motif_features (list): tuples within a list of lists with spectral features assigned per motif and their given motif importance
        
    RETURNS:
        motif_spectra (list): list of matchms spectrum objects; one for each motif
    """

    motif_spectra = []
        
    for k, motif_k_features in enumerate(motif_features):
        motif_spectrum = create_spectrum(motif_k_features, k, charge=charge, motifset=motifset_name)
        motif_spectra.append(motif_spectrum)

    return motif_spectra




if __name__ == "__main__":
    documents = [
        ["frag@24.33", "frag@34.23", "loss@18.01", "loss@18.01"],
        ["frag@24.33", "frag@65.87", "loss@121.30", "frag@24.33"],
        ["frag@74.08", "frag@34.23", "loss@18.01", "loss@18.01", "loss@18.01"],
        ["frag@74.08", "frag@121.30", "loss@34.01"]
        ] 
    
    model = define_model(2) 
    model = train_model(model, documents)
    motifs = extract_motifs(model)
    motif_spectra = create_motif_spectra(motifs)
    print(motif_spectra[0])
    print("simple test")


    # example with emulating fixed motifs
    print()

    from matchms import Spectrum
    import numpy as np
    fixed_motifs = [
        Spectrum(mz=np.array([74.08]),
                intensities=np.array([1.0]),
                metadata={'id': 'spectrum1',
                        'precursor_mz': 201.}),
    ]

    model = define_model(3)
    model = train_model(model, documents)
    motifs = extract_motifs(model)
    motif_spectra = create_motif_spectra(motifs)
    print(motif_spectra[0].peaks.mz)
    print(motif_spectra[0].peaks.intensities)

   



