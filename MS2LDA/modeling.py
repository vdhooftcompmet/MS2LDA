import tomotopy as tp
import numpy as np

from matchms import Spectrum, Fragments
from matchms import set_matchms_logger_level; set_matchms_logger_level("ERROR")


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


def create_motif_spectra(motif_features):
    """creates a matchms spectrum object for the found motifs
    
    ARGS:
        motif_features (list): tuples within a list of lists with spectral features assigned per motif and their given motif importance
        
    RETURNS:
        motif_spectra (list): list of matchms spectrum objects; one for each motif
    """

    motif_spectra = []
        
    for k, motif_k_features in enumerate(motif_features):

        # extract fragments and losses
        fragments = [ (float(feature[5:]), float(importance)) for feature, importance in motif_k_features if feature.startswith("frag@") ]
        losses = [ (float(feature[5:]), float(importance)) for feature, importance in motif_k_features if feature.startswith("loss@") ]

        # sort features based on mz value
        sorted_fragments, sorted_fragments_intensities = zip(*sorted(fragments)) if fragments else (np.array([]), np.array([]))
        sorted_losses, sorted_losses_intensities = zip(*sorted(losses)) if losses else (np.array([]), np.array([]))

        # normalize intensity over fragments and losses
        intensities = list(sorted_fragments_intensities) + list(sorted_losses_intensities)
        min_intensity, max_intensity = np.min(intensities), np.max(intensities)
        
        normalized_intensities = [(intensity - min_intensity) / (max_intensity - min_intensity) for intensity in intensities]

        normalized_frag_intensities = normalized_intensities[:len(sorted_fragments)]
        normalized_loss_intensities = normalized_intensities[len(sorted_fragments):]

        # create motif spectrum
        motif_spectrum = Spectrum(
            mz=np.array(sorted_fragments),
            intensities=np.array(normalized_frag_intensities),
            metadata={
                "id": f"motif_{k}",
            }
        )
        motif_spectrum.losses = Fragments(mz=np.array(sorted_losses), intensities=np.array(normalized_loss_intensities))

        motif_spectra.append(motif_spectrum)

    return motif_spectra


if __name__ == "__main__":
    documents = [
        ["frag@24.33", "frag@34.23", "loss@18.01", "loss@18.01"],
        ["frag@24.33", "frag@65.87", "loss@121.30", "frag@24.33"],
        ["frag@74.08", "frag@34.23", "loss@18.01", "loss@18.01", "loss@18.01"]
        ] 
    
    model = define_model(2) 
    model = train_model(model, documents)
    motifs = extract_motifs(model)
    motif_spectra = create_motif_spectra(motifs)
    print(motif_spectra)
