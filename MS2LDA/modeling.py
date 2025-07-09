import tomotopy as tp
import numpy as np
from MS2LDA.utils import create_spectrum
import warnings
from tqdm import tqdm


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


def train_model(
    model,
    documents,
    iterations=100,
    train_parameters={},
    convergence_parameters={
        "type": "entropy_history_doc",
        "threshold": 0.01,
        "window_size": 3,
        "step_size": 10,
    },
):
    """trains the LDA model on the given documents

    ARGS:
        model: tomotopy LDAModel class
        documents (list): list of lists with frag@/loss@ strings representing spectral features
        iterations (int): number of iterations in the training
        train_parameters (dict): defines all further parameters that can be set in the tomotopy training function (see https://bab2min.github.io/tomotopy/v0.12.6/en/#tomotopy.LDAModel.train)

    RETURNS:
        model: tomotopy LDAModel class
        convergence_curve (list): list containing the model perplexity values for after every 10 iterations
    """

    for doc in documents:
        model.add_doc(doc)

    convergence_history = {
        "entropy_history_doc": [],
        "entropy_history_topic": [],
        "perplexity_history": [],
        "log_likelihood_history": [],
    }

    # entropy_history_doc = []
    # entropy_history_topic = []
    # perplexity_history = []
    # log_likelihood_history = []

    for _ in tqdm(range(0, iterations, convergence_parameters["step_size"])):
        model.train(convergence_parameters["step_size"], **train_parameters)

        # calculate perplexity score and saves it
        perplexity = model.perplexity
        convergence_history["perplexity_history"].append(perplexity)

        # calculates log likelihood score and save it
        log_likelihood = model.ll_per_word
        convergence_history["log_likelihood_history"].append(log_likelihood)

        # calculates the document topic entropy and saves it
        current_doc_entropy = calculate_document_entropy(model)
        convergence_history["entropy_history_doc"].append(current_doc_entropy)

        # calculates the topic word entropy and saves it
        current_topic_entropy = calculate_topic_entropy(model)
        convergence_history["entropy_history_topic"].append(current_topic_entropy)

        # Check convergence criteria
        model_converged = len(
            convergence_history[convergence_parameters["type"]]
        ) > convergence_parameters["window_size"] and check_convergence(
            convergence_history[convergence_parameters["type"]],
            epsilon=convergence_parameters["threshold"],
            n=convergence_parameters["window_size"],
        )

        # early stopping
        if model_converged:
            print("Model has converged")
            return model, convergence_history

    else:
        print("model did not converge")
        return model, convergence_history


# check if model converged
def calculate_document_entropy(model):
    """Entropy for Document-Topic Distribution"""
    entropy_values = []

    for doc in model.docs:
        topic_dist = doc.get_topic_dist()
        topic_dist = np.where(np.array(topic_dist) == 0, 1e-12, topic_dist)

        entropy = -np.sum(topic_dist * np.log(topic_dist))
        entropy_values.append(entropy)

    return np.mean(entropy_values)


def calculate_topic_entropy(model):
    """Entropy for Topic-Word Distribution"""
    entropy_values = []

    for k in range(model.k):
        word_dist = model.get_topic_word_dist(k)
        word_dist = np.where(np.array(word_dist) == 0, 1e-12, word_dist)

        entropy = -np.sum(word_dist * np.log(word_dist))
        entropy_values.append(entropy)

    return np.mean(entropy_values)


def check_convergence(entropy_history, epsilon=0.001, n=3):
    """no"""
    changes = [
        abs(entropy_history[i] - entropy_history[i - 1]) / entropy_history[i - 1]
        for i in range(1, len(entropy_history))
    ]

    return all(change < epsilon for change in changes[-n:])


def extract_motifs(model, top_n=50):
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


def create_motif_spectra(
    motif_features, charge=1, motifset_name="unknown", significant_digits=2
):
    """creates a matchms spectrum object for the found motifs

    ARGS:
        motif_features (list): tuples within a list of lists with spectral features assigned per motif and their given motif importance

    RETURNS:
        motif_spectra (list): list of matchms spectrum objects; one for each motif
    """

    motif_spectra = []

    for k, motif_k_features in enumerate(motif_features):
        motif_spectrum = create_spectrum(
            motif_k_features,
            k,
            charge=charge,
            motifset=motifset_name,
            significant_digits=significant_digits,
        )
        motif_spectra.append(motif_spectrum)

    return motif_spectra


if __name__ == "__main__":
    documents = [
        ["frag@24.33", "frag@34.23", "loss@18.01", "loss@18.01"],
        ["frag@24.33", "frag@65.87", "loss@121.30", "frag@24.33"],
        ["frag@74.08", "frag@34.23", "loss@18.01", "loss@18.01", "loss@18.01"],
        ["frag@74.08", "frag@121.30", "loss@34.01"],
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
        Spectrum(
            mz=np.array([74.08]),
            intensities=np.array([1.0]),
            metadata={"id": "spectrum1", "precursor_mz": 201.0},
        ),
    ]

    model = define_model(3)
    model = train_model(model, documents)
    motifs = extract_motifs(model)
    motif_spectra = create_motif_spectra(motifs)
    print(motif_spectra[0].peaks.mz)
    print(motif_spectra[0].peaks.intensities)
