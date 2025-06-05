import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel

from Preprocessing.load_and_clean import load_mgf
from Preprocessing.load_and_clean import clean_spectra
from Preprocessing.generate_corpus import features_to_words
from Preprocessing.generate_corpus import combine_features

from MS2LDA.modeling import define_model
from MS2LDA.modeling import train_model


def compute_coherence_values(spectra_path, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        spectra = load_mgf(spectra_path)
        cleaned_spectra = clean_spectra(spectra)
        fragment_words, loss_words = features_to_words(cleaned_spectra)
        feature_words = combine_features(fragment_words, loss_words)
        ms2lda = define_model(n_motifs=n_motifs)
        train_parameters = {"parallel": 4}
        trained_ms2lda = train_model(
            ms2lda, feature_words, iterations=300, train_parameters=train_parameters
        )
        model_list.append(trained_ms2lda)
        coherence_model_lda = CoherenceModel(
            model=lda_model, texts=spectra, dictionary=id2word, coherence="c_v"
        )
        coherence_values.append(coherence_model_lda.get_coherence())
    # Plotting
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc="best")
    plt.ylim(0, 1)
    plt.show()

    return None
