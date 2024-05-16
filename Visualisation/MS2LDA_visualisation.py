import pyLDAvis.gensim
pyLDAvis.enable_notebook()
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from MS2LDA_core import run_lda



def create_pyLDAvis(spectra_path, list_no_motif, dataset_name):
    """ Creates different html files for each number of topics in the list_no_motif
    Args:
    - spectra_path: spectra_path
    - list_no_motif: no. motifs to be used in LDA

    Returns:
    - html with the different no. motifs on the list

    """
    for num_topics in list_no_motif:
        lda_model, corpus, id2word, spectra= run_lda(spectra_path=spectra_path, num_motifs=num_topics, iterations=100)
        vis_data=pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word, mds='tsne')
        pyLDAvis.save_html(vis_data, f'{dataset_name}_{num_topics}.html')
    return None
  

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
        lda_model, corpus, id2word, spectra= run_lda(spectra_path=spectra_path, num_motifs=num_topics, iterations=100)
        model_list.append(lda_model)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=spectra, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherence_model_lda.get_coherence())        
      # Plotting
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.ylim(0, 1)
    plt.show()

    return None
