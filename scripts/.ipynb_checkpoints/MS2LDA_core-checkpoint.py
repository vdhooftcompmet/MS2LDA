from gensim.models.ldamodel import LdaModel
from gensim.models import EnsembleLda
from gensim.corpora import Dictionary

import numpy as np
from itertools import chain
from operator import itemgetter

from matchms.importing import load_from_mgf
import matchms.filtering as msfilters
from matchms import Spectrum
from matchms.Fragments import Fragments

def load_mgf(spectra_path):
    """loads spectra from a mgf file

    ARGS:
        spectra_path (str): path to the spectra.mgf file

     :
        spectra (generator): matchms generator object with the loaded spectra
    """

    spectra = load_from_mgf(spectra_path)

    return spectra


def clean_spectra(spectra):
    """uses matchms to normalize intensities, add information and add losses to the spectra
    
    ARGS:
        spectra (generator): generator object of matchms.Spectrum.objects loaded via matchms in python
    
    RETURNS:
        cleaned_spectra (list): list of matchms.Spectrum.objects; spectra that do not fit will be removed
    """
    cleaned_spectra = []

    for spectrum in spectra:
        # metadata filters
        spectrum = msfilters.default_filters(spectrum)
        spectrum = msfilters.add_retention_index(spectrum)
        spectrum = msfilters.add_retention_time(spectrum)
        spectrum = msfilters.require_precursor_mz(spectrum)

        # normalize and filter peaks
        spectrum = msfilters.normalize_intensities(spectrum)
        spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
        spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
        spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
        spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)
        spectrum = msfilters.add_losses(spectrum)

        if spectrum:
            spectrum_binned = Spectrum(mz=np.array([round(mz_, 2) for mz_ in spectrum.peaks.mz]),
                                intensities=spectrum.peaks.intensities,
                                metadata=spectrum.metadata
            ) # replaces peaks.mz with binned peaks.mz
            spectrum_binned.losses = Fragments(mz=np.array([round(mz_, 2) for mz_ in spectrum.losses.mz if mz_ > 0.01]),
                                               intensities=np.array([intensity for intensity, mz_ in zip(spectrum.losses.intensities, spectrum.losses.mz) if mz_ > 0.01])
            ) # replaces losses.mz with binned losses.mz
            cleaned_spectra.append(spectrum_binned)

    return cleaned_spectra



def frag_and_loss2word(spectra): #You should write some unittests for this function; seems to be error prone
    """generates a list of lists for fragments and losses for a dataset

    ARGS:
        spectra (list): list of matchms.Spectrum.objects; they should be cleaned beforehand e.g. intensity normalization, add losses

    RETURNS:
        dataset_frag (list): is a list of lists where each list represents fragements from one spectrum
        dataset_loss (list): is a list of lists where each list represents the losses from one spectrum
    """
    dataset_frag = []
    dataset_loss = []

    for spectrum in spectra:
        intensities_from_0_to_100 = (spectrum.peaks.intensities * 100).round()

        frag_with_2_digits = [ [str(round(mz, 2))+"+"] for mz in spectrum.peaks.mz] # every fragment is in a list; BINNING NOT NEEDED ANYMORE
        frag_multiplied_intensities = [frag * int(intensity) for frag, intensity in zip(frag_with_2_digits, intensities_from_0_to_100)]
        frag_flattend = list(chain(*frag_multiplied_intensities))

        if frag_flattend not in dataset_frag: # if the exact peaks were already found the spectrum will be removed
            dataset_frag.append(frag_flattend)

            loss_with_2_digits = [ [str(round(mz, 2))] for mz in spectrum.losses.mz] # every fragment is in a list; BINNING NOT NEEDED ANYMORE
            loss_multiplied_intensities = [loss * int(intensity) for loss, intensity in zip(loss_with_2_digits, intensities_from_0_to_100)]
            loss_flattend = list(chain(*loss_multiplied_intensities))
            loss_without_zeros = list(filter(lambda loss: float(loss) > 0.01, loss_flattend)) # removes 0 or negative loss values
            dataset_loss.append(loss_without_zeros)

    return dataset_frag, dataset_loss



def combine_frag_loss(dataset_frag, dataset_loss):
    """combines fragments and losses for a list of spectra

    ARGS:
        dataset_frag(list): is a list of lists where each list represents fragements from one spectrum
        dataset_loss (list): is a list of lists where each list represents the losses from one spectrum

    RETURNS:
        frag_and_loss (list): is a list of list where each list represents the fragments and losses from one spectrum
    """

    dataset_frag_and_loss = []
    for spectrum_frag, spectrum_loss in zip(dataset_frag, dataset_loss):
        dataset_frag_and_loss.append(spectrum_frag + spectrum_loss)

    return dataset_frag_and_loss


def generate_corpus(dataset_frag_and_loss, id2dataset_frag_and_loss=None):
    """generates a corpus (dictionary) for the lda model

    ARGS:
        frag_and_loss (list): is a list of list where each list represents the fragments and losses from one spectrum

    RETURNS:
        corpus4frag_and_loss (list): list of tuple with the count and id of frag or loss
        id2dataset_frag_and_loss (dict): Dictionary with id for fragments and losses
    """

    if id2dataset_frag_and_loss == None:
        id2dataset_frag_and_loss = Dictionary(dataset_frag_and_loss)
    
    corpus4dataset_frag_and_loss = []
    for spectrum_frag_and_loss in dataset_frag_and_loss:
        id_count_per_spectrum = id2dataset_frag_and_loss.doc2bow(spectrum_frag_and_loss)
        corpus4dataset_frag_and_loss.append(id_count_per_spectrum)

    return corpus4dataset_frag_and_loss, id2dataset_frag_and_loss


def run_lda(spectra_path, num_motifs, iterations=300, update_every=1):
    """runs lda and needed scripts to clean and prepare data
    - load mgf file
    - clean spectra (matchms preprocessing)
    - convert fragments and losses to words and integrate intensities
    - combine fragments and losses in one document
    - build corpus with fragments and losses document
    - run LDA

    ARGS:
        spectra_path (str): path to the mgf file
        num_motifs (int): number of motifs/topics for LDA algorithm
        iterations (int): number of LDA iterations to find the right topic for a given document
        update_every (int): see gensim doc

    RETURNS:
        lda_model (gensim.model): trained LDA model
        corpus4dataset_frag_and_loss (list(tuples)): list of tuples where the first entry is the words id and the second the count per doc
        id2dataset_frag_and_loss (dictionary): dictionary where each words gets a unique identifier
    """

    spectra = load_mgf(spectra_path)
    cleaned_spectra = clean_spectra(spectra)
    dataset_frag, dataset_loss = frag_and_loss2word(cleaned_spectra)
    dataset_frag_and_loss = combine_frag_loss(dataset_frag, dataset_loss)
    corpus4dataset_frag_and_loss, id2dataset_frag_and_loss = generate_corpus(dataset_frag_and_loss)

    lda_model = LdaModel(corpus=corpus4dataset_frag_and_loss,
                     id2word=id2dataset_frag_and_loss,
                     num_topics=num_motifs, 
                     random_state=73,
                     update_every=update_every,
                     iterations=iterations,
                     alpha="auto",
                     eta="auto",
                     decay=0.8,
                     eval_every=1,
                     #gamma_threshold=0.8,
                     #minimum_probability=0.7,
                     #offset=0.8,
                     #decay=0.9,
                     ##num_models=4,
                     ) # there are more here!!!
    
    return lda_model, corpus4dataset_frag_and_loss, id2dataset_frag_and_loss




def retrieve_smiles_from_spectra(cleaned_spectra):
    """retrieves smiles from matchms ojects
    
    ARGS:
        cleaned_spectra (list): list of matchms.Spectrum.objects after matchms preprocessing

    RETURNS:
        dataset_smiles (list(str)): a list of SMLIES retrieved from the matchms.spectrum.object metadata
    """
    dataset_smiles = []
    for spectrum in cleaned_spectra:
        try:
            smiles = spectrum.metadata["smiles"]
            dataset_smiles.append(smiles)
        except ValueError:
            dataset_smiles.append(None)

    return dataset_smiles


def predict_with_lda(lda_model, spectra_path, id2dataset_frag_and_loss):
    """Uses a pretrained LDA model to assign spectra to earlier (from the pretrained LDA model) defined motifs and retrieve SMILES for assigned spectra; 
    Therefore all cleaned steps as for the LDA training and SMILES are retrieved from the spectra
    - load mgf file
    - clean spectra (matchms preprocessing)
    - retrieve SMILES form matchms.spectrum.object metadata
    - convert fragments and losses to words and integrate intensities
    - combine fragments and losses in one document
    !- NO new corpus is build since the one from the given model is used
    - predict motifs with given LDA model (currently to the most likely motif, means highest score)

    ARGS:
        lda_model (gensim.object): pretrained lda_model
        spectra_path (str): path to the mgf file
        id2dataset_frag_and_loss (dictionary): dictionary where each words gets a unique identifier

    RETURNS:
        smiles_per_motifs (list(list(str))): list of lists with SMILES per motif (all SMILES that belong to one motif)
        predicted_motifs  (list(list(tuple))): list of lists of tuples with [0] predicted motif and [1] the certainty of the predicted motif
        predicted_motifs_distribution (list(list(list(tuple)))): extend the predicted_motifs, so you can see all predicted motifs and not only the one with the highest score
        spectra_per_motifs (list(list(matchms.spectrum.object))): list of lists of matchms.spectrum.objects per motifs (all spctra that belong to one motif)
    
    """

    # preprocessing
    spectra = load_mgf(spectra_path)
    cleaned_spectra = clean_spectra(spectra)
    dataset_smiles = retrieve_smiles_from_spectra(cleaned_spectra)
    dataset_frag, dataset_loss = frag_and_loss2word(cleaned_spectra)
    dataset_frag_and_loss = combine_frag_loss(dataset_frag, dataset_loss)

    # predict motifs
    predicted_motifs = []
    for spectrum_frag_and_loss in dataset_frag_and_loss:
        corpus4spectrum_frag_and_loss, _ = generate_corpus([spectrum_frag_and_loss], id2dataset_frag_and_loss=id2dataset_frag_and_loss)
    
        transformed_corpus = lda_model[corpus4spectrum_frag_and_loss]

        for predicted_motif in transformed_corpus:
            predicted_motifs.append(predicted_motif)

    # add smiles
    num_motifs = max([max(predicted_motif)[0] for predicted_motif in predicted_motifs]) + 1
    smiles_per_motifs = [list() for i in range(num_motifs)]
    predicted_motifs_distribution = [list() for i in range(num_motifs)]
    spectra_per_motifs = [list() for i in range(num_motifs)]
    for smiles, predicted_motif, cleaned_spectrum in zip(dataset_smiles, predicted_motifs, cleaned_spectra):
        most_likely_topic = max(predicted_motif, key=itemgetter(1))[0] # find the most likely topic
        smiles_per_motifs[most_likely_topic].append(smiles) # append SMILES per motif
        predicted_motifs_distribution[most_likely_topic].append(predicted_motif) # append motifs distribution per motif
        spectra_per_motifs[most_likely_topic].append(cleaned_spectrum) # append cleaned spectra per motif


    return smiles_per_motifs, predicted_motifs, predicted_motifs_distribution, spectra_per_motifs



if __name__ == "__main__":
    # test run
    spectra_path = r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\Spec2Struc\Project_SubformulaAnnotation\raw_data\_RAWdata1\GNPS-SCIEX-LIBRARY.mgf"
    run_lda(spectra_path, 12)