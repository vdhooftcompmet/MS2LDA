from itertools import chain


def features_to_words(spectra, significant_figures=2): #You should write some unittests for this function; seems to be error prone
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

        frag_with_n_digits = [ ["frag@" + str(round(mz, significant_figures))] for mz in spectrum.peaks.mz] # round mz and add identifier -> frag@
        frag_multiplied_intensities = [frag * int(intensity) for frag, intensity in zip(frag_with_n_digits, intensities_from_0_to_100)] # weight fragments
        frag_flattend = list(chain(*frag_multiplied_intensities)) # flatten lists
        dataset_frag.append(frag_flattend)

        loss_with_n_digits = [ ["loss@" + str(round(mz, significant_figures))] for mz in spectrum.losses.mz] # round mz and add identifier -> loss@
        loss_multiplied_intensities = [loss * int(intensity) for loss, intensity in zip(loss_with_n_digits, intensities_from_0_to_100)] # weight losses
        loss_flattend = list(chain(*loss_multiplied_intensities)) # flatten lists
        loss_without_zeros = list(filter(lambda loss: float(loss[5:]) > 0.01, loss_flattend)) # removes 0 or negative loss values
        dataset_loss.append(loss_without_zeros)

    return dataset_frag, dataset_loss



def combine_features(dataset_frag, dataset_loss):
    """combines fragments and losses for a list of spectra

    ARGS:
        dataset_frag(list): list of lists where each list represents fragements from one spectrum
        dataset_loss (list): list of lists where each list represents the losses from one spectrum

    RETURNS:
        frag_and_loss (list): list of list where each list represents the fragments and losses from one spectrum
    """

    dataset_features = []
    for spectrum_frag, spectrum_loss in zip(dataset_frag, dataset_loss):
        dataset_features.append(spectrum_frag + spectrum_loss)

    return dataset_features



if __name__ == "__main__":
    from matchms import Spectrum
    from matchms.filtering import add_losses
    import numpy as np
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

    dataset_frag, dataset_loss = features_to_words(spectra)
    dataset_features = combine_features(dataset_frag, dataset_loss)
    print(dataset_features[0])