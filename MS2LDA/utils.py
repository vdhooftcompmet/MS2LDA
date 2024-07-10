from rdkit import Chem
from matchms import Spectrum, Fragments
from matchms.filtering import normalize_intensities
import numpy as np


from matchms import set_matchms_logger_level; set_matchms_logger_level("ERROR")


def create_spectrum(motif_k_features, k, frag_tag="frag@", loss_tag="loss@", significant_digits=2):

    # identify slicing start
    frag_start = len(frag_tag)
    loss_start = len(loss_tag)

    # extract fragments and losses
    fragments = [ ( round(float(feature[frag_start:]),significant_digits), float(importance) ) for feature, importance in motif_k_features if feature.startswith(frag_tag) ]
    losses = [ ( round(float(feature[loss_start:]),significant_digits), float(importance) ) for feature, importance in motif_k_features if feature.startswith(loss_tag) ]

    # sort features based on mz value
    sorted_fragments, sorted_fragments_intensities = zip(*sorted(fragments)) if fragments else (np.array([]), np.array([]))
    sorted_losses, sorted_losses_intensities = zip(*sorted(losses)) if losses else (np.array([]), np.array([]))

    # normalize intensity over fragments and losses
    intensities = list(sorted_fragments_intensities) + list(sorted_losses_intensities)
    max_intensity = np.max(intensities)
    normalized_intensities = np.array(intensities) / max_intensity

    # split fragments and losses
    normalized_frag_intensities = normalized_intensities[:len(sorted_fragments)]
    normalized_loss_intensities = normalized_intensities[len(sorted_fragments):]

    # create spectrum object
    spectrum = Spectrum(
        mz=np.array(sorted_fragments),
        intensities=np.array(normalized_frag_intensities),
        metadata={
            "id": f"motif_{k}",
        }
    )
    spectrum.losses = Fragments(mz=np.array(sorted_losses), intensities=np.array(normalized_loss_intensities))

    return spectrum


def match_frags_and_losses(motif_spectrum, analog_spectra):
    """matches fragments and losses between analog and motif spectrum and returns them
    
    ARGS:
        motif_spectrum (matchms.spectrum.object): spectrum build from the found motif
        analog_spectra (list): list of matchms.spectrum.objects which normally are identified by Spec2Vec

    RETURNS:
        matching_frags (list): a list of sets with fragments that are present in analog spectra and the motif spectra: each set represents one analog spectrum
        matching_losses (list) a list of sets with losses that are present in analog spectra and the motif spectra: each set represents one analog spectrum
        
    """

    motif_frags = set(motif_spectrum.peaks.mz)
    motif_losses = set(motif_spectrum.losses.mz)

    matching_frags = []
    matching_losses = []

    for analog_spectrum in analog_spectra:
        analog_frag = set(analog_spectrum.peaks.mz)
        analog_loss = set(analog_spectrum.losses.mz)

        matching_frag = motif_frags.intersection(analog_frag)
        matching_loss = motif_losses.intersection(analog_loss)

        matching_frags.append(matching_frag)
        matching_losses.append(matching_loss)

    return matching_frags, matching_losses