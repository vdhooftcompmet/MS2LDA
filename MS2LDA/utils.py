from rdkit import Chem
from matchms import Spectrum, Fragments
from matchms.filtering import normalize_intensities
import numpy as np

def smiles2mols(smiles_per_motif):
    """convert smiles to rdkit mol object
    
    ARGS:
        smiles_per_motif (list(str)): list of smiles that are associated with one motif

    RETURNS:
        mols (list(rdkit.mol.objects)): list of rdkit.mol.objects from the smiles
    
    !!! Currently only valid smiles are allowed; program could break if invalid smiles are given
    """
    mols = []
    for smiles in smiles_per_motif:
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

    return mols


def motifs2spectra(lda_model): ### testing is needed!!
    """converts found motifs into matchms spectrum objects"""

    def sort_dependent_lists(loss_or_frag, intensities):
        """sorts two lists based on the values of one of the lists"""

        combined_lists = list(zip(loss_or_frag, intensities))
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
        sorted_loss_or_frag, sorted_intensities = zip(*sorted_combined_lists)

        return np.array(sorted_loss_or_frag), np.array(sorted_intensities)
    
    def normalize_motifs(motif_spectrum):
        """normalizes peaks and losses together"""
        fragments_mz = motif_spectrum.peaks.mz
        fragments_intens = list(motif_spectrum.peaks.intensities)

        losses_mz = motif_spectrum.losses.mz
        losses_intens = list(motif_spectrum.losses.intensities)

        fragments_losses_intens = np.array(fragments_intens + losses_intens)
        min_intens = np.min(fragments_losses_intens)
        max_intens = np.max(fragments_losses_intens)

        normalized_fragments_intens = [(intens - min_intens) / (max_intens - min_intens) for intens in fragments_intens]
        normalized_losses_intens = [(intens - min_intens) / (max_intens - min_intens) for intens in losses_intens]


        motif_spectrum.peaks = Fragments(mz=fragments_mz, intensities=np.array(normalized_fragments_intens))
        motif_spectrum.losses = Fragments(mz=losses_mz, intensities=np.array(normalized_losses_intens))

        return motif_spectrum

        
        

    motif_word_distribution = lda_model.show_topics(num_topics=-1, num_words=20, formatted=False) # number of words should be replace by the importance threshold or something like that

    motif_spectra = []
    for motif_id, motif_words in motif_word_distribution:
        
        losses = []
        losses_intens = []
        fragments = []
        fragments_intens = []

        for word, importance in motif_words:
            if word.startswith("frag@"):
                fragments.append(float(word[5:]))
                fragments_intens.append(float(importance))
            
            elif word.startswith("loss@"):
                losses.append(float(word[5:]))
                losses_intens.append(float(importance))

        sorted_fragments, sorted_fragmnets_intens = sort_dependent_lists(fragments, fragments_intens)
        sorted_losses, sorted_losses_intens = sort_dependent_lists(losses, losses_intens)

        motif_spectrum = Spectrum(
            mz=sorted_fragments,
            intensities=sorted_fragmnets_intens,
            metadata={
                "id": f"motif_{motif_id}",
                "precursor_mz": max(fragments),
            }
        )

        motif_spectrum.losses = Fragments(
            mz=sorted_losses,
            intensities=sorted_losses_intens,
        )

        normalized_motif_spectrum = normalize_motifs(motif_spectrum) # the normal normalization relies on a intensity dependence of fragments and losses
        motif_spectra.append(normalized_motif_spectrum)

    return motif_spectra


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
