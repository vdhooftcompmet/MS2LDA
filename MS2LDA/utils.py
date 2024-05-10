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