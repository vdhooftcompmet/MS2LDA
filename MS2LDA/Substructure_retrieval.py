from FP_calculation.adaptive_fps import generate_fingerprint
from rdkit import Chem
from itertools import chain
import numpy as np


def retrieve_substructures(fp_per_motifs, smiles_per_motifs):
    """ retrieves the SMARTS patterns from the adaptive fingerprint based on the motif fingerprints alignments

    ARGS:
        fp_per_motifs (list or arrays): List of np.arrays, where every array is a motif fingerprints
        smiles_per_motifs (list of lists): list of SMILES associated with the same motif

    RETURNS:
        substructure_matches (list of lists): retrieved substructures from adaptive fingerprint where the motif fingerprint had 1 as a bit
    """
    
    all_mols = list(chain(*smiles_per_motifs))
    frequent_substructures = generate_fingerprint([Chem.MolFromSmiles(mol) for mol in all_mols])

    substructure_matches = [list() for i in range(len(fp_per_motifs))]  
    for i, fp_per_motif in enumerate(fp_per_motifs):
        substructures_per_motif_indices = np.where(fp_per_motif == 1)[0]
        for idx in substructures_per_motif_indices:
            substructure_match = frequent_substructures[idx]
            substructure_matches[i].append(substructure_match)

    return substructure_matches


if __name__ == "__main__":
    pass