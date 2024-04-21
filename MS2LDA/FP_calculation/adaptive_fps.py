from FP_calculation.rdkit_fps import efficient_array

from rdkit import DataStructs
from rdkit import Chem
from rdkit import RDPaths
import os
import sys

ifg_path = os.path.join(RDPaths.RDContribDir, "IFG")
sys.path.append(ifg_path)
import ifg

from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from collections import Counter




def functional_group_finder(mols):
    """Uses the ERTL algorithm to find functional groups and returns their smiles
    
    ARGS: 
        mols (list(rdkit.mol.objects)): list of molecules converted to rdkit molecule objects

    RETURNS:
        fgs (Counter dict): counter object that counts the appearance of functional groups across all molecules. The same functional group will only be counted once per molecule.
    """
    fgs = Counter()
    for mol in mols:
        fgs_per_mol = ifg.identify_functional_groups(mol)

        seen_fgs = set()

        for fg in fgs_per_mol:
            atoms = fg.atoms

            if atoms not in seen_fgs:
                fgs[atoms] += 1
                seen_fgs.add(atoms)

    return fgs




def scaffold_finder(mols):
    """Uses the Murck Scaffold definition to find the molecular scaffold and returns their smiles
    
    ARGS:
        mols (list(rdkit.mol.objects)): list of molecules converted to rdkit molecule objects

    RETURNS:
        scaffolds (Counter dict): counter object that counts the appearance of Murcko Scaffolds across all molecules
    """
    
    scaffolds = Counter()
    for mol in mols:
        scaffold = GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)

        scaffolds[scaffold_smiles] += 1

    return scaffolds



def greedy_substructure_finder(mols):
    """breaks all single bonds within molecules which are not part of a ring system and returns the resulting substructures
    
    ARGS:
        mols (list(rdkit.mol.objects)): list of molecules converted to rdkit molecule objects

    RETURNS:
        substructures (Counter dict): counter object that counts the appearance of all substructures (as defined above) across all molecules
    """

    substructures = Counter()
    for mol in mols:
        num_bonds = mol.GetNumBonds()
        bonds_index = list(range(num_bonds))
        for bond_index in bonds_index:
            with Chem.RWMol(mol) as rwmol:
                selected_bond = rwmol.GetBondWithIdx(bond_index)
                if selected_bond.GetBondType() == Chem.BondType.SINGLE and selected_bond.IsInRing() == False:
                    rwmol.RemoveBond(selected_bond.GetBeginAtomIdx(), selected_bond.GetEndAtomIdx())
                else:
                    continue
            substructure_pair = Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False) 
            substructure_1 = Chem.MolToSmiles(substructure_pair[0])
            substructure_2 = Chem.MolToSmiles(substructure_pair[1])
    
            substructures[substructure_1] += 1
            substructures[substructure_2] += 1

    return substructures


@efficient_array
def calc_adaptive(mols, smarts):
    """Generate fingerprints for molecules based on the most common fgs and scaffolds SMARTS patterns

    ARGS:
        mols (list): List of RDKit Mol objects.
        smarts (list): List of SMARTS patterns.

    RETURNS:
        adaptive_fps (list): List of fingerprints for each molecule.
    """
    
    smarts = [Chem.MolFromSmiles(sma, sanitize=False) for sma in smarts]
    n_bits = len(smarts)

    adaptive_fps = [DataStructs.ExplicitBitVect(n_bits) for _ in mols]

    for mol_idx, mol in enumerate(mols):
        for i, sma in enumerate(smarts):
            if mol.HasSubstructMatch(sma):
                adaptive_fps[mol_idx].SetBit(i)

    return adaptive_fps, len(mols), n_bits




def generate_fingerprint(mols):
    """uses ERTL algorithm + Murcho Scaffold Finder to generate fingerprints based on common substructures
    
    - functional_group_finder (ERTL algorithm)
    - scaffold_finder (Murcko Scaffold algorithm)
    - generate_fingerprints (Fingerprints based on ERTL algorithm and Murcko scaffold algorithm)

    ARGS:
        mols (list): List of RDKit Mol objects.

    RETURNS:
        adaptive_fps (list): List of RDKit explitcit bit vectors (fingerprints)
    """

    #fgs = functional_group_finder(mols)
    #scaffolds = scaffold_finder(mols)
    #substructures = fgs + scaffolds

    substructures = greedy_substructure_finder(mols)

    n_mols = len(mols)
    min_frequency = int(n_mols * 0.005) # substructure must be present in 0.5% of molecules to be part of the fingerprint

    frequent_substructures = [match[0] for match in substructures.most_common() if match[1] > min_frequency] 

    return frequent_substructures


if __name__ == "__main__":
    pass # have an example here!!!