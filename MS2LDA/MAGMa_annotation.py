import urllib.request
import urllib.parse
import json

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula



def mass2formula(mass=77.04, mfRange='C0-8H0-8O0-2'):
    chemcalcURL = 'https://www.chemcalc.org/chemcalc/em'

    # Define the parameters and send them to Chemcalc
    params = {
        'mfRange': mfRange,
        'monoisotopicMass': mass,
        'integerUnsaturation': False,
    }

    # Encode the parameters
    data = urllib.parse.urlencode(params)
    data = data.encode('ascii')

    # Make the HTTP request
    with urllib.request.urlopen(chemcalcURL, data=data) as response:
        # Read the output and convert it from JSON into a Python dictionary
        jsondata = response.read()
        data = json.loads(jsondata)
        #print(data)
        mf = data["results"][0]["mf"]
        print(mf)

    return mf

masses = [77.04, 105.03, 136.15]
smiles = 'O=C(OC)c1ccccc1'
mf = mass2formula(mass=105.03)

n_atoms = 0
for i in mf:
    if i.isnumeric():
        n_atoms += int(i)


mol = Chem.MolFromSmiles(smiles)

img = Draw.MolToImage(mol)
img.show()

n_bonds = mol.GetNumBonds()
print(n_bonds)

fragments = []
for bond in range(n_bonds):
    with Chem.RWMol(mol) as rwmol:
        b = rwmol.GetBondWithIdx(bond)
        if b.GetBondType() == Chem.BondType.SINGLE:
            rwmol.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        else:
            continue
    fragment = Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False)
    fragments.append(fragment[0])
    fragments.append(fragment[1])

print(fragments)

print("------------------------")
for fragment in fragments:
    formula = CalcMolFormula(fragment)
    print(formula)
    if formula == mf:
        img =Draw.MolToImage(fragment)
        img.show()

    
