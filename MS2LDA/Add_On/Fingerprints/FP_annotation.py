from MS2LDA.Add_On.Fingerprints.FP_calculation.rdkit_fps import *
#from Add_On.Fingerprints.FP_calculation.minhash_fps import *
from MS2LDA.Add_On.Fingerprints.FP_calculation.adaptive_fps import generate_fingerprint
from itertools import chain
from rdkit.Chem import MolFromSmiles


def mols2fps(smiles_per_motif, selected_fp_type, smarts=None): 
    """calculates the selected fingerprint for a given list of rdkit mol objects
    
    ARGS:
        mols_per_motif (list(rdkit.mol.objects)): list of rdkit.mol.objects associated with one motif
        fp_type (str): a name of fingerprint that will be calculated

    RETURNS:
        fps (numpy.array): a multi-array numpy error containing all fingerprints
    """

    fps_type = ["adaptive", "pubchem", "daylight", "kr", "lingo", "estate", "dfs", "asp", "lstar", "rad2d", 
                "ph2", "ph3", "ecfp", "avalon", "tt", "maccs", "fcfp", "ap", "rdkit", "map4", "mhfp"]
    
    selected_fp_type = selected_fp_type.lower()

    # self developed dynamic fingerprint
    if selected_fp_type == "adaptive":
        from FP_calculation.adaptive_fps import calc_adaptive
        mols_per_motif = smiles2mols(smiles_per_motif)
        fps = calc_adaptive(mols_per_motif, smarts)

    # cdk based fingerprints
    elif selected_fp_type in fps_type[1:6]:
        from FP_calculation.cdk_fps import calc_PUBCHEM, calc_DAYLIGHT, calc_KR, calc_LINGO, calc_ESTATE

        if selected_fp_type == "pubchem":
            fps = calc_PUBCHEM(smiles_per_motif)
        elif selected_fp_type == "daylight":
            fps = calc_DAYLIGHT(smiles_per_motif)
        elif selected_fp_type == "kr":
            fps = calc_KR(smiles_per_motif)
        elif selected_fp_type == "lingo":
            fps = calc_LINGO(smiles_per_motif)
        elif selected_fp_type == "estate":
            fps = calc_ESTATE(smiles_per_motif)
    
    # jmap based fingerprints
    elif selected_fp_type in fps_type[6:10]:
        from FP_calculation.jmap_fps import calc_DFS, calc_ASP, calc_LSTAR, calc_RAD2D, calc_PH2, calc_PH3
        
        if selected_fp_type == "dfs":
            fps = calc_DFS(smiles_per_motif)
        elif selected_fp_type == "asp":
            fps = calc_ASP(smiles_per_motif)
        elif selected_fp_type == "lstar":
            fps = calc_LSTAR(smiles_per_motif)
        elif selected_fp_type == "rad2d":
            fps = calc_RAD2D(smiles_per_motif)
        elif selected_fp_type == "ph2":
            fps = calc_PH2(smiles_per_motif)
        elif selected_fp_type == "ph3":
            fps = calc_PH3(smiles_per_motif)

    # rdkit based fingerprints
    elif selected_fp_type in fps_type[10:19]:
        mols_per_motif = smiles2mols(smiles_per_motif)

        if selected_fp_type == "ecfp":
            fps = calc_ECFP(mols_per_motif)
        elif selected_fp_type == "avalon":
            fps = calc_AVALON(mols_per_motif)
        elif selected_fp_type == "maccs":
            fps = calc_MACCS(mols_per_motif)
        elif selected_fp_type == "fcfp":
            fps = calc_FCFP(mols_per_motif)
        elif selected_fp_type == "ap":
            fps = calc_AP(mols_per_motif)
        elif selected_fp_type == "rdkit":
            fps = calc_RDKIT(mols_per_motif)
      
        
    # minhash based fingerprints
    elif selected_fp_type in fps_type[19:]:
        from FP_calculation.minhash_fps import calc_MAP4, calc_MHFP
        mols_per_motif = smiles2mols(smiles_per_motif)
        if selected_fp_type == "map4":
            fps = calc_MAP4(mols_per_motif)
        elif selected_fp_type == "mhfp":
            fps = calc_MHFP(mols_per_motif)

    else:
        raise Exception (f"One of the following fingerprint types need to be selected: {fps_type}")

    return fps


def smiles2mols(smiles):
    """converts SMILES to rdkit mol objects
    
    ARGS:
        smiles (list): list of SMILES strings
        
    RETURNS:
        mols (list): list of rdkit mol objects
    """
    mols = []

    for smi in smiles:
        mol = MolFromSmiles(smi)
        if mol:
            mols.append(mol)

    return mols
        
        
    
def scale_fps(fps_per_motif): 
    """calculates the percentage of the presents of every fingerprint bit in a motif
    
    ARGS:
        fps_per_motif (pandas.dataframe): a dataframe (rows are molecules and columns are fingerprint bit) for all molecular fingerprints

    RETURNS:
        scaled_fps (np.array): a fingerprint array with values between 0 and 1 showing the presents of substructures within a motif
    
    """
    n_fps_per_motif = len(fps_per_motif)
    combined_fps = sum(fps_per_motif)

    scaled_fps = combined_fps/n_fps_per_motif
    # error with Nan if cluster is empty
    return scaled_fps


def fps2motif(scaled_fps, threshold):
    """overlaps fingerprints of compounds allocated to the same topic/motif
    
    ARGS:
        scaled_fps (np.array): a fingerprint array with values between 0 and 1 showing the presents of substructures within a motif
        threshold (float; 0 > x <= 1): number that defines if a bit in the fingerprint with be set to zero (below threshold) or to one (above threshold)

    RETURNS:
        scaled_fps (np.array): could also be called motif_fps, because it represents the most common fingerprint bits in a motif (bits above the threshold)
    """
    # above_threshold_indices = np.where(scaled_fps > threshold)[0] # useful for retrieval, but maybe you can do it in another function
    # maybe you can use the masking also for the retrieveal of SMARTS patterns

    lower_as_threshold = scaled_fps < threshold
    higher_as_threshold = scaled_fps >= threshold

    scaled_fps[lower_as_threshold] = 0
    scaled_fps[higher_as_threshold] = 1

    return scaled_fps






def annotate_motifs(smiles_per_motifs, fp_type="maccs", threshold=0.8): # can be simplyfied
    """runs all the scripts to generate a selected fingerprint for a motif

    - smiles2mol: convert smiles to mol objects
    - mols2fps: convert mol objects to selected fingerprint
    - scale_fps: check present of fingerprints bits across motif
    - fps2motif: make the motif fingerprint binary based on given threshold
    - fps2smarts: retrieve SMARTS for found motif fingerprint bits

    - motifs2tanimotoScore: calculated motif similarity based on motif fingerprints using tanimoto similarity


    ARGS:
        smiles_per_motifs: list(list(str)): SMILES for every motif in a different list
        fp_type (CDK_pywrapper.fp_type.object): a object that represents a type of fingerprint that will be calculated
        threshold (float; 0 > x <= 1): number that defines if a bit in the fingerprint with be set to zero (below threshold) or to one (above threshold)

    RETURNS:
        fps_motifs (list(list(np.array))): binary fingerprint for motifs, based on given threshold for including/excluding bits on their presents in a motif
        smarts_per_motifs (list(list(rdkit.mol.object))): mol object for the present bits in fps_motifs (SMARTS pattern)
        motifs_similarities (list): tanimoto score for every motif combination
    """
    fps_motifs = []
    all_mols = list(chain(*smiles_per_motifs))
    smarts = generate_fingerprint([Chem.MolFromSmiles(mol) for mol in all_mols])


    for smiles_per_motif in smiles_per_motifs:
        fps_per_motif = mols2fps(smiles_per_motif, fp_type, smarts)
        #print(fps_per_motif)
        scaled_fps = scale_fps(fps_per_motif)
        #print(scaled_fps)
        fps_motif = fps2motif(scaled_fps, threshold)
        #print(fps_motif)
        fps_motifs.append(fps_motif)

    return fps_motifs



if __name__ == "__main__":

    smiles_per_motifs = [["O=C(C)Oc1ccccc1C(=O)O", "COC(=O)C1CCC(C1)C(=O)O"]]
    fps_motif = annotate_motifs(smiles_per_motifs, fp_type="pubchem")
