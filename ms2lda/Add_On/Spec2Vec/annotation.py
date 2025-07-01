import numpy as np
import faiss
from tqdm import tqdm
import sqlite3
import pickle
from rdkit.Chem import MolFromSmiles, MolToInchi, InchiToInchiKey
from gensim.models import Word2Vec
from spec2vec import Spec2Vec
from ms2lda.Mass2MotifDocument import Mass2MotifDocument


def load_s2v_model(path_model):
    """Load Spec2Vec model."""
    s2v = Word2Vec.load(path_model)
    s2v_similarity = Spec2Vec(
        model=s2v, intensity_weighting_power=0.5, allowed_missing_percentage=100.0
    )
    return s2v_similarity


def calc_embeddings(s2v_similarity, spectra):
    """Calculate spectral embeddings for a list of spectra."""
    spectral_embeddings = [
        np.array(s2v_similarity._calculate_embedding(Mass2MotifDocument(spectrum)))
        for spectrum in spectra
    ]
    return np.array(spectral_embeddings)


def normalize_embeddings(embeddings):
    """Normalize embeddings safely to avoid overflow and NaN issues."""
    embeddings = np.nan_to_num(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # norms[norms == 0] = 1  # Prevent division by zero
    embeddings /= norms

    return embeddings


def load_spectrum_from_db(db_path, spectrum_id):
    """Load a spectrum from the SQLite database by ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT smiles, spectrum FROM spectra WHERE id = ?
    """,
        (int(spectrum_id),),
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        smiles, spectrum_blob = result
        spectrum = pickle.loads(spectrum_blob)
        return {"smiles": smiles, "spectrum": spectrum}
    else:
        return None


def calc_similarity_faiss(embeddings_A, embeddings_B, k=None):
    """Calculate cosine similarity using Faiss, ensuring proper normalization."""
    # Ensure embeddings are float32
    embeddings_A = embeddings_A.astype(np.float32)
    embeddings_B = embeddings_B.astype(np.float32)

    # Normalize embeddings (to ensure cosine similarity calculation)
    embeddings_A = normalize_embeddings(embeddings_A)
    embeddings_B = normalize_embeddings(embeddings_B)

    # Create a Faiss index with Inner Product (IP)
    index = faiss.IndexFlatIP(embeddings_B.shape[1])
    index.add(embeddings_B)

    # If k is not set, use all references
    if k is None:
        k = embeddings_B.shape[0]

    # Perform the search
    similarities, indices = index.search(embeddings_A, k)

    return similarities, indices


def get_library_matches_per_motif(
    similarities, indices, db_path, motif_number=0, top_n=10, unique_mols=True
):
    """Return similarity scores, SMILES, and spectra for top n matches for one motif."""
    top_smiles = []
    top_spectra = []
    top_scores = []
    top_inchikeys = []

    i = 0  # Index for iterating over ranked molecules
    while (
        len(top_smiles) < top_n and i < indices.shape[1]
    ):  # Ensure we collect 10 molecules
        score = similarities[motif_number, i]
        spectrum_id = indices[motif_number, i]
        spectrum_data = load_spectrum_from_db(db_path, spectrum_id)
        if spectrum_data is None:
            i += 1
            continue  # Skip missing data

        smi = spectrum_data["smiles"]
        spectrum = spectrum_data["spectrum"]

        if unique_mols:
            mol = MolFromSmiles(smi)
            inchi = MolToInchi(mol)
            inchikey = InchiToInchiKey(inchi)
            if inchikey in top_inchikeys:
                i += 1
                continue  # Skip duplicates
            else:
                top_inchikeys.append(inchikey)

        # Add the molecule to the results
        top_scores.append(score)
        top_smiles.append(smi)
        top_spectra.append(spectrum)

        i += 1  # Move to the next candidate

    return top_smiles, top_spectra, top_scores


def get_library_matches(similarities, indices, db_path, top_n=10, unique_mols=True):
    """Return similarity scores, SMILES, and spectra for top n matches for all motifs."""
    num_motifs = similarities.shape[0]
    library_matches = [
        get_library_matches_per_motif(
            similarities,
            indices,
            db_path,
            motif_number=i,
            top_n=top_n,
            unique_mols=unique_mols,
        )
        for i in tqdm(range(num_motifs))
    ]
    return library_matches


# Example usage
if __name__ == "__main__":
    from matchms.filtering import add_losses
    from matchms import Spectrum

    # Generate dummy spectra
    spectrum_1 = Spectrum(
        mz=np.array([100, 150, 200.0]),
        intensities=np.array([0.7, 0.2, 0.1]),
        metadata={"id": "spectrum1", "precursor_mz": 201.0},
    )
    spectrum_2 = Spectrum(
        mz=np.array([100, 140, 190.0]),
        intensities=np.array([0.4, 0.2, 0.1]),
        metadata={"id": "spectrum2", "precursor_mz": 233.0},
    )
    spectrum_3 = Spectrum(
        mz=np.array([110, 140, 195.0]),
        intensities=np.array([0.6, 0.2, 0.1]),
        metadata={"id": "spectrum3", "precursor_mz": 214.0},
    )
    spectrum_4 = Spectrum(
        mz=np.array([100, 150, 200.0]),
        intensities=np.array([0.6, 0.1, 0.6]),
        metadata={"id": "spectrum4", "precursor_mz": 265.0},
    )
    motif_spectra = [
        add_losses(spectrum_1),
        add_losses(spectrum_2),
        add_losses(spectrum_3),
        add_losses(spectrum_4),
    ]

    # Load data
    s2v_similarity = load_s2v_model(
        r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\WP1\MS2LDA\ms2lda\Add_On\Spec2Vec\model_positive_mode\020724_Spec2Vec_pos_CleanedLibraries.model"
    )

    embeddings = np.load(
        r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\WP1\MS2LDA\ms2lda\Add_On\Spec2Vec\model_positive_mode_fast\020724_Spec2Vec_pos_embeddings.npy"
    )
    embeddings = embeddings.reshape((426280, 300))

    motif_embeddings = calc_embeddings(s2v_similarity, motif_spectra)
    similarities, indices = calc_similarity_faiss(motif_embeddings, embeddings)

    # Get matches for all motifs
    matching_settings = {
        "similarities": similarities,
        "indices": indices,
        "db_path": r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\WP1\MS2LDA\ms2lda\Add_On\Spec2Vec\model_positive_mode_fast\XXXXXX_cleaned_library_spectra.db",
        "top_n": 10,
        "unique_mols": True,
    }
    library_matches = get_library_matches(**matching_settings)
    print(library_matches)
