#from rdkit import Chem
#from matchms import Spectrum, Fragments
#from matchms.filtering import normalize_intensities
import os, requests
from tqdm import tqdm
import numpy as np
import hashlib
from MS2LDA.Mass2Motif import Mass2Motif


from matchms import set_matchms_logger_level; set_matchms_logger_level("ERROR")


def create_spectrum(motif_k_features, k, frag_tag="frag@", loss_tag="loss@", significant_digits=2, charge=1, motifset="unknown"):
    """creates a spectrum from fragments and losses text representations like frag@123.45 or fragment_67.89

    ARGS:
        motif_k_features (list): for motif number k all features in a unified representation
        k (int): motif id
        frag_tag (str): unified pre-fragments tag
        loss_tag (str): unified pre-loss tag
        significant_digits (int): number of significant digits that should be used for each fragment and loss

    RETURNS:
        spectrum: matchms spectrum object
    """

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
    spectrum = Mass2Motif(
        frag_mz=np.array(sorted_fragments),
        frag_intensities=np.array(normalized_frag_intensities),
        loss_mz=np.array(sorted_losses),
        loss_intensities=np.array(normalized_loss_intensities),
        metadata={
            "id": f"motif_{k}".strip(),
            "charge": charge,
            "ms2accuracy": (1 / (10 ** significant_digits)) / 2 ,
            "motifset": motifset,
        }
    )

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


def retrieve_spec4doc(doc2spec_map, ms2lda, doc_id):
    """
    Retrieves the orginal spectrum for a given document based on a hashmap

    ARGS:
        doc2spec_map (dict): hash-spectrum pairs
        ms2lda (tomotopy object): LDA model
        doc_id (int): id of document

    RETURNS:
        retrieved_spec: matchms spectrum object
    """
    original_doc = ""
    for word_index in ms2lda.docs[doc_id].words:
        original_doc += ms2lda.vocabs[word_index]

    hashed_feature_word = hashlib.md5(original_doc.encode('utf-8')).hexdigest()
    retrieved_spec = doc2spec_map[hashed_feature_word]
    return retrieved_spec



def download_model_and_data(file_urls=[
    "https://zenodo.org/records/15003249/files/150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy?download=1",
    "https://zenodo.org/records/15003249/files/150225_Spec2Vec_pos_CleanedLibraries.model?download=1",
    "https://zenodo.org/records/15003249/files/150225_Spec2Vec_pos_CleanedLibraries.model.syn1neg.npy?download=1",
    "https://zenodo.org/records/15003249/files/150225_Spec2Vec_pos_CleanedLibraries.model.wv.vectors.npy?download=1",
    "https://zenodo.org/records/15003249/files/150225_CombLibraries_spectra.db?download=1",
], mode="positive"):
    """Downloads the Spec2Vec model/data to Add_On/Spec2Vec/model_{mode}_mode.
       Skips files already present and shows a tqdm progress bar per file.
    """


    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(script_directory, f"Add_On/Spec2Vec/model_{mode}_mode")
    os.makedirs(save_directory, exist_ok=True)

    download_count = 0
    skip_count = 0

    for url in file_urls:
        file_name = os.path.basename(url.split("?download")[0])
        file_path = os.path.join(save_directory, file_name)

        if os.path.exists(file_path):
            skip_count += 1
            print(f"File {file_name} already exists, skipping download.")
            continue

        print(f"Downloading {file_name} ...")
        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code} while fetching {url}")

            total = int(response.headers.get("content-length", 0))
            with open(file_path, "wb") as fh, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=file_name,
                initial=0,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:                       # skip keep-alive chunks
                        fh.write(chunk)
                        bar.update(len(chunk))

        download_count += 1
        print(f"Downloaded {file_name} successfully.")

    return f"Done. Downloaded {download_count} files, skipped {skip_count} existing."




