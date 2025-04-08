import gzip
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
from matchms import Spectrum


def generate_corpusjson_from_tomotopy(model, documents, spectra, doc_metadata,
                                      min_prob_to_keep_beta=1e-3,
                                      min_prob_to_keep_phi=1e-2,
                                      min_prob_to_keep_theta=1e-2,
                                      filename=None):
    """
    Generates lda_dict in the similar format as in the previous MS2LDA app.
    """

    # Build doc names
    if spectra is not None and len(spectra) == len(documents):
        doc_names = [spec.get("id") for spec in spectra]
    else:
        doc_names = list(doc_metadata.keys())

    # Gather all unique tokens from `documents`:
    unique_words = set()
    for doc in documents:
        unique_words.update(doc)
    word_list = sorted(unique_words)
    word_index = {word: idx for idx, word in enumerate(word_list)}
    n_words = len(word_index)

    # doc_index
    doc_index = {name: i for i, name in enumerate(doc_names)}
    n_docs = len(doc_index)

    # corpus
    corpus = {}
    for doc_name, doc_words in zip(doc_names, documents):
        word_counts = {}
        for w in doc_words:
            word_counts[w] = word_counts.get(w, 0) + 1
        corpus[doc_name] = word_counts

    # Number of topics and alpha
    K = model.k
    alpha = list(map(float, model.alpha))

    # Map each token => model’s vocab index if present
    model_vocab = model.used_vocabs
    model_vocab_index = {w: i for i, w in enumerate(model_vocab)}
    word_to_model_idx = {}
    for w in word_index:
        if w in model_vocab_index:
            word_to_model_idx[word_index[w]] = model_vocab_index[w]

    # Construct beta_matrix (K x n_words)
    beta_matrix = np.zeros((K, n_words), dtype=float)
    for k_idx in range(K):
        word_probs = model.get_topic_word_dist(k_idx)
        for w_i in range(n_words):
            if w_i in word_to_model_idx:
                model_wi = word_to_model_idx[w_i]
                beta_matrix[k_idx, w_i] = word_probs[model_wi]
            else:
                beta_matrix[k_idx, w_i] = 0.0

    # Construct gamma_matrix (doc-topic distribution)
    gamma_matrix = np.zeros((n_docs, K), dtype=float)
    for d_idx, doc in enumerate(model.docs):
        gamma_matrix[d_idx, :] = doc.get_topic_dist()

    # Construct phi_matrix
    phi_matrix = {}
    for d_idx, doc in enumerate(model.docs):
        doc_name = doc_names[d_idx]
        phi_matrix[doc_name] = defaultdict(lambda: np.zeros(K, dtype=float))
        for (word_id, topic_id) in zip(doc.words, doc.topics):
            w_str = model.vocabs[word_id]
            phi_matrix[doc_name][w_str][topic_id] += 1.0
        # Normalize each word’s topic distribution
        for w_str, topic_vec in phi_matrix[doc_name].items():
            total = topic_vec.sum()
            if total > 0.0:
                phi_matrix[doc_name][w_str] = topic_vec / total

    # Build topic_index + metadata
    topic_index = {f"motif_{k}": k for k in range(K)}
    topic_metadata = {f"motif_{k}": {"name": f"motif_{k}", "type": "learnt"} for k in range(K)}

    # For convenience, extract “features” if they look like "frag@X" or "loss@X".
    features_to_mz = {}
    for w in word_list:
        if w.startswith("frag@") or w.startswith("loss@"):
            try:
                val = float(w.split("@")[1])
                features_to_mz[w] = (val, val)
            except:
                pass

    # Prepare the final dictionary
    lda_dict = {
        "corpus": corpus,
        "word_index": word_index,
        "doc_index": doc_index,
        "K": K,
        "alpha": alpha,
        "beta": {},
        "doc_metadata": doc_metadata,
        "topic_index": topic_index,
        "topic_metadata": topic_metadata,
        "features": features_to_mz,
        "gamma": [list(map(float, row)) for row in gamma_matrix],
    }

    reverse_word_index = {v: k for k, v in word_index.items()}
    reverse_topic_index = {v: k for k, v in topic_index.items()}

    # Fill in beta
    for k_idx in range(K):
        t_name = reverse_topic_index[k_idx]  # e.g. "motif_49"
        t_dict = {}
        for w_i in range(n_words):
            val = beta_matrix[k_idx, w_i]
            if val > min_prob_to_keep_beta:
                w_str = reverse_word_index[w_i]
                t_dict[w_str] = float(val)
        lda_dict["beta"][t_name] = t_dict

    # Build theta from gamma
    e_theta = gamma_matrix / gamma_matrix.sum(axis=1, keepdims=True)
    lda_dict["theta"] = {}
    for d_idx in range(n_docs):
        doc_name = doc_names[d_idx]
        row = {}
        for k_idx in range(K):
            val = e_theta[d_idx, k_idx]
            if val > min_prob_to_keep_theta:
                row[reverse_topic_index[k_idx]] = float(val)
        lda_dict["theta"][doc_name] = row

    # Compute overlap_scores
    overlap_scores = {}
    for doc_name, phi_dict in phi_matrix.items():
        doc_overlaps = {}
        for w_str, topic_vec in phi_dict.items():

            # For each topic that has p >= min_prob_to_keep_phi, check if we pass that threshold
            for k_idx, p in enumerate(topic_vec):
                if p < min_prob_to_keep_phi:
                    continue

                t = reverse_topic_index[k_idx]
                if t not in lda_dict["beta"]:
                    continue
                # add a zero if not in doc_overlaps
                if t not in doc_overlaps:
                    doc_overlaps[t] = 0.0
                # multiply
                doc_overlaps[t] += lda_dict["beta"][t].get(w_str, 0.0) * p

        overlap_scores[doc_name] = {}
        for t in doc_overlaps:
            overlap_scores[doc_name][t] = float(doc_overlaps[t])
    lda_dict["overlap_scores"] = overlap_scores

    # 17) Optionally save
    if filename:
        with open(filename, "w") as f:
            json.dump(lda_dict, f, indent=2)

    return lda_dict


def save_visualization_data(
        trained_ms2lda,
        cleaned_spectra,
        optimized_motifs,
        doc2spec_map,
        output_folder,
        filename="ms2lda_viz.json",
        min_prob_to_keep_beta=1e-3,
        min_prob_to_keep_phi=1e-2,
        min_prob_to_keep_theta=1e-2,
        run_parameters=None
):
    """
    Creates the final data structure needed by the MS2LDA UI
    (clustered_smiles_data, optimized_motifs_data, lda_dict, spectra_data)
    and saves it to JSON in `output_folder/filename`.

    Args:
        trained_ms2lda (tomotopy.LDAModel): the trained LDA model in memory
        cleaned_spectra (list of Spectrum): final cleaned spectra
        optimized_motifs (list of Spectrum): annotated + optimized motifs
        doc2spec_map (dict): doc-hash to original Spectrum map
        output_folder (str): folder path for saving the .json
        filename (str): name of the saved JSON (default "ms2lda_viz.json")
        min_prob_to_keep_beta (float): threshold for storing topic-word distribution in beta
        min_prob_to_keep_phi (float): threshold for storing word-topic distribution in phi (used for overlap calc)
        min_prob_to_keep_theta (float): threshold for doc-topic distribution in theta

    Returns:
        None
    """

    # 1) Build "documents" & doc_metadata from the model in memory
    documents = []
    for doc_idx, doc in enumerate(trained_ms2lda.docs):
        tokens = [trained_ms2lda.vocabs[word_id] for word_id in doc.words]
        documents.append(tokens)

    doc_metadata = {}
    for i, doc in enumerate(trained_ms2lda.docs):
        doc_name = f"spec_{i}"
        doc_metadata[doc_name] = {"placeholder": f"Doc {i}"}

    # 2) Generate the main LDA dictionary (which does *not* store phi in final dict)
    lda_dict = generate_corpusjson_from_tomotopy(
        model=trained_ms2lda,
        documents=documents,
        spectra=None,  # skip re-cleaning for token consistency
        doc_metadata=doc_metadata,
        min_prob_to_keep_beta=min_prob_to_keep_beta,
        min_prob_to_keep_phi=min_prob_to_keep_phi,
        min_prob_to_keep_theta=min_prob_to_keep_theta,
        filename=None,  # we won't save it here
    )

    # 3) Convert each cleaned spectrum to a dictionary
    def spectrum_to_dict(s):
        dct = {
            "metadata": s.metadata.copy(),
            "mz": [float(mz) for mz in s.peaks.mz],
            "intensities": [float(i) for i in s.peaks.intensities],
        }
        if s.losses:
            dct["metadata"]["losses"] = [
                {"loss_mz": float(mz_), "loss_intensity": float(int_)}
                for mz_, int_ in zip(s.losses.mz, s.losses.intensities)
            ]
        return dct

    # 4) Convert each optimized motif to a dictionary
    def motif_to_dict(m):
        dct = {
            "metadata": m.metadata.copy(),
            "mz": [float(x) for x in m.peaks.mz],
            "intensities": [float(x) for x in m.peaks.intensities],
        }
        if m.losses:
            dct["metadata"]["losses"] = [
                {"loss_mz": float(mz_), "loss_intensity": float(int_)}
                for mz_, int_ in zip(m.losses.mz, m.losses.intensities)
            ]
        return dct

    optimized_motifs_data = [motif_to_dict(m) for m in optimized_motifs]
    spectra_data = [spectrum_to_dict(s) for s in cleaned_spectra]

    # 5) Gather short_annotation from each optimized motif for "clustered_smiles_data"
    clustered_smiles_data = []
    for motif in optimized_motifs:
        ann = motif.get("auto_annotation")
        if isinstance(ann, list):
            clustered_smiles_data.append(ann)
        elif ann is None:
            clustered_smiles_data.append([])
        else:
            # single string
            clustered_smiles_data.append([ann])

    # 6) Build the final dictionary
        # build doc→spec index from doc2spec_map
        # Map Spectrum object -> integer index
        spectrum_to_idx = {spec: i for i, spec in enumerate(cleaned_spectra)}
        doc_to_spec_index = {}

        for d_idx, doc in enumerate(trained_ms2lda.docs):
            words = [trained_ms2lda.vocabs[w_id] for w_id in doc.words]
            doc_text = "".join(words)
            hashed = hashlib.md5(doc_text.encode("utf-8")).hexdigest()
            if hashed in doc2spec_map:
                real_spec = doc2spec_map[hashed]
                doc_to_spec_index[str(d_idx)] = spectrum_to_idx.get(real_spec, -1)
            else:
                doc_to_spec_index[str(d_idx)] = -1 # hash is not found

        lda_dict["doc_to_spec_index"] = doc_to_spec_index

    final_data = {
        "clustered_smiles_data": clustered_smiles_data,
        "optimized_motifs_data": optimized_motifs_data,
        "lda_dict": lda_dict,
        "spectra_data": spectra_data,
        "run_parameters": run_parameters if run_parameters else {},
    }

    # 7) Compress & Save as .json.gz
    os.makedirs(output_folder, exist_ok=True)
    outpath = os.path.join(output_folder, filename + ".gz")  # e.g. ms2lda_viz.json.gz
    with gzip.open(outpath, "wt", encoding="utf-8") as gz_file:
        json.dump(final_data, gz_file, indent=2)

    print(f"Visualization data saved (gzipped) to: {outpath}")

