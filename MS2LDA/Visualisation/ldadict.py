import json
from collections import defaultdict

import numpy as np


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

    # Build phi
    lda_dict["phi"] = {}
    for doc_name in phi_matrix:
        lda_dict["phi"][doc_name] = {}
        for w_str, topic_vec in phi_matrix[doc_name].items():
            t_sub = {}
            for k_idx in range(K):
                p = topic_vec[k_idx]
                if p >= min_prob_to_keep_phi:
                    t_sub[reverse_topic_index[k_idx]] = float(p)
            if len(t_sub) > 0:
                lda_dict["phi"][doc_name][w_str] = t_sub

    # Compute overlap_scores
    overlap_scores = {}
    for doc_name, phi_dict in lda_dict["phi"].items():
        # Instead of only doc topics, let's just do an empty dict & fill as needed
        doc_overlaps = {}
        for w_str, topic_probs in phi_dict.items():
            for t in topic_probs:
                # Skip if not in lda_dict["beta"] for some reason
                if t not in lda_dict["beta"]:
                    continue
                # Add a zero if we haven't seen it
                if t not in doc_overlaps:
                    doc_overlaps[t] = 0.0
                # Multiply
                doc_overlaps[t] += lda_dict["beta"][t].get(w_str, 0.0) * topic_probs[t]
        # Save
        overlap_scores[doc_name] = {}
        for t in doc_overlaps:
            overlap_scores[doc_name][t] = float(doc_overlaps[t])
    lda_dict["overlap_scores"] = overlap_scores

    # 17) Optionally save
    if filename:
        with open(filename, "w") as f:
            json.dump(lda_dict, f, indent=2)

    return lda_dict