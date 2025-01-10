import json
from collections import defaultdict

import numpy as np
def generate_corpusjson_from_tomotopy(model, documents, spectra, doc_metadata,
                                      min_prob_to_keep_beta=1e-3,
                                      min_prob_to_keep_phi=1e-2,
                                      min_prob_to_keep_theta=1e-2,
                                      filename=None):
    """
    A robust version that prevents KeyError if phi refers to a motif not
    in doc's top topics or if 'motif_XX' is missing from beta.
    """
    import numpy as np
    from collections import defaultdict
    import json

    # 1) Build doc names
    if spectra is not None and len(spectra) == len(documents):
        doc_names = [spec.get("id") for spec in spectra]
    else:
        doc_names = list(doc_metadata.keys())

    # 2) Gather all unique tokens from `documents`:
    unique_words = set()
    for doc in documents:
        unique_words.update(doc)
    word_list = sorted(unique_words)
    word_index = {word: idx for idx, word in enumerate(word_list)}
    n_words = len(word_index)

    # 3) doc_index
    doc_index = {name: i for i, name in enumerate(doc_names)}
    n_docs = len(doc_index)

    # 4) corpus
    corpus = {}
    for doc_name, doc_words in zip(doc_names, documents):
        word_counts = {}
        for w in doc_words:
            word_counts[w] = word_counts.get(w, 0) + 1
        corpus[doc_name] = word_counts

    # 5) Number of topics and alpha
    K = model.k
    alpha = list(map(float, model.alpha))

    # 6) Map each token => model’s vocab index if present
    model_vocab = model.used_vocabs
    model_vocab_index = {w: i for i, w in enumerate(model_vocab)}
    word_to_model_idx = {}
    for w in word_index:
        if w in model_vocab_index:
            word_to_model_idx[word_index[w]] = model_vocab_index[w]

    # 7) Construct beta_matrix (K x n_words)
    beta_matrix = np.zeros((K, n_words), dtype=float)
    for k_idx in range(K):
        word_probs = model.get_topic_word_dist(k_idx)
        for w_i in range(n_words):
            if w_i in word_to_model_idx:
                model_wi = word_to_model_idx[w_i]
                beta_matrix[k_idx, w_i] = word_probs[model_wi]
            else:
                beta_matrix[k_idx, w_i] = 0.0

    # 8) gamma_matrix (doc-topic distribution)
    gamma_matrix = np.zeros((n_docs, K), dtype=float)
    for d_idx, doc in enumerate(model.docs):
        gamma_matrix[d_idx, :] = doc.get_topic_dist()

    # 9) Build phi_matrix
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

    # 10) Build topic_index + metadata
    topic_index = {f"motif_{k}": k for k in range(K)}
    topic_metadata = {f"motif_{k}": {"name": f"motif_{k}", "type": "learnt"} for k in range(K)}

    # 11) For convenience, extract “features” if they look like "frag@X" or "loss@X".
    features_to_mz = {}
    for w in word_list:
        if w.startswith("frag@") or w.startswith("loss@"):
            try:
                val = float(w.split("@")[1])
                features_to_mz[w] = (val, val)
            except:
                pass

    # 12) Prepare the final dictionary
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

    # 13) Fill in beta
    for k_idx in range(K):
        t_name = reverse_topic_index[k_idx]  # e.g. "motif_49"
        t_dict = {}
        for w_i in range(n_words):
            val = beta_matrix[k_idx, w_i]
            if val > min_prob_to_keep_beta:
                w_str = reverse_word_index[w_i]
                t_dict[w_str] = float(val)
        lda_dict["beta"][t_name] = t_dict

    # 14) Build theta from gamma
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

    # 15) Build phi
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

    # 16) Compute overlap_scores – now robust to “missing” topic IDs
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




def generate_corpusjson(corpus, word_index, doc_index, K, alpha, beta_matrix, gamma_matrix,
                        phi_matrix, doc_metadata, topic_index, topic_metadata, features,
                        min_prob_to_keep_beta=1e-3, min_prob_to_keep_phi=1e-2,
                        min_prob_to_keep_theta=1e-2, filename=None):
    """
    Generates a corpusjson dictionary compatible with the LDA output format, including overlap scores.

    Args:
        corpus (dict):
            Dictionary representing the corpus with document and feature structure.
            Format: {doc_name: {word: count/intensity, ...}, ...}
        word_index (dict):
            Dictionary mapping each feature (word) to a unique integer index.
            Format: {word: index, ...}
        doc_index (dict):
            Dictionary mapping each document to a unique integer index.
            Format: {doc_name: index, ...}
        K (int):
            Number of topics.
        alpha (array-like):
            Dirichlet hyperparameters for each topic (length K).
        beta_matrix (numpy.ndarray):
            Topic-word distribution matrix from LDA.
            Shape: (K, number of words)
        gamma_matrix (numpy.ndarray):
            Document-topic distribution matrix from LDA.
            Shape: (number of documents, K)
        phi_matrix (dict):
            Nested dictionary of word-topic probabilities per document.
            Format: {doc_name: {word: numpy.ndarray of size K}, ...}
        doc_metadata (dict):
            Metadata for each document.
            Format: {doc_name: {metadata_key: value, ...}, ...}
        topic_index (dict):
            Dictionary mapping each topic to a unique index.
            Format: {topic_name: index, ...}
        topic_metadata (dict):
            Metadata about each topic.
            Format: {topic_name: {metadata_key: value, ...}, ...}
        features (dict):
            Dictionary mapping each feature (word) to its m/z range.
            Format: {word: (min_mz, max_mz), ...}
        min_prob_to_keep_beta (float, optional):
            Minimum probability threshold to include a word in the beta (topic-word) distributions.
            Defaults to 1e-3.
        min_prob_to_keep_phi (float, optional):
            Minimum probability threshold to include a word in the phi (word-topic) distributions.
            Defaults to 1e-2.
        min_prob_to_keep_theta (float, optional):
            Minimum probability threshold to include a topic in the theta (document-topic) distributions.
            Defaults to 1e-2.
        filename (str, optional):
            Optional path to save the generated corpusjson dictionary as a JSON file.
            If None, the dictionary is not saved to disk.

    Returns:
        dict: A dictionary representing the corpusjson structure compatible with LDA outputs.
              This includes the corpus, indices, topic distributions, and overlap scores.
    """

    # Initialize the corpusjson dictionary
    lda_dict = {
        "corpus": corpus,
        "word_index": word_index,
        "doc_index": doc_index,
        "K": K,
        "alpha": [float(a) for a in alpha],  # Ensure alpha is a list of floats
        "beta": {},
        "doc_metadata": doc_metadata,
        "topic_index": topic_index,
        "topic_metadata": topic_metadata,
        "features": features,
        "gamma": [list(map(float, gamma_row)) for gamma_row in gamma_matrix]
    }

    # Reverse lookup for word and document names based on indices
    reverse_word_index = {v: k for k, v in word_index.items()}
    reverse_doc_index = {v: k for k, v in doc_index.items()}
    reverse_topic_index = {v: k for k, v in topic_index.items()}

    # Build beta: word probabilities per topic
    for k in range(K):
        topic_name = reverse_topic_index.get(k, f'motif_{k}')  # Ensure topic naming consistency
        lda_dict["beta"][topic_name] = {
            reverse_word_index[w]: float(beta_matrix[k, w])
            for w in np.where(beta_matrix[k, :] > min_prob_to_keep_beta)[0]
        }

    # Build theta: topic distribution per document
    lda_dict["theta"] = {}
    e_theta = gamma_matrix / gamma_matrix.sum(axis=1)[:, None]  # Normalize gamma to get theta
    for d in range(len(e_theta)):
        doc_name = reverse_doc_index[d]
        lda_dict["theta"][doc_name] = {
            reverse_topic_index.get(k, f'motif_{k}'): float(e_theta[d, k])
            for k in np.where(e_theta[d, :] > min_prob_to_keep_theta)[0]
        }

    # Build phi: word-topic distribution per document
    lda_dict["phi"] = {}
    for doc_name in corpus:
        lda_dict["phi"][doc_name] = {}
        for word in corpus[doc_name]:
            lda_dict["phi"][doc_name][word] = {
                reverse_topic_index.get(k, f'motif_{k}'): float(phi_matrix[doc_name][word][k])
                for k in np.where(phi_matrix[doc_name][word] >= min_prob_to_keep_phi)[0]
            }

    # Compute overlap scores
    lda_dict["overlap_scores"] = compute_overlap_scores_from_dict(lda_dict)

    # Save to file if filename is provided
    if filename:
        with open(filename, "w") as f:
            json.dump(lda_dict, f)

    return lda_dict


def compute_overlap_scores_from_dict(lda_dictionary):
    """
    Compute the overlap scores for the LDA model in dictionary format.
    Overlap scores measure the contribution of each topic to each document based on word probabilities.

    Args:
        lda_dictionary (dict):
            The dictionary containing LDA outputs, including 'phi', 'theta', and 'beta'.

    Returns:
        dict: A nested dictionary of overlap scores.
              Format: {doc_name: {topic_name: overlap_score, ...}, ...}
    """
    overlap_scores = {}
    for doc, phi in lda_dictionary["phi"].items():
        motifs = lda_dictionary["theta"][doc].keys()
        doc_overlaps = {m: 0.0 for m in motifs}
        for word, probs in phi.items():
            for m in motifs:
                if word in lda_dictionary["beta"][m] and m in probs:
                    doc_overlaps[m] += lda_dictionary["beta"][m][word] * probs[m]
        overlap_scores[doc] = {}
        for m in doc_overlaps:
            overlap_scores[doc][m] = doc_overlaps[m]
    return overlap_scores
