import json

import numpy as np


def generate_corpusjson_from_tomotopy(model, documents, spectra,
                                      doc_metadata, min_prob_to_keep_beta=1e-3,
                                      min_prob_to_keep_phi=1e-2, min_prob_to_keep_theta=1e-2,
                                      filename=None):
    """
    Generates the corpusjson dictionary from a trained tomotopy LDA model.

    Args:
        model (tomotopy.LDAModel): The trained tomotopy LDA model.
        documents (list of list of str): List of documents, where each document is a list of words.
        spectra (list of matchms.Spectrum): List of Spectrum objects corresponding to the documents.
        doc_metadata (dict): Metadata for each document.
        min_prob_to_keep_beta (float): Minimum probability to include a word in beta (topic-word distributions).
        min_prob_to_keep_phi (float): Minimum probability to include a topic in phi (word-topic distributions per document).
        min_prob_to_keep_theta (float): Minimum probability to include a topic in theta (document-topic distributions).
        filename (str, optional): Path to save the generated corpusjson dictionary. Defaults to None.

    Returns:
        dict: The corpusjson dictionary compatible with the LDA output format.
    """
    # Build word_index
    unique_words = set()
    for doc in documents:
        unique_words.update(doc)
    word_list = sorted(unique_words)
    word_index = {word: idx for idx, word in enumerate(word_list)}
    n_words = len(word_index)

    # Build doc_index
    doc_names = [spec.get("id") for spec in spectra]
    doc_index = {name: idx for idx, name in enumerate(doc_names)}
    n_docs = len(doc_index)

    # Build corpus
    corpus = {}
    for doc_name, doc_words in zip(doc_names, documents):
        word_counts = {}
        for word in doc_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        corpus[doc_name] = word_counts

    # Number of topics
    K = model.k  # Number of topics in the model

    # Alpha (Dirichlet priors for each topic)
    alpha = list(map(float, model.alpha))  # Ensure it's a list of floats

    # Build mapping between our word_index and model's vocabulary
    model_vocab = model.used_vocabs  # List of words in model's vocabulary
    model_vocab_index = {word: idx for idx, word in enumerate(model_vocab)}

    # Map our word indices to model's vocabulary indices
    word_to_model_vocab_idx = {word_index[word]: model_vocab_index[word] for word in word_index}

    # Extract beta_matrix (topic-word distributions)
    beta_matrix = np.zeros((K, n_words))
    for k in range(K):
        # Get word probabilities for topic k
        word_probs = model.get_topic_word_dist(k)
        # Map probabilities to our word order
        for word_idx in word_index.values():
            model_word_idx = word_to_model_vocab_idx[word_idx]
            beta_matrix[k, word_idx] = word_probs[model_word_idx]
    # Now, beta_matrix[k, w] is P(word w | topic k)

    # Extract gamma_matrix (document-topic distributions)
    gamma_matrix = np.zeros((n_docs, K))
    for d_idx, doc in enumerate(model.docs):
        gamma_matrix[d_idx, :] = doc.get_topic_dist()

    phi_matrix = {}
    for d_idx, doc in enumerate(model.docs):
        doc_name = doc_names[d_idx]
        phi_matrix[doc_name] = {}
        word_topic_dict = {}
        # Iterate over words and their assigned topics
        for word_id, topic_id in zip(doc.words, doc.topics):
            word = model.vocabs[word_id]
            if word not in word_topic_dict:
                word_topic_dict[word] = np.zeros(K)
            word_topic_dict[word][topic_id] += 1
        # For each word in the document, compute phi
        for word in corpus[doc_name]:
            phi_values = np.zeros(K)
            if word in word_topic_dict:
                phi_values = word_topic_dict[word]
            # Normalize phi over topics
            total_phi = phi_values.sum()
            if total_phi > 0:
                phi_values = phi_values / total_phi
            else:
                phi_values = np.ones(K) / K  # Assign uniform distribution if zero
            phi_matrix[doc_name][word] = phi_values

    # Build topic_index and topic_metadata
    topic_index = {'motif_{0}'.format(k): k for k in range(K)}
    topic_metadata = {'motif_{0}'.format(k): {'name': 'motif_{0}'.format(k), 'type': 'learnt'} for k in range(K)}

    # Construct features_to_mz_range
    features_to_mz_range = {}
    for word in word_index:
        if word.startswith("frag@") or word.startswith("loss@"):
            try:
                mz_value = float(word.split("@")[1])
                features_to_mz_range[word] = (mz_value, mz_value)
            except ValueError:
                pass
        else:
            pass  # Handle other cases if necessary

    # Use the generate_corpusjson function
    lda_dict = generate_corpusjson(
        corpus=corpus,
        word_index=word_index,
        doc_index=doc_index,
        K=K,
        alpha=alpha,
        beta_matrix=beta_matrix,
        gamma_matrix=gamma_matrix,
        phi_matrix=phi_matrix,
        doc_metadata=doc_metadata,
        topic_index=topic_index,
        topic_metadata=topic_metadata,
        features=features_to_mz_range,
        min_prob_to_keep_beta=min_prob_to_keep_beta,
        min_prob_to_keep_phi=min_prob_to_keep_phi,
        min_prob_to_keep_theta=min_prob_to_keep_theta,
        filename=filename
    )

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
