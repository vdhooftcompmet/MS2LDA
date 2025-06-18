from MS2LDA import modeling


def _docs():
    return [["frag@100", "loss@10"], ["frag@100", "frag@110"]]


def test_define_model_sets_topics():
    m = modeling.define_model(2)
    assert m.k == 2  # noqa: S101,PLR2004


def test_check_convergence():
    hist = [1.0, 0.9, 0.89, 0.888]
    assert modeling.check_convergence(hist, epsilon=0.02, n=2)  # noqa: S101


def test_train_extract_and_create():
    m = modeling.define_model(2)
    m, history = modeling.train_model(
        m,
        _docs(),
        iterations=10,
        convergence_parameters={
            "type": "perplexity_history",
            "threshold": 0.001,
            "window_size": 1,
            "step_size": 5,
        },
    )
    motifs = modeling.extract_motifs(m, top_n=2)
    spectra = modeling.create_motif_spectra(motifs)
    assert len(spectra) == 2  # noqa: S101,PLR2004
    assert all(s.peaks.mz.size > 0 for s in spectra)  # noqa: S101

