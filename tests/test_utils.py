import hashlib
from types import SimpleNamespace

import numpy as np
from matchms import Fragments, Spectrum

from MS2LDA import utils


def test_create_spectrum_normalizes():
    features = [("frag@100", 1.0), ("loss@10", 0.5)]
    spec = utils.create_spectrum(features, k=1)
    assert np.isclose(spec.intensities.max(), 1.0)  # noqa: S101


def test_match_frags_and_losses():
    motif = utils.create_spectrum([("frag@100", 1.0), ("loss@10", 0.5)], k=0)
    analog = Spectrum(
        mz=np.array([100.0], dtype=float),
        intensities=np.array([1.0], dtype=float),
        metadata={"precursor_mz": 110.0},
    )
    analog._losses = Fragments(  # noqa: SLF001
        mz=np.array([10.0], dtype=float),
        intensities=np.array([1.0], dtype=float),
    )
    frags, losses = utils.match_frags_and_losses(motif, [analog])
    assert frags[0] == {100.0}  # noqa: S101
    assert losses[0] == {10.0}  # noqa: S101


def test_retrieve_spec4doc_roundtrip():
    model = SimpleNamespace(
        docs=[SimpleNamespace(words=[0])],
        vocabs=["frag@100"],
    )
    key = hashlib.md5(b"frag@100").hexdigest()  # noqa: S324
    mapping = {key: Spectrum(mz=np.array([100.0]), intensities=np.array([1.0]))}
    retrieved = utils.retrieve_spec4doc(mapping, model, 0)
    assert np.allclose(retrieved.mz, [100.0])  # noqa: S101

