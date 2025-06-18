import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "generate_corpus",
    Path(__file__).resolve().parents[1]
    / "MS2LDA"
    / "Preprocessing"
    / "generate_corpus.py",
)
_generate_corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_generate_corpus)

features_to_words = _generate_corpus.features_to_words
map_doc2spec = _generate_corpus.map_doc2spec

@dataclass
class DummyPeaks:
    mz: np.ndarray
    intensities: np.ndarray

class DummySpectrum:
    def __init__(self, mz, intensities, precursor_mz):
        self.peaks = DummyPeaks(
            np.asarray(mz, dtype=float),
            np.asarray(intensities, dtype=float),
        )
        self._losses = DummyPeaks(
            precursor_mz - self.peaks.mz,
            self.peaks.intensities,
        )

    @property
    def losses(self):
        return self._losses


def _example_spectra():
    s1 = DummySpectrum([100, 150, 200], [0.7, 0.2, 0.1], 201.0)
    s2 = DummySpectrum([110, 140, 195], [0.6, 0.2, 0.2], 215.0)
    return [s1, s2]


def test_features_to_words_ddacreates_frag_loss():
    spectra = _example_spectra()
    docs = features_to_words(spectra, significant_figures=1, acquisition_type="DDA")
    assert len(docs) == len(spectra)  # noqa: S101
    assert any(tok.startswith("loss@") for tok in docs[0])  # noqa: S101
    assert all(tok.startswith(("frag@", "loss@")) for doc in docs for tok in doc)  # noqa: S101


def test_features_to_words_dia_only_fragments():
    spectra = _example_spectra()
    docs = features_to_words(spectra, significant_figures=1, acquisition_type="DIA")
    assert all(tok.startswith("frag@") for doc in docs for tok in doc)  # noqa: S101


def test_map_doc2spec_roundtrip():
    spectra = _example_spectra()
    docs = features_to_words(spectra, significant_figures=1)
    mapping = map_doc2spec(docs, spectra)
    assert len(mapping) == len(spectra)  # noqa: S101
    for doc, spec in zip(docs, spectra):
        key = hashlib.md5("".join(doc).encode("utf-8")).hexdigest()  # noqa: S324
        assert mapping[key] is spec  # noqa: S101
