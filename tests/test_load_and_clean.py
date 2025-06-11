import numpy as np
from matchms import Spectrum

from MS2LDA.Preprocessing.load_and_clean import clean_spectra


def _dummy_spectra():
    return [
        Spectrum(
            mz=np.array([100.0, 150.0, 200.0]),
            intensities=np.array([0.1, 0.2, 0.7]),
            metadata={"precursor_mz": 250.0, "id": "orig1"},
        ),
        Spectrum(
            mz=np.array([110.0, 140.0, 195.0]),
            intensities=np.array([0.2, 0.4, 0.4]),
            metadata={"precursor_mz": 260.0, "id": "orig2"},
        ),
    ]


def test_clean_spectra_assigns_new_ids():
    spectra = clean_spectra(iter(_dummy_spectra()))
    assert [s.get("id") for s in spectra] == ["spec_0", "spec_1"]  # noqa: S101


def test_clean_spectra_respects_min_frags():
    spectra = clean_spectra(iter(_dummy_spectra()), {"min_frags": 4})
    assert spectra == []  # noqa: S101

