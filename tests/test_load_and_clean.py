import unittest
import sys
import os
from MS2LDA.Preprocessing.load_and_clean import load_mgf
from MS2LDA.Preprocessing.load_and_clean import clean_spectra



class TestLoadAndClean(unittest.TestCase):

    def setUp(self) -> None:
        self.spectra_path = os.path.join(TEST_DIR, "datasets/pos_ache_inhibitors_pesticides.mgf")
        self.empty_spectra_path = os.path.join(TEST_DIR, "datasets/empty_spectra.mgf")

    def test_load_mgf(self):
        self.assertTrue(os.path.exists(self.spectra_path))
        spectra = list(load_mgf(self.spectra_path))
        self.assertGreater(len(spectra), 0)
        self.assertEqual(len(spectra), 705)

    def test_clean_spectra(self):
        spectra = list(load_mgf(self.spectra_path))
        cleaned_spectra = list(clean_spectra(spectra))

        for spectrum in cleaned_spectra:
            self.assertTrue(hasattr(spectrum, 'losses') and spectrum.losses, "add_losses function did not apply correctly")

        self.assertTrue(len(cleaned_spectra) < len(spectra), "The length of clean_spectra should be less than the length of spectra")



if __name__ == '__main__':
    unittest.main()