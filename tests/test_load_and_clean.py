import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Preprocessing')))
from load_and_clean import load_mgf
from load_and_clean import clean_spectra

TEST_DIR = os.path.dirname(os.path.abspath(__file__))



class TestLoadAndClean(unittest.TestCase):

    def setUp(self) -> None:
        self.spectra_path = os.path.join(TEST_DIR, "test_data/pos_ache_inhibitors_pesticides.mgf")
        self.empty_spectra_path = os.path.join(TEST_DIR, "test_data/empty_spectra.mgf")

    def test_load_mgf(self):
        spectra = list(load_mgf(self.spectra_path))
        self.assertGreater(len(spectra), 0)

    def test_clean_spectra(self):
        spectra = list(load_mgf(self.spectra_path))
        cleaned_spectra = list(clean_spectra(spectra))
        self.assertTrue(len(cleaned_spectra) < len(spectra), "The length of clean_spectra should be less than the length of spectra")

if __name__ == '__main__':
    unittest.main()