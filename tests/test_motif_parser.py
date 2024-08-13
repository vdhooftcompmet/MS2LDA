import unittest
import sys
import os
from matchms import Spectrum
from matchms.filtering import add_losses
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../MS2LDA/')))



from motif_parser import store_m2m_file

non_existing_folder_name = "notebooks/test_folder"

class TestMotifParser(unittest.TestCase):
    def test_store_m2m_file(self):
        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                        intensities=np.array([0.7, 0.2, 0.1]),
                        metadata={'id': 'spectrum1',
                                    'precursor_mz': 201.})
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                            intensities=np.array([0.4, 0.2, 0.1]),
                            metadata={'id': 'spectrum2',
                                    'precursor_mz': 233.})
        spectrum_3 = Spectrum(mz=np.array([110, 140, 195.]),
                            intensities=np.array([0.6, 0.2, 0.1]),
                            metadata={'id': 'spectrum3',
                                    'precursor_mz': 214.})
        spectrum_4 = Spectrum(mz=np.array([100, 150, 200.]),
                            intensities=np.array([0.6, 0.1, 0.6]),
                            metadata={'id': 'spectrum4',
                                    'precursor_mz': 265.})
        motif_spectrum = [add_losses(spectrum_1), add_losses(spectrum_2), add_losses(spectrum_3), add_losses(spectrum_4)]
        self.assertTrue(store_m2m_file(motif_spectrum, 3, non_existing_folder_name))

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main()