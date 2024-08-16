import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Preprocessing')))
from matchms import Spectrum
from matchms.filtering import add_losses
import numpy as np

from generate_corpus import features_to_words
from generate_corpus import combine_features

class TestGenerateCorpus(unittest.TestCase):
    def test_features_to_words(self):
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
    
        spectra = [add_losses(spectrum_1), add_losses(spectrum_2), add_losses(spectrum_3), add_losses(spectrum_4)]
        dataset_frag, dataset_loss = features_to_words(spectra)
        self.assertGreater(len(dataset_frag), 0)
        self.assertGreater(len(dataset_loss), 0)
        self.assertTrue(all(isinstance(item, str) and item.startswith('frag')  for sublist in dataset_frag for item in sublist), 
                        "Not all elements are strings")
        self.assertTrue(all(isinstance(item, str) and item.startswith('loss')  for sublist in dataset_loss for item in sublist), 
                        "Not all elements are strings")
        
    def test_combine_features(self):
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
    
        spectra = [add_losses(spectrum_1), add_losses(spectrum_2), add_losses(spectrum_3), add_losses(spectrum_4)]
        dataset_frag, dataset_loss = features_to_words(spectra)
        dataset_features = combine_features(dataset_frag, dataset_loss)
        print(dataset_features[:10])
        self.assertTrue(all(isinstance(item, str) for sublist in dataset_features for item in sublist), 
                "Not all elements are strings")
        self.assertEqual(len(dataset_features),4)

if __name__ == '__main__':
    unittest.main()