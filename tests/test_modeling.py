import unittest
import sys
import os
import tomotopy as tp

from MS2LDA.modeling import define_model
from MS2LDA.modeling import train_model
from MS2LDA.modeling import extract_motifs
from MS2LDA.modeling import create_motif_spectra
from MS2LDA.modeling import define_model

from MS2LDA.Visualisation import create_network

class TestModeling(unittest.TestCase):
    def setUp(self):
        self.n_motifs = 5
        self.documents =  [
        ["frag@24.33", "frag@34.23", "loss@18.01", "loss@18.01"],
        ["frag@24.33", "frag@65.87", "loss@121.30", "frag@24.33"],
        ["frag@74.08", "frag@34.23", "loss@18.01", "loss@18.01", "loss@18.01"],
        ["frag@74.08", "frag@121.30", "loss@34.01"]
        ]
        self.documents_emtpy =  [
        ]  
        self.model= define_model(self.n_motifs)

    def test_define_model_default_parameters(self):
        self.assertIsInstance(self.model, tp.LDAModel)
        self.assertEqual(self.model.k, self.n_motifs)
    
    def test_define_model_invalid_parameters(self):
        invalid_params = {
            'invalid_param': 6
        }
        with self.assertRaises(TypeError):
            define_model(self.n_motifs, **invalid_params)

    def test_documents_added_to_model(self):
        model = train_model(self.model, self.documents, iterations=10)
        self.assertEqual(len(model.docs), len(self.documents), "Not all documents were added to the model")

    def test_model_training(self):
        model = train_model(self.model, self.documents, iterations=10)
        self.assertTrue(model.num_words > 0, "Model training did not increase the number of words in the model")

    def test_extract_motifs(self):
        model = train_model(self.model, self.documents, iterations=10)
        motifs = extract_motifs(model, top_n=3)
        self.assertIsInstance(motifs, list)
        self.assertEqual(len(motifs), 5)

    def test_create_motif_spectra(self):
        model = train_model(self.model, self.documents, iterations=10)
        motifs = extract_motifs(model, top_n=3)
        motif_spectra = create_motif_spectra(motifs)
        self.assertEqual(len(list(motif_spectra)), 5)


if __name__ == '__main__':
    unittest.main()
