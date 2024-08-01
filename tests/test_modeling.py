import unittest
import sys
import os
import tomotopy as tp


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../MS2LDA/')))
from modeling import define_model
from modeling import train_model


class TestModeling(unittest.TestCase):
    def test_define_model_default_parameters(self):
        n_motifs = 5
        model = define_model(n_motifs)
        self.assertIsInstance(model, tp.LDAModel)
        self.assertEqual(model.k, n_motifs)
    
    def test_define_model_invalid_parameters(self):
        n_motifs = 5
        invalid_params = {
            'invalid_param': 6
        }
        with self.assertRaises(TypeError):
            define_model(n_motifs, **invalid_params)

    def test_train_model(self):
        documents = [
        ["frag@24.33", "frag@34.23", "loss@18.01", "loss@18.01"],
        ["frag@24.33", "frag@65.87", "loss@121.30", "frag@24.33"],
        ["frag@74.08", "frag@34.23", "loss@18.01", "loss@18.01", "loss@18.01"],
        ["frag@74.08", "frag@121.30", "loss@34.01"]
        ]
        model = define_model(2) 
        model = train_model(model, documents)
        self.assertEqual(self.model.add_doc.call_count, len(self.documents))
        #for doc in self.documents:
#            self.model.add_doc.assert_any_call(doc)

        


if __name__ == '__main__':
    unittest.main()
