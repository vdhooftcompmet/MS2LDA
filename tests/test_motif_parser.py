import unittest
import pickle
import os
import re 
import shutil  # to remove the folder after testing
from MS2LDA.motif_parser import store_m2m_file
from MS2LDA.motif_parser import store_m2m_folder
from MS2LDA.motif_parser import load_m2m_file

file_path = "MS2LDA/tests/motif_spectra.pkl"
existing_folder_name = "MS2LDA/tests"
new_folder_name = "MS2LDA/tests/test_folder_new"

class TestMotifParser(unittest.TestCase):
    def setUp(self):
        with open(file_path, 'rb') as file:
            self.motif_spectra = pickle.load(file)
            self.motif_spectra_empty= []

    def test_exist_m2m_file(self):
        self.assertIsInstance(self.motif_spectra, list)
        self.assertTrue(len(self.motif_spectra) > 0)
        self.assertEqual(len(self.motif_spectra_empty), 0)

    def test_store_m2m_file(self):
        store_m2m_file(self.motif_spectra[0], 0, existing_folder_name)
        self.assertTrue(os.path.exists(f"{existing_folder_name}/tests_motif_0.m2m"))

    def test_store_m2m_folder(self): # Ensure the folder does not already exist
        if os.path.exists(new_folder_name):
            shutil.rmtree(new_folder_name)

        result = store_m2m_folder(self.motif_spectra, new_folder_name)
        self.assertTrue(os.path.exists(new_folder_name))
        self.assertEqual(len(os.listdir(new_folder_name)), len(self.motif_spectra))

    def test_store_m2m_folder_exists(self): # Test for the case where the folder already exists
        os.makedirs(new_folder_name, exist_ok=True)
        with self.assertRaises(Exception) as context:
            store_m2m_folder(self.motif_spectra, new_folder_name)

        self.assertTrue('Folder already exists' in str(context.exception))
        shutil.rmtree(new_folder_name)


    def test_header(self):
        file_path_correct = "MS2LDA/tests/tests_motif_0.m2m"  
        with open(file_path_correct, 'r') as file:
            lines = file.readlines()
        
        self.assertEqual(lines[0].strip(), "#MS2ACCURACY 0.005")
        self.assertEqual(lines[1].strip(), "#MOTIFSET MS2LDA/tests")
        self.assertEqual(lines[2].strip(), "#CHARGE 1")
        self.assertEqual(lines[3].strip(), "#NAME tests_motif_0")
        self.assertEqual(lines[4].strip(), "#ANNOTATION None")
        self.assertEqual(lines[5].strip(), "#SHORT_ANNOTATION None")
        self.assertEqual(lines[6].strip(), "#COMMENT None")
    
    def test_fragments(self):
        file_path_correct = "MS2LDA/tests/tests_motif_0.m2m"  
        with open(file_path_correct, 'r') as file:
            lines = file.readlines()
        
        fragment_pattern = re.compile(r'^fragment_\d+\.\d+,\d+\.\d+$')
        loss_pattern = re.compile(r'^loss_\d+\.\d+,\d+\.\d+$')

        for line in lines[7:]:
            line = line.strip()
            if line.startswith('fragment'):
                self.assertIsNotNone(fragment_pattern.match(line))
            elif line.startswith('loss'):
                self.assertIsNotNone(loss_pattern.match(line))
            else:
                self.fail(f"Unexpected line format: {line}")

    def load_m2m_file(self):
        # Call the function with the mock file
        motif_spectrum = load_m2m_file(mock_file_path)

        # Assertions to check if the returned motif_spectrum has correct attributes
        self.assertEqual(motif_spectrum.name, "001")  # Check name parsing
        self.assertEqual(motif_spectrum.get("short_annotation"), "Example short annotation")
        self.assertEqual(motif_spectrum.get("ms2accuracy"), "0.01")
        self.assertEqual(motif_spectrum.get("motifset"), "example_motifset")
        self.assertEqual(motif_spectrum.get("charge"), "2")
        self.assertEqual(motif_spectrum.get("annotation"), "Example full annotation")

        # Check that the features were parsed correctly
        expected_features = [
            ["frag", "100", "1", "some data"],
            ["loss", "200", "0.5", "some other data"]
        ]
        self.assertEqual(motif_spectrum.features, expected_features)

        
    
if __name__ == '__main__':
    unittest.main()
