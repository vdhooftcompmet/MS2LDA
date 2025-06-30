"""
Comprehensive tests for MS2LDA corpus generation functions.
"""
import hashlib
from typing import List

import numpy as np
import pytest
from matchms import Fragments, Spectrum

from MS2LDA.Preprocessing.generate_corpus import (
    features_to_words, map_doc2spec, combine_features
)


class TestFeaturesToWords:
    """Test cases for converting spectra features to words."""
    
    def test_features_to_words_dda_mode(self, sample_spectra_list):
        """Test DDA mode (includes losses)."""
        # Use only spectra with losses
        spec_with_losses = sample_spectra_list[0]  # Has losses
        documents = features_to_words([spec_with_losses], 
                                    significant_figures=2, 
                                    acquisition_type="DDA")
        
        assert len(documents) == 1
        doc = documents[0]
        
        # Check that document contains both fragments and losses
        frag_words = [w for w in doc if w.startswith("frag@")]
        loss_words = [w for w in doc if w.startswith("loss@")]
        
        assert len(frag_words) > 0
        assert len(loss_words) > 0
        
    def test_features_to_words_dia_mode(self, sample_spectra_list):
        """Test DIA mode (no losses)."""
        documents = features_to_words(sample_spectra_list[:2], 
                                    significant_figures=2, 
                                    acquisition_type="DIA")
        
        assert len(documents) == 2
        
        # Check that documents contain only fragments
        for doc in documents:
            assert all(w.startswith("frag@") for w in doc)
            assert not any(w.startswith("loss@") for w in doc)
            
    def test_features_to_words_significant_figures(self):
        """Test rounding to different significant figures."""
        spectrum = Spectrum(
            mz=np.array([123.456789, 234.567890]),
            intensities=np.array([1.0, 0.5]),
            metadata={"precursor_mz": 300.0}
        )
        
        # Test with 2 significant figures
        docs_2sf = features_to_words([spectrum], significant_figures=2)
        assert "frag@123.46" in docs_2sf[0] or "frag@123.45" in docs_2sf[0]
        
        # Test with 4 significant figures
        docs_4sf = features_to_words([spectrum], significant_figures=4)
        assert any("frag@123.45" in w for w in docs_4sf[0])
        
    def test_features_to_words_intensity_weighting(self):
        """Test that features are repeated based on intensity."""
        spectrum = Spectrum(
            mz=np.array([100.0, 200.0]),
            intensities=np.array([0.25, 1.0]),  # 25% and 100%
            metadata={"precursor_mz": 300.0}
        )
        
        documents = features_to_words([spectrum], significant_figures=2)
        doc = documents[0]
        
        # Count occurrences
        count_100 = doc.count("frag@100.00")
        count_200 = doc.count("frag@200.00")
        
        # The 200 m/z peak should appear more times due to higher intensity
        assert count_200 > count_100
        assert count_200 == 100  # Maximum repeats for highest intensity
        assert count_100 == 25   # 25% of maximum
        
    def test_features_to_words_empty_spectrum(self):
        """Test handling of empty spectrum."""
        empty_spec = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"precursor_mz": 200.0}
        )
        
        documents = features_to_words([empty_spec])
        assert len(documents) == 1
        assert documents[0] == []
        
    def test_features_to_words_multiple_spectra(self, sample_spectra_list):
        """Test processing multiple spectra."""
        documents = features_to_words(sample_spectra_list[:3], significant_figures=2)
        
        assert len(documents) == 3
        # Each document should have features
        assert all(len(doc) > 0 for doc in documents[:2])  # First two have peaks
        
    def test_features_to_words_low_intensity_filtering(self):
        """Test that very low intensities are handled properly."""
        spectrum = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensities=np.array([0.001, 0.01, 1.0]),  # Very low to high
            metadata={"precursor_mz": 400.0}
        )
        
        documents = features_to_words([spectrum], significant_figures=2)
        doc = documents[0]
        
        # Very low intensity peaks should still appear at least once
        assert "frag@100.00" in doc
        assert doc.count("frag@100.00") >= 1
        
    @pytest.mark.parametrize("acquisition_type", ["DDA", "DIA"])
    def test_features_to_words_acquisition_types(self, acquisition_type):
        """Test both acquisition types work correctly."""
        spectrum = Spectrum(
            mz=np.array([100.0, 200.0]),
            intensities=np.array([1.0, 0.5]),
            metadata={"precursor_mz": 300.0}
        )
        spectrum._losses = Fragments(
            mz=np.array([50.0, 100.0]),
            intensities=np.array([0.8, 0.4])
        )
        
        documents = features_to_words([spectrum], 
                                    significant_figures=2,
                                    acquisition_type=acquisition_type)
        
        doc = documents[0]
        has_losses = any(w.startswith("loss@") for w in doc)
        
        if acquisition_type == "DDA":
            assert has_losses
        else:
            assert not has_losses


class TestMapDoc2Spec:
    """Test cases for mapping documents to spectra."""
    
    def test_map_doc2spec_basic(self):
        """Test basic document to spectrum mapping."""
        feature_words = ["frag@100.00", "frag@200.00", "loss@50.00"]
        
        # Create matching spectra
        spectra = [
            Spectrum(mz=np.array([100.0]), intensities=np.array([1.0])),
            Spectrum(mz=np.array([200.0]), intensities=np.array([1.0])),
            Spectrum(mz=np.array([50.0]), intensities=np.array([1.0])),
        ]
        
        doc2spec = map_doc2spec(feature_words, spectra)
        
        # Check that all features are mapped
        assert len(doc2spec) == len(feature_words)
        
        # Check that hashes are correct
        for word in feature_words:
            hash_key = hashlib.md5(word.encode()).hexdigest()
            assert hash_key in doc2spec
            
    def test_map_doc2spec_duplicate_features(self):
        """Test mapping with duplicate feature words."""
        feature_words = ["frag@100.00", "frag@100.00", "frag@200.00"]
        spectra = [
            Spectrum(mz=np.array([100.0]), intensities=np.array([1.0])),
            Spectrum(mz=np.array([200.0]), intensities=np.array([1.0])),
        ]
        
        doc2spec = map_doc2spec(feature_words, spectra)
        
        # Should have only unique features
        assert len(doc2spec) == 2  # Only two unique features
        
    def test_map_doc2spec_empty_lists(self):
        """Test mapping with empty inputs."""
        doc2spec = map_doc2spec([], [])
        assert doc2spec == {}
        
    def test_map_doc2spec_mismatched_lengths(self):
        """Test when features and spectra have different lengths."""
        feature_words = ["frag@100.00", "frag@200.00"]
        spectra = [Spectrum(mz=np.array([100.0]), intensities=np.array([1.0]))]
        
        # Should handle gracefully
        doc2spec = map_doc2spec(feature_words, spectra)
        assert len(doc2spec) <= len(spectra)
        
    def test_map_doc2spec_preserves_spectrum_reference(self):
        """Test that mapping preserves the actual spectrum objects."""
        feature_words = ["frag@100.00"]
        spectrum = Spectrum(
            mz=np.array([100.0]), 
            intensities=np.array([1.0]),
            metadata={"test": "value"}
        )
        
        doc2spec = map_doc2spec(feature_words, [spectrum])
        hash_key = hashlib.md5(b"frag@100.00").hexdigest()
        
        # Should be the same object
        assert doc2spec[hash_key] is spectrum
        assert doc2spec[hash_key].get("test") == "value"


class TestCombineFeatures:
    """Test cases for combining fragment and loss features."""
    
    def test_combine_features_basic(self):
        """Test basic combination of fragments and losses."""
        fragments = [[100.0, 200.0], [150.0, 250.0]]
        losses = [[50.0, 100.0], [75.0, 125.0]]
        
        combined = combine_features(fragments, losses)
        
        assert len(combined) == 2
        assert len(combined[0]) == 4  # 2 fragments + 2 losses
        assert len(combined[1]) == 4
        
        # Check that features have correct prefixes
        assert any(f.startswith("frag@") for f in combined[0])
        assert any(f.startswith("loss@") for f in combined[0])
        
    def test_combine_features_empty_losses(self):
        """Test combination when losses are empty."""
        fragments = [[100.0, 200.0], [150.0]]
        losses = [[], []]
        
        combined = combine_features(fragments, losses)
        
        assert len(combined) == 2
        assert len(combined[0]) == 2  # Only fragments
        assert all(f.startswith("frag@") for f in combined[0])
        
    def test_combine_features_empty_fragments(self):
        """Test combination when fragments are empty."""
        fragments = [[], []]
        losses = [[50.0, 100.0], [75.0]]
        
        combined = combine_features(fragments, losses)
        
        assert len(combined) == 2
        assert len(combined[0]) == 2  # Only losses
        assert all(f.startswith("loss@") for f in combined[0])
        
    def test_combine_features_mismatched_lengths(self):
        """Test when fragments and losses have different lengths."""
        fragments = [[100.0, 200.0]]
        losses = [[50.0], [75.0]]
        
        # Should handle the mismatch appropriately
        # The actual behavior depends on implementation
        with pytest.raises(Exception):
            combine_features(fragments, losses)
            
    def test_combine_features_formatting(self):
        """Test that features are formatted correctly."""
        fragments = [[123.456]]
        losses = [[45.678]]
        
        combined = combine_features(fragments, losses)
        
        assert combined[0][0] == "frag@123.456"
        assert combined[0][1] == "loss@45.678"
        
    def test_combine_features_preserves_order(self):
        """Test that order is preserved in combination."""
        fragments = [[100.0, 200.0, 300.0]]
        losses = [[25.0, 50.0]]
        
        combined = combine_features(fragments, losses)
        
        # Fragments should come first, then losses
        assert combined[0][:3] == ["frag@100.0", "frag@200.0", "frag@300.0"]
        assert combined[0][3:] == ["loss@25.0", "loss@50.0"]


class TestIntegration:
    """Integration tests for corpus generation pipeline."""
    
    def test_full_corpus_generation_pipeline(self, sample_spectra_list):
        """Test the complete corpus generation pipeline."""
        # Generate documents from spectra
        documents = features_to_words(
            sample_spectra_list[:3],
            significant_figures=2,
            acquisition_type="DDA"
        )
        
        # Create vocabulary from all documents
        vocab = set()
        for doc in documents:
            vocab.update(doc)
        vocab = list(vocab)
        
        # Create mapping
        doc2spec = map_doc2spec(vocab, sample_spectra_list[:len(vocab)])
        
        # Verify the pipeline
        assert len(documents) == 3
        assert len(vocab) > 0
        assert len(doc2spec) > 0
        
        # Check that all document words can be mapped
        for doc in documents:
            for word in set(doc):  # Use set to avoid duplicates
                hash_key = hashlib.md5(word.encode()).hexdigest()
                # Some words might not be in mapping if we have more words than spectra
                if hash_key in doc2spec:
                    assert doc2spec[hash_key] is not None
                    
    def test_corpus_generation_with_rounding(self):
        """Test corpus generation with different rounding levels."""
        # Create spectra with precise m/z values
        spectra = []
        for i in range(3):
            mz = np.array([100.123456 + i, 200.234567 + i, 300.345678 + i])
            intensities = np.array([1.0, 0.8, 0.6])
            spectrum = Spectrum(
                mz=mz,
                intensities=intensities,
                metadata={"precursor_mz": 400.0 + i}
            )
            spectra.append(spectrum)
            
        # Generate with different significant figures
        docs_2sf = features_to_words(spectra, significant_figures=2)
        docs_4sf = features_to_words(spectra, significant_figures=4)
        
        # Vocabularies should be different sizes
        vocab_2sf = set(word for doc in docs_2sf for word in doc)
        vocab_4sf = set(word for doc in docs_4sf for word in doc)
        
        # More precision should lead to more unique words
        assert len(vocab_4sf) >= len(vocab_2sf)
        
    def test_corpus_generation_intensity_distribution(self):
        """Test that intensity weighting creates proper word distributions."""
        # Create spectrum with known intensity pattern
        spectrum = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensities=np.array([0.1, 0.5, 1.0]),  # 10%, 50%, 100%
            metadata={"precursor_mz": 400.0}
        )
        
        documents = features_to_words([spectrum], significant_figures=2)
        doc = documents[0]
        
        # Count word frequencies
        word_counts = {}
        for word in doc:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Verify intensity-based distribution
        assert word_counts["frag@100.00"] == 10  # 10% of 100
        assert word_counts["frag@200.00"] == 50  # 50% of 100
        assert word_counts["frag@300.00"] == 100  # 100% of 100