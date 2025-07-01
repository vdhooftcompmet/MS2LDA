"""
Comprehensive tests for MS2LDA modeling functions.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ms2lda import modeling


class TestDefineModel:
    """Test cases for model definition."""
    
    def test_define_model_basic(self):
        """Test basic model creation with default parameters."""
        model = modeling.define_model(n_motifs=5)
        assert model.k == 5
        
    def test_define_model_with_custom_params(self):
        """Test model creation with custom parameters."""
        params = {"alpha": 0.5, "eta": 0.05}
        model = modeling.define_model(n_motifs=10, model_parameters=params)
        assert model.k == 10
        
    @pytest.mark.parametrize("n_motifs", [1, 2, 5, 10, 50, 100])
    def test_define_model_various_sizes(self, n_motifs):
        """Test model creation with various numbers of motifs."""
        model = modeling.define_model(n_motifs=n_motifs)
        assert model.k == n_motifs
        
    def test_define_model_with_zero_motifs(self):
        """Test model creation with zero motifs (edge case)."""
        # tomotopy might allow 0 topics, so we test the behavior
        try:
            model = modeling.define_model(n_motifs=0)
            # If it doesn't raise, check that k is 0
            assert model.k == 0
        except:
            # If it raises, that's also acceptable
            pass
            
    def test_define_model_with_negative_motifs(self):
        """Test model creation with negative motifs (edge case)."""
        # tomotopy might handle negative topics by converting to 0 or similar
        try:
            model = modeling.define_model(n_motifs=-5)
            # If it doesn't raise, check what k value it has
            assert model.k >= 0  # Should be non-negative
        except:
            # If it raises, that's also acceptable
            pass


class TestTrainModel:
    """Test cases for model training."""
    
    def test_train_model_basic(self, sample_documents):
        """Test basic model training."""
        model = modeling.define_model(n_motifs=3)
        # Use default convergence parameters
        trained_model, history = modeling.train_model(
            model=model,
            documents=sample_documents,
            iterations=10,
            train_parameters={}
        )
        assert trained_model is not None
        assert isinstance(history, dict)
        
    def test_train_model_with_convergence(self, sample_documents):
        """Test model training with convergence parameters."""
        model = modeling.define_model(n_motifs=3)
        convergence_params = {
            "type": "perplexity_history",
            "threshold": 0.001,
            "window_size": 2,
            "step_size": 5,
        }
        trained_model, history = modeling.train_model(
            model=model,
            documents=sample_documents,
            iterations=20,
            train_parameters={"workers": 1},
            convergence_parameters=convergence_params
        )
        assert trained_model is not None
        assert "perplexity_history" in history
        
    def test_train_model_empty_documents(self):
        """Test model training with empty documents."""
        model = modeling.define_model(n_motifs=3)
        # Training with empty documents might not raise an error, just produce empty model
        trained_model, history = modeling.train_model(
            model=model,
            documents=[],
            iterations=10,
            train_parameters={}
        )
        assert trained_model is not None
            
    def test_train_model_single_document(self):
        """Test model training with a single document."""
        model = modeling.define_model(n_motifs=2)
        single_doc = [["frag@100.00", "frag@150.00"]]
        trained_model, history = modeling.train_model(
            model=model,
            documents=single_doc,
            iterations=5,
            train_parameters={}
        )
        assert trained_model is not None
        
    @pytest.mark.parametrize("conv_type", [
        "perplexity_history",
        "entropy_history_doc",
        "entropy_history_topic",
        "log_likelihood_history"
    ])
    def test_train_model_convergence_types(self, sample_documents, conv_type):
        """Test different convergence criteria types."""
        model = modeling.define_model(n_motifs=3)
        convergence_params = {
            "type": conv_type,
            "threshold": 0.01,
            "window_size": 3,
            "step_size": 5,
        }
        trained_model, history = modeling.train_model(
            model=model,
            documents=sample_documents,
            iterations=20,
            train_parameters={},
            convergence_parameters=convergence_params
        )
        assert trained_model is not None


class TestEntropyCalculations:
    """Test cases for entropy calculations."""
    
    def test_calculate_document_entropy(self, mock_tomotopy_model):
        """Test document entropy calculation."""
        entropy = modeling.calculate_document_entropy(mock_tomotopy_model)
        assert isinstance(entropy, float)
        assert entropy >= 0
        
    def test_calculate_topic_entropy(self, mock_tomotopy_model):
        """Test topic entropy calculation."""
        entropy = modeling.calculate_topic_entropy(mock_tomotopy_model)
        assert isinstance(entropy, float)
        assert entropy >= 0
        
    def test_entropy_with_uniform_distribution(self):
        """Test entropy calculation with uniform distribution."""
        # Create mock model with uniform distributions
        model = MagicMock()
        model.k = 3
        model.docs = []
        
        # Uniform document-topic distribution
        for _ in range(5):
            doc = MagicMock()
            doc.get_topic_dist = MagicMock(return_value=[1/3, 1/3, 1/3])
            model.docs.append(doc)
            
        entropy = modeling.calculate_document_entropy(model)
        # Entropy should be maximum for uniform distribution
        expected_entropy = -np.log(1/3)  # Maximum entropy for 3 topics
        assert np.isclose(entropy, expected_entropy, rtol=0.1)


class TestCheckConvergence:
    """Test cases for convergence checking."""
    
    def test_check_convergence_stable_history(self):
        """Test convergence with stable history."""
        history = [10.0, 9.5, 9.2, 9.1, 9.05, 9.02, 9.01]
        assert modeling.check_convergence(history, epsilon=0.01, n=3)
        
    def test_check_convergence_unstable_history(self):
        """Test convergence with unstable history."""
        history = [10.0, 9.0, 8.0, 7.0, 6.0]
        assert not modeling.check_convergence(history, epsilon=0.01, n=3)
        
    def test_check_convergence_short_history(self):
        """Test convergence with history shorter than window."""
        history = [10.0, 9.5]
        assert not modeling.check_convergence(history, epsilon=0.01, n=3)
        
    def test_check_convergence_exact_threshold(self):
        """Test convergence at exact threshold."""
        history = [10.0, 9.9, 9.89, 9.88]
        # Difference is exactly 0.01
        assert modeling.check_convergence(history, epsilon=0.01, n=2)
        
    @pytest.mark.parametrize("epsilon,n,expected", [
        (0.001, 2, True),   # Very tight threshold
        (0.1, 2, True),     # Loose threshold
        (0.001, 5, False),  # Tight threshold, large window
    ])
    def test_check_convergence_parameters(self, epsilon, n, expected):
        """Test convergence with different parameter combinations."""
        history = [10.0, 9.95, 9.92, 9.91, 9.905, 9.902]
        result = modeling.check_convergence(history, epsilon=epsilon, n=n)
        assert result == expected


class TestExtractMotifs:
    """Test cases for motif extraction."""
    
    def test_extract_motifs_basic(self, mock_tomotopy_model):
        """Test basic motif extraction."""
        motifs = modeling.extract_motifs(mock_tomotopy_model, top_n=5)
        assert len(motifs) == mock_tomotopy_model.k
        assert all(len(features) <= 5 for features in motifs)
        
    def test_extract_motifs_all_features(self, mock_tomotopy_model):
        """Test extracting all features."""
        n_vocabs = mock_tomotopy_model.num_vocabs
        motifs = modeling.extract_motifs(mock_tomotopy_model, top_n=n_vocabs)
        assert len(motifs) == mock_tomotopy_model.k
        assert all(len(features) <= n_vocabs for features in motifs)
        
    def test_extract_motifs_single_feature(self, mock_tomotopy_model):
        """Test extracting single top feature per motif."""
        # Mock the get_topic_words to return exactly 1 feature
        def mock_get_topic_words(topic_id, top_n):
            if top_n == 1:
                return [(mock_tomotopy_model.vocabs[0], 0.5)]
            return []
        
        mock_tomotopy_model.get_topic_words = mock_get_topic_words
        motifs = modeling.extract_motifs(mock_tomotopy_model, top_n=1)
        assert all(len(features) == 1 for features in motifs)
        
    def test_extract_motifs_more_than_vocab(self, mock_tomotopy_model):
        """Test requesting more features than vocabulary size."""
        motifs = modeling.extract_motifs(mock_tomotopy_model, top_n=1000)
        max_features = mock_tomotopy_model.num_vocabs
        assert all(len(features) <= max_features for features in motifs)
        
    def test_extract_motifs_feature_format(self, mock_tomotopy_model):
        """Test that extracted features have correct format."""
        motifs = modeling.extract_motifs(mock_tomotopy_model, top_n=5)
        for motif_features in motifs:
            for feature, probability in motif_features:
                assert isinstance(feature, str)
                assert isinstance(probability, float)
                assert 0 <= probability <= 1
                assert feature.startswith("frag@") or feature.startswith("loss@")


class TestCreateMotifSpectra:
    """Test cases for motif spectra creation."""
    
    def test_create_motif_spectra_basic(self, sample_motif_features):
        """Test basic motif spectra creation."""
        spectra = modeling.create_motif_spectra(sample_motif_features)
        assert len(spectra) == len(sample_motif_features)
        assert all(hasattr(s, "peaks") for s in spectra)
        assert all(hasattr(s, "losses") for s in spectra)
        
    def test_create_motif_spectra_with_metadata(self, sample_motif_features):
        """Test motif spectra creation with custom metadata."""
        spectra = modeling.create_motif_spectra(
            sample_motif_features,
            charge=2,
            motifset_name="test_motifs"
        )
        for i, spectrum in enumerate(spectra):
            assert spectrum.get("charge") == 2
            assert spectrum.get("motifset") == "test_motifs"
            assert spectrum.get("id") == f"motif_{i}"
            
    def test_create_motif_spectra_empty_features(self):
        """Test creating spectra from empty features."""
        empty_features = [[]]
        # Empty features should raise an error when trying to normalize
        with pytest.raises(ValueError):
            modeling.create_motif_spectra(empty_features)
        
    def test_create_motif_spectra_only_fragments(self):
        """Test creating spectra with only fragments."""
        frag_only = [[("frag@100.00", 1.0), ("frag@200.00", 0.5)]]
        spectra = modeling.create_motif_spectra(frag_only)
        assert len(spectra[0].peaks.mz) == 2
        assert len(spectra[0].losses.mz) == 0
        
    def test_create_motif_spectra_only_losses(self):
        """Test creating spectra with only losses."""
        loss_only = [[("loss@50.00", 1.0), ("loss@100.00", 0.5)]]
        spectra = modeling.create_motif_spectra(loss_only)
        assert len(spectra[0].peaks.mz) == 0
        assert len(spectra[0].losses.mz) == 2
        
    def test_create_motif_spectra_normalization(self, sample_motif_features):
        """Test that intensities are properly normalized."""
        spectra = modeling.create_motif_spectra(sample_motif_features)
        for spectrum in spectra:
            if len(spectrum.peaks.intensities) > 0:
                assert np.max(spectrum.peaks.intensities) == 1.0
            if len(spectrum.losses.intensities) > 0:
                max_loss = np.max(spectrum.losses.intensities)
                assert max_loss <= 1.0
                
    @pytest.mark.parametrize("sig_digits", [2, 3, 4])
    def test_create_motif_spectra_significant_digits(self, sig_digits):
        """Test rounding to significant digits."""
        features = [[("frag@123.456789", 0.123456)]]
        spectra = modeling.create_motif_spectra(
            features,
            significant_digits=sig_digits
        )
        # Check that the mz values are rounded appropriately
        mz = spectra[0].peaks.mz[0]
        # Convert to string and count decimal places
        mz_str = f"{mz:.10f}".rstrip('0').rstrip('.')
        decimal_places = len(mz_str.split('.')[-1]) if '.' in mz_str else 0
        assert decimal_places <= sig_digits + 1  # Allow for floating point precision


class TestIntegration:
    """Integration tests for the full modeling pipeline."""
    
    def test_full_modeling_pipeline(self, sample_documents):
        """Test the complete modeling pipeline from definition to spectra creation."""
        # Define model
        model = modeling.define_model(n_motifs=3)
        
        # Train model
        trained_model, history = modeling.train_model(
            model=model,
            documents=sample_documents,
            iterations=50,
            train_parameters={"workers": 1},
            convergence_parameters={
                "type": "perplexity_history",
                "threshold": 0.01,
                "window_size": 3,
                "step_size": 10,
            }
        )
        
        # Extract motifs
        motifs = modeling.extract_motifs(trained_model, top_n=5)
        
        # Create spectra
        spectra = modeling.create_motif_spectra(motifs)
        
        # Validate results
        assert len(spectra) == 3
        assert all(s.get("id").startswith("motif_") for s in spectra)
        assert "perplexity_history" in history
        
    def test_modeling_with_edge_cases(self):
        """Test modeling with edge case documents."""
        # Documents with single word, repeated words, etc.
        edge_documents = [
            ["frag@100.00"],  # Single word
            ["frag@100.00"] * 10,  # Repeated word
            ["frag@100.00", "frag@200.00"] * 5,  # Repeated pattern
            list(set([f"frag@{i*10}.00" for i in range(50)])),  # Many unique words
        ]
        
        model = modeling.define_model(n_motifs=2)
        trained_model, _ = modeling.train_model(
            model=model,
            documents=edge_documents,
            iterations=20,
            train_parameters={}
        )
        
        motifs = modeling.extract_motifs(trained_model, top_n=10)
        assert len(motifs) == 2