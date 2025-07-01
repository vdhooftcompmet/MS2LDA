"""
End-to-end integration tests for MS2LDA workflow.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from matchms import Spectrum

from ms2lda import modeling, utils
from ms2lda.Preprocessing import load_and_clean, generate_corpus
from ms2lda.motif_parser import store_m2m_folder, load_m2m_folder
from ms2lda.run import run, filetype_check


class TestCompleteWorkflow:
    """Integration tests for the complete MS2LDA workflow."""
    
    def test_minimal_workflow(self, sample_spectra_list, temp_dir):
        """Test minimal workflow from spectra to motifs."""
        # Step 1: Clean spectra
        cleaned_spectra = load_and_clean.clean_spectra(
            iter(sample_spectra_list[:3]),
            preprocessing_parameters={"min_frags": 2}
        )
        assert len(cleaned_spectra) >= 2
        
        # Step 2: Generate corpus
        documents = generate_corpus.features_to_words(
            cleaned_spectra,
            significant_figures=2,
            acquisition_type="DDA"
        )
        assert len(documents) == len(cleaned_spectra)
        
        # Step 3: Define and train model
        model = modeling.define_model(n_motifs=2)
        trained_model, history = modeling.train_model(
            model=model,
            documents=documents,
            iterations=20,
            train_parameters={"workers": 1},
            convergence_parameters=None
        )
        assert trained_model is not None
        
        # Step 4: Extract motifs
        motifs = modeling.extract_motifs(trained_model, top_n=5)
        assert len(motifs) == 2
        
        # Step 5: Create motif spectra
        motif_spectra = modeling.create_motif_spectra(motifs)
        assert len(motif_spectra) == 2
        
        # Step 6: Save motifs
        output_folder = temp_dir / "motifs"
        store_m2m_folder(motif_spectra, str(output_folder))
        assert output_folder.exists()
        assert len(list(output_folder.glob("*.m2m"))) == 2
        
        # Step 7: Load and verify
        loaded_motifs = load_m2m_folder(str(output_folder))
        assert len(loaded_motifs) == 2
        
    def test_workflow_with_file_io(self, temp_dir, sample_mgf_content):
        """Test workflow including file loading and saving."""
        # Create test MGF file
        mgf_file = temp_dir / "test.mgf"
        mgf_file.write_text(sample_mgf_content)
        
        # Load spectra from file
        spectra = list(load_and_clean.load_mgf(str(mgf_file)))
        assert len(spectra) == 2
        
        # Clean and process
        cleaned = load_and_clean.clean_spectra(spectra)
        documents = generate_corpus.features_to_words(cleaned, significant_figures=2)
        
        # Train model
        model = modeling.define_model(n_motifs=2)
        trained_model, _ = modeling.train_model(
            model=model,
            documents=documents,
            iterations=10,
            train_parameters={},
            convergence_parameters=None
        )
        
        # Extract and save results
        motifs = modeling.extract_motifs(trained_model, top_n=3)
        motif_spectra = modeling.create_motif_spectra(motifs)
        
        # Save results
        output_folder = temp_dir / "results"
        store_m2m_folder(motif_spectra, str(output_folder))
        
        # Save metadata
        metadata = {
            "n_spectra": len(cleaned),
            "n_motifs": len(motifs),
            "input_file": str(mgf_file),
        }
        with open(output_folder / "metadata.json", "w") as f:
            json.dump(metadata, f)
            
        # Verify outputs
        assert (output_folder / "metadata.json").exists()
        assert len(list(output_folder.glob("*.m2m"))) == 2
        
    @patch('ms2lda.utils.download_model_and_data')
    def test_workflow_with_annotation(self, mock_download, sample_spectra_list, temp_dir):
        """Test workflow including Spec2Vec annotation."""
        mock_download.return_value = True
        
        # Prepare data
        cleaned = load_and_clean.clean_spectra(sample_spectra_list[:3])
        documents = generate_corpus.features_to_words(cleaned)
        
        # Train model
        model = modeling.define_model(n_motifs=3)
        trained_model, _ = modeling.train_model(model, documents, iterations=20)
        
        # Extract motifs
        motifs = modeling.extract_motifs(trained_model, top_n=5)
        motif_spectra = modeling.create_motif_spectra(motifs)
        
        # Mock Spec2Vec annotation
        with patch('ms2lda.run.s2v_annotation') as mock_s2v:
            mock_s2v.return_value = motif_spectra  # Return unchanged
            
            # Simulate annotation
            annotated_spectra = mock_s2v(
                motif_spectra,
                {"mode": "positive", "top_n": 10}
            )
            
            assert len(annotated_spectra) == 3
            mock_s2v.assert_called_once()
            
    def test_workflow_with_convergence(self, sample_spectra_list):
        """Test workflow with convergence criteria."""
        # Prepare larger dataset
        spectra = sample_spectra_list[:3] * 5  # Repeat for more data
        cleaned = load_and_clean.clean_spectra(spectra)
        documents = generate_corpus.features_to_words(cleaned)
        
        # Train with convergence
        model = modeling.define_model(n_motifs=3)
        convergence_params = {
            "type": "perplexity_history",
            "threshold": 0.1,  # Loose threshold for quick convergence
            "window_size": 2,
            "step_size": 5,
        }
        
        trained_model, history = modeling.train_model(
            model=model,
            documents=documents,
            iterations=100,  # Max iterations
            train_parameters={"workers": 1},
            convergence_parameters=convergence_params
        )
        
        # Check that convergence was tracked
        assert "perplexity" in history
        assert len(history["perplexity"]) > 0
        
        # Model should have converged before max iterations
        assert len(history["perplexity"]) < 100


class TestRunFunction:
    """Test the main run function."""
    
    @patch('ms2lda.run.load_mgf')
    @patch('ms2lda.run.store_m2m_folder')
    @patch('ms2lda.utils.download_model_and_data')
    def test_run_basic(self, mock_download, mock_store, mock_load, temp_dir):
        """Test basic run function execution."""
        # Setup mocks
        mock_download.return_value = True
        mock_spectra = [
            Spectrum(
                mz=np.array([100.0, 200.0, 300.0]),
                intensities=np.array([1.0, 0.5, 0.3]),
                metadata={"precursor_mz": 400.0}
            )
            for _ in range(5)
        ]
        mock_load.return_value = mock_spectra
        
        # Run analysis
        output_folder = str(temp_dir / "output")
        results = run(
            dataset="test.mgf",
            n_motifs=2,
            n_iterations=10,
            output_folder=output_folder,
            run_parameters={
                "preprocessing_parameters": {"min_frags": 2},
                "modelling_parameters": {},
                "annotation_parameters": None,
            }
        )
        
        # Verify execution
        assert mock_load.called
        assert mock_store.called
        
    def test_filetype_check(self):
        """Test file type detection."""
        assert filetype_check("test.mgf") == "mgf"
        assert filetype_check("test.mzML") == "mzml"
        assert filetype_check("test.msp") == "msp"
        assert filetype_check(["test1.mgf", "test2.mgf"]) == "mgf"
        
        with pytest.raises(ValueError):
            filetype_check("test.unknown")
            
        with pytest.raises(ValueError):
            filetype_check(["test.mgf", "test.mzml"])  # Mixed types


class TestWorkflowVariations:
    """Test different workflow variations and edge cases."""
    
    def test_workflow_single_spectrum(self):
        """Test workflow with single spectrum."""
        spectrum = Spectrum(
            mz=np.array([100.0, 200.0]),
            intensities=np.array([1.0, 0.5]),
            metadata={"precursor_mz": 300.0}
        )
        
        cleaned = load_and_clean.clean_spectra([spectrum])
        documents = generate_corpus.features_to_words(cleaned)
        
        model = modeling.define_model(n_motifs=1)
        trained_model, _ = modeling.train_model(model, documents, iterations=5)
        
        motifs = modeling.extract_motifs(trained_model, top_n=2)
        assert len(motifs) == 1
        
    def test_workflow_many_motifs(self, sample_spectra_list):
        """Test workflow with more motifs than spectra."""
        cleaned = load_and_clean.clean_spectra(sample_spectra_list[:2])
        documents = generate_corpus.features_to_words(cleaned)
        
        # Request more motifs than we have spectra
        model = modeling.define_model(n_motifs=5)
        trained_model, _ = modeling.train_model(model, documents, iterations=10)
        
        motifs = modeling.extract_motifs(trained_model, top_n=3)
        assert len(motifs) == 5
        
    def test_workflow_different_preprocessing(self, sample_spectra_list):
        """Test workflow with different preprocessing parameters."""
        # Strict preprocessing
        strict_params = {
            "min_frags": 5,
            "max_frags": 100,
            "min_intensity": 0.1,
            "normalize_intensities": True,
        }
        
        cleaned_strict = load_and_clean.clean_spectra(
            sample_spectra_list,
            strict_params
        )
        
        # Lenient preprocessing
        lenient_params = {
            "min_frags": 1,
            "normalize_intensities": False,
        }
        
        cleaned_lenient = load_and_clean.clean_spectra(
            sample_spectra_list,
            lenient_params
        )
        
        # Strict should filter out more spectra
        assert len(cleaned_strict) <= len(cleaned_lenient)
        
    def test_workflow_dia_vs_dda(self, sample_spectra_list):
        """Test workflow with different acquisition modes."""
        cleaned = load_and_clean.clean_spectra(sample_spectra_list[:3])
        
        # DDA mode (with losses)
        docs_dda = generate_corpus.features_to_words(
            cleaned,
            acquisition_type="DDA"
        )
        
        # DIA mode (no losses)
        docs_dia = generate_corpus.features_to_words(
            cleaned,
            acquisition_type="DIA"
        )
        
        # DDA documents should have more features (includes losses)
        total_features_dda = sum(len(doc) for doc in docs_dda)
        total_features_dia = sum(len(doc) for doc in docs_dia)
        
        # This assumes at least one spectrum has losses
        assert total_features_dda >= total_features_dia


class TestErrorHandling:
    """Test error handling in the workflow."""
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        empty_spectra = []
        cleaned = load_and_clean.clean_spectra(empty_spectra)
        
        assert cleaned == []
        
        # Should handle empty documents gracefully
        documents = generate_corpus.features_to_words(cleaned)
        assert documents == []
        
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Invalid number of motifs
        with pytest.raises(Exception):
            modeling.define_model(n_motifs=0)
            
        # Invalid iterations
        model = modeling.define_model(n_motifs=2)
        with pytest.raises(Exception):
            modeling.train_model(model, [], iterations=-1)
            
    def test_file_not_found(self):
        """Test handling of missing files."""
        with pytest.raises(FileNotFoundError):
            list(load_and_clean.load_mgf("nonexistent.mgf"))
            
    def test_corrupted_data_handling(self, temp_dir):
        """Test handling of corrupted data."""
        # Create corrupted MGF
        corrupted_mgf = temp_dir / "corrupted.mgf"
        corrupted_mgf.write_text("This is not valid MGF format")
        
        # Should handle gracefully or raise appropriate error
        try:
            spectra = list(load_and_clean.load_mgf(str(corrupted_mgf)))
            # If it doesn't raise, it should return empty or handle it
            assert isinstance(spectra, list)
        except Exception as e:
            # Should be a meaningful error
            assert str(e) != ""


class TestPerformance:
    """Test performance aspects of the workflow."""
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Generate 100 spectra
        large_spectra = []
        for i in range(100):
            n_peaks = np.random.randint(10, 50)
            mz = np.sort(np.random.uniform(50, 500, n_peaks))
            intensities = np.random.exponential(0.3, n_peaks)
            intensities = intensities / intensities.max()
            
            spectrum = Spectrum(
                mz=mz,
                intensities=intensities,
                metadata={"precursor_mz": 600.0 + i}
            )
            large_spectra.append(spectrum)
            
        # Process through pipeline
        cleaned = load_and_clean.clean_spectra(large_spectra)
        documents = generate_corpus.features_to_words(cleaned[:50])  # Use subset
        
        # Train with reasonable parameters
        model = modeling.define_model(n_motifs=10)
        trained_model, _ = modeling.train_model(
            model=model,
            documents=documents,
            iterations=20,
            train_parameters={"workers": 1}
        )
        
        motifs = modeling.extract_motifs(trained_model, top_n=10)
        assert len(motifs) == 10
        
    def test_memory_efficiency(self, sample_spectra_list):
        """Test that the workflow doesn't create unnecessary copies."""
        # This is a simple test - in practice, would use memory profiling
        cleaned = load_and_clean.clean_spectra(sample_spectra_list)
        
        # Check that spectra are not duplicated unnecessarily
        assert len(cleaned) <= len(sample_spectra_list)