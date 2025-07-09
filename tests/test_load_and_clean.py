"""
Comprehensive tests for MS2LDA data loading and cleaning functions.
"""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
from matchms import Spectrum

from MS2LDA.Preprocessing.load_and_clean import (
    clean_spectra, load_mgf, load_mzml, load_msp
)


class TestLoadMGF:
    """Test cases for MGF file loading."""
    
    def test_load_mgf_basic(self, temp_dir, sample_mgf_content):
        """Test loading a basic MGF file."""
        mgf_file = temp_dir / "test.mgf"
        mgf_file.write_text(sample_mgf_content)
        
        spectra = list(load_mgf(str(mgf_file)))
        assert len(spectra) == 2
        assert spectra[0].get("precursor_mz") == 300.0
        assert spectra[1].get("precursor_mz") == 400.0
        
    def test_load_mgf_empty_file(self, temp_dir):
        """Test loading an empty MGF file."""
        mgf_file = temp_dir / "empty.mgf"
        mgf_file.write_text("")
        
        spectra = list(load_mgf(str(mgf_file)))
        assert len(spectra) == 0
        
    def test_load_mgf_missing_file(self):
        """Test loading a non-existent MGF file."""
        with pytest.raises(FileNotFoundError):
            list(load_mgf("nonexistent.mgf"))
            
    def test_load_mgf_malformed_content(self, temp_dir):
        """Test loading MGF with malformed content."""
        malformed_content = """BEGIN IONS
TITLE=Bad Spectrum
PEPMASS=not_a_number
100.0 1000
END IONS"""
        mgf_file = temp_dir / "malformed.mgf"
        mgf_file.write_text(malformed_content)
        
        # Should raise ValueError for malformed content
        with pytest.raises(ValueError):
            list(load_mgf(str(mgf_file)))
        
    @patch('MS2LDA.Preprocessing.load_and_clean.load_from_mgf')
    def test_load_mgf_with_mock(self, mock_load):
        """Test MGF loading with mocked matchms function."""
        mock_spectra = [MagicMock(spec=Spectrum) for _ in range(3)]
        mock_load.return_value = mock_spectra
        
        result = list(load_mgf("test.mgf"))
        assert len(result) == 3
        mock_load.assert_called_once_with("test.mgf")


class TestLoadMZML:
    """Test cases for mzML file loading."""
    
    @patch('MS2LDA.Preprocessing.load_and_clean.load_from_mzml')
    def test_load_mzml_basic(self, mock_load):
        """Test basic mzML loading."""
        mock_spectra = [MagicMock(spec=Spectrum) for _ in range(2)]
        mock_load.return_value = mock_spectra
        
        result = list(load_mzml("test.mzML"))
        assert len(result) == 2
        mock_load.assert_called_once()
        
    def test_load_mzml_missing_file(self):
        """Test loading a non-existent mzML file."""
        with pytest.raises(FileNotFoundError):
            list(load_mzml("nonexistent.mzML"))
            
    @patch('MS2LDA.Preprocessing.load_and_clean.load_from_mzml')
    def test_load_mzml_empty_result(self, mock_load):
        """Test mzML loading returning empty results."""
        mock_load.return_value = []
        
        result = list(load_mzml("empty.mzML"))
        assert len(result) == 0


class TestLoadMSP:
    """Test cases for MSP file loading."""
    
    def test_load_msp_basic(self, temp_dir, sample_msp_content):
        """Test loading a basic MSP file."""
        msp_file = temp_dir / "test.msp"
        msp_file.write_text(sample_msp_content)
        
        spectra = list(load_msp(str(msp_file)))
        assert len(spectra) == 2
        assert spectra[0].get("precursor_mz") == 300.0
        assert spectra[1].get("precursor_mz") == 400.0
        
    @patch('MS2LDA.Preprocessing.load_and_clean.load_from_msp')
    def test_load_msp_with_mock(self, mock_load):
        """Test MSP loading with mocked matchms function."""
        mock_spectra = [MagicMock(spec=Spectrum) for _ in range(4)]
        mock_load.return_value = mock_spectra
        
        result = list(load_msp("test.msp"))
        assert len(result) == 4
        mock_load.assert_called_once()


class TestCleanSpectra:
    """Test cases for spectra cleaning."""
    
    def test_clean_spectra_basic(self, sample_spectra_list):
        """Test basic spectra cleaning with default parameters."""
        cleaned = clean_spectra(iter(sample_spectra_list[:-1]))  # Exclude empty spectrum
        
        # Check IDs are reassigned
        assert all(s.get("id") == f"spec_{i}" for i, s in enumerate(cleaned))
        # Check that some spectra are returned (default filtering may remove some)
        assert len(cleaned) >= 1
        
    def test_clean_spectra_min_fragments_filter(self, sample_spectra_list):
        """Test filtering by minimum number of fragments."""
        params = {"min_frags": 5}
        cleaned = clean_spectra(iter(sample_spectra_list), params)
        
        # Only spectrum 3 has >= 5 fragments
        assert len(cleaned) == 1
        assert cleaned[0].get("id") == "spec_0"
        
    def test_clean_spectra_max_fragments_filter(self, sample_spectra_list):
        """Test filtering by maximum number of fragments."""
        params = {"max_frags": 3}
        cleaned = clean_spectra(iter(sample_spectra_list), params)
        
        # Spectra 1 and 2 have <= 3 fragments
        assert len(cleaned) == 2
        
    def test_clean_spectra_intensity_filters(self):
        """Test filtering by relative intensity thresholds."""
        spectra = [
            Spectrum(
                mz=np.array([100.0, 150.0, 200.0]),
                intensities=np.array([0.001, 0.01, 1.0]),  # Mix of intensities
                metadata={"precursor_mz": 300.0}
            ),
            Spectrum(
                mz=np.array([100.0, 150.0]),
                intensities=np.array([0.5, 1.0]),  # All high intensities
                metadata={"precursor_mz": 200.0}
            ),
        ]
        
        # Use relative intensity thresholds (0-1 range)
        params = {"min_intensity": 0.1, "max_intensity": 1.0}
        cleaned = clean_spectra(iter(spectra), params)
        
        # Check that filtering worked
        for spectrum in cleaned:
            # All peaks should have relative intensity >= 0.1
            assert np.all(spectrum.peaks.intensities >= 0.1)
        
    def test_clean_spectra_precursor_mz_filters(self):
        """Test that precursor m/z filtering is not applied by default."""
        # Use spectra with at least 3 peaks (default min_frags)
        spectra = [
            Spectrum(
                mz=np.array([100.0, 150.0, 200.0]),
                intensities=np.array([0.5, 0.7, 1.0]),
                metadata={"precursor_mz": 30.0}  # Low
            ),
            Spectrum(
                mz=np.array([100.0, 150.0, 200.0]),
                intensities=np.array([0.5, 0.7, 1.0]),
                metadata={"precursor_mz": 500.0}  # Medium
            ),
            Spectrum(
                mz=np.array([100.0, 150.0, 200.0]),
                intensities=np.array([0.5, 0.7, 1.0]),
                metadata={"precursor_mz": 3000.0}  # High
            ),
        ]
        
        # Note: precursor_mz filtering is commented out in the code
        params = {}
        cleaned = clean_spectra(iter(spectra), params)
        
        # All spectra should pass since precursor filtering is not active
        assert len(cleaned) == 3
        
    def test_clean_spectra_normalize_intensities(self):
        """Test intensity normalization."""
        spectrum = Spectrum(
            mz=np.array([100.0, 150.0, 200.0]),
            intensities=np.array([50.0, 100.0, 25.0]),
            metadata={"precursor_mz": 300.0}
        )
        
        params = {"normalize_intensities": True}
        cleaned = clean_spectra([spectrum], params)
        
        # Check that max intensity is 1.0
        assert np.max(cleaned[0].peaks.intensities) == 1.0
        # Check relative intensities are preserved
        assert cleaned[0].peaks.intensities[0] == 0.5
        assert cleaned[0].peaks.intensities[2] == 0.25
        
    def test_clean_spectra_empty_input(self):
        """Test cleaning with empty input."""
        cleaned = clean_spectra(iter([]))
        assert cleaned == []
        
    def test_clean_spectra_all_filtered_out(self):
        """Test when all spectra are filtered out."""
        spectra = [
            Spectrum(
                mz=np.array([100.0]),
                intensities=np.array([1.0]),
                metadata={"precursor_mz": 200.0}
            )
        ]
        
        params = {"min_frags": 10}  # Impossible to satisfy
        cleaned = clean_spectra(iter(spectra), params)
        assert cleaned == []
        
    def test_clean_spectra_preserve_metadata(self, sample_spectrum):
        """Test that metadata is preserved during cleaning."""
        original_metadata = sample_spectrum.metadata.copy()
        cleaned = clean_spectra([sample_spectrum])
        
        # ID should be changed, but other metadata preserved
        assert cleaned[0].get("id") == "spec_0"
        assert cleaned[0].get("precursor_mz") == original_metadata["precursor_mz"]
        assert cleaned[0].get("charge") == original_metadata["charge"]
        assert cleaned[0].get("ionmode") == original_metadata["ionmode"]
        
    def test_clean_spectra_combined_filters(self):
        """Test multiple filters applied together."""
        spectra = []
        # Create diverse spectra to test multiple filters
        for i in range(10):
            n_peaks = i + 1
            mz = np.linspace(50, 200, n_peaks)
            intensities = np.random.random(n_peaks) * (i + 1) * 0.1
            spectra.append(
                Spectrum(
                    mz=mz,
                    intensities=intensities,
                    metadata={"precursor_mz": 100 + i * 50}
                )
            )
            
        params = {
            "min_frags": 3,
            "max_frags": 7,
            "min_intensity": 0.01,  # Lower threshold for relative intensity
            "max_intensity": 1.0,
            "normalize_intensities": True,
        }
        
        cleaned = clean_spectra(iter(spectra), params)
        
        # Verify all filters were applied
        for spectrum in cleaned:
            n_peaks = len(spectrum.peaks.mz)
            assert 3 <= n_peaks <= 7
            # Precursor filtering is not active
            assert np.max(spectrum.peaks.intensities) == 1.0
            
    @pytest.mark.parametrize("param_name,param_value,expected_count", [
        ("min_frags", 1, 3),  # All pass except empty spectrum
        ("min_frags", 10, 1),  # Only spectrum 3 passes
        ("max_frags", 2, 0),   # None pass due to default min_frags=3
        ("min_frags", 3, 2),   # Spectra 1 and 3 pass
        ("min_frags", 5, 1),   # Only spectrum 3 passes
    ])
    def test_clean_spectra_individual_params(
        self, sample_spectra_list, param_name, param_value, expected_count
    ):
        """Test individual parameter effects."""
        params = {param_name: param_value}
        cleaned = clean_spectra(iter(sample_spectra_list), params)
        assert len(cleaned) == expected_count


class TestIntegration:
    """Integration tests for loading and cleaning pipeline."""
    
    def test_load_and_clean_mgf_pipeline(self, temp_dir, sample_mgf_content):
        """Test complete pipeline from MGF loading to cleaning."""
        # Create MGF file
        mgf_file = temp_dir / "test.mgf"
        mgf_file.write_text(sample_mgf_content)
        
        # Load spectra
        spectra = load_mgf(str(mgf_file))
        
        # Clean spectra
        params = {
            "min_frags": 2,
            "normalize_intensities": True,
        }
        cleaned = clean_spectra(spectra, params)
        
        # Verify results
        assert len(cleaned) == 2
        assert all(s.get("id").startswith("spec_") for s in cleaned)
        assert all(np.max(s.peaks.intensities) == 1.0 for s in cleaned)
        
    def test_mixed_file_loading(self, temp_dir, sample_mgf_content, sample_msp_content):
        """Test loading different file formats."""
        # Create test files
        mgf_file = temp_dir / "test.mgf"
        mgf_file.write_text(sample_mgf_content)
        msp_file = temp_dir / "test.msp"
        msp_file.write_text(sample_msp_content)
        
        # Load from both formats
        mgf_spectra = list(load_mgf(str(mgf_file)))
        msp_spectra = list(load_msp(str(msp_file)))
        
        # Combine and clean
        all_spectra = mgf_spectra + msp_spectra
        cleaned = clean_spectra(iter(all_spectra))
        
        # Should have 4 spectra total with sequential IDs
        assert len(cleaned) == 4
        assert [s.get("id") for s in cleaned] == ["spec_0", "spec_1", "spec_2", "spec_3"]