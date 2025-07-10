"""Unit tests for MassQL functionality in MS2LDA"""

import numpy as np
import pandas as pd
import pytest
from MS2LDA.Mass2Motif import Mass2Motif
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import motifs2motifDB
from massql4motifs import msql_engine


class TestMassQLMetafilter:
    """Test METAFILTER functionality in MassQL queries"""
    
    @pytest.fixture
    def sample_motifs(self):
        """Create sample motif data for testing"""
        motifs = []
        
        # Create motif_12
        motif_12 = Mass2Motif(
            frag_mz=np.array([100.0, 200.0]),
            frag_intensities=np.array([50.0, 100.0]),
            loss_mz=np.array([18.0]),
            loss_intensities=np.array([30.0]),
            metadata={
                "id": "motif_12",
                "charge": 1,
                "annotation": "Test motif 12",
                "short_annotation": "TM12"
            }
        )
        motifs.append(motif_12)
        
        # Create motif_123
        motif_123 = Mass2Motif(
            frag_mz=np.array([150.0, 250.0]),
            frag_intensities=np.array([60.0, 120.0]),
            loss_mz=np.array([17.0]),
            loss_intensities=np.array([40.0]),
            metadata={
                "id": "motif_123",
                "charge": 1,
                "annotation": "Test motif 123",
                "short_annotation": "TM123"
            }
        )
        motifs.append(motif_123)
        
        # Create motif with whitespace in ID (edge case)
        motif_space = Mass2Motif(
            frag_mz=np.array([300.0]),
            frag_intensities=np.array([200.0]),
            loss_mz=np.array([]),
            loss_intensities=np.array([]),
            metadata={
                "id": "motif_456 ",  # Trailing space
                "charge": 2,
                "annotation": "Test motif with space",
                "short_annotation": "TM456"
            }
        )
        motifs.append(motif_space)
        
        return motifs
    
    @pytest.fixture
    def massql_dataframes(self, sample_motifs):
        """Convert motifs to MassQL dataframes"""
        ms1_df, ms2_df = motifs2motifDB(sample_motifs)
        return ms1_df, ms2_df
    
    def test_metafilter_exact_match(self, massql_dataframes):
        """Test METAFILTER with exact ID match"""
        ms1_df, ms2_df = massql_dataframes
        
        # Test motif_12
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_12"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert "motif_id" in result.columns
        assert result["motif_id"].unique()[0] == "motif_12"
        assert len(result) == 1
        
        # Test motif_123 (this was the failing case)
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_123"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert "motif_id" in result.columns
        assert result["motif_id"].unique()[0] == "motif_123"
        assert len(result) == 1
    
    def test_metafilter_with_whitespace(self, massql_dataframes):
        """Test METAFILTER handles whitespace correctly"""
        ms1_df, ms2_df = massql_dataframes
        
        # Query with trailing space should still match
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_456 "
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert len(result) == 1
        
        # Query without space should also match the motif with trailing space
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_456"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert len(result) == 1
    
    def test_metafilter_no_match(self, massql_dataframes):
        """Test METAFILTER with non-existent ID"""
        ms1_df, ms2_df = massql_dataframes
        
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_999"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        # Should return empty result, not error
        assert result.empty or "motif_id" not in result.columns or len(result) == 0
    
    def test_metafilter_multiple_filters(self, massql_dataframes):
        """Test multiple METAFILTER conditions"""
        ms1_df, ms2_df = massql_dataframes
        
        # Filter by motif_id and charge
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_12 METAFILTER:charge=1"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert result["motif_id"].unique()[0] == "motif_12"
        assert result["charge"].unique()[0] == 1
        
        # Filter that should return no results
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_12 METAFILTER:charge=2"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert result.empty or len(result) == 0
    
    def test_metafilter_other_columns(self, massql_dataframes):
        """Test METAFILTER on columns other than motif_id"""
        ms1_df, ms2_df = massql_dataframes
        
        # Filter by short_annotation
        query = "QUERY scaninfo(MS2DATA) METAFILTER:short_annotation=TM123"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert result["motif_id"].unique()[0] == "motif_123"
        
        # Filter by annotation (with spaces)
        query = "QUERY scaninfo(MS2DATA) METAFILTER:annotation=Test motif 12"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert result["motif_id"].unique()[0] == "motif_12"
    
    def test_metafilter_regex_extraction(self):
        """Test the regex extraction of METAFILTER parameters"""
        import re
        
        # Test the regex pattern used in msql_engine
        metafilter_pattern = r"METAFILTER:([\w_]+)=(.*?)(?=\s+METAFILTER:|$)"
        
        # Test single METAFILTER
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_123"
        matches = re.findall(metafilter_pattern, query)
        assert len(matches) == 1
        assert matches[0] == ('motif_id', 'motif_123')
        
        # Test multiple METAFILTERs
        query = "QUERY scaninfo(MS2DATA) METAFILTER:motif_id=motif_123 METAFILTER:charge=1"
        matches = re.findall(metafilter_pattern, query)
        assert len(matches) == 2
        assert matches[0] == ('motif_id', 'motif_123')
        assert matches[1] == ('charge', '1')
        
        # Test METAFILTER with spaces in value
        query = "QUERY scaninfo(MS2DATA) METAFILTER:annotation=Test motif 123"
        matches = re.findall(metafilter_pattern, query)
        assert len(matches) == 1
        assert matches[0] == ('annotation', 'Test motif 123')


class TestMassQLMotifQueries:
    """Test other MassQL query functionality with motifs"""
    
    @pytest.fixture
    def sample_motifs_with_masses(self):
        """Create motifs with specific fragment masses for testing"""
        motifs = []
        
        # Motif with fragment at 178.03
        motif_178 = Mass2Motif(
            frag_mz=np.array([178.03, 250.0]),
            frag_intensities=np.array([100.0, 50.0]),
            loss_mz=np.array([]),
            loss_intensities=np.array([]),
            metadata={"id": "motif_178", "charge": 1}
        )
        motifs.append(motif_178)
        
        # Motif without 178.03
        motif_other = Mass2Motif(
            frag_mz=np.array([200.0, 300.0]),
            frag_intensities=np.array([100.0, 100.0]),
            loss_mz=np.array([]),
            loss_intensities=np.array([]),
            metadata={"id": "motif_other", "charge": 1}
        )
        motifs.append(motif_other)
        
        return motifs
    
    def test_ms2prod_query(self, sample_motifs_with_masses):
        """Test MS2PROD queries on motif data"""
        ms1_df, ms2_df = motifs2motifDB(sample_motifs_with_masses)
        
        # Query for specific fragment mass
        query = "QUERY scaninfo(MS2DATA) WHERE MS2PROD=178.03:TOLERANCEMZ=0.05"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert "motif_id" in result.columns
        assert "motif_178" in result["motif_id"].values
        assert "motif_other" not in result["motif_id"].values
    
    def test_combined_ms2prod_and_metafilter(self, sample_motifs_with_masses):
        """Test combining MS2PROD and METAFILTER queries"""
        ms1_df, ms2_df = motifs2motifDB(sample_motifs_with_masses)
        
        # Query for fragment mass AND specific motif
        query = "QUERY scaninfo(MS2DATA) WHERE MS2PROD=178.03:TOLERANCEMZ=0.05 METAFILTER:motif_id=motif_178"
        result = msql_engine.process_query(query, ms1_df=ms1_df, ms2_df=ms2_df)
        
        assert not result.empty
        assert len(result) == 1
        assert result["motif_id"].iloc[0] == "motif_178"