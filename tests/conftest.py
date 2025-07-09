"""
Shared test fixtures and configuration for MS2LDA tests.
"""
import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from matchms import Fragments, Spectrum


@pytest.fixture
def sample_spectrum():
    """Create a single sample spectrum for testing."""
    return Spectrum(
        mz=np.array([100.0, 150.0, 200.0, 250.0, 300.0]),
        intensities=np.array([0.1, 0.2, 0.5, 0.8, 1.0]),
        metadata={
            "id": "test_spectrum_1",
            "precursor_mz": 350.0,
            "charge": 1,
            "ionmode": "positive",
            "scan_number": 100,
        },
    )


@pytest.fixture
def sample_spectra_list():
    """Create a list of diverse sample spectra for testing."""
    spectra = []
    
    # Spectrum 1: Normal spectrum with losses
    spec1 = Spectrum(
        mz=np.array([100.0, 150.0, 200.0, 250.0]),
        intensities=np.array([0.2, 0.4, 0.6, 1.0]),
        metadata={
            "id": "spectrum_1",
            "precursor_mz": 300.0,
            "charge": 1,
            "ionmode": "positive",
        },
    )
    spec1._losses = Fragments(
        mz=np.array([50.0, 100.0, 150.0]),
        intensities=np.array([0.3, 0.5, 0.7]),
    )
    spectra.append(spec1)
    
    # Spectrum 2: Spectrum with few peaks
    spectra.append(
        Spectrum(
            mz=np.array([120.0, 180.0]),
            intensities=np.array([0.5, 1.0]),
            metadata={
                "id": "spectrum_2",
                "precursor_mz": 250.0,
                "charge": 2,
                "ionmode": "positive",
            },
        )
    )
    
    # Spectrum 3: Spectrum with many peaks
    mz_values = np.linspace(50, 500, 50)
    intensities = np.random.exponential(0.3, 50)
    intensities = intensities / intensities.max()
    spectra.append(
        Spectrum(
            mz=mz_values,
            intensities=intensities,
            metadata={
                "id": "spectrum_3",
                "precursor_mz": 550.0,
                "charge": 1,
                "ionmode": "negative",
            },
        )
    )
    
    # Spectrum 4: Empty spectrum (edge case)
    spectra.append(
        Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={
                "id": "spectrum_4",
                "precursor_mz": 200.0,
                "charge": 1,
                "ionmode": "positive",
            },
        )
    )
    
    return spectra


@pytest.fixture
def sample_documents():
    """Create sample documents for LDA modeling."""
    return [
        ["frag@100.00", "frag@150.00", "loss@50.00", "loss@50.00"],
        ["frag@100.00", "frag@200.00", "frag@200.00", "loss@100.00"],
        ["frag@150.00", "frag@250.00", "loss@50.00"],
        ["frag@100.00", "frag@150.00", "frag@200.00", "frag@250.00"],
        ["loss@50.00", "loss@100.00", "loss@150.00"],
    ]


@pytest.fixture
def sample_motif_features():
    """Create sample motif features for testing."""
    return [
        [("frag@100.00", 0.8), ("frag@150.00", 0.6), ("loss@50.00", 0.4)],
        [("frag@200.00", 1.0), ("frag@250.00", 0.7), ("loss@100.00", 0.5)],
        [("frag@300.00", 0.9), ("loss@50.00", 0.8), ("loss@150.00", 0.3)],
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_mgf_content():
    """Sample MGF file content for testing."""
    return """BEGIN IONS
TITLE=Spectrum 1
PEPMASS=300.0
CHARGE=1+
MSLEVEL=2
100.0 1000
150.0 2000
200.0 3000
250.0 5000
END IONS

BEGIN IONS
TITLE=Spectrum 2
PEPMASS=400.0
CHARGE=2+
MSLEVEL=2
120.0 500
180.0 1500
240.0 2500
300.0 4000
END IONS
"""


@pytest.fixture
def sample_msp_content():
    """Sample MSP file content for testing."""
    return """Name: Test Compound 1
Precursor_type: [M+H]+
Precursor_MZ: 300.0
Num Peaks: 4
100.0 100
150.0 200
200.0 300
250.0 500

Name: Test Compound 2
Precursor_type: [M+H]+
Precursor_MZ: 400.0
Num Peaks: 3
120.0 50
180.0 150
240.0 250
"""


@pytest.fixture
def mock_tomotopy_model():
    """Create a mock tomotopy LDA model."""
    model = MagicMock()
    model.k = 3  # 3 topics
    model.num_vocabs = 10
    model.vocabs = [
        "frag@100.00", "frag@150.00", "frag@200.00", "frag@250.00", "frag@300.00",
        "loss@50.00", "loss@100.00", "loss@150.00", "loss@200.00", "loss@250.00"
    ]
    
    # Mock document-topic distributions
    model.docs = []
    for i in range(5):
        doc = MagicMock()
        doc.get_topic_dist = MagicMock(return_value=[0.3, 0.5, 0.2])
        doc.words = [0, 1, 5]  # Word indices
        model.docs.append(doc)
    
    # Mock topic-word distributions
    def get_topic_word_dist(topic_id):
        if topic_id == 0:
            return np.array([0.3, 0.2, 0.1, 0.05, 0.05, 0.2, 0.05, 0.03, 0.01, 0.01])
        elif topic_id == 1:
            return np.array([0.1, 0.1, 0.3, 0.2, 0.1, 0.05, 0.1, 0.03, 0.01, 0.01])
        else:
            return np.array([0.05, 0.05, 0.1, 0.1, 0.3, 0.1, 0.15, 0.1, 0.03, 0.02])
    
    model.get_topic_word_dist = get_topic_word_dist
    
    # Mock training methods
    model.train = MagicMock()
    model.perplexity = 10.0
    model.ll_per_word = -2.5
    
    return model


@pytest.fixture
def mock_spec2vec_model():
    """Create a mock Spec2Vec model."""
    model = MagicMock()
    
    # Mock embedding calculation
    def mock_embedding(spectrum):
        # Return a consistent embedding based on spectrum size
        embedding_size = 300
        if hasattr(spectrum, 'peaks') and hasattr(spectrum.peaks, 'mz'):
            seed = len(spectrum.peaks.mz)
        else:
            seed = 42
        np.random.seed(seed)
        return np.random.randn(embedding_size)
    
    model.model.calculate_embedding = mock_embedding
    return model


@pytest.fixture
def preprocessing_params():
    """Standard preprocessing parameters for tests."""
    return {
        "min_frags": 3,
        "max_frags": 500,
        "min_intensity": 0.01,
        "max_intensity": 1e6,
        "min_precursor_mz": 50.0,
        "max_precursor_mz": 2000.0,
        "normalize_intensities": True,
    }


@pytest.fixture
def modeling_params():
    """Standard modeling parameters for tests."""
    return {
        "n_motifs": 3,
        "iterations": 100,
        "model_parameters": {
            "alpha": 0.1,
            "eta": 0.01,
        },
        "train_parameters": {
            "workers": 1,
        },
        "convergence_parameters": {
            "type": "perplexity_history",
            "threshold": 0.001,
            "window_size": 5,
            "step_size": 10,
        },
    }


@pytest.fixture
def sample_m2m_content():
    """Sample Mass2Motif file content."""
    return """#FeatureID\tProbability\tFeatureName\tFeatureType
frag_100.0000\t0.8000\tfrag@100.00\tfragment
frag_150.0000\t0.6000\tfrag@150.00\tfragment
loss_50.0000\t0.4000\tloss@50.00\tloss
#Metadata
MotifID\tmotif_0
MotifSet\ttest_motifs
Charge\t1
IonMode\tpositive
"""


# Helper functions for test data generation
def generate_random_spectrum(
    n_peaks: int = 10,
    mz_range: Tuple[float, float] = (50, 500),
    precursor_mz: float = None,
    add_losses: bool = False,
) -> Spectrum:
    """Generate a random spectrum with specified parameters."""
    mz = np.sort(np.random.uniform(mz_range[0], mz_range[1], n_peaks))
    intensities = np.random.exponential(0.3, n_peaks)
    intensities = intensities / intensities.max()
    
    if precursor_mz is None:
        precursor_mz = mz_range[1] + 50
    
    spectrum = Spectrum(
        mz=mz,
        intensities=intensities,
        metadata={
            "precursor_mz": precursor_mz,
            "charge": 1,
            "ionmode": "positive",
        },
    )
    
    if add_losses and n_peaks > 3:
        n_losses = n_peaks // 2
        loss_mz = np.sort(np.random.uniform(10, 200, n_losses))
        loss_intensities = np.random.exponential(0.2, n_losses)
        loss_intensities = loss_intensities / loss_intensities.max()
        spectrum._losses = Fragments(mz=loss_mz, intensities=loss_intensities)
    
    return spectrum


def generate_test_documents(
    n_docs: int = 10,
    vocab_size: int = 20,
    doc_length_range: Tuple[int, int] = (5, 20),
) -> List[List[str]]:
    """Generate random documents for LDA testing."""
    # Create vocabulary
    vocab = []
    for i in range(vocab_size // 2):
        vocab.append(f"frag@{100 + i*50:.2f}")
    for i in range(vocab_size // 2):
        vocab.append(f"loss@{20 + i*30:.2f}")
    
    # Generate documents
    documents = []
    for _ in range(n_docs):
        doc_length = np.random.randint(doc_length_range[0], doc_length_range[1])
        doc = np.random.choice(vocab, doc_length).tolist()
        documents.append(doc)
    
    return documents


@pytest.fixture
def mock_download_function(monkeypatch):
    """Mock the download_model_and_data function."""
    def mock_download(*args, **kwargs):
        # Create dummy files instead of downloading
        return True
    
    monkeypatch.setattr("MS2LDA.utils.download_model_and_data", mock_download)
    return mock_download