# Tutorials

This section contains step-by-step tutorials for using MS2LDA.

## Basic MS2LDA Analysis

A complete walkthrough of running MS2LDA on your mass spectrometry data.

### Loading Data

```python
from MS2LDA.Preprocessing import load_and_clean

# Load MGF file
spectra = load_and_clean.load_mgf("your_data.mgf")
```

### Running MS2LDA

```python
from MS2LDA import run

# Run with default parameters
results = run.run_ms2lda(spectra, n_motifs=200)
```

### Visualizing Results

```python
from MS2LDA.Visualisation import MS2LDA_visualisation

# Create visualization
MS2LDA_visualisation.plot_motifs(results)
```

## Advanced Topics

- Working with MotifDB
- Custom preprocessing pipelines
- Integration with other tools

For more examples, see the [notebooks](https://github.com/vdhooftcompmet/MS2LDA/tree/main/notebooks) directory in the repository.