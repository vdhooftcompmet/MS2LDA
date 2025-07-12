# API Reference

Welcome to the MS2LDA API reference documentation. This section provides detailed information about all the modules, classes, and functions available in MS2LDA.

## Overview

MS2LDA is organized into several main components:

- **Core Modules**: The fundamental components for running MS2LDA analysis
- **Add-On Modules**: Additional functionality for enhanced analysis
- **Visualization**: Tools for visualizing results

## Core Modules

### [MS2LDA Run](run.md)
The main entry point for running MS2LDA analysis. Contains functions for executing the complete workflow.

### [Modeling](modeling.md)
Core LDA modeling functionality including model training and inference.

### [Mass2Motif](mass2motif.md)
Classes and functions for handling Mass2Motif objects and operations.

### [Preprocessing](preprocessing.md)
Data preprocessing utilities for preparing mass spectrometry data for analysis.

## Add-On Modules

### [Fingerprints](fingerprints.md)
Molecular fingerprint calculation and substructure retrieval functionality.

### [MassQL](massql.md)
MassQL query language support for searching and filtering mass spectrometry data.

### [NTS](nts.md)
Non-target screening tools for environmental analysis.

### [Spec2Vec](spec2vec.md)
Spec2Vec integration for spectral similarity calculations and annotation.

## Visualization

### [Visualization Tools](visualization.md)
Comprehensive visualization utilities for exploring MS2LDA results.

## Quick Start

To use MS2LDA programmatically:

```python
from MS2LDA import run_ms2lda
from MS2LDA.Preprocessing import load_and_clean

# Load and preprocess data
spectra = load_and_clean.load_mgf("your_data.mgf")

# Run MS2LDA
results = run_ms2lda(spectra, n_motifs=200)
```

For detailed usage examples, see the [Examples](../examples/tutorials.md) section.