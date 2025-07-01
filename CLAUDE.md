# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL GIT RULES

**NEVER PUSH ANYTHING TO THE MAIN BRANCH WITHOUT EXPLICIT USER APPROVAL. THIS IS EXTREMELY IMPORTANT!**

### Commit Message Guidelines
- Keep commit messages to 1-2 sentences maximum
- No emojis, no Claude signatures, no "Co-Authored-By" lines
- Just describe what changed concisely

## Repository Overview

MS2LDA is an advanced bioinformatics tool for unsupervised substructure discovery in mass spectrometry data. It uses topic modeling techniques to identify recurring structural patterns (called "motifs") in mass spectrometry fragmentation data without requiring prior compound identification.

## Key Components

- **MS2LDA/**: Core functionality including topic modeling engine, preprocessing, and visualization
- **App/**: Dash-based web application for interactive exploration
- **MS2LDA/Add_On/**: Additional modules (Spec2Vec, Fingerprints, MassQL, NTS)
- **tests/**: Pytest-based test suite

## Development Setup and Commands

### Initial Setup
```bash
# Install Poetry first: https://python-poetry.org/docs/#installation
poetry install

# Download required data and models
poetry run ./run_analysis.sh --only-download

# For ARM Macs or RDKit issues:
poetry install --extras lite --no-binary rdkit
```

### Common Development Commands

```bash
# Testing
poetry run pytest                          # Run all tests
poetry run pytest -v                       # Verbose output
poetry run pytest -k test_name             # Run specific test
poetry run pytest tests/test_modeling.py   # Run specific file

# Code Quality
poetry run black .                         # Format code
poetry run flake8                          # Lint code
poetry run ruff check .                    # Additional linting
poetry run mypy .                          # Type checking
poetry run pre-commit run --all-files      # Run all pre-commit hooks

# Running MS2LDA
poetry run ms2lda --dataset datasets/mushroom_spectra.mgf --n-motifs 200 --n-iterations 5000 --output-folder cli_results

# Running Web Interface
poetry run ms2lda-viz                      # Opens at http://localhost:8050

# Build Package
poetry build                               # Creates wheel and sdist in dist/
```

## Architecture Notes

### Technology Stack
- Python 3.11-3.12
- Core: tomotopy (LDA), matchms (MS data), spec2vec (embeddings)
- Web UI: Dash, Plotly, Dash-Cytoscape
- Chemistry: RDKit for molecular fingerprints
- Build: Poetry for dependency management

### Key Patterns
- **Data Flow**: MS data → Preprocessing → LDA modeling → Motif discovery → Annotation → Visualization
- **Web App Structure**: Multi-tab Dash app with separate modules for each functionality
- **API Design**: Each tab has its own callback structure in App/
- **Testing**: Pytest with fixtures for MS data and models

### Important Files and Entry Points
- **pyproject.toml**: Poetry configuration and dependencies
- **MS2LDA/modeling.py**: Core LDA implementation
- **scripts/ms2lda_runfull.py**: Main entry point for CLI analysis
- **run_analysis.sh/.bat**: Wrapper scripts that execute ms2lda_runfull.py
- **App/app.py**: Main Dash application entry point
- **run_ms2ldaviz.sh/.bat**: Wrapper scripts that launch the Dash app
- **MS2LDA/Add_On/Spec2Vec/**: Neural embeddings for motif annotation
- **MS2LDA/MotifDB/**: Database of known motifs (fetched at runtime)

Note: Poetry creates convenient command aliases:
- `ms2lda`: Maps to scripts.ms2lda_runfull:main (CLI analysis)
- `ms2lda-viz`: Maps to App.app:main (Dash web interface)

## Usage Modes

1. **CLI**: `ms2lda` command for batch processing
2. **Web**: `ms2lda-viz` for interactive exploration
3. **API**: Direct Python imports for custom workflows

## Vendored Dependencies

- **massql4motifs**: Vendored from https://github.com/j-a-dietrich/MassQueryLanguage4Mass2Motifs.git to enable PyPI distribution. See `massql4motifs/README.vendored.md` for details on provenance and update process. This is temporary until the package is published to PyPI or merged into upstream MassQL.

## Notes

- Large model files (Spec2Vec models, MotifDB) are excluded from the package and downloaded on first use
- The project supports multiple MS data formats: MGF, mzML, MSP
- Convergence monitoring includes multiple criteria (perplexity, log-likelihood, motif stability)
- Pre-commit hooks are configured for code quality

## Dash App Refactoring Plan

A comprehensive plan for refactoring the monolithic Dash app has been created in `plan/PLANS.md`. This plan outlines:
- Breaking down the 3,400+ line callback.py and 2,600+ line layout.py files into modular components
- Creating a service layer for business logic separation
- Implementing comprehensive testing to ensure functionality is preserved
- Step-by-step migration strategy for each tab

Refer to `plan/PLANS.md` for detailed implementation steps and progress tracking.
