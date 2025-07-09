# MS2LDA Project Plans

## Tasks

- [x] **Dash Application Refactoring** - Modularize the monolithic Dash app - **Status: Completed (2025-07-04)**
- [x] **Vendor massql4motif library** - Import massql4motif into this repo - **Status: Completed (2025-07-01, updated 2025-07-04)**
- [ ] **Fix notebook installation issues** - **Status: Blocking PyPI release**
  - Issue 1: After conda install, cannot import MS2LDA in notebooks (module not found)
  - Issue 2: `pip install .` fails due to tomotopy installation issues
- [ ] **Create Python package** - Package ms2lda for PyPI - **Status: ⚠️ BLOCKED by notebook installation issues**
- [x] **Fix MassQL query bug** - Fixed METAFILTER whitespace handling - **Status: Completed (2025-07-04)**
- [ ] **Setup GitHub CI** - Configure continuous integration
- [ ] **Automated documentation** - Build docs automatically in CI

## Notes

### ⚠️ Notebook Installation Issues (reported by Jonas Dietrich) - BLOCKING PyPI RELEASE
- When installing ms2lda with conda, the package installs but cannot be imported in notebooks
- When installing with `pip install .`, installation fails (Rosina had issues installing tomotopy)
- **These issues MUST be resolved before PyPI release**

### Completed Items
1. Dash Application Refactoring - Split monolithic layout.py and callbacks.py into modular components
2. Vendored massql4motifs package to enable PyPI distribution
3. Fixed MassQL METAFILTER query bug (whitespace handling issue)
4. Added unit tests for MassQL functionality