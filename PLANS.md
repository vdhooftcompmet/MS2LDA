# MS2LDA Project Plans

## Tasks

- [x] **Dash Application Refactoring** - Modularize the monolithic Dash app - **Status: Completed (2025-07-04)**
- [x] **Vendor massql4motif library** - Import massql4motif into this repo - **Status: Completed (2025-07-01, updated 2025-07-04)**
- [x] **Fix notebook installation issues** - **Status: Resolved (2025-07-09)**
  - Issue 1: Fixed by removing conflicting setup.py and adding installation instructions
  - Issue 2: Added troubleshooting guide for tomotopy C++ compiler requirements
- [ ] **Create Python package** - Package ms2lda for PyPI - **Status: Ready to proceed**
- [x] **Fix MassQL query bug** - Fixed METAFILTER whitespace handling - **Status: Completed (2025-07-04)**
- [ ] **Setup GitHub CI** - Configure continuous integration
- [ ] **Automated documentation** - Build docs automatically in CI

## Notes

### ✅ Notebook Installation Issues (reported by Jonas Dietrich) - RESOLVED
- Fixed package name conflict between setup.py (MS2LDA) and pyproject.toml (ms2lda)
- Added explicit installation instructions for both Conda and Poetry users
- Added comprehensive troubleshooting guide for tomotopy C++ compiler requirements
- **PyPI release can now proceed**

### Completed Items
1. Dash Application Refactoring - Split monolithic layout.py and callbacks.py into modular components
2. Vendored massql4motifs package to enable PyPI distribution
3. Fixed MassQL METAFILTER query bug (whitespace handling issue)
4. Added unit tests for MassQL functionality