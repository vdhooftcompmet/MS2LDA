# Test Data Directory

This directory contains sample data files for testing MS2LDA functionality.

## Files

### sample.mgf
- Contains 5 test spectra in MGF format
- Includes various test cases:
  - Spectrum 1-3: Normal spectra with different numbers of peaks
  - Spectrum 4: Empty spectrum (no peaks)
  - Spectrum 5: Low intensity spectrum

### sample.msp
- Contains 4 test compounds in MSP format
- Includes metadata like compound names, formulas, and InChIKeys
- Test cases:
  - Compounds 1-3: Full metadata and peak lists
  - Compound 4: Minimal metadata

## Usage

These files are used by the test suite to verify:
- File loading functionality
- Data parsing correctness
- Edge case handling
- Integration testing

## Note

These are synthetic test files created for testing purposes only.
They do not represent real mass spectrometry data.