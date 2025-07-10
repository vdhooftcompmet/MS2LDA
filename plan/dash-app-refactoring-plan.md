# MS2LDA Dash App Refactoring Plan

## Overview

The MS2LDA Dash app currently has two monolithic files:
- `App/callbacks.py` (3,443 lines) - Contains all callback functions
- `App/layout.py` (2,634 lines) - Contains all UI components

This makes the code difficult to maintain, test, and extend. This plan details a systematic refactoring to modularize the app while ensuring it continues to work correctly.

## Goals

1. **Maintainability**: Break monolithic files into manageable modules (~400-500 lines each)
2. **Testability**: Enable unit testing of business logic separate from UI
3. **No Breaking Changes**: App must work exactly the same after refactoring
4. **Reuse Existing Code**: Use existing MS2LDA functions where possible

## New Structure

```
App/
├── __init__.py
├── app.py (main entry point)
├── app_instance.py (app configuration)
├── layout/
│   ├── __init__.py
│   ├── run_analysis.py      # Layout + callbacks for Run Analysis tab
│   ├── load_results.py      # Layout + callbacks for Load Results tab
│   ├── motif_rankings.py    # Layout + callbacks for Motif Rankings tab
│   ├── motif_details.py     # Layout + callbacks for Motif Details tab
│   ├── spectra_search.py    # Layout + callbacks for Spectra Search tab
│   ├── network_view.py      # Layout + callbacks for Network View tab
│   └── motif_search.py      # Layout + callbacks for Motif Search tab
├── services/
│   ├── __init__.py
│   ├── analysis.py          # Business logic for analysis and data loading
│   ├── motifs.py           # Motif-related calculations
│   ├── spectra.py          # Spectrum processing and search
│   ├── network.py          # Network generation
│   └── visualization.py    # Dash-specific visualization functions
└── tests/
    ├── test_app_integrity.py   # Tests for duplicate callbacks, missing components
    ├── test_baseline.py        # Capture current behavior
    ├── test_services/          # Unit tests for service functions
    └── test_integration.py     # End-to-end tests
```

## Step 0: Background Knowledge Gathering

Before starting the refactoring, familiarize yourself with the codebase:

- [ ] Read `App/app.py` to understand the main entry point
- [ ] Read `App/app_instance.py` to understand app configuration
- [ ] Examine `App/layout.py` to identify the 7 main tabs:
  - [ ] `create_run_analysis_tab()` (lines ~16-1034)
  - [ ] `create_load_results_tab()` (lines ~1035-1195)  
  - [ ] `create_motif_rankings_tab()` (lines ~1398-1693)
  - [ ] `create_motif_details_tab()` (lines ~1694-2058)
  - [ ] `create_spectra_search_tab()` (lines ~2291-2634)
  - [ ] `create_cytoscape_network_tab()` (lines ~1196-1397)
  - [ ] `create_screening_tab()` (lines ~2059-2290)
- [ ] Examine `App/callbacks.py` to identify key functions:
  - [ ] Helper functions: `calculate_motif_shares()`, `make_spectrum_plot()`, `apply_common_layout()`
  - [ ] Main callbacks: `handle_run_or_load()`, `update_motif_rankings_table()`, etc.
- [ ] Check existing MS2LDA functions we can reuse:
  - [ ] `MS2LDA.run.run_ms2lda()` - Core analysis function
  - [ ] `MS2LDA.Preprocessing.load_and_clean.*` - File loading
  - [ ] `MS2LDA.utils.*` - Utility functions
- [ ] Note which functions in `MS2LDA.Visualisation.visualisation.py` are used by Dash
- [ ] Locate test data: `datasets/mushroom_spectra.mgf` (2.7MB file we'll use for testing)

## Phase 1: Testing Infrastructure Setup (Day 1)

### 1.1 Create Test Structure

- [ ] Create `tests/` directory if it doesn't exist
- [ ] Create `tests/test_app_integrity.py` for app health checks
- [ ] Create `tests/test_baseline.py` for capturing current behavior
- [ ] Create `tests/test_services/` directory for unit tests
- [ ] Create `tests/test_integration.py` for end-to-end tests
- [ ] Create `tests/baseline_data/` directory for storing baseline outputs

### 1.2 Implement App Integrity Tests

Create `tests/test_app_integrity.py`:

```python
# tests/test_app_integrity.py
from collections import Counter
from dash.testing.application_runners import import_app

def test_no_duplicate_callbacks():
    """Ensure no duplicate callback outputs."""
    app = import_app("App.app")
    app._setup_server()
    
    base_outputs = [key.split("@")[0] for key in app.callback_map.keys()]
    duplicates = [out for out, count in Counter(base_outputs).items() if count > 1]
    
    assert not duplicates, f"Duplicate callback outputs detected: {duplicates}"

def test_all_callback_components_exist():
    """Ensure all callback inputs/outputs reference existing components."""
    app = import_app("App.app")
    app._setup_server()
    
    def get_all_ids(layout):
        ids = set()
        def traverse(component):
            if hasattr(component, 'id') and component.id:
                ids.add(component.id)
            if hasattr(component, 'children'):
                if isinstance(component.children, list):
                    for child in component.children:
                        traverse(child)
                elif component.children:
                    traverse(component.children)
        traverse(layout)
        return ids
    
    layout_ids = get_all_ids(app.layout)
    missing = []
    
    for callback in app.callback_map.values():
        for inp in getattr(callback, 'inputs', []):
            if inp.component_id not in layout_ids:
                missing.append(f"Input: {inp.component_id}")
        for out in getattr(callback, 'outputs', []):
            if out.component_id not in layout_ids:
                missing.append(f"Output: {out.component_id}")
    
    assert not missing, f"Missing components: {missing}"
```

- [ ] Run `pytest tests/test_app_integrity.py -v` to verify current app is healthy

### 1.3 Capture Baseline Behavior

Create `tests/test_baseline.py`:

```python
# tests/test_baseline.py
import json
import pickle
import base64
from pathlib import Path

def capture_baseline():
    """Capture current app behavior with mushroom dataset."""
    from App import callbacks
    
    baseline_dir = Path("tests/baseline_data")
    baseline_dir.mkdir(exist_ok=True)
    
    # Load mushroom file
    with open("datasets/mushroom_spectra.mgf", "rb") as f:
        contents = base64.b64encode(f.read()).decode()
    
    # Test run_analysis callback
    print("Capturing run_analysis baseline...")
    result = callbacks.handle_run_or_load(
        n_clicks=1,
        n_clicks_load=None,
        n_clicks_demo=None,
        contents=f"data:application/octet-stream;base64,{contents}",
        filename="mushroom_spectra.mgf",
        n_motifs=30,  # Use 30 for faster tests
        acquisition_type="DDA",
        polarity="positive",
        contents_load=None,
        filename_load=None,
        motif_ranking_state=None,
        n_spec2vec=10,
        s2v_downloaded=True,
        advanced_params={}
    )
    
    # Save outputs
    with open(baseline_dir / "run_analysis_output.pkl", "wb") as f:
        pickle.dump(result, f)
    
    # Save key metrics
    metrics = {
        "num_spectra": len(result.get("spectra-store", [])),
        "num_motifs": len(result.get("lda-dict-store", {}).get("beta", {})),
        "theta_keys": list(result.get("lda-dict-store", {}).get("theta", {}).keys())
    }
    
    with open(baseline_dir / "run_analysis_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Baseline captured: {metrics['num_spectra']} spectra, {metrics['num_motifs']} motifs")

if __name__ == "__main__":
    capture_baseline()
```

- [ ] Run `python tests/test_baseline.py` to capture baseline
- [ ] Verify files created in `tests/baseline_data/`

### 1.4 Create Integration Test

Create `tests/test_integration.py`:

```python
# tests/test_integration.py
from dash.testing.application_runners import import_app
from pathlib import Path
import time

def test_mushroom_analysis_flow(dash_duo):
    """Test complete analysis flow with mushroom dataset."""
    app = import_app("App.app")
    dash_duo.start_server(app)
    
    # Upload mushroom file
    mushroom_path = str(Path("datasets/mushroom_spectra.mgf").absolute())
    upload = dash_duo.find_element("#upload-data")
    upload.send_keys(mushroom_path)
    
    # Wait for file info
    dash_duo.wait_for_contains_text("#file-upload-info", "mushroom_spectra.mgf", timeout=5)
    
    # Set motifs to 30
    n_motifs = dash_duo.find_element("#n-motifs")
    n_motifs.clear()
    n_motifs.send_keys("30")
    
    # Run analysis
    dash_duo.find_element("#run-button").click()
    
    # Wait for completion
    dash_duo.wait_for_text_to_equal("#tabs .tab-selected", "Motif Rankings", timeout=60)
    
    # Verify results
    table = dash_duo.find_element("#motif-rankings-table")
    assert table is not None
```

- [ ] Run `pytest tests/test_integration.py -v` to verify test works

## Phase 2: Create Service Layer (Day 2)

### 2.1 Create Service Directory Structure

- [ ] Create `App/services/` directory
- [ ] Create `App/services/__init__.py`
- [ ] Create service modules:
  - [ ] `App/services/analysis.py`
  - [ ] `App/services/motifs.py`
  - [ ] `App/services/spectra.py`
  - [ ] `App/services/network.py`
  - [ ] `App/services/visualization.py`

### 2.2 Move Visualization Functions

Extract these functions from `App/callbacks.py` to `App/services/visualization.py`:

- [ ] Copy `calculate_motif_shares()` (lines ~56-155)
- [ ] Copy `make_spectrum_plot()` (lines ~156-452)
- [ ] Copy `apply_common_layout()` (lines ~453-468)
- [ ] Copy `make_spectrum_from_dict()` (lines ~2534-2563)
- [ ] Copy `filter_and_normalize_spectra()` (lines ~2564-2629)
- [ ] Copy any required imports
- [ ] Create unit tests in `tests/test_services/test_visualization.py`

### 2.3 Create Analysis Service

Create `App/services/analysis.py`:

```python
# App/services/analysis.py
"""Analysis and data loading services."""
import base64
import tempfile
from MS2LDA import run
from MS2LDA.Preprocessing import load_and_clean

def execute_analysis(file_contents, filename, n_motifs, acquisition_type, 
                    polarity, advanced_params):
    """Run MS2LDA analysis using existing run function."""
    # Parse uploaded file
    spectra = parse_uploaded_file(file_contents, filename)
    
    # Use existing run_ms2lda function
    lda_dict, spectra_dicts, doc2spec_dict = run.run_ms2lda(
        spectra=spectra,
        n_motifs=n_motifs,
        acquisition_type=acquisition_type,
        **advanced_params
    )
    
    # Calculate additional data for UI
    motif_spectra_ids = {}
    for motif in lda_dict["beta"]:
        motif_spectra_ids[motif] = [
            doc for doc, theta in lda_dict["theta"].items()
            if theta.get(motif, 0) > 0.1
        ]
    
    return {
        "lda-dict-store": lda_dict,
        "spectra-store": spectra_dicts,
        "selected-motif-store": "motif_0",
        "motif-spectra-ids-store": motif_spectra_ids
    }

def parse_uploaded_file(contents, filename):
    """Parse uploaded file using existing loaders."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    with tempfile.NamedTemporaryFile(suffix=filename, delete=False) as tmp:
        tmp.write(decoded)
        tmp_path = tmp.name
    
    # Use existing loaders
    if filename.endswith('.mgf'):
        return list(load_and_clean.load_mgf(tmp_path))
    elif filename.endswith('.mzML'):
        return list(load_and_clean.load_mzml(tmp_path))
    elif filename.endswith('.msp'):
        return list(load_and_clean.load_msp(tmp_path))
    else:
        raise ValueError(f"Unsupported file type: {filename}")
```

- [ ] Create tests in `tests/test_services/test_analysis.py`
- [ ] Verify service outputs match baseline

### 2.4 Create Other Services

Following the same pattern:

- [ ] Create `App/services/motifs.py`:
  - [ ] Move `compute_motif_degrees()` from callbacks
  - [ ] Add helper functions for motif operations
  
- [ ] Create `App/services/spectra.py`:
  - [ ] Add spectrum search functions
  - [ ] Add spectrum processing functions
  
- [ ] Create `App/services/network.py`:
  - [ ] Move `create_cytoscape_elements()` from callbacks
  - [ ] Move `compute_motif_degrees()` if network-related

## Phase 3: Tab-by-Tab Migration (Days 3-6)

### 3.1 Migrate Run Analysis Tab (Day 3)

- [ ] Create `App/layout/run_analysis.py`
- [ ] Copy layout from `App/layout.py` `create_run_analysis_tab()` function
- [ ] Move related callbacks from `App/callbacks.py`:
  - [ ] `handle_run_or_load()`
  - [ ] `toggle_advanced_settings()`
  - [ ] Any other Run Analysis specific callbacks
- [ ] Update callbacks to use service functions
- [ ] Update `App/app.py` to import from new location:
  ```python
  from App.layout import run_analysis
  # In layout, replace:
  # layout.create_run_analysis_tab()
  # with:
  # run_analysis.create_layout()
  ```
- [ ] Run tests:
  - [ ] `pytest tests/test_app_integrity.py -v`
  - [ ] `pytest tests/test_integration.py -v`
- [ ] Manual test in browser

### 3.2 Migrate Load Results Tab (Day 3)

- [ ] Create `App/layout/load_results.py`
- [ ] Copy layout from `create_load_results_tab()`
- [ ] Move related callbacks
- [ ] Update imports in `App/app.py`
- [ ] Run all tests

### 3.3 Migrate Motif Rankings Tab (Day 4)

- [ ] Create `App/layout/motif_rankings.py`
- [ ] Copy layout from `create_motif_rankings_tab()`
- [ ] Move callbacks:
  - [ ] `update_motif_rankings_table()`
  - [ ] `handle_massql_query()`
  - [ ] Related toggle callbacks
- [ ] Update to use services
- [ ] Update imports
- [ ] Run all tests

### 3.4 Migrate Motif Details Tab (Day 4)

- [ ] Create `App/layout/motif_details.py`
- [ ] Copy layout from `create_motif_details_tab()`
- [ ] Move callbacks:
  - [ ] `update_motif_details()`
  - [ ] `update_spectrum_plot()`
  - [ ] Related callbacks
- [ ] Update imports
- [ ] Run all tests

### 3.5 Migrate Spectra Search Tab (Day 5)

- [ ] Create `App/layout/spectra_search.py`
- [ ] Copy layout from `create_spectra_search_tab()`
- [ ] Move callbacks:
  - [ ] `update_spectra_search_table()`
  - [ ] Search-related callbacks
- [ ] Update imports
- [ ] Run all tests

### 3.6 Migrate Network View Tab (Day 5)

- [ ] Create `App/layout/network_view.py`
- [ ] Copy layout from `create_cytoscape_network_tab()`
- [ ] Move callbacks:
  - [ ] `update_cytoscape()`
  - [ ] Network-related callbacks
- [ ] Update imports
- [ ] Run all tests

### 3.7 Migrate Motif Search Tab (Day 6)

- [ ] Create `App/layout/motif_search.py`
- [ ] Copy layout from `create_screening_tab()`
- [ ] Move callbacks:
  - [ ] `compute_spec2vec_screening()`
  - [ ] Screening-related callbacks
- [ ] Update imports
- [ ] Run all tests

## Phase 4: Cleanup and Documentation (Day 7)

### 4.1 Remove Old Code

- [ ] Backup original files:
  ```bash
  cp App/layout.py App/layout.py.backup
  cp App/callbacks.py App/callbacks.py.backup
  ```
- [ ] Remove old layout creation functions from `App/layout.py`
- [ ] Remove migrated callbacks from `App/callbacks.py`
- [ ] Delete empty files if all code has been moved

### 4.2 Update Documentation

- [ ] Update README with new structure
- [ ] Add docstrings to all service functions
- [ ] Create developer guide for adding new features

### 4.3 Final Testing

- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Manual testing of all features:
  - [ ] Upload and analyze mushroom dataset
  - [ ] View motif rankings
  - [ ] Click through to motif details
  - [ ] Search spectra
  - [ ] View network
  - [ ] Run motif search
- [ ] Performance check - ensure no degradation

### 4.4 Commit and Push

- [ ] Review all changes
- [ ] Commit with message: "Refactor Dash app into modular structure"
- [ ] Push to branch

## Success Criteria

- ✅ All tests pass
- ✅ No duplicate callbacks detected
- ✅ App behavior unchanged (verified against baseline)
- ✅ Code files under 500 lines each
- ✅ Business logic separated from UI
- ✅ Each module has unit tests

## Rollback Plan

If any step fails:
1. Restore from backup files
2. Revert git changes: `git checkout -- App/`
3. Identify issue and fix before retrying

## Important Notes

1. **Test after each migration** - Don't move to next tab until current one works
2. **Keep imports minimal** - Only import what's needed in each module
3. **Preserve callback signatures** - Don't change Input/Output/State definitions
4. **Use existing MS2LDA functions** - Don't reimplement what already exists
5. **Maintain component IDs** - Changing IDs will break callbacks

## Test Data Reference

Primary test file: `datasets/mushroom_spectra.mgf`
- Size: 2.7MB
- Use for all testing to ensure consistency
- Set n_motifs=30 for faster tests