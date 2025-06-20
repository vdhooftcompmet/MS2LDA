![header](App/assets/MS2LDA_LOGO_white.jpg)
![Maintainer](https://img.shields.io/badge/maintainer-Rosina_Torres_Ortega-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Jonas_Dietrich-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Joe_Wandy-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12625409.svg)](https://doi.org/10.5281/zenodo.11394248)

**MS2LDA** is an advanced tool designed for unsupervised substructure discovery in mass spectrometry data, utilizing topic modeling and providing automated annotation of discovered motifs. This tool significantly enhances the capabilities described in the [original MS2LDA paper](https://www.pnas.org/doi/abs/10.1073/pnas.1608041113) (2016), offering users an integrated workflow with improved usability, detailed visualizations, and a searchable motif database (MotifDB).

Mass spectrometry fragmentation patterns hold abundant structural information vital for analytical chemistry, natural product research, and food safety assessments. However, interpreting this data remains challenging, and only a fraction of available information is traditionally utilized. MS2LDA addresses this by identifying recurring substructures (motifs) across spectral datasets without relying on prior compound identification, thus accelerating structure elucidation and analysis.

---

# MS2LDA Command Line Tool Usage

MS2LDA provides powerful command-line tools for batch processing and analysis of mass spectrometry data. This guide explains how to use these tools effectively.

## Wrapper Scripts: run_analysis.sh and run_analysis.bat

The repository includes wrapper scripts that simplify running MS2LDA from the command line:

- **run_analysis.sh** - For Linux/macOS users
- **run_analysis.bat** - For Windows users

These scripts serve as convenient wrappers around the main Python script `ms2lda_runfull.py`. They:

1. Set up the proper Python environment by configuring PYTHONPATH
2. Execute the main Python script with all provided arguments
3. Handle platform-specific environment settings

## The ms2lda_runfull.py Script

The `ms2lda_runfull.py` script is the core command-line interface for MS2LDA. It provides a comprehensive set of options for running MS2LDA analyses.

### Important: Downloading Required Data First

Before running any analysis, you must download the necessary auxiliary data files:

```bash
# For Linux/macOS:
./run_analysis.sh --only-download

# For Windows:
run_analysis.bat --only-download
```

This command downloads:
- Spec2Vec models and embeddings
- Fingerprint calculation JARs
- MotifDB reference sets

**This step is required before running any analysis.**

### Basic Usage

```bash
./run_analysis.sh --dataset <input_file> --n-motifs <number> --n-iterations <number> --output-folder <folder>
```

### Key Parameters

- `--dataset`: Input MS data file (.mgf, .mzml, .msp)
- `--n-motifs`: Number of motifs (topics) to discover
- `--n-iterations`: Number of LDA training iterations
- `--output-folder`: Directory where results are saved
- `--config`: Optional JSON config file for advanced parameters
- `--run-name`: Optional custom name for the analysis run
- `--download-data`: Download auxiliary data before running the analysis
- `--only-download`: Just download auxiliary data then exit

### Advanced Configuration

For advanced settings, you can provide a JSON configuration file:

```bash
./run_analysis.sh --dataset input.mgf --n-motifs 200 --n-iterations 5000 --output-folder results --config my_config.json
```

#### Configuration File Structure

The configuration file should be a JSON file with the following structure:

```json
{
    "dataset_parameters": { ... },
    "train_parameters": { ... },
    "model_parameters": { ... },
    "convergence_parameters": { ... },
    "annotation_parameters": { ... },
    "preprocessing_parameters": { ... },
    "motif_parameter": 50,
    "fingerprint_parameters": { ... }
}
```

#### Parameter Groups

1. **dataset_parameters**: Controls dataset-specific settings
   - `acquisition_type`: Type of MS acquisition (e.g., "DDA")
   - `charge`: Charge state of ions (default: 1)
   - `significant_digits`: Number of significant digits for m/z values (default: 2)
   - `name`: Name for the analysis run (default: "ms2lda_cli_run")
   - `output_folder`: Directory where results are saved (default: "ms2lda_results")

2. **train_parameters**: Controls training parallelization
   - `parallel`: Number of parallel processes for training (default: 3)
   - `workers`: Number of worker threads (0 = auto) (default: 0)

3. **model_parameters**: Controls LDA model hyperparameters
   - `rm_top`: Number of top fragments to remove (default: 0)
   - `min_cf`: Minimum count frequency (default: 0)
   - `min_df`: Minimum document frequency (default: 3)
   - `alpha`: Dirichlet prior on document-topic distributions (default: 0.6)
   - `eta`: Dirichlet prior on topic-word distributions (default: 0.01)
   - `seed`: Random seed for reproducibility (default: 42)

4. **convergence_parameters**: Controls when to stop training
   - `step_size`: Number of iterations between convergence checks (default: 50)
   - `window_size`: Window size for convergence calculation (default: 10)
   - `threshold`: Convergence threshold (default: 0.005)
   - `type`: Type of convergence criterion (default: "perplexity_history")

5. **annotation_parameters**: Controls motif annotation
   - `criterium`: Criterion for annotation (default: "biggest")
   - `cosine_similarity`: Minimum cosine similarity for annotation (default: 0.90)
   - `n_mols_retrieved`: Number of molecules to retrieve (default: 5)
   - `s2v_model_path`: Path to Spec2Vec model
   - `s2v_library_embeddings`: Path to Spec2Vec embeddings
   - `s2v_library_db`: Path to Spec2Vec library database

6. **preprocessing_parameters**: Controls data preprocessing
   - `min_mz`: Minimum m/z value to consider (default: 0)
   - `max_mz`: Maximum m/z value to consider (default: 2000)
   - `max_frags`: Maximum number of fragments per spectrum (default: 1000)
   - `min_frags`: Minimum number of fragments per spectrum (default: 5)
   - `min_intensity`: Minimum relative intensity (default: 0.01)
   - `max_intensity`: Maximum relative intensity (default: 1.0)

7. **motif_parameter**: Number of motifs to discover (default: 50)

8. **fingerprint_parameters**: Controls fingerprint calculation
   - `fp_type`: Type of fingerprint to use (default: "maccs")
   - `threshold`: Threshold for fingerprint matching (default: 0.8)

#### Example Configuration

Here's a complete example configuration file:

```json
{
    "dataset_parameters": {
        "acquisition_type": "DDA",
        "charge": 1,
        "significant_digits": 2,
        "name": "my_analysis",
        "output_folder": "my_results"
    },
    "train_parameters": {
        "parallel": 4,
        "workers": 0
    },
    "model_parameters": {
        "rm_top": 0,
        "min_cf": 0,
        "min_df": 3,
        "alpha": 0.6,
        "eta": 0.01,
        "seed": 42
    },
    "convergence_parameters": {
        "step_size": 50,
        "window_size": 10,
        "threshold": 0.005,
        "type": "perplexity_history"
    },
    "annotation_parameters": {
        "criterium": "biggest",
        "cosine_similarity": 0.90,
        "n_mols_retrieved": 5,
        "s2v_model_path": "../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_Spec2Vec_pos_CleanedLibraries.model",
        "s2v_library_embeddings": "../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy",
        "s2v_library_db": "../MS2LDA/MS2LDA/Add_On/Spec2Vec/model_positive_mode/150225_CombLibraries_spectra.db"
    },
    "preprocessing_parameters": {
        "min_mz": 0,
        "max_mz": 2000,
        "max_frags": 1000,
        "min_frags": 5,
        "min_intensity": 0.01,
        "max_intensity": 1.0
    },
    "motif_parameter": 100,
    "fingerprint_parameters": {
        "fp_type": "maccs",
        "threshold": 0.8
    }
}
```

#### Configuration Tips

- You don't need to specify all parameters in your configuration file. Any parameters not specified will use the default values.
- The `--n-motifs` command-line argument overrides the `motif_parameter` in the configuration file.
- The `--output-folder` command-line argument overrides the `output_folder` in the configuration file.
- The `--run-name` command-line argument overrides the `name` in the configuration file.

#### Use Cases

1. **High-Resolution Data**: For high-resolution MS data, increase the `significant_digits` parameter:
   ```json
   "dataset_parameters": {
       "significant_digits": 4
   }
   ```

2. **Large Datasets**: For large datasets, adjust preprocessing to reduce complexity:
   ```json
   "preprocessing_parameters": {
       "min_intensity": 0.05,
       "min_frags": 10
   }
   ```

3. **Fine-Tuning LDA**: To fine-tune the LDA model:
   ```json
   "model_parameters": {
       "alpha": 0.3,
       "eta": 0.005
   }
   ```

4. **Stricter Convergence**: For stricter convergence criteria:
   ```json
   "convergence_parameters": {
       "threshold": 0.001,
       "window_size": 20
   }
   ```

---

## Documentation

For more comprehensive guidance, refer to our complete [MS2LDA Documentation](https://ms2lda.org/), providing detailed instructions, FAQs, and additional resources.

---

## Citing MS2LDA

Please cite our work if you use MS2LDA in your research. *(Citation details coming soon.)*

---

## Our Research Group

[![GitHub Logo](https://github.com/vdhooftcompmet/group-website/blob/main/website/custom/logo/logo.png?raw=true)](https://vdhooftcompmet.github.io)
[![Github Logo](App/assets/WUR_RGB_standard_2021.png?raw=true)](https://www.wur.nl/en.htm)

---
