#!/usr/bin/env python
"""
Command-line interface for running MS2LDA analysis.

This script allows running the core MS2LDA workflow with specified parameters,
optionally loading advanced settings from a JSON configuration file.
It also provides an option to download necessary Spec2Vec data files.
"""

import os
# If running on mac, force single-thread numeric ops for tomotopy to prevent crashing. Maybe can remove this later.
import platform

if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import sys
import traceback

# Check if everything can be imported correctly
try:
    from MS2LDA.run import run as run_ms2lda
    from MS2LDA.utils import download_model_and_data
except ModuleNotFoundError:
    print("ERROR: Could not import the MS2LDA package.", file=sys.stderr)
    print("Please ensure that MS2LDA is installed correctly and that", file=sys.stderr)
    print("you are running this script from the project root directory", file=sys.stderr)
    print("or have set the PYTHONPATH environment variable.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# Default parameters mirroring the Dash app
DEFAULT_PARAMS = {
    "dataset_parameters": {
        "acquisition_type": "DDA",
        "charge": 1,
        "significant_digits": 2,
        "name": "ms2lda_cli_run",
        "output_folder": "ms2lda_results"
    },
    "train_parameters": {
        "parallel": 3,
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
    "motif_parameter": 50,
    "fingerprint_parameters": {
        "fp_type": "maccs",
        "threshold": 0.8
    }
}


def deep_update(source, overrides):
    """
    Helper function to deeply update nested dictionaries.
    Modifies 'source' in place.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source


def main():
    """Parses arguments, merges parameters, downloads data if requested, and runs MS2LDA."""

    # EARLY EXIT: pure download mode triggered by --only-download
    if "--only-download" in sys.argv:
        try:
            print("\n--- Downloading Spec2Vec Data (if missing) ---")
            print(download_model_and_data(mode="positive"))  # TODO: support negative mode later
            print("--------------------------------------------\n")
        except Exception as e:
            print(f"ERROR during download: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Run MS2LDA analysis from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the input mass spectrometry data file (e.g., .mgf, .mzml, .msp)")
    parser.add_argument("--n-motifs", type=int, required=True,
                        help="Number of motifs (topics) to discover.")
    parser.add_argument("--n-iterations", type=int, required=True,
                        help="Number of LDA training iterations.")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Path to the directory where results should be saved.")

    # Optional arguments
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to an optional JSON configuration file for advanced parameters.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional name for the run (used in output filenames). Overrides default/config.")

    # Utility arguments
    parser.add_argument("--download-spec2vec", action="store_true",
                        help="Download the required Spec2Vec model and data files before running the analysis.")
    parser.add_argument("--only-download", action="store_true",
                        help="Only perform the download specified by --download-spec2vec and then exit.")

    args = parser.parse_args()

    # Download Spec2Vec data if needed
    if args.download_spec2vec:
        try:
            print("\n--- Downloading Spec2Vec Data (if missing) ---")
            status_message = download_model_and_data(mode="positive")  # TODO: support negative mode later
            print(status_message)
            print("--------------------------------------------\n")
        except Exception as e:
            print(f"\nERROR: Failed during Spec2Vec data download: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    # Parameter Loading and Merging
    current_params = DEFAULT_PARAMS.copy()  # Start with defaults

    # 1. Load from config file (if provided)
    if args.config:
        if not os.path.exists(args.config):
            print(f"Warning: Configuration file specified but not found at {args.config}. Using defaults.",
                  file=sys.stderr)
        else:
            with open(args.config, 'r') as f:
                config_params = json.load(f)
            print(f"Loaded configuration from: {args.config}")
            # Deep update defaults with config file values
            current_params = deep_update(current_params, config_params)

    # 2. Override with specific required CLI arguments
    current_params["dataset_parameters"]["output_folder"] = args.output_folder
    if args.run_name:
        current_params["dataset_parameters"]["name"] = args.run_name
    elif "name" not in current_params["dataset_parameters"] or current_params["dataset_parameters"]["name"] == \
            DEFAULT_PARAMS["dataset_parameters"]["name"]:
        if args.dataset:
            base_name = os.path.splitext(os.path.basename(args.dataset))[0]
            current_params["dataset_parameters"]["name"] = f"ms2lda_{base_name}"

    # Check if Spec2Vec files exist
    s2v_model = current_params["annotation_parameters"].get("s2v_model_path")
    s2v_embed = current_params["annotation_parameters"].get("s2v_library_embeddings")
    s2v_db = current_params["annotation_parameters"].get("s2v_library_db")

    if not all(os.path.exists(p) for p in [s2v_model, s2v_embed, s2v_db]):
        print("\nWarning: One or more specified Spec2Vec files do not exist:", file=sys.stderr)
        print(f"  Model: {s2v_model} {'(Exists)' if os.path.exists(s2v_model) else '(Missing!)'}", file=sys.stderr)
        print(f"  Embeddings: {s2v_embed} {'(Exists)' if os.path.exists(s2v_embed) else '(Missing!)'}", file=sys.stderr)
        print(f"  DB: {s2v_db} {'(Exists)' if os.path.exists(s2v_db) else '(Missing!)'}", file=sys.stderr)
        print("Annotation step might fail. Consider running with --download-spec2vec first.", file=sys.stderr)

    print("\n--- Running MS2LDA with the following parameters ---")
    print(f"Dataset: {args.dataset}")
    print(f"Number of Motifs: {args.n_motifs}")
    print(f"Number of Iterations: {args.n_iterations}")
    print("Effective Parameters:")
    print(json.dumps(current_params, indent=2))
    print("----------------------------------------------------\n")

    # Call MS2LDA Run
    try:
        print("Starting MS2LDA analysis...")
        # Pass CLI args directly, and unpacked dictionaries for the rest
        run_ms2lda(
            dataset=args.dataset,
            n_motifs=args.n_motifs,
            n_iterations=args.n_iterations,
            dataset_parameters=current_params["dataset_parameters"],
            train_parameters=current_params["train_parameters"],
            model_parameters=current_params["model_parameters"],
            convergence_parameters=current_params["convergence_parameters"],
            annotation_parameters=current_params["annotation_parameters"],
            preprocessing_parameters=current_params["preprocessing_parameters"],
            motif_parameter=current_params["motif_parameter"],
            fingerprint_parameters=current_params["fingerprint_parameters"],
            save=True  # Explicitly save results from CLI run
        )
        print("\nMS2LDA analysis completed successfully.")
    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found: {e}", file=sys.stderr)
        print("This might be the input dataset or a Spec2Vec file. Please check paths.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during MS2LDA execution: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
