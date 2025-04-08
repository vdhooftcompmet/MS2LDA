#!/usr/bin/env python
"""
Command-line interface for running MS2LDA analysis.

This script allows running the core MS2LDA workflow with specified parameters,
optionally loading advanced settings from a JSON configuration file.
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

from MS2LDA.run import run as run_ms2lda

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
    """Parses arguments, merges parameters, and runs MS2LDA."""
    parser = argparse.ArgumentParser(description="Run MS2LDA analysis from the command line.")

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
                        help="Optional name for the run (used in output filenames). Defaults to a name derived from the dataset.")

    args = parser.parse_args()

    # Load params
    current_params = DEFAULT_PARAMS.copy()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_params = json.load(f)
            print(f"Loaded configuration from: {args.config}")
            # Deep update defaults with config file values
            current_params = deep_update(current_params, config_params)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {args.config}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON configuration file {args.config}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration file {args.config}: {e}", file=sys.stderr)
            sys.exit(1)

    # Override params with CLI arguments
    # Note: n_motifs and n_iterations are passed directly to run_ms2lda,
    # others update the nested dictionaries.
    current_params["dataset_parameters"]["output_folder"] = args.output_folder
    if args.run_name:
        current_params["dataset_parameters"]["name"] = args.run_name
    elif args.dataset:
        base_name = os.path.splitext(os.path.basename(args.dataset))[0]
        current_params["dataset_parameters"]["name"] = f"ms2lda_{base_name}"

    print("\n--- Running MS2LDA with the following parameters ---")
    print(f"Dataset: {args.dataset}")
    print(f"Number of Motifs: {args.n_motifs}")
    print(f"Number of Iterations: {args.n_iterations}")
    print(json.dumps(current_params, indent=2))
    print("----------------------------------------------------\n")

    # Call MS2LDA Run
    try:
        print("Starting MS2LDA analysis...")
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
            save=True
        )
        print("\nMS2LDA analysis completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during MS2LDA execution: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
