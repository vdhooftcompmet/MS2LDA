#!/usr/bin/env python
"""
Command-line interface for running MS2LDA analysis.

This script allows running the core MS2LDA workflow with specified parameters,
optionally loading advanced settings from a JSON configuration file.
It also provides an option to download necessary Spec2Vec data files.
Adds a unified “aux-data” downloader that fetches:
  • Spec2Vec positive-mode model + embeddings + DB
  • The two fingerprint JARs (cdk-2.2.jar, jCMapperCLI.jar)
  • All MotifDB JSON reference sets
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
from pathlib import Path
from typing import List
from tqdm import tqdm

import requests

try:
    from ms2lda.run import run as run_ms2lda
    from ms2lda.utils import download_model_and_data
    import ms2lda  # locate package root
except ModuleNotFoundError:
    print("ERROR: Could not import the MS2LDA package.", file=sys.stderr)
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
        "s2v_model_path": "../ms2lda/ms2lda/Add_On/Spec2Vec/model_positive_mode/150225_Spec2Vec_pos_CleanedLibraries.model",
        "s2v_library_embeddings": "../ms2lda/ms2lda/Add_On/Spec2Vec/model_positive_mode/150225_CleanedLibraries_Spec2Vec_pos_embeddings.npy",
        "s2v_library_db": "../ms2lda/ms2lda/Add_On/Spec2Vec/model_positive_mode/150225_CombLibraries_spectra.db"
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

# ----------------------------------------------------------------------
# Helper download functions
# ----------------------------------------------------------------------


def _github_file_download(repo: str, file_path: str, local_dst: Path) -> str:
    """
    Download a single file from a public GitHub repo (main branch) **with a
    tqdm progress-bar**, mirroring the behaviour of download_model_and_data().
    Returns a status string.
    """
    api = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"

    # Skip if already present
    if local_dst.exists():
        return f"✓ {local_dst} (exists)"

    # Stream download with progress bar
    with requests.get(api, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        local_dst.parent.mkdir(parents=True, exist_ok=True)
        with open(local_dst, "wb") as fh, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=local_dst.name,
            initial=0,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # skip keep-alive chunks
                    fh.write(chunk)
                    bar.update(len(chunk))

    return f"✓ {local_dst} (downloaded)"


def _github_dir_download(repo: str, dir_path: str, local_dst: Path) -> List[str]:
    """
    Download every file in a directory of a public GitHub repo
    using the Contents API. Returns status strings.
    """
    api = f"https://api.github.com/repos/{repo}/contents/{dir_path}?ref=main"
    r = requests.get(api, timeout=60)
    r.raise_for_status()
    log: List[str] = []

    for item in r.json():
        if item["type"] != "file":
            continue
        tgt = local_dst / item["name"]
        log.append(_github_file_download(repo, f"{dir_path}/{item['name']}", tgt))
    return log


def download_all_aux_data() -> str:
    """
    Download Spec2Vec assets, two FP-calculation JARs, and all MotifDB JSONs.
    Returns a printable summary.
    """
    pkg_root = Path(ms2lda.__file__).resolve().parent
    repo = "vdhooftcompmet/MS2LDA"
    summary: List[str] = []

    # 1. Spec2Vec (positive mode)
    summary.append("→ Spec2Vec assets:")
    summary.append(download_model_and_data(mode="positive").strip())

    # 2. Fingerprint JARs
    summary.append("\n→ Fingerprint JARs:")
    jar_base = pkg_root / "Add_On" / "Fingerprints" / "FP_calculation"
    summary.append(
        _github_file_download(
            repo,
            "ms2lda/Add_On/Fingerprints/FP_calculation/cdk-2.2.jar",
            jar_base / "cdk-2.2.jar",
        )
    )
    summary.append(
        _github_file_download(
            repo,
            "ms2lda/Add_On/Fingerprints/FP_calculation/jCMapperCLI.jar",
            jar_base / "jCMapperCLI.jar",
        )
    )

    # 3. MotifDB JSONs
    summary.append("\n→ MotifDB JSONs:")
    motif_dst = pkg_root / "MotifDB"
    summary.extend(_github_dir_download(repo, "ms2lda/MotifDB", motif_dst))

    return "\n".join(summary)


def deep_update(src, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and k in src and isinstance(src[k], dict):
            deep_update(src[k], v)
        else:
            src[k] = v
    return src


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    if "--only-download" in sys.argv:
        try:
            print("\n--- Downloading auxiliary MS2LDA data ---")
            print(download_all_aux_data())
            print("-----------------------------------------\n")
        except Exception as e:
            print(f"ERROR during download: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    p = argparse.ArgumentParser(
        description="Run MS2LDA analysis from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", required=True, help="Input MS data file (.mgf, .mzml, .msp)"
    )
    p.add_argument(
        "--n-motifs",
        required=True,
        type=int,
        help="Number of motifs (topics) to discover.",
    )
    p.add_argument(
        "--n-iterations",
        required=True,
        type=int,
        help="Number of LDA training iterations.",
    )
    p.add_argument(
        "--output-folder", required=True, help="Directory where results are saved."
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        help="Optional JSON config file for advanced parameters.",
    )
    p.add_argument(
        "--run-name", type=str, help="Optional run name (overrides default/config)."
    )
    p.add_argument(
        "--download-data",
        action="store_true",
        help="Download auxiliary data before running.",
    )
    p.add_argument(
        "--only-download",
        action="store_true",
        help="Just download auxiliary data then exit.",
    )

    args = p.parse_args()

    if args.download_data:
        try:
            print("\n--- Downloading auxiliary MS2LDA data ---")
            print(download_all_aux_data())
            print("-----------------------------------------\n")
        except Exception as e:
            print(f"\nERROR: Failed during data download: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    params = DEFAULT_PARAMS.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            params = deep_update(params, json.load(f))

    params["dataset_parameters"]["output_folder"] = args.output_folder
    if args.run_name:
        params["dataset_parameters"]["name"] = args.run_name
    elif (
        params["dataset_parameters"]["name"]
        == DEFAULT_PARAMS["dataset_parameters"]["name"]
    ):
        base = os.path.splitext(os.path.basename(args.dataset))[0]
        params["dataset_parameters"]["name"] = f"ms2lda_{base}"

    print("\n--- Running MS2LDA with parameters ---")
    print(json.dumps(params, indent=2))
    print("--------------------------------------\n")

    try:
        run_ms2lda(
            dataset=args.dataset,
            n_motifs=args.n_motifs,
            n_iterations=args.n_iterations,
            dataset_parameters=params["dataset_parameters"],
            train_parameters=params["train_parameters"],
            model_parameters=params["model_parameters"],
            convergence_parameters=params["convergence_parameters"],
            annotation_parameters=params["annotation_parameters"],
            preprocessing_parameters=params["preprocessing_parameters"],
            motif_parameter=params["motif_parameter"],
            fingerprint_parameters=params["fingerprint_parameters"],
            save=True,
        )
        print("\nMS2LDA analysis completed successfully.")
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
