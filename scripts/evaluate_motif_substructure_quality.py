#!/usr/bin/env python
"""Evaluate motif substructure annotation quality against known molecules.

Usage:
    python scripts/evaluate_motif_substructure_quality.py annotations.csv memberships.csv --out-dir results/my_model
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MS2LDA.Add_On.Fingerprints.FP_annotation import annotate_motifs
from MS2LDA.Add_On.Fingerprints.FP_calculation.rdkit_fps import calc_MACCS


ANNOTATION_THRESHOLD = 0.9
FINGERPRINT_TYPE = "maccs"
MEMBERSHIP_THRESHOLD = 0.5
CANDIDATE_RANGES = [
    ("1", 0, 1),
    ("2-4", 2, 4),
    ("5-7", 5, 7),
    ("8-10", 8, 10),
]


@dataclass
class MotifInput:
    motif_id: str
    annotation_smiles: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score motif substructure annotations by comparing annotation "
            "fingerprints with known molecules associated to each motif."
        )
    )
    parser.add_argument(
        "annotations_csv",
        type=Path,
        help="CSV with columns motif_id and annotation_smiles.",
    )
    parser.add_argument(
        "memberships_csv",
        type=Path,
        help="CSV with columns motif_id, smiles, and membership_score.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where score CSVs and summary JSON will be written.",
    )
    return parser.parse_args()


def normalise_smiles_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            values = list(parsed)
        elif "|" in stripped:
            values = stripped.split("|")
        else:
            values = [stripped]
    else:
        values = [str(value)]
    return [str(item).strip() for item in values if str(item).strip()]


def safe_mol(smiles: str):
    try:
        return MolFromSmiles(smiles)
    except Exception:
        return None


def canonical_smiles(mol) -> str:
    try:
        return MolToSmiles(mol)
    except Exception:
        return ""


def motif_annotation_fingerprint(annotation_smiles: list[str]) -> np.ndarray | None:
    if not annotation_smiles:
        return None
    try:
        fp = annotate_motifs(
            [annotation_smiles],
            fp_type=FINGERPRINT_TYPE,
            threshold=ANNOTATION_THRESHOLD,
        )[0]
    except Exception:
        return None
    return np.asarray(fp, dtype=bool)


def molecule_fingerprints(mols: list) -> np.ndarray:
    if not mols:
        return np.empty((0, 0), dtype=bool)
    return np.asarray(calc_MACCS(mols), dtype=bool)


def calculate_sos(annotation_fp: np.ndarray, molecule_fp: np.ndarray) -> float:
    denom = int(annotation_fp.sum())
    if denom == 0:
        return 0.0
    overlap = int(np.logical_and(annotation_fp, molecule_fp).sum())
    return float(overlap / denom) if overlap > 0 else 0.0


def load_csv_inputs(
    annotations_csv: Path, memberships_csv: Path, membership_threshold: float
) -> tuple[list[MotifInput], dict[str, list[str]], dict]:
    annotations = pd.read_csv(annotations_csv)
    memberships = pd.read_csv(memberships_csv)

    required_annotation_cols = {"motif_id", "annotation_smiles"}
    required_membership_cols = {"motif_id", "smiles", "membership_score"}
    missing_annotation = required_annotation_cols - set(annotations.columns)
    missing_membership = required_membership_cols - set(memberships.columns)
    if missing_annotation:
        raise ValueError(f"Missing annotation CSV columns: {sorted(missing_annotation)}")
    if missing_membership:
        raise ValueError(f"Missing membership CSV columns: {sorted(missing_membership)}")

    motif_inputs = []
    for motif_id, group in annotations.groupby("motif_id", sort=False):
        annotation_smiles = []
        for value in group["annotation_smiles"]:
            annotation_smiles.extend(normalise_smiles_list(value))
        motif_inputs.append(
            MotifInput(
                motif_id=str(motif_id),
                annotation_smiles=annotation_smiles,
            )
        )

    filtered = memberships[memberships["membership_score"] >= membership_threshold]
    associated_smiles_by_motif = defaultdict(list)
    invalid_smiles = 0
    for row in filtered.itertuples(index=False):
        mol = safe_mol(str(row.smiles))
        if mol is None:
            invalid_smiles += 1
            continue
        associated_smiles_by_motif[str(row.motif_id)].append(canonical_smiles(mol))

    metadata = {
        "input_type": "csv",
        "annotations_csv": str(annotations_csv),
        "memberships_csv": str(memberships_csv),
        "motif_count": len(motif_inputs),
        "membership_rows": len(memberships),
        "membership_rows_above_threshold": len(filtered),
        "invalid_smiles": invalid_smiles,
    }
    return motif_inputs, dict(associated_smiles_by_motif), metadata


def quality_bin(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    if value <= 0.6:
        return "low"
    if value <= 0.8:
        return "intermediate"
    return "high"


def score_motifs(
    motif_inputs: list[MotifInput],
    associated_smiles_by_motif: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    associated_rows = []

    for motif in tqdm(motif_inputs, desc=f"Scoring motifs ({FINGERPRINT_TYPE})"):
        annotation_mols = [safe_mol(smiles) for smiles in motif.annotation_smiles]
        valid_annotation_smiles = [
            canonical_smiles(mol) for mol in annotation_mols if mol is not None
        ]
        annotation_fp = motif_annotation_fingerprint(valid_annotation_smiles)

        associated_smiles = associated_smiles_by_motif.get(motif.motif_id, [])
        associated_mols = [safe_mol(smiles) for smiles in associated_smiles]
        associated_mols = [mol for mol in associated_mols if mol is not None]
        associated_canonical_smiles = [canonical_smiles(mol) for mol in associated_mols]
        associated_count = len(associated_mols)

        for idx, smiles in enumerate(associated_canonical_smiles):
            associated_rows.append(
                {
                    "motif_id": motif.motif_id,
                    "associated_molecule_index": idx,
                    "associated_molecule_smiles": smiles,
                }
            )

        sos = np.nan
        if annotation_fp is not None and associated_count > 0:
            fps = molecule_fingerprints(associated_mols)
            if fps.size > 0:
                sos_values = [calculate_sos(annotation_fp, fp) for fp in fps]
                sos = float(np.mean(sos_values))

        for range_label, min_count, max_count in CANDIDATE_RANGES:
            included = (
                not pd.isna(sos)
                and associated_count >= min_count
                and associated_count <= max_count
            )
            rows.append(
                {
                    "motif_id": motif.motif_id,
                    "candidate_range": range_label,
                    "candidate_min_count": min_count,
                    "candidate_max_count": max_count,
                    "included_in_range": bool(included),
                    "associated_molecule_count": associated_count,
                    "annotation_smiles": "|".join(valid_annotation_smiles),
                    "annotation_smiles_count": len(valid_annotation_smiles),
                    "associated_molecule_smiles": "|".join(associated_canonical_smiles),
                    "sos": sos if included else np.nan,
                    "quality_bin": quality_bin(sos) if included else "",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(associated_rows)


def summarize_scores(scores: pd.DataFrame, metadata: dict, output_dir: Path) -> dict:
    included = scores[scores["included_in_range"]].copy()
    quality_counts = included["quality_bin"].value_counts().to_dict()
    panel_counts = included.groupby("candidate_range").size().to_dict()
    median_sos = included["sos"].median()
    mean_sos = included["sos"].mean()
    total_motifs = int(metadata.get("motif_count", scores["motif_id"].nunique()))

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "fingerprint": FINGERPRINT_TYPE,
            "membership_threshold": MEMBERSHIP_THRESHOLD,
            "annotation_threshold": ANNOTATION_THRESHOLD,
            "candidate_ranges": ",".join(label for label, _, _ in CANDIDATE_RANGES),
        },
        "input": metadata,
        "metrics": {
            "total_motifs": total_motifs,
            "total_rows": int(len(scores)),
            "evaluated_rows": int(len(included)),
            "evaluated_unique_motifs": int(included["motif_id"].nunique()),
            "coverage_fraction": (
                float(included["motif_id"].nunique() / total_motifs)
                if total_motifs
                else 0.0
            ),
            "mean_sos": None if pd.isna(mean_sos) else float(mean_sos),
            "median_sos": None if pd.isna(median_sos) else float(median_sos),
            "quality_adjusted_coverage": (
                float(included["sos"].sum() / total_motifs) if total_motifs else 0.0
            ),
            "quality_counts": {
                "low": int(quality_counts.get("low", 0)),
                "intermediate": int(quality_counts.get("intermediate", 0)),
                "high": int(quality_counts.get("high", 0)),
            },
            "candidate_range_counts": {
                str(key): int(value) for key, value in panel_counts.items()
            },
        },
        "outputs": {
            "scores_csv": str(output_dir / "motif_substructure_scores.csv"),
            "associated_molecules_csv": str(output_dir / "associated_molecules.csv"),
            "summary_json": str(output_dir / "motif_substructure_summary.json"),
        },
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    motif_inputs, associated_smiles_by_motif, metadata = load_csv_inputs(
        args.annotations_csv.resolve(),
        args.memberships_csv.resolve(),
        membership_threshold=MEMBERSHIP_THRESHOLD,
    )

    scores, associated_molecules = score_motifs(
        motif_inputs=motif_inputs,
        associated_smiles_by_motif=associated_smiles_by_motif,
    )
    scores.to_csv(args.out_dir / "motif_substructure_scores.csv", index=False)
    associated_molecules.to_csv(args.out_dir / "associated_molecules.csv", index=False)

    summary = summarize_scores(scores, metadata, args.out_dir)
    (args.out_dir / "motif_substructure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
