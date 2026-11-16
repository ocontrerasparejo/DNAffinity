#!/usr/bin/env python3
"""Verify canonical PCA inputs and produce clean electrostatic/combined PCA."""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_config import (  # noqa: E402
    FEATURE_GROUPS_CLEAN_TETRAMER, FEATURE_GROUPS_CLEAN_HEXAMER,
    FEATURES_CLEAN_TETRAMER, FEATURES_CLEAN_HEXAMER,
)

DATA_DIR = ROOT / "data" / "selex"
OUT_DIR = ROOT / "outputs" / "eda" / "pca"
THRESHOLDS = (0.80, 0.90, 0.95)


def load_table(length: str):
    groups = FEATURE_GROUPS_CLEAN_TETRAMER if length == "tetramer" else FEATURE_GROUPS_CLEAN_HEXAMER
    features = FEATURES_CLEAN_TETRAMER if length == "tetramer" else FEATURES_CLEAN_HEXAMER
    expected_rows = 256 if length == "tetramer" else 4096
    data = pd.read_csv(DATA_DIR / length / f"input_{length}_clean.csv")
    if data.columns.tolist() != ["SEQUENCE", *features]:
        raise ValueError(f"{length} clean columns do not match canonical feature configuration.")
    if len(data) != expected_rows or data.SEQUENCE.nunique() != expected_rows:
        raise ValueError(f"{length} sequence coverage validation failed.")
    if data.isna().any().any() or data.SEQUENCE.duplicated().any():
        raise ValueError(f"{length} clean table has missing values or duplicate sequences.")
    k = 4 if length == "tetramer" else 6
    expected_sequences = {"".join(bases) for bases in product("ACGT", repeat=k)}
    if set(data.SEQUENCE) != expected_sequences:
        raise ValueError(f"{length} clean table does not contain every expected DNA sequence.")
    return data, groups, features


def validate_clean_against_original(length: str, data: pd.DataFrame, features: list[str]) -> dict:
    """Compare clean values with original values by sequence identifier, exactly."""
    original_path = DATA_DIR / length / f"input_{length}.dat"
    original = pd.read_csv(original_path, sep=r"\s+")
    if "SEQUENCE" not in original or original.SEQUENCE.duplicated().any():
        raise ValueError(f"Original {length} table has missing or repeated sequence IDs.")
    if set(original.SEQUENCE) != set(data.SEQUENCE):
        raise ValueError(f"Clean and original {length} sequence coverage differs.")
    aligned = original.set_index("SEQUENCE").loc[data.SEQUENCE, features]
    if not np.array_equal(aligned.to_numpy(), data.loc[:, features].to_numpy()):
        raise ValueError(f"Clean {length} descriptor values differ from original table by sequence.")
    return {
        "Sequence_length": length,
        "Original_rows": len(original),
        "Clean_rows": len(data),
        "Canonical_feature_count": len(features),
        "Sequence_ID_sets_match": True,
        "Values_match_exactly_by_sequence": True,
        "Original_missing_values": int(original[features].isna().sum().sum()),
        "Clean_missing_values": int(data[features].isna().sum().sum()),
    }


def make_plots(length: str, config_slug: str, label: str, variance: np.ndarray,
               loadings: pd.DataFrame, scores: np.ndarray, sequences: pd.Series) -> None:
    plot_dir = OUT_DIR / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"pca_{length}_{config_slug}"
    pcs = np.arange(1, len(variance) + 1)
    cumulative = np.cumsum(variance)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcs, variance, marker="o")
    ax.set(title=f"Scree plot: {label} ({length}s)", xlabel="Principal component", ylabel="Explained variance ratio")
    ax.set_xticks(pcs)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_scree.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcs, cumulative, marker="o")
    for threshold in THRESHOLDS:
        ax.axhline(threshold, linestyle="--", linewidth=1, label=f"{threshold:.0%}")
    ax.set(title=f"Cumulative explained variance: {label} ({length}s)",
           xlabel="Number of principal components", ylabel="Cumulative explained variance", ylim=(0, 1.05))
    ax.set_xticks(pcs)
    ax.legend(title="Threshold")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_cumulative_variance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(variance) * 0.35), max(7, len(loadings) * 0.28)))
    sns.heatmap(loadings.T, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "PCA loading"})
    ax.set(title=f"PCA loadings: {label} ({length}s)", xlabel="Canonical descriptor", ylabel="Principal component")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_loadings.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    size = 28 if length == "tetramer" else 10
    ax.scatter(scores[:, 0], scores[:, 1], s=size, alpha=0.45, color="steelblue")
    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.6)
    ax.axvline(0, color="grey", linewidth=0.6, alpha=0.6)
    ax.set(title=f"PCA scores: {label} ({length}s)",
           xlabel=f"PC1 ({variance[0]:.1%} variance explained)",
           ylabel=f"PC2 ({variance[1]:.1%} variance explained)")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_scores.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_pca(length: str, data: pd.DataFrame, groups: dict[str, list[str]], config_name: str,
            feature_groups: list[str], sequence_col: str = "SEQUENCE") -> dict:
    features = [feature for group in feature_groups for feature in groups[group]]
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate feature names in PCA input: {config_name}")
    if not set(features).issubset(data.columns):
        raise KeyError(f"Missing canonical PCA feature in {config_name}.")
    X = data.loc[:, features]
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in features) or X.isna().any().any():
        raise ValueError(f"Invalid numeric PCA input in {length}: {config_name}")
    scaled = StandardScaler().fit_transform(X)
    model = PCA()
    scores = model.fit_transform(scaled)
    variance = model.explained_variance_ratio_
    cumulative = np.cumsum(variance)
    loadings = pd.DataFrame(
        model.components_.T * np.sqrt(model.explained_variance_),
        index=features,
        columns=[f"PC{i}" for i in range(1, len(variance) + 1)],
    )
    slug = config_name.lower().replace(" + ", "_").replace("-", "")
    label = config_name.replace("Electrostatic", "electrostatic")
    variance_table = pd.DataFrame({
        "Sequence_length": length,
        "Configuration": config_name,
        "N_sequences": len(data),
        "N_features": len(features),
        "Principal_component": [f"PC{i}" for i in range(1, len(variance) + 1)],
        "Explained_variance_ratio": variance,
        "Cumulative_explained_variance": cumulative,
    })
    threshold_table = pd.DataFrame({
        "Sequence_length": length,
        "Configuration": config_name,
        "Threshold": THRESHOLDS,
        "Components_required": [int(np.searchsorted(cumulative, t) + 1) for t in THRESHOLDS],
        "N_features": len(features),
    })
    variance_table.to_csv(OUT_DIR / f"{length}_{slug}_explained_variance.csv", index=False)
    threshold_table.to_csv(OUT_DIR / f"{length}_{slug}_thresholds.csv", index=False)
    loadings.to_csv(OUT_DIR / f"{length}_{slug}_loadings.csv", index_label="Feature")
    score_frame = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, scores.shape[1] + 1)])
    score_frame.insert(0, sequence_col, data[sequence_col].to_numpy())
    score_frame.to_csv(OUT_DIR / f"{length}_{slug}_scores.csv", index=False)
    make_plots(length, slug, label, variance, loadings, scores, data[sequence_col])
    result = {
        "Sequence_length": length, "Configuration": config_name,
        "N_sequences": len(data), "N_features": len(features),
        "PC1_variance": float(variance[0]), "PC2_variance": float(variance[1]),
        "PCs_for_80_percent": int(np.searchsorted(cumulative, 0.80) + 1),
        "PCs_for_90_percent": int(np.searchsorted(cumulative, 0.90) + 1),
        "PCs_for_95_percent": int(np.searchsorted(cumulative, 0.95) + 1),
        "Feature_names_match_canonical": True,
    }
    print(f"{length} | {config_name}: {len(features)} features; PCs 80/90/95% = "
          f"{result['PCs_for_80_percent']}/{result['PCs_for_90_percent']}/{result['PCs_for_95_percent']}", flush=True)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows, results, input_checks = [], [], []
    for length in ("tetramer", "hexamer"):
        data, groups, flat = load_table(length)
        input_checks.append(validate_clean_against_original(length, data, flat))
        old_counts = {
            "AVG": len(groups["AVG"]), "FCE": len(groups["FCE"]),
            "Electrostatic": 5, "Ion-related": len(groups["Ion-related"]),
        }
        for group, canonical in groups.items():
            audit_rows.append({
                "Sequence_length": length, "Group": group,
                "Canonical_feature_count": len(canonical),
                "Existing_notebook_feature_count": old_counts[group],
                "Existing_notebook_input_matches_canonical": old_counts[group] == len(canonical),
                "Action": "Recompute" if old_counts[group] != len(canonical) else "Retain as feature-consistent",
            })
        existing_combinations = {
            "AVG + FCE": ["AVG", "FCE"],
            "AVG + FCE + Electrostatic": ["AVG", "FCE", "Electrostatic"],
            "AVG + FCE + Electrostatic + Ion-related": ["AVG", "FCE", "Electrostatic", "Ion-related"],
        }
        for combination, selected_groups in existing_combinations.items():
            old_count = sum(old_counts[group] for group in selected_groups)
            canonical_count = sum(len(groups[group]) for group in selected_groups)
            audit_rows.append({
                "Sequence_length": length, "Group": combination,
                "Canonical_feature_count": canonical_count,
                "Existing_notebook_feature_count": old_count,
                "Existing_notebook_input_matches_canonical": old_count == canonical_count,
                "Action": "Recompute" if old_count != canonical_count else "Retain as feature-consistent",
            })
        configurations = [
            ("Electrostatic", ["Electrostatic"]),
            ("AVG + FCE", ["AVG", "FCE"]),
            ("AVG + FCE + Electrostatic", ["AVG", "FCE", "Electrostatic"]),
            ("AVG + FCE + Electrostatic + Ion-related", ["AVG", "FCE", "Electrostatic", "Ion-related"]),
        ]
        for name, selected in configurations:
            results.append(run_pca(length, data, groups, name, selected))
    pd.DataFrame(audit_rows).to_csv(OUT_DIR / "pca_feature_audit.csv", index=False)
    pd.DataFrame(input_checks).to_csv(OUT_DIR / "canonical_input_validation.csv", index=False)
    summary = pd.DataFrame(results)
    summary.to_csv(OUT_DIR / "pca_summary.csv", index=False)
    if len(summary) != 8 or not summary.Feature_names_match_canonical.all():
        raise AssertionError("PCA summary incomplete or canonical feature validation failed.")
    expected = {
        ("tetramer", "Electrostatic"): 3,
        ("hexamer", "Electrostatic"): 3,
        ("tetramer", "AVG + FCE"): 27,
        ("hexamer", "AVG + FCE"): 27,
        ("tetramer", "AVG + FCE + Electrostatic"): 30,
        ("hexamer", "AVG + FCE + Electrostatic"): 30,
        ("tetramer", "AVG + FCE + Electrostatic + Ion-related"): 36,
        ("hexamer", "AVG + FCE + Electrostatic + Ion-related"): 40,
    }
    for _, row in summary.iterrows():
        if row.N_features != expected[(row.Sequence_length, row.Configuration)]:
            raise AssertionError(f"Unexpected PCA feature count: {row.to_dict()}")
    print(f"Saved {len(summary)} canonical PCA summaries and {len(list((OUT_DIR / 'figures').glob('*.png')))} figures.")


if __name__ == "__main__":
    main()
