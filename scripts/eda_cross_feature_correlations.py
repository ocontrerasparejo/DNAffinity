#!/usr/bin/env python3
"""Canonical within- and cross-group descriptor correlation audit."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_config import (  # noqa: E402
    FEATURES_CLEAN_TETRAMER, FEATURES_CLEAN_HEXAMER,
    FEATURE_GROUPS_CLEAN_TETRAMER, FEATURE_GROUPS_CLEAN_HEXAMER,
)

DATA_DIR = ROOT / "data" / "selex"
OUT_DIR = ROOT / "outputs" / "eda" / "cross_feature"
GROUP_ORDER = ["AVG", "FCE", "Electrostatic", "Ion-related"]
STRONG_THRESHOLD = 0.7


def load_clean(length_name: str):
    if length_name == "tetramer":
        path = DATA_DIR / "tetramer" / "input_tetramer_clean.csv"
        groups, features = FEATURE_GROUPS_CLEAN_TETRAMER, FEATURES_CLEAN_TETRAMER
        expected_rows = 256
    else:
        path = DATA_DIR / "hexamer" / "input_hexamer_clean.csv"
        groups, features = FEATURE_GROUPS_CLEAN_HEXAMER, FEATURES_CLEAN_HEXAMER
        expected_rows = 4096
    data = pd.read_csv(path)
    if data.columns.tolist() != ["SEQUENCE", *features]:
        raise ValueError(f"{path} feature names/order do not match src/feature_config.py.")
    if len(data) != expected_rows or data["SEQUENCE"].nunique() != expected_rows:
        raise ValueError(f"Unexpected sequence coverage in {path}.")
    if data["SEQUENCE"].duplicated().any() or data.isna().any().any():
        raise ValueError(f"Missing or duplicate rows in {path}.")
    numeric = data.loc[:, features]
    if not all(pd.api.types.is_numeric_dtype(numeric[c]) for c in features):
        raise TypeError(f"Non-numeric feature in {path}.")
    ordered_groups = {group: list(groups[group]) for group in GROUP_ORDER}
    flattened = [feature for group in GROUP_ORDER for feature in ordered_groups[group]]
    if flattened != list(features):
        raise AssertionError(f"Canonical feature group order does not reproduce {path} columns.")
    return data, ordered_groups, list(features)


def pair_tables(correlation: pd.DataFrame, feature_groups: dict[str, list[str]], method: str, length: str):
    feature_group = {name: group for group, names in feature_groups.items() for name in names}
    rows = []
    features = list(correlation.columns)
    for i, first in enumerate(features):
        for second in features[i + 1:]:
            rows.append({
                "Sequence_length": length,
                "Method": method,
                "Feature_1": first,
                "Feature_2": second,
                "Group_1": feature_group[first],
                "Group_2": feature_group[second],
                "Correlation": float(correlation.loc[first, second]),
                "Absolute_correlation": abs(float(correlation.loc[first, second])),
            })
    pairs = pd.DataFrame(rows)
    pairs["Relationship"] = pairs.apply(
        lambda r: "Within-group" if r.Group_1 == r.Group_2 else "Cross-group", axis=1
    )
    possible = pairs.groupby(["Group_1", "Group_2"], as_index=False).size().rename(columns={"size": "Possible_pairs"})
    summary = (
        pairs.assign(Strong=lambda d: d.Absolute_correlation >= STRONG_THRESHOLD)
        .groupby(["Group_1", "Group_2"], as_index=False)
        .agg(Strong_pairs=("Strong", "sum"), Median_absolute_correlation=("Absolute_correlation", "median"),
             Maximum_absolute_correlation=("Absolute_correlation", "max"))
        .merge(possible, on=["Group_1", "Group_2"], validate="one_to_one")
    )
    summary["Strong_pair_fraction"] = summary["Strong_pairs"] / summary["Possible_pairs"]
    summary.insert(0, "Sequence_length", length)
    summary.insert(1, "Method", method)
    return pairs, summary


def save_heatmaps(correlation: pd.DataFrame, groups: dict[str, list[str]], method: str, length: str) -> None:
    heatmap_dir = OUT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(13, len(correlation) * 0.34), max(11, len(correlation) * 0.30)))
    sns.heatmap(correlation, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax,
                cbar_kws={"label": f"{method.title()} correlation"}, xticklabels=True, yticklabels=True)
    offset = 0
    for group in GROUP_ORDER:
        n = len(groups[group])
        if offset:
            ax.axhline(offset, color="black", linewidth=1)
            ax.axvline(offset, color="black", linewidth=1)
        offset += n
    ax.set_title(f"{method.title()} correlations: canonical {length} descriptors")
    ax.set_xlabel("Features ordered by canonical descriptor group")
    ax.set_ylabel("Features ordered by canonical descriptor group")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelrotation=0, labelsize=6)
    fig.tight_layout()
    fig.savefig(heatmap_dir / f"{length}_{method}_all_features.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for i, group_1 in enumerate(GROUP_ORDER):
        for group_2 in GROUP_ORDER[i + 1:]:
            block = correlation.loc[groups[group_1], groups[group_2]]
            fig, ax = plt.subplots(figsize=(max(7, len(groups[group_2]) * 0.42), max(5, len(groups[group_1]) * 0.38)))
            sns.heatmap(block, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax,
                        cbar_kws={"label": f"{method.title()} correlation"})
            ax.set_title(f"{method.title()}: {group_1} vs {group_2} ({length}s)")
            ax.set_xlabel(f"{group_2} features")
            ax.set_ylabel(f"{group_1} features")
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            ax.tick_params(axis="y", labelrotation=0, labelsize=7)
            fig.tight_layout()
            slug = f"{group_1.lower().replace('-', '')}_{group_2.lower().replace('-', '')}"
            fig.savefig(heatmap_dir / f"{length}_{method}_{slug}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs, all_summaries = [], []
    validation = []
    for length in ("tetramer", "hexamer"):
        data, groups, features = load_clean(length)
        validation.append({
            "Sequence_length": length, "N_sequences": len(data), "N_features": len(features),
            "Feature_groups": "; ".join(f"{g}={len(groups[g])}" for g in GROUP_ORDER),
            "Canonical_columns_match": True, "Missing_values": int(data[features].isna().sum().sum()),
            "Unique_sequences": int(data.SEQUENCE.nunique()),
        })
        values = data.loc[:, features]
        for method in ("pearson", "spearman"):
            matrix = values.corr(method=method)
            matrix.to_csv(OUT_DIR / f"{length}_{method}_correlation_matrix.csv")
            pairs, summary = pair_tables(matrix, groups, method.title(), length)
            all_pairs.append(pairs)
            all_summaries.append(summary)
            save_heatmaps(matrix, groups, method, length)
    pair_df = pd.concat(all_pairs, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    pair_df.to_csv(OUT_DIR / "all_feature_pairs.csv", index=False)
    pair_df[pair_df.Absolute_correlation >= STRONG_THRESHOLD].sort_values(
        ["Sequence_length", "Method", "Absolute_correlation"], ascending=[True, True, False]
    ).to_csv(OUT_DIR / "strong_feature_pairs.csv", index=False)
    summary_df.to_csv(OUT_DIR / "group_pair_summary.csv", index=False)
    summary_df[summary_df.Group_1 == summary_df.Group_2].to_csv(OUT_DIR / "within_group_summary.csv", index=False)
    summary_df[summary_df.Group_1 != summary_df.Group_2].to_csv(OUT_DIR / "cross_group_summary.csv", index=False)
    pair_df[pair_df.Group_1 != pair_df.Group_2].sort_values(
        ["Sequence_length", "Method", "Absolute_correlation"], ascending=[True, True, False]
    ).groupby(["Sequence_length", "Method", "Group_1", "Group_2"], sort=False).head(10).to_csv(
        OUT_DIR / "top_cross_group_pairs.csv", index=False
    )
    pd.DataFrame(validation).to_csv(OUT_DIR / "validation_summary.csv", index=False)
    if pair_df["Correlation"].isna().any():
        raise ValueError("Missing values found in correlation matrices.")
    expected_pairs = {"tetramer": 36 * 35 // 2, "hexamer": 40 * 39 // 2}
    for row in validation:
        observed = len(pair_df[(pair_df.Sequence_length == row["Sequence_length"]) & (pair_df.Method == "Pearson")])
        if observed != expected_pairs[row["Sequence_length"]]:
            raise AssertionError(f"Incomplete pair output for {row['Sequence_length']}.")
    print(f"Saved {len(pair_df):,} pair-method rows; {len(list((OUT_DIR / 'heatmaps').glob('*.png')))} heatmaps.")


if __name__ == "__main__":
    main()
