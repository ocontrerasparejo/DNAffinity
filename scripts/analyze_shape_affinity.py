#!/usr/bin/env python3
"""Exploratory correlations between 10-mer shape descriptors and HT-SELEX labels.

For each selected transcription-factor cycle, this script averages canonical
clean tetramer/hexamer descriptor values over the valid overlapping windows in
an unchanged 10-mer. It writes correlation summaries and heatmaps only; it does
not fit predictive models or write per-sequence intermediate tables.
"""

from __future__ import annotations

import sys
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_config import (  # noqa: E402
    FEATURES_CLEAN_TETRAMER,
    FEATURES_CLEAN_HEXAMER,
    FEATURE_GROUPS_CLEAN_TETRAMER,
    FEATURE_GROUPS_CLEAN_HEXAMER,
)

SELEX_DIR = ROOT / "data" / "selex"
OUTPUT_DIR = ROOT / "outputs" / "shape_affinity"

# Same selected cycle mapping used in notebooks/preprocessing_selex_2.ipynb.
TF_CYCLES = {
    "Gata4": 4,
    "Zfp24": 5,
    "arid3": 6,
    "hoxd1": 3,
    "nkx3": 6,
    "pax3-like": 5,
    "six1": 5,
    "tbx15": 5,
    "telf": 5,
    "uncxb": 5,
    "xbpc": 5,
}

OUTCOMES = ["Affinity", "ObservedCount", "Log10ObservedCount", "SE"]
COMPOSITION_NAMES = [
    "GC_content", "A_frequency", "C_frequency", "G_frequency", "T_frequency"
]
CHUNK_SIZE = 100_000


def load_clean_descriptor_table(sequence_length: int) -> tuple[pd.Index, np.ndarray, list[str], dict[str, str]]:
    """Load a clean table and return exact sequence-key lookup values."""
    if sequence_length == 4:
        size_name = "tetramer"
        features = FEATURES_CLEAN_TETRAMER
        feature_groups = FEATURE_GROUPS_CLEAN_TETRAMER
    elif sequence_length == 6:
        size_name = "hexamer"
        features = FEATURES_CLEAN_HEXAMER
        feature_groups = FEATURE_GROUPS_CLEAN_HEXAMER
    else:
        raise ValueError("Only tetramer and hexamer descriptor tables are supported.")

    path = SELEX_DIR / size_name / f"input_{size_name}_clean.csv"
    data = pd.read_csv(path)
    if "SEQUENCE" not in data.columns:
        raise KeyError(f"SEQUENCE column missing from {path}")
    if data["SEQUENCE"].duplicated().any():
        raise ValueError(f"Duplicate sequence IDs found in {path}")
    if data["SEQUENCE"].isna().any():
        raise ValueError(f"Missing sequence IDs found in {path}")
    if data.columns.tolist() != ["SEQUENCE", *features]:
        raise ValueError(f"Columns in {path} do not match the canonical feature list.")
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in features):
        raise TypeError(f"Non-numeric descriptor value found in {path}")
    if data[features].isna().any().any():
        raise ValueError(f"Missing descriptor value found in {path}")

    group_features = [feature for group in feature_groups.values() for feature in group]
    if group_features != features:
        raise AssertionError(f"Feature groups and flat canonical list differ for {size_name}.")
    feature_to_group = {
        feature: group_name
        for group_name, group in feature_groups.items()
        for feature in group
    }
    return (
        pd.Index(data["SEQUENCE"], name="SEQUENCE"),
        data.loc[:, features].to_numpy(dtype=np.float32),
        list(features),
        feature_to_group,
    )


def average_over_windows(
    sequences: np.ndarray,
    descriptor_index: pd.Index,
    descriptor_values: np.ndarray,
    sequence_length: int,
    output: np.ndarray,
) -> int:
    """Fill output with mean descriptor values across every valid window."""
    n_windows = 10 - sequence_length + 1
    for start in range(n_windows):
        window_keys = [sequence[start:start + sequence_length] for sequence in sequences]
        row_codes = descriptor_index.get_indexer(window_keys)
        if np.any(row_codes < 0):
            missing = sorted({window_keys[i] for i in np.flatnonzero(row_codes < 0)[:10]})
            raise KeyError(f"Missing {sequence_length}-mer descriptor lookups: {missing}")
        np.add(output, descriptor_values[row_codes], out=output)
    output /= n_windows
    return n_windows


def add_sequence_composition(sequences: np.ndarray, output: np.ndarray, offset: int) -> None:
    """Append GC and A/C/G/T frequencies, each measured over ten bases."""
    sequence_bytes = "".join(sequences).encode("ascii")
    bases = np.frombuffer(sequence_bytes, dtype="S1").reshape(len(sequences), 10)
    frequency_columns = {}
    for base in "ACGT":
        frequency = (bases == base.encode("ascii")).mean(axis=1).astype(np.float32)
        frequency_columns[base] = frequency
    output[:, offset] = frequency_columns["C"] + frequency_columns["G"]
    for position, base in enumerate("ACGT", start=offset + 1):
        output[:, position] = frequency_columns[base]


def pearson_feature_outcome_correlations(features: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Compute predictor-to-outcome Pearson r in chunks to limit temporary memory."""
    n_rows, n_features = features.shape
    n_outcomes = outcomes.shape[1]
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)
    sum_y = np.zeros(n_outcomes, dtype=np.float64)
    sum_y2 = np.zeros(n_outcomes, dtype=np.float64)
    sum_xy = np.zeros((n_features, n_outcomes), dtype=np.float64)

    for start in range(0, n_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_rows)
        x_chunk = features[start:end].astype(np.float64, copy=False)
        y_chunk = outcomes[start:end]
        sum_x += x_chunk.sum(axis=0)
        sum_x2 += np.einsum("ij,ij->j", x_chunk, x_chunk)
        sum_y += y_chunk.sum(axis=0)
        sum_y2 += np.einsum("ij,ij->j", y_chunk, y_chunk)
        sum_xy += x_chunk.T @ y_chunk

    covariance = sum_xy - np.outer(sum_x, sum_y) / n_rows
    variance_x = sum_x2 - sum_x**2 / n_rows
    variance_y = sum_y2 - sum_y**2 / n_rows
    denominator = np.sqrt(np.outer(variance_x, variance_y))
    correlations = np.full((n_features, n_outcomes), np.nan, dtype=np.float64)
    valid = denominator > 0
    correlations[valid] = covariance[valid] / denominator[valid]
    return np.clip(correlations, -1.0, 1.0)


def spearman_feature_outcome_correlations(features: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Rank one predictor at a time; Spearman r is Pearson r on average ranks."""
    n_rows, n_features = features.shape
    ranked_outcomes = np.column_stack([
        rankdata(outcomes[:, column], method="average")
        for column in range(outcomes.shape[1])
    ])
    ranked_outcomes -= ranked_outcomes.mean(axis=0)
    outcome_sumsquares = np.einsum("ij,ij->j", ranked_outcomes, ranked_outcomes)
    correlations = np.full((n_features, outcomes.shape[1]), np.nan, dtype=np.float64)

    for feature_index in range(n_features):
        ranked_feature = rankdata(features[:, feature_index], method="average")
        ranked_feature -= ranked_feature.mean()
        feature_sumsquares = np.dot(ranked_feature, ranked_feature)
        denominator = np.sqrt(feature_sumsquares * outcome_sumsquares)
        if feature_sumsquares > 0:
            valid = denominator > 0
            correlations[feature_index, valid] = (
                ranked_feature @ ranked_outcomes[:, valid]
            ) / denominator[valid]
    return np.clip(correlations, -1.0, 1.0)


def make_heatmap(correlation_table: pd.DataFrame, tf_name: str, method: str, feature_groups: list[str]) -> Path:
    """Save an 81-feature by 4-outcome correlation heatmap."""
    matrix = correlation_table.pivot(
        index="Feature", columns="Outcome", values=method
    ).reindex(columns=OUTCOMES)
    matrix = matrix.loc[correlation_table["Feature"].drop_duplicates().tolist()]

    fig_height = max(14, len(matrix.index) * 0.23)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
        cbar_kws={"label": f"{method.title()} correlation"},
    )
    boundaries = []
    previous_group = feature_groups[0]
    for row, group in enumerate(feature_groups[1:], start=1):
        if group != previous_group:
            boundaries.append(row)
        previous_group = group
    for boundary in boundaries:
        ax.axhline(boundary, color="black", linewidth=0.8)
    ax.set_title(f"{method.title()} descriptor–outcome correlations: {tf_name}")
    ax.set_xlabel("HT-SELEX outcome")
    ax.set_ylabel("Mean descriptor or sequence composition")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    fig.tight_layout()
    path = OUTPUT_DIR / f"{tf_name.lower()}_{method}_heatmap.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def analyze_factor(
    tf_name: str,
    cycle: int,
    tetra_index: pd.Index,
    tetra_values: np.ndarray,
    tetra_features: list[str],
    tetra_feature_groups: dict[str, str],
    hexa_index: pd.Index,
    hexa_values: np.ndarray,
    hexa_features: list[str],
    hexa_feature_groups: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create one TF's aggregate feature matrix and calculate both correlations."""
    label_path = SELEX_DIR / "transcription_factors" / tf_name / f"{cycle}.txt"
    required_columns = {"Kmer", "ObservedCount", "Affinity", "SE"}
    labels = pd.read_csv(label_path, sep="\t", usecols=lambda column: column in required_columns)
    missing_columns = required_columns - set(labels.columns)
    if missing_columns:
        raise KeyError(f"Missing label columns in {label_path}: {sorted(missing_columns)}")

    if tf_name == "hoxd1" and "#" in labels.columns:
        labels = labels.drop(columns="#")
    sequences = labels["Kmer"].astype(str).to_numpy()
    valid_sequences = labels["Kmer"].str.fullmatch(r"[ACGT]{10}", na=False).to_numpy()
    if not valid_sequences.all():
        raise ValueError(f"Invalid or non-10-mer Kmer found for {tf_name}.")
    duplicate_sequences = int(labels["Kmer"].duplicated().sum())
    missing_label_values = int(labels[["Affinity", "ObservedCount", "SE"]].isna().sum().sum())
    if duplicate_sequences or missing_label_values:
        raise ValueError(
            f"{tf_name}: {duplicate_sequences} duplicate Kmers and "
            f"{missing_label_values} missing target values."
        )
    target_values = labels[["Affinity", "ObservedCount", "SE"]].to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise ValueError(f"Non-finite target value found for {tf_name}.")
    if (labels["ObservedCount"] <= 0).any():
        raise ValueError(f"ObservedCount must be positive to calculate log10 for {tf_name}.")

    n_rows = len(labels)
    predictor_names = (
        [f"tetramer__{feature}" for feature in tetra_features]
        + [f"hexamer__{feature}" for feature in hexa_features]
        + COMPOSITION_NAMES
    )
    predictor_groups = (
        [f"Tetramer {tetra_feature_groups[feature]}" for feature in tetra_features]
        + [f"Hexamer {hexa_feature_groups[feature]}" for feature in hexa_features]
        + ["Sequence composition"] * len(COMPOSITION_NAMES)
    )
    if len(predictor_names) != 81 or len(set(predictor_names)) != len(predictor_names):
        raise AssertionError("Expected 81 unique shape/composition predictors.")

    predictor_matrix = np.zeros((n_rows, len(predictor_names)), dtype=np.float32)
    tetra_count = len(tetra_features)
    hexa_count = len(hexa_features)
    tetra_windows = average_over_windows(
        sequences, tetra_index, tetra_values, 4,
        predictor_matrix[:, :tetra_count],
    )
    hexa_start = tetra_count
    hexa_end = hexa_start + hexa_count
    hexa_windows = average_over_windows(
        sequences, hexa_index, hexa_values, 6,
        predictor_matrix[:, hexa_start:hexa_end],
    )
    add_sequence_composition(sequences, predictor_matrix, hexa_end)

    observed_count = labels["ObservedCount"].to_numpy(dtype=np.float64)
    outcome_matrix = np.column_stack([
        labels["Affinity"].to_numpy(dtype=np.float64),
        observed_count,
        np.log10(observed_count),
        labels["SE"].to_numpy(dtype=np.float64),
    ])
    pearson_values = pearson_feature_outcome_correlations(predictor_matrix, outcome_matrix)
    spearman_values = spearman_feature_outcome_correlations(predictor_matrix, outcome_matrix)

    records = []
    for feature_index, (feature, group) in enumerate(zip(predictor_names, predictor_groups)):
        for outcome_index, outcome in enumerate(OUTCOMES):
            records.append({
                "TranscriptionFactor": tf_name,
                "Cycle": cycle,
                "N_sequences": n_rows,
                "Feature": feature,
                "Feature_group": group,
                "Outcome": outcome,
                "Pearson": pearson_values[feature_index, outcome_index],
                "Spearman": spearman_values[feature_index, outcome_index],
            })
    result = pd.DataFrame(records)

    validation = {
        "TranscriptionFactor": tf_name,
        "Cycle": cycle,
        "Label_file": str(label_path.relative_to(ROOT)),
        "N_sequences": n_rows,
        "Duplicate_Kmers": duplicate_sequences,
        "Invalid_or_wrong_length_Kmers": int((~valid_sequences).sum()),
        "Missing_target_values": missing_label_values,
        "Min_ObservedCount": int(observed_count.min()),
        "Tetramer_windows_per_sequence": tetra_windows,
        "Hexamer_windows_per_sequence": hexa_windows,
        "Tetramer_lookup_coverage": True,
        "Hexamer_lookup_coverage": True,
        "Predictor_count": len(predictor_names),
        "Correlation_values_missing": int(result[["Pearson", "Spearman"]].isna().sum().sum()),
        "Composition_sum_error_max": float(
            np.max(np.abs(
                predictor_matrix[:, -4:].sum(axis=1, dtype=np.float64) - 1.0
            ))
        ),
    }
    return result, validation


def build_summaries(correlations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create cross-TF top-association, feature-group, and TF comparison tables."""
    top_rows = []
    group_rows = []
    factor_rows = []
    for (tf_name, outcome), block in correlations.groupby(
        ["TranscriptionFactor", "Outcome"], sort=False
    ):
        for method in ("Pearson", "Spearman"):
            ranked = block.assign(AbsoluteCorrelation=block[method].abs()).sort_values(
                "AbsoluteCorrelation", ascending=False, na_position="last"
            )
            top_rows.extend(
                {
                    "TranscriptionFactor": row.TranscriptionFactor,
                    "Outcome": row.Outcome,
                    "Method": method,
                    "Rank": rank,
                    "Feature": row.Feature,
                    "Feature_group": row.Feature_group,
                    "Correlation": getattr(row, method),
                    "AbsoluteCorrelation": row.AbsoluteCorrelation,
                }
                for rank, row in enumerate(ranked.head(10).itertuples(index=False), start=1)
            )
            for group_name, group in block.groupby("Feature_group", sort=False):
                absolute = group[method].abs()
                best = absolute.idxmax()
                best_row = group.loc[best]
                group_rows.append({
                    "TranscriptionFactor": tf_name,
                    "Outcome": outcome,
                    "Method": method,
                    "Feature_group": group_name,
                    "Feature_count": len(group),
                    "Max_abs_correlation": float(absolute.loc[best]),
                    "Top_feature": best_row["Feature"],
                    "Top_feature_correlation": float(best_row[method]),
                    "Median_abs_correlation": float(absolute.median()),
                })
            best_row = ranked.iloc[0]
            factor_rows.append({
                "TranscriptionFactor": tf_name,
                "Outcome": outcome,
                "Method": method,
                "Top_feature": best_row["Feature"],
                "Feature_group": best_row["Feature_group"],
                "Top_correlation": best_row[method],
                "Absolute_correlation": best_row["AbsoluteCorrelation"],
            })
    return pd.DataFrame(top_rows), pd.DataFrame(group_rows), pd.DataFrame(factor_rows)


def write_readme(
    correlations: pd.DataFrame,
    validation: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> None:
    """Write method, validation results, and concise descriptive findings."""
    largest = correlations.assign(
        AbsoluteCorrelation=correlations[["Pearson", "Spearman"]].abs().max(axis=1)
    )
    top_by_outcome = (
        largest.sort_values("AbsoluteCorrelation", ascending=False)
        .groupby("Outcome", sort=False)
        .head(5)
    )
    lines = [
        "# Shape–affinity exploratory correlation analysis",
        "",
        "## Method",
        "",
        "For each selected HT-SELEX cycle, the script maps each 10-mer to its seven overlapping 4-mer windows and five overlapping 6-mer windows. It retrieves values from the clean descriptor tables using exact sequence IDs and averages each canonical feature over the valid windows. Sequence composition is calculated as GC content and A/C/G/T frequencies. Affinity, ObservedCount, log10(ObservedCount), and SE are retained as outcomes. Pearson and Spearman coefficients are calculated separately for each factor. No models are fitted and no per-sequence feature files are written.",
        "",
        "Canonical feature names are imported from `src/feature_config.py`. Hoxd1's auxiliary `#` field is excluded; the four requested label columns are read directly from each selected-cycle TSV. All 10-mers are kept without count filtering or reverse-complement collapsing.",
        "",
        "## Validation",
        "",
        f"- Factors processed: {len(validation)} of {len(TF_CYCLES)}.",
        f"- Total input rows: {int(validation['N_sequences'].sum()):,}.",
        f"- Invalid/incorrect-length Kmers: {int(validation['Invalid_or_wrong_length_Kmers'].sum())}.",
        f"- Duplicate Kmers: {int(validation['Duplicate_Kmers'].sum())}.",
        f"- Missing target values: {int(validation['Missing_target_values'].sum())}.",
        f"- Descriptor lookups missing: {int((~validation['Tetramer_lookup_coverage']).sum() + (~validation['Hexamer_lookup_coverage']).sum())}.",
        f"- Missing correlation coefficients: {int(validation['Correlation_values_missing'].sum())}.",
        "- A/C/G/T frequencies sum to 1 for every sequence (within floating-point precision); GC content equals C plus G frequency.",
        "",
        "The complete validation values are in `validation_by_tf.csv`.",
        "",
        "## Main associations",
        "",
        "The table below lists the five largest absolute correlations per outcome across all factor–feature pairs and both methods. It is descriptive: effect sizes should be compared within TF, and the selected maxima should not be interpreted as confirmatory findings.",
        "",
        "| Outcome | TF | Feature | Group | Pearson r | Spearman rho | |r|/|rho| max |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for _, row in top_by_outcome.iterrows():
        lines.append(
            f"| {row['Outcome']} | {row['TranscriptionFactor']} | {row['Feature']} | "
            f"{row['Feature_group']} | {row['Pearson']:.4f} | {row['Spearman']:.4f} | "
            f"{row['AbsoluteCorrelation']:.4f} |"
        )
    lines.extend([
        "",
        "The per-TF and per-feature-group comparison tables are `top_associations_by_tf.csv`, `top_associations_by_group.csv`, and `top_association_by_tf_outcome.csv`. All predictor-to-outcome coefficients are in `correlations_long.csv`.",
        "",
        "## Interpretation limits",
        "",
        "Averaging overlapping windows removes position-specific information. Composition measures obey exact sum constraints, and Spearman correlations for ObservedCount and its strictly increasing log10 transform are mathematically identical (positive counts were validated). Pearson coefficients use raw ObservedCount as requested and may be sensitive to its skew. The standard error is an uncertainty measure and should be interpreted separately from affinity. With many TFs, features and outcomes, the strongest correlations are exploratory and do not establish causation or biological mechanism.",
        "",
        "## Files",
        "",
        "- `correlations_long.csv`: one row per TF, predictor and outcome, with both correlation methods.",
        "- `top_associations_by_tf.csv`: top ten absolute associations for each TF, outcome and method.",
        "- `top_associations_by_group.csv`: strongest and median absolute associations by TF, group, outcome and method.",
        "- `top_association_by_tf_outcome.csv`: strongest feature for each TF, outcome and method.",
        "- `validation_by_tf.csv`: input/feature lookup validation.",
        "- `{tf}_{Pearson|Spearman}_heatmap.png`: per-TF descriptor-by-outcome heatmaps.",
    ])
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tetra_index, tetra_values, tetra_features, tetra_feature_groups = load_clean_descriptor_table(4)
    hexa_index, hexa_values, hexa_features, hexa_feature_groups = load_clean_descriptor_table(6)

    all_correlations = []
    validation_rows = []
    for tf_name, cycle in TF_CYCLES.items():
        print(f"Processing {tf_name} cycle {cycle}...")
        correlations, validation = analyze_factor(
            tf_name,
            cycle,
            tetra_index,
            tetra_values,
            tetra_features,
            tetra_feature_groups,
            hexa_index,
            hexa_values,
            hexa_features,
            hexa_feature_groups,
        )
        all_correlations.append(correlations)
        validation_rows.append(validation)
        for method in ("Pearson", "Spearman"):
            make_heatmap(
                correlations,
                tf_name,
                method,
                (
                    [f"Tetramer {tetra_feature_groups[feature]}" for feature in tetra_features]
                    + [f"Hexamer {hexa_feature_groups[feature]}" for feature in hexa_features]
                    + ["Sequence composition"] * len(COMPOSITION_NAMES)
                ),
            )
        del correlations
        gc.collect()

    correlations = pd.concat(all_correlations, ignore_index=True)
    validation = pd.DataFrame(validation_rows)
    top_by_tf, top_by_group, top_by_factor = build_summaries(correlations)

    # Validate the complete correlation output schema and one row per planned comparison.
    expected_rows = len(TF_CYCLES) * 81 * len(OUTCOMES)
    if len(correlations) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} correlation rows, found {len(correlations)}")
    if correlations.duplicated(["TranscriptionFactor", "Feature", "Outcome"]).any():
        raise AssertionError("Duplicate TF/feature/outcome correlation rows.")
    if len(validation) != len(TF_CYCLES):
        raise AssertionError("Validation table does not contain all 11 TFs.")

    correlations.to_csv(OUTPUT_DIR / "correlations_long.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_by_tf.csv", index=False)
    top_by_tf.to_csv(OUTPUT_DIR / "top_associations_by_tf.csv", index=False)
    top_by_group.to_csv(OUTPUT_DIR / "top_associations_by_group.csv", index=False)
    top_by_factor.to_csv(OUTPUT_DIR / "top_association_by_tf_outcome.csv", index=False)
    write_readme(correlations, validation, top_by_group)

    # Reload tables from disk to verify saved dimensions and numeric coefficients.
    saved_correlations = pd.read_csv(OUTPUT_DIR / "correlations_long.csv")
    saved_validation = pd.read_csv(OUTPUT_DIR / "validation_by_tf.csv")
    if len(saved_correlations) != expected_rows or len(saved_validation) != len(TF_CYCLES):
        raise AssertionError("Saved summary table row counts failed validation.")
    if saved_correlations[["Pearson", "Spearman"]].isna().any().any():
        raise AssertionError("Unexpected missing correlations in saved table.")

    print(f"Saved outputs to {OUTPUT_DIR.relative_to(ROOT)}")
    print("Rows in correlations_long.csv:", len(saved_correlations))
    print("Heatmaps:", len(list(OUTPUT_DIR.glob("*_heatmap.png"))))
    print("Validation summary:")
    print(saved_validation.to_string(index=False))
    print("Top association summary (largest absolute coefficient by outcome):")
    print(
        top_by_factor.assign(AbsoluteCorrelation=top_by_factor["Top_correlation"].abs())
        .sort_values("AbsoluteCorrelation", ascending=False)
        .groupby("Outcome", sort=False)
        .head(3)
        [["Outcome", "TranscriptionFactor", "Method", "Top_feature", "Top_correlation"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
