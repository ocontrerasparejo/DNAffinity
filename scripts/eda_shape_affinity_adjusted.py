#!/usr/bin/env python3
"""Composition-adjusted, exploratory shape–affinity associations.

Uses sequence-level means of canonical tetramer and hexamer descriptors and
reports marginal, linearly composition-adjusted, and quadratic-adjusted
Pearson/Spearman associations with the SELEX-derived Affinity label. The
ObservedCount >= 2 analysis is a prespecified sensitivity analysis.
"""

from __future__ import annotations

import gc
import sys
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
    FEATURE_GROUPS_CLEAN_TETRAMER,
    FEATURE_GROUPS_CLEAN_HEXAMER,
    FEATURES_CLEAN_TETRAMER,
    FEATURES_CLEAN_HEXAMER,
)

SELEX_DIR = ROOT / "data" / "selex"
OUT_DIR = ROOT / "outputs" / "eda" / "shape_affinity"
TF_CYCLES = {
    "Gata4": 4, "Zfp24": 5, "arid3": 6, "hoxd1": 3, "nkx3": 6,
    "pax3-like": 5, "six1": 5, "tbx15": 5, "telf": 5, "uncxb": 5,
    "xbpc": 5,
}
LOW_COUNT_THRESHOLD = 2
GROUP_ORDER = ["AVG", "FCE", "Electrostatic", "Ion-related", "Sequence composition"]
BASE_CODE = np.full(256, 255, dtype=np.uint8)
for _base, _code in ((ord("A"), 0), (ord("C"), 1), (ord("G"), 2), (ord("T"), 3)):
    BASE_CODE[_base] = _code


def load_descriptor_lookup(k: int) -> tuple[np.ndarray, list[str], list[str]]:
    if k == 4:
        path = SELEX_DIR / "tetramer" / "input_tetramer_clean.csv"
        groups, features = FEATURE_GROUPS_CLEAN_TETRAMER, FEATURES_CLEAN_TETRAMER
    else:
        path = SELEX_DIR / "hexamer" / "input_hexamer_clean.csv"
        groups, features = FEATURE_GROUPS_CLEAN_HEXAMER, FEATURES_CLEAN_HEXAMER
    data = pd.read_csv(path)
    if data.columns.tolist() != ["SEQUENCE", *features]:
        raise ValueError(f"{path} columns do not match src/feature_config.py.")
    if data["SEQUENCE"].duplicated().any() or data["SEQUENCE"].isna().any():
        raise ValueError(f"{path} has missing or duplicated sequence identifiers.")
    if data[features].isna().any().any() or not all(
        pd.api.types.is_numeric_dtype(data[name]) for name in features
    ):
        raise ValueError(f"{path} has non-numeric or missing descriptor values.")

    lookup = np.empty((4**k, len(features)), dtype=np.float32)
    seen = np.zeros(4**k, dtype=bool)
    for row, sequence in enumerate(data["SEQUENCE"]):
        code = 0
        for base in sequence.encode("ascii"):
            code = (code << 2) | int(BASE_CODE[base])
        lookup[code] = data.loc[row, features].to_numpy(dtype=np.float32)
        seen[code] = True
    if not seen.all():
        raise ValueError(f"{path} does not contain the complete {k}-mer space.")
    feature_groups = [group for group, names in groups.items() for _ in names]
    return lookup, list(features), feature_groups


def make_predictors(
    sequences: np.ndarray,
    tetra_lookup: np.ndarray,
    hexa_lookup: np.ndarray,
    tetra_features: list[str],
    tetra_groups: list[str],
    hexa_features: list[str],
    hexa_groups: list[str],
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    names = [f"tetramer__{name}" for name in tetra_features]
    names.extend(f"hexamer__{name}" for name in hexa_features)
    groups = tetra_groups + hexa_groups
    X = np.zeros((len(sequences), len(names)), dtype=np.float32)
    raw = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    bases = raw.reshape(len(sequences), 10)
    for start in range(7):
        code = np.zeros(len(sequences), dtype=np.uint16)
        for position in range(start, start + 4):
            code = (code << 2) | BASE_CODE[bases[:, position]]
        X[:, :len(tetra_features)] += tetra_lookup[code]
    for start in range(5):
        code = np.zeros(len(sequences), dtype=np.uint16)
        for position in range(start, start + 6):
            code = (code << 2) | BASE_CODE[bases[:, position]]
        X[:, len(tetra_features):len(tetra_features) + len(hexa_features)] += hexa_lookup[code]
    X[:, :len(tetra_features)] /= 7
    X[:, len(tetra_features):len(tetra_features) + len(hexa_features)] /= 5

    comp = np.column_stack([
        (bases == ord("A")).mean(axis=1),
        (bases == ord("C")).mean(axis=1),
        (bases == ord("G")).mean(axis=1),
        (bases == ord("T")).mean(axis=1),
    ]).astype(np.float32)
    composition = np.column_stack([comp[:, 1] + comp[:, 2], comp]).astype(np.float32)
    X = np.column_stack([X, composition])
    names.extend(["GC_content", "A_frequency", "C_frequency", "G_frequency", "T_frequency"])
    groups.extend(["Sequence composition"] * 5)
    if np.max(np.abs(comp.sum(axis=1, dtype=np.float64) - 1)) > 1e-6:
        raise AssertionError("Base frequencies do not sum to one.")
    return X, names, groups, comp


def composition_design(comp: np.ndarray, quadratic: bool) -> np.ndarray:
    """Use independent A/C/G frequencies; T is fixed by the sum constraint."""
    a, c, g = comp[:, 0].astype(np.float64), comp[:, 1].astype(np.float64), comp[:, 2].astype(np.float64)
    columns = [np.ones(len(comp)), a, c, g]
    if quadratic:
        columns.extend([a * a, c * c, g * g, a * c, a * g, c * g])
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Composition adjustment design is rank deficient.")
    return design


def orthonormal_basis(design: np.ndarray) -> np.ndarray:
    return np.linalg.qr(design, mode="reduced")[0]


def residualize(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - q @ (q.T @ values)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def compute_factor(
    tf: str,
    cycle: int,
    tetra_lookup: np.ndarray,
    hexa_lookup: np.ndarray,
    tetra_features: list[str],
    tetra_groups: list[str],
    hexa_features: list[str],
    hexa_groups: list[str],
) -> tuple[list[dict], dict, dict, dict]:
    path = SELEX_DIR / "transcription_factors" / tf / f"{cycle}.txt"
    labels = pd.read_csv(path, sep="\t", usecols=lambda col: col in {"Kmer", "ObservedCount", "Affinity", "SE"})
    required = {"Kmer", "ObservedCount", "Affinity", "SE"}
    if not required.issubset(labels.columns):
        raise KeyError(f"{path} is missing requested columns: {sorted(required - set(labels.columns))}")
    seq = labels["Kmer"].astype(str).to_numpy()
    valid = labels["Kmer"].str.fullmatch(r"[ACGT]{10}", na=False).to_numpy()
    if not valid.all() or labels["Kmer"].duplicated().any():
        raise ValueError(f"Invalid or duplicated 10-mer in {path}.")
    if labels[["ObservedCount", "Affinity", "SE"]].isna().any().any():
        raise ValueError(f"Missing target values in {path}.")
    count = labels["ObservedCount"].to_numpy(dtype=np.int64)
    y = labels["Affinity"].to_numpy(dtype=np.float64)
    if (count < 1).any():
        raise ValueError(f"ObservedCount must be positive in {path}.")
    X, names, groups, comp = make_predictors(
        seq, tetra_lookup, hexa_lookup, tetra_features, tetra_groups, hexa_features, hexa_groups
    )
    feature_X = X[:, :len(names) - 5]

    validation = {
        "TranscriptionFactor": tf, "Cycle": cycle, "N_sequences": len(seq),
        "Duplicate_Kmers": int(labels["Kmer"].duplicated().sum()),
        "Invalid_or_wrong_length_Kmers": int((~valid).sum()),
        "Missing_target_values": int(labels[["ObservedCount", "Affinity", "SE"]].isna().sum().sum()),
        "Count_ge_2_retained": int((count >= LOW_COUNT_THRESHOLD).sum()),
        "Count_ge_2_removed": int((count < LOW_COUNT_THRESHOLD).sum()),
        "Count_ge_2_removed_fraction": float((count < LOW_COUNT_THRESHOLD).mean()),
        "Tetramer_windows_per_sequence": 7, "Hexamer_windows_per_sequence": 5,
        "Canonical_predictor_count": feature_X.shape[1],
        "Composition_sum_error_max": float(np.max(np.abs(comp.sum(axis=1) - 1))),
        "GC_equals_C_plus_G_error_max": float(np.max(np.abs(X[:, -5] - (comp[:, 1] + comp[:, 2])))),
    }

    records: list[dict] = []
    model_fit = []
    for filter_name, keep in (("All_sequences", np.ones(len(seq), dtype=bool)), ("Count_ge_2", count >= LOW_COUNT_THRESHOLD)):
        n = int(keep.sum())
        if n < 20:
            raise ValueError(f"Too few observations after {filter_name} for {tf}.")
        x_sub = feature_X[keep]
        y_sub = y[keep]
        comp_sub = comp[keep]
        y_rank = rankdata(y_sub, method="average")
        raw_designs = {
            "Linear": composition_design(comp_sub, quadratic=False),
            "Quadratic": composition_design(comp_sub, quadratic=True),
        }
        bases = {name: orthonormal_basis(design) for name, design in raw_designs.items()}
        for adjustment, design in raw_designs.items():
            beta = np.linalg.lstsq(design, y_sub, rcond=None)[0]
            fitted = design @ beta
            ss_total = np.sum((y_sub - y_sub.mean()) ** 2)
            r_squared = 1 - np.sum((y_sub - fitted) ** 2) / ss_total
            model_fit.append({
                "TranscriptionFactor": tf, "Cycle": cycle, "Filter": filter_name,
                "Composition_model": adjustment, "N_sequences": n,
                "Composition_only_Affinity_R_squared": float(r_squared),
                "Composition_covariates": "A_frequency, C_frequency, G_frequency; T omitted (sum constraint)",
            })

        y_residuals = {
            name: residualize(y_sub, q) for name, q in bases.items()
        }
        y_rank_residuals = {
            name: residualize(y_rank, q) for name, q in bases.items()
        }
        for j, feature_name in enumerate(names[:feature_X.shape[1]]):
            x = x_sub[:, j].astype(np.float64)
            x_rank = rankdata(x, method="average")
            values = {
                "Pearson": {"Unadjusted": corr(x, y_sub)},
                "Spearman": {"Unadjusted": corr(x_rank, y_rank)},
            }
            for adjustment, q in bases.items():
                values["Pearson"][f"{adjustment}_composition_adjusted"] = corr(
                    residualize(x, q), y_residuals[adjustment]
                )
                values["Spearman"][f"{adjustment}_composition_adjusted"] = corr(
                    residualize(x_rank, q), y_rank_residuals[adjustment]
                )
            for method, adjustments in values.items():
                for adjustment, coefficient in adjustments.items():
                    records.append({
                        "TranscriptionFactor": tf, "Cycle": cycle, "Filter": filter_name,
                        "N_sequences": n, "Feature": feature_name, "Feature_group": groups[j],
                        "Method": method, "Adjustment": adjustment,
                        "Correlation": coefficient,
                    })
    del X, feature_X
    gc.collect()
    return records, validation, model_fit, {"n": len(seq)}


def build_summaries(correlations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shape = correlations[correlations["Feature_group"] != "Sequence composition"].copy()
    by_tf = (
        shape.groupby(["TranscriptionFactor", "Filter", "Method", "Adjustment", "Feature_group"], as_index=False)
        .agg(
            Feature_count=("Feature", "nunique"), Mean_correlation=("Correlation", "mean"),
            Median_correlation=("Correlation", "median"), Median_absolute_correlation=("Correlation", lambda x: x.abs().median()),
            Positive_fraction=("Correlation", lambda x: (x > 0).mean()),
        )
    )
    feature_consistency = (
        shape.groupby(["Filter", "Method", "Adjustment", "Feature_group", "Feature"], as_index=False)
        .agg(
            Median_across_TFs=("Correlation", "median"),
            Min_across_TFs=("Correlation", "min"), Max_across_TFs=("Correlation", "max"),
            Positive_TFs=("Correlation", lambda x: int((x > 0).sum())),
            Negative_TFs=("Correlation", lambda x: int((x < 0).sum())),
        )
    )
    feature_consistency["Direction_consistency"] = feature_consistency[["Positive_TFs", "Negative_TFs"]].max(axis=1) / shape["TranscriptionFactor"].nunique()
    across_tf = (
        feature_consistency.groupby(["Filter", "Method", "Adjustment", "Feature_group"], as_index=False)
        .agg(
            Feature_count=("Feature", "nunique"),
            Median_feature_effect=("Median_across_TFs", "median"),
            Median_absolute_effect=("Median_across_TFs", lambda x: x.abs().median()),
            Mean_direction_consistency=("Direction_consistency", "mean"),
            Features_with_consistent_direction=("Direction_consistency", lambda x: int((x >= 0.8).sum())),
        )
    )
    return by_tf, across_tf


def save_heatmaps(correlations: pd.DataFrame) -> None:
    heatmap_dir = OUT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    data = correlations[correlations["Feature_group"] != "Sequence composition"]
    for filter_name in ("All_sequences", "Count_ge_2"):
        for method in ("Pearson", "Spearman"):
            for adjustment in ("Unadjusted", "Linear_composition_adjusted", "Quadratic_composition_adjusted"):
                subset = data[(data.Filter == filter_name) & (data.Method == method) & (data.Adjustment == adjustment)]
                if subset.empty:
                    continue
                matrix = subset.pivot(index="Feature", columns="TranscriptionFactor", values="Correlation")
                fig, ax = plt.subplots(figsize=(10, max(11, len(matrix) * 0.17)))
                sns.heatmap(matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                            cbar_kws={"label": f"{method} association with Affinity"})
                ax.set_title(f"{method}: {adjustment.replace('_', ' ')} ({filter_name.replace('_', ' ')})")
                ax.set_xlabel("Transcription factor")
                ax.set_ylabel("Canonical descriptor")
                ax.tick_params(axis="y", labelsize=6)
                fig.tight_layout()
                fig.savefig(heatmap_dir / f"affinity_{filter_name}_{method}_{adjustment}.png", dpi=180, bbox_inches="tight")
                plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tetra_lookup, tetra_features, tetra_groups = load_descriptor_lookup(4)
    hexa_lookup, hexa_features, hexa_groups = load_descriptor_lookup(6)
    all_records, validations, model_fits = [], [], []
    for tf, cycle in TF_CYCLES.items():
        print(f"Analyzing {tf} cycle {cycle}...", flush=True)
        records, validation, fit, _ = compute_factor(
            tf, cycle, tetra_lookup, hexa_lookup,
            tetra_features, tetra_groups, hexa_features, hexa_groups,
        )
        all_records.extend(records)
        validations.append(validation)
        model_fits.extend(fit)
        print(f"  n={validation['N_sequences']:,}; count>=2 n={validation['Count_ge_2_retained']:,}", flush=True)

    correlations = pd.DataFrame(all_records)
    validation = pd.DataFrame(validations)
    model_fit = pd.DataFrame(model_fits)
    by_tf, across_tf = build_summaries(correlations)
    correlations.to_csv(OUT_DIR / "affinity_correlations_long.csv", index=False)
    validation.to_csv(OUT_DIR / "sample_sizes_and_validation.csv", index=False)
    model_fit.to_csv(OUT_DIR / "composition_model_fit.csv", index=False)
    by_tf.to_csv(OUT_DIR / "descriptor_group_summary_by_tf.csv", index=False)
    across_tf.to_csv(OUT_DIR / "descriptor_group_consistency_across_tf.csv", index=False)
    save_heatmaps(correlations)

    if correlations["Correlation"].isna().any():
        raise ValueError("Missing correlation values detected.")
    if validation["TranscriptionFactor"].nunique() != 11:
        raise AssertionError("Expected all 11 TFs.")
    expected_features = len(FEATURES_CLEAN_TETRAMER) + len(FEATURES_CLEAN_HEXAMER)
    counts = correlations[correlations.Filter == "All_sequences"].groupby(["TranscriptionFactor", "Method", "Adjustment"])["Feature"].nunique()
    if not (counts == expected_features).all():
        raise AssertionError("A TF/method/adjustment does not contain all canonical features.")
    print(f"Saved {len(correlations):,} correlation rows; {len(list((OUT_DIR / 'heatmaps').glob('*.png')))} heatmaps.")


if __name__ == "__main__":
    main()
