# SELEX descriptor data alignment audit

**Scope:** source tables and combined tetramer/hexamer tables used in `notebooks/preprocessing_selex_2.ipynb`. This audit was read-only. Sequence-keyed source maps were reconstructed in memory using the parsing and join logic in `scripts/make_selex_input.py`; no source or combined data files were written.

## Summary status

| Check | Status | Finding |
|---|---|---|
| Expected sequence coverage | **PASS** | All 256 possible 4-mers and all 4,096 possible 6-mers occur in every source and combined table. |
| Duplicate, invalid, or wrong-length sequence IDs | **PASS** | None found in any source or combined table. |
| Source table dimensions and identifiers | **PASS** | All four descriptor sources have one row for each expected sequence. FCE sequence IDs are the first token in data rows; their header does not name that field. |
| Tetramer electrostatic load with default `pandas.read_csv` | **WARNING** | The raw file has no header. Default header inference treats the first row, `AAAA`, as column names, leaving 255 data rows. `AAAA` is present in the original file and in `input_tetramer.dat`. |
| Combined table types and missing values | **PASS** | All descriptor columns are numeric; there are no missing values. |
| Sequence-to-descriptor alignment | **PASS** | Each combined descriptor value exactly matches its source-table value after joining by sequence ID. The existing builder also joins keyed maps, intersects IDs and sorts the resulting sequences. |
| Combined tables suitable for subsequent analysis | **PASS** | Both combined files are complete and consistent with all source tables under the builder’s sequence-keyed join. |

## Table dimensions

Dimensions below are **sequence rows × descriptor columns**; the sequence identifier is included in each source record but not counted as a descriptor.

| Sequence length | Table | Dimensions | Unique IDs | Missing / duplicate / invalid / wrong length |
|---|---|---:|---:|---:|
| Tetramer | AVG | 256 × 6 | 256 | 0 / 0 / 0 / 0 |
| Tetramer | FCE | 256 × 36 | 256 | 0 / 0 / 0 / 0 |
| Tetramer | Electrostatic | 256 × 5 | 256 | 0 / 0 / 0 / 0 |
| Tetramer | Ion-related | 256 × 6 | 256 | 0 / 0 / 0 / 0 |
| Hexamer | AVG | 4,096 × 6 | 4,096 | 0 / 0 / 0 / 0 |
| Hexamer | FCE | 4,096 × 36 | 4,096 | 0 / 0 / 0 / 0 |
| Hexamer | Electrostatic | 4,096 × 5 | 4,096 | 0 / 0 / 0 / 0 |
| Hexamer | Ion-related | 4,096 × 10 | 4,096 | 0 / 0 / 0 / 0 |
| Tetramer | `input_tetramer.dat` | 256 × 53 | 256 | 0 / 0 / 0 / 0 |
| Hexamer | `input_hexamer.dat` | 4,096 × 57 | 4,096 | 0 / 0 / 0 / 0 |

The combined files have dimensions **256 × 54** and **4,096 × 58** when the `SEQUENCE` column is included. Their 53 and 57 descriptor columns are numeric, with zero missing values.

## Alignment details

- Expected IDs were generated from all strings over `A`, `C`, `G`, and `T` at lengths four and six. Every source and combined table matched its complete expected set exactly.
- AVG, electrostatic and ion-related tables have identifiable sequence fields in their records. FCE files also contain IDs: the first token of each data row is the sequence, although its header begins with descriptor names and has no `SEQUENCE` label.
- The tetramer electrostatic file begins with a data row, not a header. A plain `pd.read_csv(path, sep=r"\s+")` produces **255 × 6** and consumes `AAAA` as the header. A sequence-aware parse reads 256 IDs and five values per ID. The constructor uses this sequence-aware parsing and names those values `e1`–`e5`.
- `scripts/make_selex_input.py` reads each sequence-keyed table into a map, intersects the sequence keys, sorts them, and writes the combined file. The audit independently compared every combined value—including all 36 FCE entries—to its source by sequence ID. It found **zero mismatches** for both lengths.
- FCE has 36 stored matrix positions, including 15 mirrored symmetric entries. This is expected in the combined tables; PCA and cross-feature analyses that require unique FCE descriptors should continue using the existing 21-element upper-triangular feature list.

## Recommendation

Use `input_tetramer.dat` and `input_hexamer.dat` as the aligned sources for subsequent analyses. When loading the individual tetramer electrostatic file directly, use a parser that treats its first row as data and supplies the `e1`–`e5` names; do not rely on default header inference. No correction to the source or combined files is indicated by this audit.
