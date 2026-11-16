# Reproducing the DNAffinity SELEX data handoff

This guide accompanies `notebooks/EDA_handoff.ipynb`. It describes the files needed to rebuild the per-TF 10-mer descriptor tables on another computer.

## Environment

The project supports Python 3.9 or newer. From the repository root, install the project dependencies and a notebook frontend if needed:

```bash
python -m pip install -e .
python -m pip install jupyterlab
jupyter lab notebooks/EDA_handoff.ipynb
```

The notebook itself uses NumPy, pandas, the Python standard library and `src/feature_config.py`. It does not require the EDA plots, PCA outputs, SciPy or a model-training framework.

## Required files

Copy the repository-relative files below without changing their names or directory structure.

### Workflow and canonical feature definitions

| Relative path | Purpose |
|---|---|
| `notebooks/EDA_handoff.ipynb` | Loads, validates, maps and exports the selected HT-SELEX datasets. |
| `src/feature_config.py` | Canonical descriptor feature names and group definitions for tetramers and hexamers. |
| `pyproject.toml` | Project metadata and declared Python dependencies; used by `pip install -e .`. |

### Clean descriptor tables

| Relative path | Purpose |
|---|---|
| `data/selex/tetramer/input_tetramer_clean.csv` | Complete sequence-keyed tetramer descriptor table; 256 sequences and 36 canonical features. |
| `data/selex/hexamer/input_hexamer_clean.csv` | Complete sequence-keyed hexamer descriptor table; 4,096 sequences and 40 canonical features. |

These clean tables are the descriptor inputs used by the handoff. The original AVG, FCE, electrostatic, ion-related and combined `.dat` files are not required to run this notebook.

### Selected HT-SELEX source files

Each file contains the selected cycle for one transcription factor. `Kmer`, `ObservedCount`, `Affinity` and `SE` are read by column name; any additional source columns are not included in the exported predictor table.

| Relative path | TF | Cycle |
|---|---|---:|
| `data/selex/transcription_factors/Gata4/4.txt` | Gata4 | 4 |
| `data/selex/transcription_factors/Zfp24/5.txt` | Zfp24 | 5 |
| `data/selex/transcription_factors/arid3/6.txt` | arid3 | 6 |
| `data/selex/transcription_factors/hoxd1/3.txt` | hoxd1 | 3 |
| `data/selex/transcription_factors/nkx3/6.txt` | nkx3 | 6 |
| `data/selex/transcription_factors/pax3-like/5.txt` | pax3-like | 5 |
| `data/selex/transcription_factors/six1/5.txt` | six1 | 5 |
| `data/selex/transcription_factors/tbx15/5.txt` | tbx15 | 5 |
| `data/selex/transcription_factors/telf/5.txt` | telf | 5 |
| `data/selex/transcription_factors/uncxb/5.txt` | uncxb | 5 |
| `data/selex/transcription_factors/xbpc/5.txt` | xbpc | 5 |

If one of these required files is missing, the notebook stops and names the missing path. It does not select a different cycle automatically.

## Processing and output

For each 10-mer, the notebook maps its seven overlapping 4-mer windows and five overlapping 6-mer windows using sequence-keyed, two-bit nucleotide lookups. It averages the descriptors separately across those windows, preserving the existing EDA methodology and original left-to-right sequence order. FCE uses the 21 independent upper-triangular entries; electrostatic features follow the canonical clean set. Tetramer and hexamer features are prefixed `tetramer__` and `hexamer__` to keep the 76 predictors distinct.

The notebook writes one gzip-compressed CSV per TF under `outputs/selex_ml_ready/`:

```text
outputs/selex_ml_ready/Gata4_cycle4_10mer_descriptors.csv.gz
outputs/selex_ml_ready/Zfp24_cycle5_10mer_descriptors.csv.gz
outputs/selex_ml_ready/arid3_cycle6_10mer_descriptors.csv.gz
outputs/selex_ml_ready/hoxd1_cycle3_10mer_descriptors.csv.gz
outputs/selex_ml_ready/nkx3_cycle6_10mer_descriptors.csv.gz
outputs/selex_ml_ready/pax3-like_cycle5_10mer_descriptors.csv.gz
outputs/selex_ml_ready/six1_cycle5_10mer_descriptors.csv.gz
outputs/selex_ml_ready/tbx15_cycle5_10mer_descriptors.csv.gz
outputs/selex_ml_ready/telf_cycle5_10mer_descriptors.csv.gz
outputs/selex_ml_ready/uncxb_cycle5_10mer_descriptors.csv.gz
outputs/selex_ml_ready/xbpc_cycle5_10mer_descriptors.csv.gz
outputs/selex_ml_ready/export_validation_summary.csv
```

Each TF table has 82 columns: `TF`, `Cycle`, `Kmer`, `Affinity`, `ObservedCount`, `SE`, and 76 descriptor means. It has exactly as many rows as its selected source cycle file. Read a compressed table with `pandas.read_csv(path, compression="gzip")`.

The exports are derived outputs, not source inputs; rerunning the notebook regenerates them. Copy `outputs/selex_ml_ready/` to the laboratory computer as well if you want the precomputed tables there without rerunning the workflow. Feature scaling, train/test splits and predictive modelling are intentionally left to the subsequent modelling stage.

### Transfer note

The current run produced 5,444,696 rows across 11 compressed TF files. The complete output directory is about 1.9 GB. The repository `.gitignore` excludes `outputs/`, so Git will not transfer these generated files. To avoid rerunning the notebook on the laboratory computer, copy `outputs/selex_ml_ready/` separately; otherwise, transfer the required source files above and regenerate the exports there.
