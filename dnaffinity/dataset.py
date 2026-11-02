"""Dataset: turns sequence/affinity tables into physics-based feature matrices.

The features are derived from tetramer-level structural and mechanical DNA
properties (averages, force-constant matrices, minor groove width,
electrostatics) obtained from atomistic molecular dynamics simulations.
"""

from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(20)

# Repo-level data folder with the legacy tetramer property tables (uPBM / gcPBM)
DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'input'
# SELEX shape, electrostatics and ion concentration (tetramer and hexamer)
SELEX_DIR = DATA_DIR.parent / 'selex'

_HELICAL = ('shift', 'slide', 'rise', 'tilt', 'roll', 'twist')


def _read_kmer_table(path):
    """Map DNA k-mer -> float vector. Header rows (SEQUENCE, SHIFT, tet, hex, ...) are skipped."""
    table = {}
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts or not set(parts[0]) <= set('ACGT'):
                continue
            table[parts[0]] = np.asarray(parts[1:], dtype=np.float64)
    if not table:
        raise ValueError(f'No k-mer rows in {path}')
    return table


class Dataset:
    def __init__(self, protein, fitxer, model_target='tetramers', randomize_fce=False,
                 chosen_features=('full_fce', 'avg'), score='Median_intensity',
                 selected_tetramers=(0, 1, 2, 3, 4, 5, 6, 7, 8), weighted=False,
                 property_dir=None, kmers=(4,)):

        self.protein = protein
        self.pbm35 = fitxer

        if weighted:
            self.pbm35 = self.pbm35[["ID_REF", "VALUE", 'WEIGHT']]
            self.pbm35.columns = ["ID_REF", "VALUE", 'WEIGHT']
        else:
            self.pbm35 = self.pbm35[["ID_REF", "VALUE"]]
            self.pbm35.columns = ["ID_REF", "VALUE"]
        self.pbm35 = self.pbm35.dropna()

        inv_seq_data = self.pbm35.copy()
        inv_seq_data["ID_REF"] = inv_seq_data["ID_REF"].apply(self.inv_seq)
        self.pbm35 = pd.concat([self.pbm35, inv_seq_data])
        self.pbm35.reset_index(inplace=True)
        # canonical-sequence group id per row: a sequence and its reverse
        # complement (and any duplicates) share a group, so splits on groups
        # keep both strands on the same side
        canon = self.pbm35['ID_REF'].map(lambda q: min(q, self.inv_seq(q)))
        self.groups = pd.factorize(canon)[0]

        self.feats = list(chosen_features)
        self.score = score
        self.selected_tetramers = list(selected_tetramers)
        self.randomize = randomize_fce
        self.model_target = model_target
        self.property_dir = Path(property_dir) if property_dir is not None else DATA_DIR
        self.kmers = tuple(kmers)
        # feature name -> channel names, concatenated k-mer by k-mer on each tetramer window
        self._channels = {}

        # Get list of all kmers for indices purposes
        self.sequences = list(self.pbm35["ID_REF"])
        self.p = len(self.sequences[0])

        # Get list of all tetramers per each octamer in order to get features
        self.featurize()

    def __str__(self):
        return f"{self.model_target}-based dataset for protein {self.protein} using features {self.feats}"

    def _table_path(self, kind, k):
        """Property file for one feature kind and k-mer length.

        `data/input` keeps the legacy tetramer names. `data/selex` stores tetramer
        and hexamer tables (avg, fce, electrostatics, ion concentration) and is the
        only source used for SELEX.
        """
        if (self.property_dir / 'tetramer').is_dir():
            folder = {4: 'tetramer', 6: 'hexamer'}[k]
            names = {'avg': f'avg_{folder}.dat', 'fce': f'fce_{folder}.dat',
                     'electrostatic': f'electrostatic_{folder}.dat',
                     'ion': f'ion_vec_{folder}.dat'}
            if kind not in names:
                raise FileNotFoundError(f'{kind} is not in {self.property_dir}')
            return self.property_dir / folder / names[kind]
        if k != 4:
            raise ValueError(f'{self.property_dir} has no {k}-mer tables')
        legacy = {'avg': 'avg_tetramer.dat', 'fce': 'fce_tetramer.dat',
                  'electrostatic': 'electrostatic.txt', 'mgw': 'mgw_rohs.txt'}
        if kind not in legacy:
            raise FileNotFoundError(
                f'{kind} is only available under data/selex (tetramer and hexamer tables)')
        return self.property_dir / legacy[kind]

    def _channel_names(self, kind, k, width):
        if kind in ('avg', 'diagonal_fce'):
            names = list(_HELICAL)
        elif kind == 'ion' and k == 4:
            names = ['m1', 'm2', 'm3', 'j1', 'j2', 'j3']
        elif kind == 'ion' and k == 6:
            names = [f'm{i}' for i in range(5)] + [f'j{i}' for i in range(5)]
        else:
            names = [f'e{i}' for i in range(width)]
        if len(names) != width:
            names = [f'c{i}' for i in range(width)]
        return names if k == 4 else [f'{n}_hex' for n in names]

    def _aligned_windows(self, table, k):
        """(n_samples, n_tetramer_windows * width), aligned to `selected_tetramers`.

        Tetramer k=4 is the window itself. Hexamer k=6 is the hexamer whose central
        tetramer is that window (start i-1); the two end windows are zero.
        """
        width = len(next(iter(table.values())))
        rows = []
        for seq in self.sequences:
            parts = []
            for i in self.selected_tetramers:
                if k == 4:
                    vec = table[seq[i:i + 4]]
                elif k == 6:
                    j = i - 1
                    vec = table[seq[j:j + 6]] if 0 <= j <= len(seq) - 6 else np.zeros(width)
                else:
                    raise ValueError(f'Unsupported k-mer length {k}')
                parts.append(np.asarray(vec, dtype=np.float64))
            rows.append(np.concatenate(parts))
        return np.asarray(rows)

    def _load_tables(self, kind, transform=None):
        tables = {}
        for k in self.kmers:
            table = _read_kmer_table(self._table_path(kind, k))
            if transform is not None:
                table = {key: transform(val) for key, val in table.items()}
            tables[k] = table
        return tables

    def _stack(self, tables, kind):
        """Concatenate every requested k-mer length onto the same tetramer-window axis."""
        blocks, names = [], []
        seq_len = len(self.selected_tetramers)
        for k in self.kmers:
            table = tables[k]
            width = len(next(iter(table.values())))
            flat = self._aligned_windows(table, k)
            blocks.append(flat.reshape(flat.shape[0], seq_len, width))
            names.extend(self._channel_names(kind, k, width))
        self._channels[kind] = names
        # channels of every k-mer sit side by side inside each tetramer window
        return np.concatenate(blocks, axis=-1).reshape(blocks[0].shape[0], -1)

    @property
    def positional_channels(self):
        """(feature, channel names) in the same order as `features`."""
        return [(feat, self._channels[feat]) for feat in self.feats if feat in self._channels]

    def featurize(self):
        if 'avg' in self.feats:
            self.tetra_avg = self._stack(self._load_tables('avg'), 'avg')
        if 'full_fce' in self.feats or 'diagonal_fce' in self.feats:
            fce = self._load_tables('fce')
            if self.randomize:
                # scramble the matchings as 'negative control', i.e. remove all physical information
                for k, table in fce.items():
                    keys = list(table)
                    perm = np.random.permutation(keys)
                    fce[k] = {key: table[val] for key, val in zip(keys, perm)}
            if 'full_fce' in self.feats:
                self.tetra_fce = self._stack(fce, 'full_fce')
            if 'diagonal_fce' in self.feats:
                diag = {k: {key: val[list(range(0, 36, 7))] for key, val in table.items()}
                        for k, table in fce.items()}
                self.tetra_fce_reduced = self._stack(diag, 'diagonal_fce')
        if 'onehot_window' in self.feats:
            # per-window one-hot of the 4 bases of each tetramer window (16 per window)
            onehot = np.array([self.onehot_encoding(otmer) for otmer in self.sequences])
            self.onehot_window = np.concatenate([onehot[:, 4 * i:4 * i + 16] for i in self.selected_tetramers],
                                                axis=1)
        if 'onehot_1mer' in self.feats:
            self.onehot_1mer = np.array([self.onehot_encoding(otmer) for otmer in self.sequences])
        if 'integer' in self.feats:
            nucleotides = product('ACGT', repeat=4)
            char_to_int = dict((''.join(c), i) for i, c in enumerate(nucleotides))
            self.integer = np.array([[char_to_int[otmer[i:i+4]] for i in self.selected_tetramers]
                                     for otmer in self.sequences]).astype(int)
        if 'presence_tetramer' in self.feats:
            self.presence_tetra = np.array([self.presence(otmer, 4) for otmer in self.sequences]).astype(np.int8)
        if 'mgw' in self.feats:
            self.mgw = self._stack(self._load_tables('mgw'), 'mgw')
        if 'electrostatic' in self.feats:
            self.electrostatic = self._stack(self._load_tables('electrostatic'), 'electrostatic')
        if 'ion' in self.feats:
            self.ion = self._stack(self._load_tables('ion'), 'ion')

    @staticmethod
    def onehot_encoding(sequence):
        nucleotides = 'ACGT'
        char_to_int = dict((c, i) for i, c in enumerate(nucleotides))
        integer_encoded = [char_to_int[char] for char in sequence]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(nucleotides))]
            letter[value] = 1
            onehot_encoded.extend(letter)
        return np.array(onehot_encoded)

    @staticmethod
    def presence(sequence, k):
        kmers = np.zeros(4 ** k)
        positions = {''.join(x): n for n, x in enumerate(list(product('ACTG', repeat=k)))}
        for i in range(len(sequence) - k + 1):
            kmers[positions[sequence[i:i+k]]] += 1
        return kmers

    @property
    def scores(self):
        keyw = "VALUE" if self.score == 'Median_intensity' else "Z-score"
        if self.model_target == "octamers":
            # raw labels; scaling is fitted on the training split inside Model
            return self.pbm35[keyw].values.astype(float).reshape(-1, 1)
        elif self.model_target == "tetramers":
            mean_scores = self.mean_score()
            vals = np.array([[mean_scores[i][otmer[i:i+4]] for i in self.selected_tetramers]
                             for otmer in self.sequences])
            return vals.astype(float)

    @staticmethod
    def inv_seq(seq):
        complementary = {"A": "T", "T": "A", "C": "G", "G": "C"}
        return ''.join([complementary[x] for x in seq[::-1]])

    @property
    def features(self):
        avail = {'avg': 'tetra_avg', 'full_fce': 'tetra_fce', 'diagonal_fce': 'tetra_fce_reduced',
                 'onehot_1mer': 'onehot_1mer', 'integer': 'integer',
                 'presence_tetramer': 'presence_tetra', 'mgw': 'mgw', 'electrostatic': 'electrostatic',
                 'ion': 'ion', 'onehot_window': 'onehot_window'}
        return np.hstack([self.__getattribute__(avail[feat]) for feat in self.feats])

    _POSITIONAL_FEATS = {
        'avg': 'tetra_avg',
        'full_fce': 'tetra_fce',
        'diagonal_fce': 'tetra_fce_reduced',
        'mgw': 'mgw',
        'electrostatic': 'electrostatic',
        'ion': 'ion',
        'onehot_window': 'onehot_window',
    }
    _NON_POSITIONAL_FEATS = frozenset(
        {'presence_tetramer', 'onehot_1mer', 'integer'}
    )

    @property
    def windowed_features(self):
        """Per-window feature tensor (n_samples, seq_len, feat_dim) for models
        that consume positional windows (CNN1D, CNNAttention, Transformer)."""
        seq_len = len(self.selected_tetramers)
        blocks = []
        for feat in self.feats:
            if feat in self._NON_POSITIONAL_FEATS:
                print(f"windowed_features: skipping non-positional feature '{feat}'")
                continue
            if feat not in self._POSITIONAL_FEATS:
                print(f"windowed_features: unknown feature '{feat}', skipping")
                continue
            attr = self._POSITIONAL_FEATS[feat]
            flat = self.__getattribute__(attr)
            n = flat.shape[0]
            if flat.shape[1] % seq_len != 0:
                raise ValueError(
                    f"Feature '{feat}' width {flat.shape[1]} is not divisible by seq_len {seq_len}"
                )
            dim = flat.shape[1] // seq_len
            blocks.append(flat.reshape(n, seq_len, dim))
        if not blocks:
            raise ValueError(
                "No positional features available for windowed_features; "
                f"chosen_features={self.feats}"
            )
        return np.concatenate(blocks, axis=-1)

    def mean_score(self):
        """
        generates a list of dictionaries, each containing the position-wise
        score per tetramer
        """
        keyw = "VALUE" if self.score == 'Median_intensity' else "Z-score"
        if self.model_target == "tetramers":
            expt_scores = self.pbm35[keyw].values
            position_scores = [defaultdict(lambda: 0) for _ in range(self.p - 3)]
            position_counts = [defaultdict(lambda: 0) for _ in range(self.p - 3)]
            for seq, score in zip(self.sequences, expt_scores):
                for i in range(self.p - 3):
                    position_scores[i][seq[i:i+4]] += score
                    position_counts[i][seq[i:i+4]] += 1
        return [{ttmer: position_scores[i][ttmer] / position_counts[i][ttmer]
                 for ttmer in position_scores[i].keys()} for i in range(self.p - 3)]
