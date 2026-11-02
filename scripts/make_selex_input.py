#!/usr/bin/env python3
"""Concatenate every SELEX property table into one file per k-mer length.

Reads only data/selex/{tetramer,hexamer}/ and writes
data/selex/tetramer/input_tetramer.dat and data/selex/hexamer/input_hexamer.dat.
Rows are joined on the k-mer (the source files are not in the same order) and
sorted alphabetically. Columns are avg, then the full force-constant matrix,
then electrostatics, then ion concentration.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELEX = ROOT / 'data' / 'selex'

# (filename, column prefix or None to keep the file's own header names)
SOURCES = (
    ('avg_{name}.dat', None),
    ('fce_{name}.dat', None),
    ('electrostatic_{name}.dat', 'e'),
    ('ion_vec_{name}.dat', None),
)


def _read(path, k, prefix):
    """Return (column names, {kmer: values}). Header rows are anything whose first token is not a k-mer."""
    rows = {}
    header = None
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        key, *values = parts
        if set(key) <= set('ACGT') and len(key) == k:
            rows[key] = values
        elif header is None:
            header = parts
    if not rows:
        raise SystemExit(f'No {k}-mers in {path}')
    width = len(next(iter(rows.values())))
    if header is not None and len(header) == width + 1:
        header = header[1:]  # leading token is the k-mer column (SEQUENCE / tet / hex)
    if header is None or len(header) != width:
        # electrostatic tetramers have no header; name columns e1..eN
        stem = prefix or 'c'
        header = [f'{stem}{i}' for i in range(1, width + 1)]
    return header, rows


def build(name, k):
    folder = SELEX / name
    columns, tables = [], []
    for pattern, prefix in SOURCES:
        header, rows = _read(folder / pattern.format(name=name), k, prefix)
        columns.extend(header)
        tables.append(rows)
    keys = sorted(set.intersection(*(set(t) for t in tables)))
    if len(keys) != 4 ** k:
        raise SystemExit(f'{name}: expected {4 ** k} k-mers, got {len(keys)}')
    out = folder / f'input_{name}.dat'
    with out.open('w') as handle:
        handle.write('SEQUENCE\t' + '\t'.join(columns) + '\n')
        for key in keys:
            values = [v for table in tables for v in table[key]]
            handle.write(key + '\t' + '\t'.join(values) + '\n')
    print(f'{out.relative_to(ROOT)}: {len(keys)} {k}-mers x {len(columns)} columns')


if __name__ == '__main__':
    build('tetramer', 4)
    build('hexamer', 6)
