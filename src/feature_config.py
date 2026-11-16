"""Canonical descriptor feature lists and exact-dependency removal logs.

FCE uses the 21 unique upper-triangular entries of the symmetric 6x6 matrix.
The lower-triangular mirrored columns are documented in the removal logs.
The electrostatic identities e3 = -e2 and e5 = -e1 - e4 hold exactly in
both stored combined tables.
"""

DEGREES_OF_FREEDOM = ["SHIFT", "SLIDE", "RISE", "TILT", "ROLL", "TWIST"]
AVG_FEATURES = DEGREES_OF_FREEDOM.copy()
FCE_FEATURES = [
    f"{first}.{second}"
    for i, first in enumerate(DEGREES_OF_FREEDOM)
    for second in DEGREES_OF_FREEDOM[i:]
]
ELECTROSTATIC_FEATURES_CLEAN = ["e1", "e2", "e4"]
ION_FEATURES_TETRAMER = ["m1", "m2", "m3", "j1", "j2", "j3"]
ION_FEATURES_HEXAMER = ["m0", "m1", "m2", "m3", "m4", "j0", "j1", "j2", "j3", "j4"]

FEATURE_GROUPS_CLEAN_TETRAMER = {
    "AVG": AVG_FEATURES.copy(),
    "FCE": FCE_FEATURES.copy(),
    "Electrostatic": ELECTROSTATIC_FEATURES_CLEAN.copy(),
    "Ion-related": ION_FEATURES_TETRAMER.copy(),
}
FEATURE_GROUPS_CLEAN_HEXAMER = {
    "AVG": AVG_FEATURES.copy(),
    "FCE": FCE_FEATURES.copy(),
    "Electrostatic": ELECTROSTATIC_FEATURES_CLEAN.copy(),
    "Ion-related": ION_FEATURES_HEXAMER.copy(),
}

FEATURES_CLEAN_TETRAMER = [
    feature
    for group in FEATURE_GROUPS_CLEAN_TETRAMER.values()
    for feature in group
]
FEATURES_CLEAN_HEXAMER = [
    feature
    for group in FEATURE_GROUPS_CLEAN_HEXAMER.values()
    for feature in group
]

_FCE_MIRROR_REMOVALS = []
for i, first in enumerate(DEGREES_OF_FREEDOM):
    for second in DEGREES_OF_FREEDOM[:i]:
        excluded = f"{first}.{second}"
        retained = f"{second}.{first}"
        _FCE_MIRROR_REMOVALS.append({
            "feature": excluded,
            "group": "FCE",
            "equation": f"{excluded} = {retained}",
            "scope": "within group",
            "max_abs_residual": 0.0,
            "recommended_retain": [retained],
            "recommended_exclude": [excluded],
            "justification": "Mirrored entry of the symmetric FCE matrix; values match for every sequence.",
        })

_ELECTROSTATIC_REMOVALS = [
    {
        "feature": "e3",
        "group": "Electrostatic",
        "equation": "e3 = -e2",
        "scope": "within group",
        "max_abs_residual": 0.0,
        "recommended_retain": ["e2"],
        "recommended_exclude": ["e3"],
        "justification": "Exact negative of e2 for every sequence.",
    },
    {
        "feature": "e5",
        "group": "Electrostatic",
        "equation": "e5 = -e1 - e4",
        "scope": "within group",
        "max_abs_residual": 0.0,
        "recommended_retain": ["e1", "e4"],
        "recommended_exclude": ["e5"],
        "justification": "Exact linear combination of e1 and e4 for every sequence.",
    },
]

REMOVAL_LOG_TETRAMER = [
    dict(entry) for entry in (_FCE_MIRROR_REMOVALS + _ELECTROSTATIC_REMOVALS)
]
REMOVAL_LOG_HEXAMER = [
    dict(entry) for entry in (_FCE_MIRROR_REMOVALS + _ELECTROSTATIC_REMOVALS)
]

# Numerical audit settings used in notebooks/audit_data_integrity.ipynb.
RANK_TOLERANCE_RULE = "max(matrix_shape) * machine_epsilon * largest_singular_value after centering and unit-variance scaling"
DEPENDENCY_RELATIVE_RESIDUAL_TOLERANCE = 1e-10
