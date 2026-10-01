from __future__ import annotations

import numpy as np
import pandas as pd

from modeling import splits as sp
from modeling.diagnostics.icbhi_robustness.baselines import permutation_map
from modeling.diagnostics.icbhi_robustness.build_splits import (
    collapsed_device_strata,
    repeated_split_frame,
)
from modeling.diagnostics.icbhi_robustness.common import (
    EXPECTED_PIPELINE_IDS,
    load_protocol,
)


def synthetic_holdout() -> tuple[sp.HoldoutSplit, pd.DataFrame]:
    rows = []
    devices = []
    patient_number = 0
    for label, device in (
        (0, "Meditron"),
        (1, "Meditron"),
        (1, "AKGC417L"),
        (1, "LittC2SE"),
    ):
        for _ in range(5):
            uid = f"p{patient_number:03d}"
            rows.append(
                {
                    "patient_uid": uid,
                    "source_dataset": "ICBHI",
                    "target_label": label,
                    "outer_role": sp.OUTER_ROLE_DEVELOPMENT,
                    "inner_fold_group": patient_number % 5,
                    "calibration_patient": False,
                }
            )
            devices.append(
                {
                    "patient_uid": uid,
                    "target_label": label,
                    "device_signature": device,
                }
            )
            patient_number += 1
    for label in (0, 1):
        uid = f"test_{label}"
        rows.append(
            {
                "patient_uid": uid,
                "source_dataset": "ICBHI",
                "target_label": label,
                "outer_role": sp.OUTER_ROLE_TEST,
                "inner_fold_group": -1,
                "calibration_patient": False,
            }
        )
    table = pd.DataFrame(rows)
    return sp.HoldoutSplit("ICBHI", table, 5), pd.DataFrame(devices)


def test_protocol_freezes_exactly_four_cofinalists():
    protocol = load_protocol()
    assert tuple(pipeline.id for pipeline in protocol.pipelines) == EXPECTED_PIPELINE_IDS
    assert protocol.n_splits == 5
    assert len(protocol.seeds) == 5


def test_repeated_split_preserves_outer_test_and_covers_development_once():
    original, devices = synthetic_holdout()
    repeated = repeated_split_frame(original, devices, 12345)
    original_test = set(original.blocked_test_patients())
    repeated_test = set(
        repeated.loc[repeated["outer_role"] == sp.OUTER_ROLE_TEST, "patient_uid"]
    )
    assert repeated_test == original_test
    development = repeated.loc[
        repeated["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT
    ]
    assert set(development["inner_fold_group"]) == set(range(5))
    assert development.groupby("patient_uid")["inner_fold_group"].nunique().eq(1).all()


def test_repeated_split_accepts_numeric_dtype_change_without_changing_test():
    original, devices = synthetic_holdout()
    original.patient_table["inner_fold_group"] = original.patient_table[
        "inner_fold_group"
    ].astype(float)
    repeated = repeated_split_frame(original, devices, 54321)
    test = repeated.loc[repeated["outer_role"] == sp.OUTER_ROLE_TEST]
    assert set(test["patient_uid"]) == set(original.blocked_test_patients())
    assert test["inner_fold_group"].astype(int).eq(-1).all()


def test_repeated_split_never_writes_folds_into_interleaved_test_rows():
    original, devices = synthetic_holdout()
    shuffled = original.patient_table.sample(frac=1.0, random_state=77).reset_index(
        drop=True
    )
    interleaved = sp.HoldoutSplit("ICBHI", shuffled, 5)
    repeated = repeated_split_frame(interleaved, devices, 24680)
    test = repeated.loc[repeated["outer_role"] == sp.OUTER_ROLE_TEST]
    development = repeated.loc[
        repeated["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT
    ]
    assert set(test["patient_uid"]) == set(interleaved.blocked_test_patients())
    assert test["inner_fold_group"].astype(int).eq(-1).all()
    assert development["inner_fold_group"].astype(int).between(0, 4).all()


def test_sparse_device_signatures_are_collapsed_without_mixing_classes():
    development = pd.DataFrame(
        {
            "target_label": [0] * 10 + [1] * 10,
            "device_signature": (
                ["Meditron"] * 10
                + ["AKGC417L"] * 4
                + ["Meditron"] * 4
                + ["AKGC417L|LittC2SE", "Litt3200|Meditron"]
            ),
        }
    )
    strata = collapsed_device_strata(development, n_splits=5)
    expected_prefixes = development["target_label"].astype(str) + ":"
    assert all(
        str(value).startswith(prefix)
        for value, prefix in zip(strata, expected_prefixes, strict=True)
    )
    assert int(strata.value_counts().min()) >= 5
    assert (strata.iloc[10:] == "1:OTHER_DEVICE_PATTERN").all()


def test_permutation_preserves_counts_and_has_both_classes_in_every_fold():
    split, _devices = synthetic_holdout()
    mapping = permutation_map(split, random_state=999, max_attempts=1000)
    assert len(mapping) == len(split.development_patients())
    assert mapping["original_target_label"].value_counts().to_dict() == (
        mapping["permuted_target_label"].value_counts().to_dict()
    )
    assert not np.array_equal(
        mapping["original_target_label"], mapping["permuted_target_label"]
    )
    for fold in range(5):
        assert set(
            mapping.loc[mapping["fold"] == fold, "permuted_target_label"]
        ) == {0, 1}

