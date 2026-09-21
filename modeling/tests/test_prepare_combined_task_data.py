"""Pruebas para el generador del dataset combinado COPD vs control.

Cubre la seleccion/armonizacion (``select_combined``), la comparacion
contra conteos esperados (``verify_expected_counts``), la copia y el
manifiesto (``build_combined``) y la proteccion contra sobrescritura del
publicado atomico reutilizado de ``prepare_task_data``.
"""

import json

import numpy as np
import pandas as pd
import pytest

from .. import prepare_combined_task_data as pctd
from .. import prepare_task_data as ptd


def _row(array_index, segment_id, audio_id, dataset, patient_uid, diagnosis, filter_value=None):
    return {
        "array_index": array_index,
        "segment_id": segment_id,
        "audio_id": audio_id,
        "dataset": dataset,
        "patient_uid": patient_uid,
        "diagnosis": diagnosis,
        "filter": filter_value,
    }


def _mixed_metadata() -> pd.DataFrame:
    """Filas elegibles (8) mezcladas con filas que deben excluirse.

    El orden de la lista esta deliberadamente desordenado respecto de
    ``array_index`` para probar que la seleccion no depende del orden de
    entrada, solo de ``array_index``.
    """

    rows = [
        # Elegibles, en orden desordenado respecto de array_index.
        _row(5, "ICBHI_P2_COPD_000", "ICBHI_P2_COPD", "ICBHI", "ICBHI_P2", "COPD"),
        _row(0, "ICBHI_P1_COPD_000", "ICBHI_P1_COPD", "ICBHI", "ICBHI_P1", "COPD"),
        _row(7, "FRAIWAN_F2_Normal_000", "FRAIWAN_F2_Normal", "FRAIWAN", "FRAIWAN_F2", "Normal", "Extended"),
        _row(2, "ICBHI_P4_Healthy_000", "ICBHI_P4_Healthy", "ICBHI", "ICBHI_P4", "Healthy"),
        _row(6, "FRAIWAN_F1_COPD_000", "FRAIWAN_F1_COPD", "FRAIWAN", "FRAIWAN_F1", "COPD", "Extended"),
        _row(1, "ICBHI_P3_Healthy_000", "ICBHI_P3_Healthy", "ICBHI", "ICBHI_P3", "Healthy"),
        _row(4, "FRAIWAN_F4_Normal_000", "FRAIWAN_F4_Normal", "FRAIWAN", "FRAIWAN_F4", "Normal", "Extended"),
        _row(3, "FRAIWAN_F3_COPD_000", "FRAIWAN_F3_COPD", "FRAIWAN", "FRAIWAN_F3", "COPD", "Extended"),
        # Excluidas: diagnostico fuera de la cohorte ICBHI.
        _row(8, "ICBHI_P5_Pneumonia_000", "ICBHI_P5_Pneumonia", "ICBHI", "ICBHI_P5", "Pneumonia"),
        # Excluidas: Fraiwan COPD con filtro distinto de Extended.
        _row(9, "FRAIWAN_F6_COPD_Bell_000", "FRAIWAN_F6_COPD_Bell", "FRAIWAN", "FRAIWAN_F6", "COPD", "Bell"),
        _row(10, "FRAIWAN_F7_COPD_Diaphragm_000", "FRAIWAN_F7_COPD_Diaphragm", "FRAIWAN", "FRAIWAN_F7", "COPD", "Diaphragm"),
        # Excluida: diagnostico fuera de la cohorte Fraiwan, aunque el filtro sea Extended.
        _row(11, "FRAIWAN_F8_Asthma_000", "FRAIWAN_F8_Asthma", "FRAIWAN", "FRAIWAN_F8", "Asthma", "Extended"),
        # Excluida: Fraiwan Normal sin filtro Extended.
        _row(12, "FRAIWAN_F9_Normal_NoFilter_000", "FRAIWAN_F9_Normal_NoFilter", "FRAIWAN", "FRAIWAN_F9", "Normal"),
    ]
    return pd.DataFrame(rows)


def test_select_combined_filters_harmonizes_orders_and_indexes():
    selected = pctd.select_combined(_mixed_metadata())

    assert len(selected) == 8
    assert set(selected["segment_id"]) == {
        "ICBHI_P1_COPD_000", "ICBHI_P2_COPD_000",
        "ICBHI_P3_Healthy_000", "ICBHI_P4_Healthy_000",
        "FRAIWAN_F1_COPD_000", "FRAIWAN_F3_COPD_000",
        "FRAIWAN_F2_Normal_000", "FRAIWAN_F4_Normal_000",
    }

    # Orden: por source_array_index ascendente, sin importar el orden de entrada.
    assert selected["source_array_index"].tolist() == list(range(8))
    assert selected["task_array_index"].tolist() == list(range(8))

    by_segment = selected.set_index("segment_id")
    assert by_segment.loc["ICBHI_P1_COPD_000", "target_label"] == 1
    assert by_segment.loc["ICBHI_P1_COPD_000", "target_name"] == "COPD"
    assert by_segment.loc["FRAIWAN_F1_COPD_000", "target_label"] == 1
    assert by_segment.loc["FRAIWAN_F1_COPD_000", "target_name"] == "COPD"
    assert by_segment.loc["ICBHI_P3_Healthy_000", "target_label"] == 0
    assert by_segment.loc["ICBHI_P3_Healthy_000", "target_name"] == "Control"
    assert by_segment.loc["FRAIWAN_F2_Normal_000", "target_label"] == 0
    assert by_segment.loc["FRAIWAN_F2_Normal_000", "target_name"] == "Control"

    # diagnosis original se conserva (Healthy/Normal), no se sobrescribe.
    assert by_segment.loc["ICBHI_P3_Healthy_000", "diagnosis"] == "Healthy"
    assert by_segment.loc["FRAIWAN_F2_Normal_000", "diagnosis"] == "Normal"


def test_select_combined_rejects_duplicate_segment_id():
    metadata = _mixed_metadata()
    duplicate = _row(13, "ICBHI_P1_COPD_000", "ICBHI_P1_COPD", "ICBHI", "ICBHI_P6", "COPD")
    metadata = pd.concat([metadata, pd.DataFrame([duplicate])], ignore_index=True)

    with pytest.raises(ValueError, match="duplicados"):
        pctd.select_combined(metadata)


def test_select_combined_rejects_patient_with_conflicting_labels():
    metadata = _mixed_metadata()
    # Mismo patient_uid que una fila COPD, pero ahora con diagnostico Healthy.
    conflict = _row(13, "ICBHI_P1_Healthy_000", "ICBHI_P1_Healthy", "ICBHI", "ICBHI_P1", "Healthy")
    metadata = pd.concat([metadata, pd.DataFrame([conflict])], ignore_index=True)

    with pytest.raises(ValueError, match="mas de una etiqueta"):
        pctd.select_combined(metadata)


def test_select_combined_rejects_shared_patient_uid_across_datasets():
    metadata = _mixed_metadata()
    # Mismo patient_uid en ICBHI y Fraiwan (ambos COPD, misma etiqueta:
    # esto no dispara la verificacion de etiquetas, solo la de fuentes).
    shared_a = _row(13, "SHARED_A_000", "SHARED_A", "ICBHI", "SHARED_001", "COPD")
    shared_b = _row(14, "SHARED_B_000", "SHARED_B", "FRAIWAN", "SHARED_001", "COPD", "Extended")
    metadata = pd.concat([metadata, pd.DataFrame([shared_a, shared_b])], ignore_index=True)

    with pytest.raises(ValueError, match="compartidos entre"):
        pctd.select_combined(metadata)


def test_select_combined_rejects_incomplete_diagnosis_set():
    # Solo ICBHI COPD, sin ninguna fila Healthy: la cohorte ICBHI queda incompleta.
    metadata = pd.DataFrame([
        _row(0, "ICBHI_P1_COPD_000", "ICBHI_P1_COPD", "ICBHI", "ICBHI_P1", "COPD"),
    ])

    with pytest.raises(ValueError, match="se esperaban los diagnosticos"):
        pctd.select_combined(metadata)


def test_select_combined_rejects_empty_selection():
    metadata = pd.DataFrame([
        _row(0, "ICBHI_P1_Pneumonia_000", "ICBHI_P1_Pneumonia", "ICBHI", "ICBHI_P1", "Pneumonia"),
    ])

    with pytest.raises(ValueError, match="vacia"):
        pctd.select_combined(metadata)


def test_verify_expected_counts_matches_injected_expectations():
    selected = pctd.select_combined(_mixed_metadata())

    expected_groups = {
        ("ICBHI", "COPD"): {"patients": 2, "recordings": 2, "segments": 2},
        ("ICBHI", "Healthy"): {"patients": 2, "recordings": 2, "segments": 2},
        ("FRAIWAN", "COPD"): {"patients": 2, "recordings": 2, "segments": 2},
        ("FRAIWAN", "Normal"): {"patients": 2, "recordings": 2, "segments": 2},
    }
    expected_total = {"patients": 8, "recordings": 8, "segments": 8}
    expected_harmonized = {
        "COPD": {"patients": 4, "recordings": 4, "segments": 4},
        "Control": {"patients": 4, "recordings": 4, "segments": 4},
    }

    groups, total, harmonized = pctd.verify_expected_counts(
        selected, expected_groups, expected_total, expected_harmonized
    )
    assert groups["ICBHI/COPD"] == {"patients": 2, "recordings": 2, "segments": 2}
    assert total == expected_total
    assert harmonized == expected_harmonized


def test_verify_expected_counts_raises_on_total_mismatch():
    selected = pctd.select_combined(_mixed_metadata())
    correct_groups = {
        ("ICBHI", "COPD"): {"patients": 2, "recordings": 2, "segments": 2},
        ("ICBHI", "Healthy"): {"patients": 2, "recordings": 2, "segments": 2},
        ("FRAIWAN", "COPD"): {"patients": 2, "recordings": 2, "segments": 2},
        ("FRAIWAN", "Normal"): {"patients": 2, "recordings": 2, "segments": 2},
    }
    correct_harmonized = {
        "COPD": {"patients": 4, "recordings": 4, "segments": 4},
        "Control": {"patients": 4, "recordings": 4, "segments": 4},
    }
    wrong_total = {"patients": 999, "recordings": 999, "segments": 999}

    with pytest.raises(ValueError, match="Total combinado"):
        pctd.verify_expected_counts(
            selected,
            expected_groups=correct_groups,
            expected_total=wrong_total,
            expected_harmonized=correct_harmonized,
        )


def test_verify_expected_counts_raises_on_group_mismatch():
    selected = pctd.select_combined(_mixed_metadata())
    wrong_groups = {("ICBHI", "COPD"): {"patients": 999, "recordings": 999, "segments": 999}}

    with pytest.raises(ValueError, match="ICBHI/COPD"):
        pctd.verify_expected_counts(selected, expected_groups=wrong_groups)


def _toy_selected_and_arrays():
    metadata = _mixed_metadata()
    selected = pctd.select_combined(metadata)
    n_source = int(selected["source_array_index"].max()) + 1
    rng = np.random.default_rng(0)
    source_arrays = {
        "no_dn": rng.standard_normal((n_source, 6)).astype(np.float32),
        "dn": rng.standard_normal((n_source, 6)).astype(np.float32),
    }
    return selected, source_arrays


def test_build_combined_copies_rows_and_writes_consistent_manifest(tmp_path):
    selected, source_arrays = _toy_selected_and_arrays()
    staging_root = tmp_path / "staging"
    staging_root.mkdir()

    manifest = pctd.build_combined(
        staging_root=staging_root,
        source_root=tmp_path / "source",
        source_arrays=source_arrays,
        selected=selected,
        source_hashes={"segments.csv": "deadbeef"},
        chunk_size=3,
        group_counts={},
        total_counts={"patients": 8, "recordings": 8, "segments": 8},
        harmonized_counts={},
    )

    task_dir = staging_root / pctd.OUTPUT_NAME
    assert manifest["verdict"] == "PASS"
    assert manifest["output_shape"] == [8, 6]
    assert manifest["dtype"] == "float32"

    no_dn = np.load(task_dir / "segments_no_dn.npy")
    dn = np.load(task_dir / "segments_dn.npy")
    source_indices = selected["source_array_index"].to_numpy()
    assert np.array_equal(no_dn, source_arrays["no_dn"][source_indices])
    assert np.array_equal(dn, source_arrays["dn"][source_indices])

    written = pd.read_csv(task_dir / "segments.csv")
    assert len(written) == len(no_dn) == len(dn)
    assert written["task_array_index"].tolist() == list(range(len(written)))

    on_disk_manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk_manifest["verdict"] == "PASS"


def test_publish_overwrite_protection(tmp_path):
    output_root = tmp_path / "copd_vs_control_combined"

    def _publish_once(seed):
        selected, source_arrays = _toy_selected_and_arrays()
        source_arrays = {branch: array + seed for branch, array in source_arrays.items()}
        staging_root = tmp_path / f"staging_{seed}"
        staging_root.mkdir()
        pctd.build_combined(
            staging_root=staging_root,
            source_root=tmp_path / "source",
            source_arrays=source_arrays,
            selected=selected,
            source_hashes={"segments.csv": "deadbeef"},
            chunk_size=4,
            group_counts={},
            total_counts={},
            harmonized_counts={},
        )
        return staging_root, source_arrays

    first_staging, first_arrays = _publish_once(seed=0.0)
    ptd.publish(first_staging, output_root, overwrite=False)
    first_no_dn = np.load(output_root / pctd.OUTPUT_NAME / "segments_no_dn.npy")
    assert np.array_equal(first_no_dn, first_arrays["no_dn"][
        pd.read_csv(output_root / pctd.OUTPUT_NAME / "segments.csv")["source_array_index"].to_numpy()
    ])

    second_staging, _ = _publish_once(seed=100.0)
    with pytest.raises(FileExistsError, match="--overwrite"):
        ptd.publish(second_staging, output_root, overwrite=False)
    # La salida existente no debe alterarse tras el intento rechazado.
    unchanged_no_dn = np.load(output_root / pctd.OUTPUT_NAME / "segments_no_dn.npy")
    assert np.array_equal(unchanged_no_dn, first_no_dn)

    third_staging, third_arrays = _publish_once(seed=200.0)
    ptd.publish(third_staging, output_root, overwrite=True)
    replaced_no_dn = np.load(output_root / pctd.OUTPUT_NAME / "segments_no_dn.npy")
    assert np.array_equal(replaced_no_dn, third_arrays["no_dn"][
        pd.read_csv(output_root / pctd.OUTPUT_NAME / "segments.csv")["source_array_index"].to_numpy()
    ])
