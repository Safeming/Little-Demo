#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_a7c_r1_2a_quotient_compositor import verify_source_file
from utils.a7c_oracle_capacity import _artifact_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _write_deterministic_npz(path: Path, arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, mode="w") as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[key]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _slice_sample_arrays(arrays, selected, sample_count):
    output = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        output[key] = (
            array[selected]
            if array.ndim > 0 and array.shape[0] == sample_count
            else array
        )
    return output


def stage_fit_only_inputs(
    *,
    contract,
    probe_path,
    teacher_path,
    evidence_path,
    r1_2b_training_dir,
    teachers_dir,
    output_dir,
):
    probe_path = Path(probe_path)
    teacher_path = Path(teacher_path)
    evidence_path = Path(evidence_path)
    r1_2b_training_dir = Path(r1_2b_training_dir)
    teachers_dir = Path(teachers_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"fit-only output already exists: {output}")
    for path, expected, label in (
        (probe_path, contract["source_probe_sha256"], "probe"),
        (teacher_path, contract["source_teacher_sha256"], "teacher"),
        (evidence_path, contract["source_evidence_sha256"], "evidence"),
    ):
        verify_source_file(path, expected, label)

    probe = _load_npz(probe_path)
    if _artifact_fingerprint(probe) != str(probe["output_fingerprint"]):
        raise ValueError("source probe artifact fingerprint mismatch")
    teacher = _load_npz(teacher_path)
    for key in ("carrier_ids", "camera_index", "frame_index"):
        if not np.array_equal(probe[key], teacher[key]):
            raise ValueError(f"source probe and teacher {key} differ")
    cameras = np.asarray(teacher["camera_index"]).reshape(-1)
    sample_count = int(cameras.size)
    if sample_count != int(contract["source_expected_sample_count"]):
        raise ValueError("source sample count differs from the frozen contract")
    fit_cameras = np.asarray(contract["fit_camera_indices"], dtype=cameras.dtype)
    selected = np.flatnonzero(np.isin(cameras, fit_cameras))
    if selected.size != int(contract["fit_only_expected_sample_count"]):
        raise ValueError("fit-only sample count differs from the frozen contract")
    if set(map(int, np.unique(cameras[selected]))) != set(map(int, fit_cameras)):
        raise ValueError("fit-only camera set differs from the frozen contract")

    artifact_paths = []
    staged_probe = _slice_sample_arrays(probe, selected, sample_count)
    staged_probe["output_fingerprint"] = np.asarray(
        _artifact_fingerprint(staged_probe)
    )
    probe_output = output / "probe/probe.npz"
    _write_deterministic_npz(probe_output, staged_probe)
    artifact_paths.append(probe_output)

    staged_teacher = {
        "carrier_ids": np.asarray(teacher["carrier_ids"]),
        "camera_index": np.asarray(teacher["camera_index"])[selected],
        "frame_index": np.asarray(teacher["frame_index"])[selected],
        "source_row_indices": selected.astype(np.int64),
        "source_sample_count": np.asarray(sample_count, np.int64),
        "output_fingerprint": np.asarray(_artifact_fingerprint({
            "carrier_ids": np.asarray(teacher["carrier_ids"]),
            "camera_index": np.asarray(teacher["camera_index"])[selected],
            "frame_index": np.asarray(teacher["frame_index"])[selected],
            "source_row_indices": selected.astype(np.int64),
            "source_sample_count": np.asarray(sample_count, np.int64),
        })),
    }
    teacher_output = output / "teacher/teacher.npz"
    _write_deterministic_npz(teacher_output, staged_teacher)
    artifact_paths.append(teacher_output)

    evidence = _load_npz(evidence_path)
    if not np.array_equal(
        evidence["renderer_sequence_camera_index"], teacher["camera_index"]
    ) or not np.array_equal(
        evidence["renderer_sequence_frame_index"], teacher["frame_index"]
    ):
        raise ValueError("source evidence and teacher manifests differ")
    staged_evidence = {
        "renderer_sequence_camera_index": np.asarray(
            evidence["renderer_sequence_camera_index"]
        )[selected],
        "renderer_sequence_frame_index": np.asarray(
            evidence["renderer_sequence_frame_index"]
        )[selected],
        "source_row_indices": selected.astype(np.int64),
        "source_sample_count": np.asarray(sample_count, np.int64),
    }
    for signal in ("target", "outer", "boundary"):
        key = f"renderer_{signal}_contribution_sequence"
        values = np.asarray(evidence[key])
        if values.shape[0] != sample_count:
            raise ValueError(f"source {signal} renderer sequence does not align")
        staged_evidence[key] = values[selected]
    evidence_output = output / "evidence/evidence.npz"
    _write_deterministic_npz(evidence_output, staged_evidence)
    artifact_paths.append(evidence_output)

    for fold in range(6):
        base_path = r1_2b_training_dir / f"fold_{fold}/predictions.npz"
        verify_source_file(
            base_path,
            contract["source_r1_2b_prediction_sha256"][fold],
            f"R1.2-B fold {fold}",
        )
        base = _load_npz(base_path)
        raw = np.asarray(base["raw_gates"])
        if raw.shape[0] != sample_count:
            raise ValueError("source R1.2-B rows differ from teacher manifest")
        base_output = output / f"training/fold_{fold}/predictions.npz"
        _write_deterministic_npz(base_output, {
            "raw_gates": raw[selected],
            "source_row_indices": selected.astype(np.int64),
            "source_sample_count": np.asarray(sample_count, np.int64),
        })
        artifact_paths.append(base_output)

        fold_relative = f"fold_{fold}/teacher.npz"
        fold_path = teachers_dir / fold_relative
        verify_source_file(
            fold_path,
            contract["source_teacher_artifacts"][fold_relative],
            f"R1.4-VP teacher fold {fold}",
        )
        fold_teacher = _load_npz(fold_path)
        gates = np.asarray(fold_teacher["teacher_gates"])
        mask = np.asarray(fold_teacher["teacher_mask"], dtype=bool)
        if gates.shape[0] != sample_count or mask.shape != (sample_count,):
            raise ValueError("source fold teacher rows differ from manifest")
        teacher_output = output / f"teachers/fold_{fold}/teacher.npz"
        staged_fold = {
            "teacher_gates": gates[selected],
            "teacher_mask": mask[selected],
            "camera_index": cameras[selected],
            "frame_index": np.asarray(teacher["frame_index"])[selected],
            "carrier_ids": np.asarray(teacher["carrier_ids"]),
            "source_row_indices": selected.astype(np.int64),
            "source_sample_count": np.asarray(sample_count, np.int64),
        }
        if "block_ids" in fold_teacher:
            staged_fold["block_ids"] = np.asarray(fold_teacher["block_ids"])[selected]
        _write_deterministic_npz(teacher_output, staged_fold)
        artifact_paths.append(teacher_output)

    manifest = {
        "schema_version": 1,
        "stage": "a7c_r1_4vp_r4b0_fit_only_inputs",
        "source_sample_count": sample_count,
        "fit_sample_count": int(selected.size),
        "fit_camera_indices": list(map(int, fit_cameras)),
        "source_row_indices": list(map(int, selected)),
        "source_sha256": {
            "probe": _sha256(probe_path),
            "teacher": _sha256(teacher_path),
            "evidence": _sha256(evidence_path),
        },
        "artifact_sha256": {
            str(path.relative_to(output)): _sha256(path)
            for path in artifact_paths
        },
        "deployment_eligible": False,
        "teacher_eligible": False,
        "paper_test_eligible": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage immutable fit-only inputs for A7c R4-B0."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--r1-2b-training-dir", type=Path, required=True)
    parser.add_argument("--teachers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest = stage_fit_only_inputs(
        contract=contract,
        probe_path=args.probe,
        teacher_path=args.teacher,
        evidence_path=args.evidence,
        r1_2b_training_dir=args.r1_2b_training_dir,
        teachers_dir=args.teachers_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
