#!/usr/bin/env bash
set -euo pipefail

STAGES=(canary evidence candidates validation loso-freeze retrospective-c21 frozen-c22-c23 paper-tables)
DEFAULT_STAGE=validation
SUBJECTS=(377 386 387 393 394)
EVIDENCE_CAMERAS=c01,c05,c09,c13
VALIDATION_CAMERAS=(17 18 19 20)
FRAME_START=0
FRAME_END=570
FRAME_STEP=5

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/exp/acceptdata/a7_temporal_reliability_v1}"
A5_METHOD_FREEZE="$ROOT/configs/semantic/frozen_a5_main_method_v1.json"
A7_CONTRACT="$ROOT/configs/semantic/frozen_a7_temporal_reliable_v1.json"
A5_BANK_ROOT="$ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723"
A5_LOSO_ROOT="$ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723"
TARGET_STAGE="$DEFAULT_STAGE"
DRY_RUN=0
ALLOW_POST_VALIDATION=0

usage() {
  printf '%s\n' "Usage: $0 [--stage STAGE] [--dry-run] [--allow-post-validation]"
  printf '%s\n' "Stages: ${STAGES[*]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      TARGET_STAGE="${2:?missing stage value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --allow-post-validation)
      ALLOW_POST_VALIDATION=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

stage_index() {
  local requested="$1" index
  for index in "${!STAGES[@]}"; do
    if [[ "${STAGES[$index]}" == "$requested" ]]; then
      printf '%s\n' "$index"
      return
    fi
  done
  echo "unknown stage: $requested" >&2
  return 2
}

TARGET_INDEX="$(stage_index "$TARGET_STAGE")"
VALIDATION_INDEX="$(stage_index validation)"
if (( TARGET_INDEX > VALIDATION_INDEX )) && [[ "$ALLOW_POST_VALIDATION" != "1" ]]; then
  echo "post-validation stages require --allow-post-validation" >&2
  exit 2
fi

subject_root() {
  case "$1" in
    377) printf '%s\n' "$ROOT/exp/acceptdata/coreview377_multisubject_strict_20260721" ;;
    386) printf '%s\n' "$ROOT/exp/acceptdata/coreview386_multisubject_strict_20260719" ;;
    387) printf '%s\n' "$ROOT/exp/acceptdata/coreview387_multisubject_strict_20260720" ;;
    393) printf '%s\n' "$ROOT/exp/acceptdata/coreview393_multisubject_strict_20260721" ;;
    394) printf '%s\n' "$ROOT/exp/acceptdata/coreview394_multisubject_strict_20260722" ;;
    *) echo "unsupported subject: $1" >&2; return 2 ;;
  esac
}

subject_config() {
  printf '%s\n' "$(subject_root "$1")/assets/test/.hydra/config.yaml"
}

checkpoint_path() {
  printf '%s\n' "$(subject_root "$1")/semantic_train_strict/ckpt42000.pth"
}

a5_bank_path() {
  printf '%s\n' "$A5_BANK_ROOT/CoreView_${1}/banks/footprint_evidence_target/part_label_bank.npz"
}

a5_loso_path() {
  printf '%s\n' "$A5_LOSO_ROOT/CoreView_${1}/loso_frozen_config.json"
}

protocol_path() {
  printf '%s\n' "$ROOT/configs/semantic/coreview${1}_strict_paper_protocol.json"
}

utc_now() {
  TZ=UTC date '+%Y-%m-%dT%H:%M:%SZ'
}

bjt_now() {
  TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z'
}

fingerprint_command_inputs() {
  local command="$1" arg
  shift
  {
    printf 'command=%s\n' "$command"
    for arg in "$@"; do
      if [[ -f "$arg" ]]; then
        printf 'input=%s:' "$arg"
        sha256sum "$arg" | awk '{print $1}'
      fi
    done
  } | sha256sum | awk '{print $1}'
}

fingerprint_output() {
  local path="$1"
  if [[ -f "$path" ]]; then
    sha256sum "$path" | awk '{print $1}'
  elif [[ -d "$path" ]]; then
    find "$path" -type f ! -name '.running' ! -name '.done' ! -name '.failed' -print0 \
      | sort -z \
      | xargs -0 -r sha256sum \
      | sha256sum \
      | awk '{print $1}'
  else
    printf '%s\n' missing
  fi
}

state_value() {
  local key="$1" path="$2"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$path"
}

write_state() {
  local path="$1" status="$2" started_utc="$3" started_bjt="$4" finished_utc="$5"
  local finished_bjt="$6" command="$7" log="$8" command_fingerprint="$9" output_fingerprint="${10}"
  {
    printf 'status=%s\n' "$status"
    printf 'started_utc=%s\n' "$started_utc"
    printf 'started_bjt=%s\n' "$started_bjt"
    printf 'finished_utc=%s\n' "$finished_utc"
    printf 'finished_bjt=%s\n' "$finished_bjt"
    printf 'pid=%s\n' "$$"
    printf 'command=%s\n' "$command"
    printf 'log=%s\n' "$log"
    printf 'command_fingerprint=%s\n' "$command_fingerprint"
    printf 'output_fingerprint=%s\n' "$output_fingerprint"
  } > "$path"
}

run_job() {
  local job_id="$1" samples="$2" output="$3"
  shift 3
  local command="" arg state_dir running done failed log started_utc started_bjt
  local current_command_fingerprint recorded_command_fingerprint
  local current_output_fingerprint recorded_output_fingerprint output_fingerprint
  for arg in "$@"; do
    printf -v command '%s %q' "$command" "$arg"
  done
  command="${command# }"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN job=%s gpu=%s samples=%s output=%s command=%s\n' \
      "$job_id" "$GPU" "$samples" "$output" "$command"
    return
  fi
  state_dir="$OUTPUT_ROOT/state/$job_id"
  running="$state_dir/.running"
  done="$state_dir/.done"
  failed="$state_dir/.failed"
  log="$state_dir/job.log"
  mkdir -p "$state_dir"
  current_command_fingerprint="$(fingerprint_command_inputs "$command" "$@")"
  if [[ -f "$done" ]]; then
    recorded_command_fingerprint="$(state_value command_fingerprint "$done")"
    recorded_output_fingerprint="$(state_value output_fingerprint "$done")"
    current_output_fingerprint="$(fingerprint_output "$output")"
    if [[ "$recorded_command_fingerprint" == "$current_command_fingerprint" \
      && "$recorded_output_fingerprint" == "$current_output_fingerprint" ]]; then
      printf 'SKIP job=%s output_fingerprint=%s\n' "$job_id" "$current_output_fingerprint"
      return
    fi
  fi
  rm -f "$running" "$done" "$failed"
  started_utc="$(utc_now)"
  started_bjt="$(bjt_now)"
  write_state "$running" running "$started_utc" "$started_bjt" "" "" \
    "$command" "$log" "$current_command_fingerprint" pending
  if "$@" > "$log" 2>&1; then
    output_fingerprint="$(fingerprint_output "$output")"
    write_state "$done" done "$started_utc" "$started_bjt" "$(utc_now)" "$(bjt_now)" \
      "$command" "$log" "$current_command_fingerprint" "$output_fingerprint"
    rm -f "$running"
    printf 'DONE job=%s output_fingerprint=%s\n' "$job_id" "$output_fingerprint"
  else
    local status=$?
    write_state "$failed" failed "$started_utc" "$started_bjt" "$(utc_now)" "$(bjt_now)" \
      "$command" "$log" "$current_command_fingerprint" "$(fingerprint_output "$output")"
    rm -f "$running"
    echo "FAILED job=$job_id status=$status log=$log" >&2
    return "$status"
  fi
}

validate_subject_inputs() {
  local subject="$1" source path
  source="$(subject_root "$subject")"
  for path in \
    "$(subject_config "$subject")" \
    "$(checkpoint_path "$subject")" \
    "$(a5_bank_path "$subject")" \
    "$(a5_loso_path "$subject")" \
    "$source/banks/raw_trained/part_label_bank.npz" \
    "$source/banks/multiview_voting/part_label_bank.npz" \
    "$source/banks/voting_evidence_target_support/part_label_bank.npz" \
    "$(protocol_path "$subject")" \
    "$A5_METHOD_FREEZE" \
    "$A7_CONTRACT"; do
    [[ -e "$path" ]] || { echo "missing input: $path" >&2; return 2; }
  done
}

plan_validation_dry_run() {
  local subject evidence_output candidate_output validation_output
  local evidence_index candidates_index
  evidence_index="$(stage_index evidence)"
  candidates_index="$(stage_index candidates)"
  printf 'dry_run=true target_stage=%s gpu_job_order=377,386,387,393,394\n' "$TARGET_STAGE"
  printf 'freeze_a5=%s freeze_a7=%s\n' "$A5_METHOD_FREEZE" "$A7_CONTRACT"
  for subject in "${SUBJECTS[@]}"; do
    evidence_output="$OUTPUT_ROOT/evidence/$subject/evidence.npz"
    candidate_output="$OUTPUT_ROOT/candidates/$subject/candidate_index.json"
    validation_output="$OUTPUT_ROOT/validation/$subject/<candidate_id>"
    if (( TARGET_INDEX >= evidence_index )); then
      printf 'PLAN subject=%s stage=evidence samples=456 cameras=%s frames=%s:%s:%s input=%s,%s,%s,%s,%s output=%s command=CUDA_VISIBLE_DEVICES=%s %s tools/build_temporal_reliability_evidence.py\n' \
        "$subject" "$EVIDENCE_CAMERAS" "$FRAME_START" "$FRAME_END" "$FRAME_STEP" \
        "$(subject_config "$subject")" "$(checkpoint_path "$subject")" "$(a5_bank_path "$subject")" \
        "$A5_METHOD_FREEZE" "$A7_CONTRACT" "$evidence_output" "$GPU" "$PYTHON_BIN"
    fi
    if (( TARGET_INDEX >= candidates_index )); then
      printf 'PLAN subject=%s stage=candidates samples=0 input=%s,%s,%s,%s output=%s command=%s tools/calibrate_temporal_reliable_a7_weights.py\n' \
        "$subject" "$evidence_output" "$(a5_bank_path "$subject")" "$A5_METHOD_FREEZE" \
        "$A7_CONTRACT" "$candidate_output" "$PYTHON_BIN"
    fi
    if (( TARGET_INDEX >= VALIDATION_INDEX )); then
      printf 'PLAN subject=%s stage=validation samples=912 cameras=c17,c18,c19,c20 frames=%s:%s:%s input=%s,%s,%s,%s,%s,%s output=%s command=CUDA_VISIBLE_DEVICES=%s %s tools/render_semantic_temporal_stability.py\n' \
        "$subject" "$FRAME_START" "$FRAME_END" "$FRAME_STEP" "$candidate_output" \
        "$(subject_config "$subject")" "$(checkpoint_path "$subject")" "$(a5_bank_path "$subject")" \
        "$(a5_loso_path "$subject")" "$A7_CONTRACT" "$validation_output" "$GPU" "$PYTHON_BIN"
    fi
  done
}

run_canary_stage() {
  local subject=377 output candidates candidate_id candidate_bank temporal_output
  output="$OUTPUT_ROOT/canary_377/evidence.npz"
  run_job canary/377/evidence 8 "$output" env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    tools/build_temporal_reliability_evidence.py \
    --config "$(subject_config "$subject")" --checkpoint "$(checkpoint_path "$subject")" \
    --a5-bank "$(a5_bank_path "$subject")" --method-freeze "$A5_METHOD_FREEZE" \
    --a7-contract "$A7_CONTRACT" --cameras c01,c05 --frame-start 0 --frame-end 20 \
    --frame-stride 5 --parts hair,face,upper,lower,shoes,skin --allow-canary-protocol \
    --output "$output"
  candidates="$OUTPUT_ROOT/canary_377/candidates"
  run_job canary/377/candidates 0 "$candidates/candidate_index.json" "$PYTHON_BIN" \
    tools/calibrate_temporal_reliable_a7_weights.py --a5-bank "$(a5_bank_path "$subject")" \
    --evidence "$output" --method-freeze "$A5_METHOD_FREEZE" --a7-contract "$A7_CONTRACT" \
    --allow-canary-evidence --output-dir "$candidates"
  candidate_id="$($PYTHON_BIN - "$candidates/candidate_index.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
shortlist = list(data.get("validation_shortlist", []))
print(shortlist[0] if shortlist else data["candidates"][0]["candidate_id"])
PY
)"
  candidate_bank="$candidates/$candidate_id/part_label_bank.npz"
  temporal_output="$OUTPUT_ROOT/canary_377/validation_c17_$candidate_id"
  run_job "canary/377/validation_c17_$candidate_id" 8 "$temporal_output" env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" tools/render_semantic_temporal_stability.py --subject "$subject" \
    --voting-bank "$(subject_root "$subject")/banks/multiview_voting/part_label_bank.npz" \
    --a5-bank "$(a5_bank_path "$subject")" --a7-bank "$candidate_bank" \
    --a7-contract "$A7_CONTRACT" --loso-config "$(a5_loso_path "$subject")" \
    --method-freeze "$A5_METHOD_FREEZE" --checkpoint "$(checkpoint_path "$subject")" \
    --config "$(subject_config "$subject")" --output-dir "$temporal_output" \
    --camera 17 --frame-start 0 --frame-end 20 --frame-step 5 --methods a5 a7 --no-videos
}

run_evidence_stage() {
  local subject output
  for subject in "${SUBJECTS[@]}"; do
    validate_subject_inputs "$subject"
    output="$OUTPUT_ROOT/evidence/$subject/evidence.npz"
    run_job "evidence/$subject" 456 "$output" env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
      tools/build_temporal_reliability_evidence.py --config "$(subject_config "$subject")" \
      --checkpoint "$(checkpoint_path "$subject")" --a5-bank "$(a5_bank_path "$subject")" \
      --method-freeze "$A5_METHOD_FREEZE" --a7-contract "$A7_CONTRACT" \
      --cameras "$EVIDENCE_CAMERAS" --frame-start "$FRAME_START" --frame-end "$FRAME_END" \
      --frame-stride "$FRAME_STEP" --parts hair,face,upper,lower,shoes,skin --resume \
      --output "$output"
  done
}

run_candidates_stage() {
  local subject output
  for subject in "${SUBJECTS[@]}"; do
    output="$OUTPUT_ROOT/candidates/$subject"
    run_job "candidates/$subject" 0 "$output/candidate_index.json" "$PYTHON_BIN" \
      tools/calibrate_temporal_reliable_a7_weights.py --a5-bank "$(a5_bank_path "$subject")" \
      --evidence "$OUTPUT_ROOT/evidence/$subject/evidence.npz" \
      --method-freeze "$A5_METHOD_FREEZE" --a7-contract "$A7_CONTRACT" \
      --max-validation-candidates 4 --output-dir "$output"
  done
}

validation_candidate_ids() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT/candidates/$1/candidate_index.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
values = list(data.get("validation_shortlist", []))
if not values:
    raise SystemExit("no valid A7 validation candidates")
for value in values:
    print(value if isinstance(value, str) else value["candidate_id"])
PY
}

write_donor_report() {
  local subject="$1" candidate_id="$2" candidate_root="$3" spatial_root="$4" output="$5"
  "$PYTHON_BIN" - "$subject" "$candidate_id" "$candidate_root" "$spatial_root" "$output" <<'PY'
import csv, json, sys
from pathlib import Path

subject, candidate_id = sys.argv[1:3]
candidate_root, spatial_root, output = map(Path, sys.argv[3:])
candidate = json.loads((candidate_root / "candidate_summary.json").read_text(encoding="utf-8"))
temporal = []
for path in sorted((candidate_root.parent.parent.parent / "validation" / subject / candidate_id / "temporal").glob("c*/summary.json")):
    temporal.append(json.loads(path.read_text(encoding="utf-8")))
if not temporal:
    raise SystemExit("missing temporal summaries")
with (spatial_root / "baseline_summary.csv").open(newline="", encoding="utf-8") as handle:
    baseline = {row["baseline"].lower(): row for row in csv.DictReader(handle)}
with (spatial_root / "spatial_guard_metrics.csv").open(newline="", encoding="utf-8") as handle:
    guards = list(csv.DictReader(handle))

def temporal_mean(method, field):
    values = []
    for summary in temporal:
        values.extend(float(metrics[field]) for metrics in summary["temporal_metrics"][method].values())
    return sum(values) / len(values)

def method_metrics(method):
    base = baseline[method]
    rows = [row for row in guards if row["baseline"].lower() == method]
    mean = lambda field: sum(float(row[field]) for row in rows) / len(rows)
    return {
        "formal_eligible_parts": int(base["evaluated_part_count"]) - int(base["empty_target_part_count"]),
        "matched_target_coverage": mean("coverage_rate"),
        "pooled_outer_burden": mean("pooled_outer_burden"),
        "pooled_boundary_burden": mean("pooled_boundary_burden"),
        "macro_miou": float(base["macro_miou"]),
        "micro_iou": float(base["micro_iou"]),
        "fixed_outer_flicker": temporal_mean(method, "fixed_strength_outer_flicker"),
        "fixed_boundary_flicker": temporal_mean(method, "fixed_strength_boundary_flicker"),
    }

payload = {
    "schema_version": 1,
    "split": "validation",
    "donor_subject": subject,
    "candidate_id": candidate_id,
    "candidate_fingerprint": candidate["candidate_config_fingerprint"],
    "a7_bank_fingerprint": candidate["output_bank_fingerprint"],
    "a5_method_freeze_fingerprint": temporal[0]["method_freeze_fingerprint"],
    "a7_contract_fingerprint": temporal[0]["a7_contract_fingerprint"],
    "parameters": candidate["parameters"],
    "weight_l1_from_a5": candidate["proxy"]["weight_l1_from_a5"],
    "a5": method_metrics("a5"),
    "a7": method_metrics("a7"),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

run_validation_stage() {
  local subject candidate_id candidate_root candidate_bank output camera source spatial donor_report
  for subject in "${SUBJECTS[@]}"; do
    source="$(subject_root "$subject")"
    while read -r candidate_id; do
      [[ -n "$candidate_id" ]] || continue
      candidate_root="$OUTPUT_ROOT/candidates/$subject/$candidate_id"
      candidate_bank="$candidate_root/part_label_bank.npz"
      for camera in "${VALIDATION_CAMERAS[@]}"; do
        output="$OUTPUT_ROOT/validation/$subject/$candidate_id/temporal/c${camera}"
        run_job "validation/$subject/$candidate_id/temporal_c${camera}" 228 "$output" \
          env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/render_semantic_temporal_stability.py \
          --subject "$subject" --voting-bank "$source/banks/multiview_voting/part_label_bank.npz" \
          --a5-bank "$(a5_bank_path "$subject")" --a7-bank "$candidate_bank" \
          --a7-contract "$A7_CONTRACT" --loso-config "$(a5_loso_path "$subject")" \
          --method-freeze "$A5_METHOD_FREEZE" --checkpoint "$(checkpoint_path "$subject")" \
          --config "$(subject_config "$subject")" --output-dir "$output" --camera "$camera" \
          --frame-start "$FRAME_START" --frame-end "$FRAME_END" --frame-step "$FRAME_STEP" \
          --methods a5 a7 --no-videos
      done
      spatial="$OUTPUT_ROOT/validation/$subject/$candidate_id/spatial"
      run_job "validation/$subject/$candidate_id/spatial" 456 "$spatial" \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/evaluate_semantic_editing_paper_protocol.py \
        --protocol "$(protocol_path "$subject")" --protocol-split validation \
        --frozen-config "$(a5_loso_path "$subject")" \
        --raw-trained-bank "$source/banks/raw_trained/part_label_bank.npz" \
        --trained-bank "$source/banks/voting_evidence_target_support/part_label_bank.npz" \
        --voting-bank "$source/banks/multiview_voting/part_label_bank.npz" \
        --footprint-bank "$(a5_bank_path "$subject")" --a7-bank "$candidate_bank" \
        --a7-contract "$A7_CONTRACT" --method-freeze "$A5_METHOD_FREEZE" \
        --checkpoint "$(checkpoint_path "$subject")" \
        --asset-root "$source/assets/validation/test-view/semantic_editable_assets" \
        --explicit-binding-render-preset none --output-dir "$spatial" \
        --baselines A5 A7 --retention-reference-baseline A5
      donor_report="$OUTPUT_ROOT/validation/$subject/$candidate_id/donor_report.json"
      run_job "validation/$subject/$candidate_id/donor_report" 0 "$donor_report" \
        write_donor_report "$subject" "$candidate_id" "$candidate_root" "$spatial" "$donor_report"
    done < <(validation_candidate_ids "$subject")
  done
}

loso_report_paths() {
  local held_out="$1"
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$held_out" "${SUBJECTS[@]}" <<'PY'
import sys
from pathlib import Path

root, held_out = Path(sys.argv[1]), sys.argv[2]
donors = [value for value in sys.argv[3:] if value != held_out]
by_donor = {}
for donor in donors:
    by_donor[donor] = {path.parent.name: path for path in (root / "validation" / donor).glob("*/donor_report.json")}
common = set.intersection(*(set(values) for values in by_donor.values()))
for candidate_id in sorted(common):
    for donor in donors:
        print(by_donor[donor][candidate_id])
PY
}

build_loso_manifest() {
  local manifest="$OUTPUT_ROOT/aggregate/loso_freeze_manifest.json"
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$manifest" "${SUBJECTS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path

root, output = Path(sys.argv[1]), Path(sys.argv[2])
entries = []
for subject in sys.argv[3:]:
    path = root / "loso" / subject / "selected_config.json"
    data = path.read_bytes()
    entries.append({"held_out_subject": subject, "selected_config": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()})
payload = {"schema_version": 1, "entries": entries}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
  chmod a-w "$manifest"
}

run_loso_freeze_stage() {
  local held_out output manifest args=() path
  for held_out in "${SUBJECTS[@]}"; do
    args=()
    while read -r path; do
      [[ -n "$path" ]] && args+=(--donor-report "$path")
    done < <(loso_report_paths "$held_out")
    output="$OUTPUT_ROOT/loso/$held_out"
    run_job "loso/$held_out" 0 "$output/selected_config.json" "$PYTHON_BIN" \
      tools/select_loso_a7_temporal_config.py --held-out-subject "$held_out" \
      "${args[@]}" --output-dir "$output"
  done
  manifest="$OUTPUT_ROOT/aggregate/loso_freeze_manifest.json"
  run_job loso/manifest 0 "$manifest" build_loso_manifest
}

require_all_selected_configs() {
  local subject path
  for subject in "${SUBJECTS[@]}"; do
    path="$OUTPUT_ROOT/loso/$subject/selected_config.json"
    [[ -s "$path" ]] || { echo "missing selected config: $path" >&2; return 2; }
  done
}

verify_loso_freeze_manifest() {
  local manifest="$OUTPUT_ROOT/aggregate/loso_freeze_manifest.json"
  [[ -s "$manifest" ]] || { echo "missing LOSO freeze manifest: $manifest" >&2; return 2; }
  if [[ -w "$manifest" && "$(stat -c '%A' "$manifest")" == *w* ]]; then
    echo "LOSO freeze manifest must be read-only: $manifest" >&2
    return 2
  fi
  "$PYTHON_BIN" - "$manifest" <<'PY'
import hashlib, json, sys
from pathlib import Path

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for row in manifest["entries"]:
    path = Path(row["selected_config"])
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    if current != row["sha256"]:
        raise SystemExit(f"selected config fingerprint mismatch: {path}")
PY
}

selected_candidate_id() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT/loso/$1/selected_config.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["candidate_id"])
PY
}

run_frozen_temporal_camera() {
  local stage="$1" subject="$2" camera="$3" candidate_id source output
  candidate_id="$(selected_candidate_id "$subject")"
  source="$(subject_root "$subject")"
  output="$OUTPUT_ROOT/$stage/$subject/c${camera}"
  run_job "$stage/$subject/c${camera}" 228 "$output" env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    tools/render_semantic_temporal_stability.py --subject "$subject" \
    --voting-bank "$source/banks/multiview_voting/part_label_bank.npz" \
    --a5-bank "$(a5_bank_path "$subject")" \
    --a7-bank "$OUTPUT_ROOT/candidates/$subject/$candidate_id/part_label_bank.npz" \
    --a7-contract "$A7_CONTRACT" --loso-config "$(a5_loso_path "$subject")" \
    --method-freeze "$A5_METHOD_FREEZE" --checkpoint "$(checkpoint_path "$subject")" \
    --config "$(subject_config "$subject")" --output-dir "$output" --camera "$camera" \
    --frame-start "$FRAME_START" --frame-end "$FRAME_END" --frame-step "$FRAME_STEP" \
    --methods a5 a7 --no-videos
}

run_retrospective_stage() {
  local subject
  require_all_selected_configs
  verify_loso_freeze_manifest
  for subject in "${SUBJECTS[@]}"; do
    run_frozen_temporal_camera retrospective-c21 "$subject" 21
  done
}

run_frozen_test_stage() {
  local subject camera
  require_all_selected_configs
  verify_loso_freeze_manifest
  for subject in "${SUBJECTS[@]}"; do
    for camera in 22 23; do
      run_frozen_temporal_camera frozen-c22-c23 "$subject" "$camera"
    done
  done
}

run_paper_tables_stage() {
  run_job paper-tables 0 "$OUTPUT_ROOT/aggregate/paper_tables" "$PYTHON_BIN" \
    tools/summarize_a7_temporal_reliability.py --output-root "$OUTPUT_ROOT"
}

if [[ "$DRY_RUN" == "1" ]]; then
  plan_validation_dry_run
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
for index in "${!STAGES[@]}"; do
  (( index <= TARGET_INDEX )) || break
  case "${STAGES[$index]}" in
    canary) run_canary_stage ;;
    evidence) run_evidence_stage ;;
    candidates) run_candidates_stage ;;
    validation) run_validation_stage ;;
    loso-freeze) run_loso_freeze_stage ;;
    retrospective-c21) run_retrospective_stage ;;
    frozen-c22-c23) run_frozen_test_stage ;;
    paper-tables) run_paper_tables_stage ;;
  esac
done
