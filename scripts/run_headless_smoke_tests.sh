#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="${REPO_ROOT}/difftactile/tasks"
OUTPUT_ROOT="${DIFFTACTILE_OUTPUT_ROOT:-/data1/determined/users/thomas/Dataset/Difftactile/output}"
CONDA_ENV="${DIFFTACTILE_CONDA_ENV:-difftactile}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/videos" "${OUTPUT_ROOT}/runs"

select_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
      sort -t, -k2 -nr |
      head -n1 |
      cut -d',' -f1 |
      tr -d ' '
  else
    echo ""
  fi
}

run_task() {
  local task="$1"
  local log_name="$2"
  local gpu="${CUDA_VISIBLE_DEVICES:-$(select_gpu)}"
  echo "=== ${task} on GPU ${gpu:-cpu/default} ==="
  (
    cd "${TASK_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DIFFTACTILE_HEADLESS=1 \
    PYOPENGL_PLATFORM=egl \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
      conda run -n "${CONDA_ENV}" python "${task}.py" \
        --use_state \
        --use_tactile \
        --headless \
        --record_video \
        --smoke \
        --output_root "${OUTPUT_ROOT}"
  ) 2>&1 | tee "${OUTPUT_ROOT}/logs/${log_name}"
}

run_task box_open box_open_smoke.log

if [[ "${1:-}" == "--all" ]]; then
  run_task surface_follow surface_follow_smoke.log
  run_task object_repose object_repose_smoke.log
  run_task cable_straightening cable_straightening_smoke.log
fi

conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/scripts/validate_outputs.py" --output-root "${OUTPUT_ROOT}"
