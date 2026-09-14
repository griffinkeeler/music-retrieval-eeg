#!/bin/bash
#SBATCH --job-name=music-eeg-train-eeg2mel-all
#SBATCH --output=logs/train_eeg2mel_%A_%a.out
#SBATCH --error=logs/train_eeg2mel_%A_%a.err
#SBATCH --partition=sdis
#SBATCH --account=sdis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-24

# Train all five folds for all five split families in one array submission.
# By default, split CSVs are read from runs/eeg2mel_uv/splits. Override the
# source run or output prefix when submitting, for example:
#   sbatch --export=ALL,SPLIT_RUN_NAME=my-run,OUTPUT_RUN_PREFIX=my-eeg2mel \
#     slurms/train_eeg2mel_folds.slurm

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-0}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Virtual-environment activation file not found: $VENV_ACTIVATE" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/configs/eeg2mel_paper.yaml}"
export SPLIT_RUN_NAME="${SPLIT_RUN_NAME:-eeg2mel_uv}"
export OUTPUT_RUN_PREFIX="${OUTPUT_RUN_PREFIX:-eeg2mel-uv-25split}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Base config not found: $CONFIG_PATH" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -z "$OUTPUT_RUN_PREFIX" || "$OUTPUT_RUN_PREFIX" == */* || \
      "$OUTPUT_RUN_PREFIX" == "." || "$OUTPUT_RUN_PREFIX" == ".." ]]; then
  echo "OUTPUT_RUN_PREFIX must be a non-blank directory name without slashes." >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
if ((TASK_ID < 0 || TASK_ID > 24)); then
  echo "Array task ID must be between 0 and 24; got $TASK_ID." >&2
  exit 1
fi

SPLIT_PREFIXES=(
  chunk_out
  random_segment_out
  song_out
  subject_out
  subject_song_out
)

SPLIT_INDEX=$((TASK_ID / 5))
export FOLD_IDX=$((TASK_ID % 5))
export SPLIT_PREFIX="${SPLIT_PREFIXES[$SPLIT_INDEX]}"
export SPLIT_FILENAME="${SPLIT_PREFIX}_fold${FOLD_IDX}.csv"
export SPLIT_PATH="$REPO_DIR/runs/$SPLIT_RUN_NAME/splits/$SPLIT_FILENAME"
export FOLD_RUN_NAME="${OUTPUT_RUN_PREFIX}-${SPLIT_PREFIX}-fold${FOLD_IDX}"
export OUTPUT_RUN_NAME="${OUTPUT_RUN_PREFIX}/${FOLD_RUN_NAME}"
export TEMP_CONFIG_PATH="${TEMP_CONFIG_PATH:-$REPO_DIR/runs/run_configs/tmp_train_eeg2mel_config_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml}"

if [[ ! -f "$SPLIT_PATH" ]]; then
  echo "Split file not found: $SPLIT_PATH" >&2
  exit 1
fi
if ! head -n 1 "$SPLIT_PATH" | tr ',' '\n' | grep -qx 'mel_path'; then
  echo "Split file is missing the required mel_path column: $SPLIT_PATH" >&2
  echo "Generate mel targets and recreate the split files before training." >&2
  exit 1
fi

echo "Split source run: $SPLIT_RUN_NAME"
echo "Split file: $SPLIT_PATH"
echo "Output group: $OUTPUT_RUN_PREFIX"
echo "Output run: $OUTPUT_RUN_NAME"

cd "$REPO_DIR"

"$PYTHON_BIN" - <<'PY'
from omegaconf import OmegaConf
from pathlib import Path
import os

config_path = Path(os.environ["CONFIG_PATH"])
temp_config_path = Path(os.environ["TEMP_CONFIG_PATH"])
cfg = OmegaConf.load(config_path)

split_path = Path(os.environ["SPLIT_PATH"])

cfg.run_name = os.environ["OUTPUT_RUN_NAME"]
cfg.training.filename = str(split_path)
cfg.testing.filename = str(split_path)

temp_config_path.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, temp_config_path)

print(f"Wrote training config to {temp_config_path}")
print(f"Run name: {cfg.run_name}")
print(f"Split file: {split_path}")
PY

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m src.evaluation.train_eeg2mel_baseline --config "$TEMP_CONFIG_PATH"

echo "Job finished at $(date)"
