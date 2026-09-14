#!/bin/bash
#SBATCH --job-name=music-eeg-test-eeg2mel-all
#SBATCH --output=logs/test_eeg2mel_%A_%a.out
#SBATCH --error=logs/test_eeg2mel_%A_%a.err
#SBATCH --partition=sdis
#SBATCH --account=sdis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-24

# Test all five folds for all five split families in one array submission.
# By default, this tests checkpoints produced by train_eeg2mel_folds.slurm:
#   runs/checkpoints/eeg2mel-uv-25split/
#     eeg2mel-uv-25split-<split-family>-fold<fold>/best.pt
# Override the split source or model run prefix when submitting, for example:
#   sbatch --export=ALL,SPLIT_RUN_NAME=my-run,MODEL_RUN_PREFIX=my-eeg2mel \
#     slurms/test_eeg2mel_folds.slurm

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-unset}"
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
export MODEL_RUN_PREFIX="${MODEL_RUN_PREFIX:-${OUTPUT_RUN_PREFIX:-eeg2mel-uv-25split}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -z "$MODEL_RUN_PREFIX" || "$MODEL_RUN_PREFIX" == */* || \
      "$MODEL_RUN_PREFIX" == "." || "$MODEL_RUN_PREFIX" == ".." ]]; then
  echo "MODEL_RUN_PREFIX must be a non-blank directory name without slashes." >&2
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
export FOLD_RUN_NAME="${MODEL_RUN_PREFIX}-${SPLIT_PREFIX}-fold${FOLD_IDX}"
export MODEL_RUN_NAME="${MODEL_RUN_PREFIX}/${FOLD_RUN_NAME}"
export CHECKPOINT_DIR="$REPO_DIR/runs/checkpoints/$MODEL_RUN_NAME"
export TEMP_CONFIG_PATH="${TEMP_CONFIG_PATH:-$REPO_DIR/runs/run_configs/tmp_test_eeg2mel_config_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Base config not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$SPLIT_PATH" ]]; then
  echo "Split file not found: $SPLIT_PATH" >&2
  exit 1
fi
if ! head -n 1 "$SPLIT_PATH" | tr ',' '\n' | grep -qx 'mel_path'; then
  echo "Split file is missing the required mel_path column: $SPLIT_PATH" >&2
  exit 1
fi

if [[ -f "$CHECKPOINT_DIR/best.pt" ]]; then
  export CHECKPOINT_PATH="$CHECKPOINT_DIR/best.pt"
elif [[ -f "$CHECKPOINT_DIR/eeg2mel.pt" ]]; then
  export CHECKPOINT_PATH="$CHECKPOINT_DIR/eeg2mel.pt"
else
  echo "No trained checkpoint found for $MODEL_RUN_NAME." >&2
  echo "Expected $CHECKPOINT_DIR/best.pt or $CHECKPOINT_DIR/eeg2mel.pt" >&2
  exit 1
fi

echo "Split source run: $SPLIT_RUN_NAME"
echo "Split file: $SPLIT_PATH"
echo "Model group: $MODEL_RUN_PREFIX"
echo "Model run: $MODEL_RUN_NAME"
echo "Checkpoint: $CHECKPOINT_PATH"

cd "$REPO_DIR"

"$PYTHON_BIN" - <<'PY'
from omegaconf import OmegaConf
from pathlib import Path
import os

config_path = Path(os.environ["CONFIG_PATH"])
temp_config_path = Path(os.environ["TEMP_CONFIG_PATH"])
split_path = Path(os.environ["SPLIT_PATH"])

cfg = OmegaConf.load(config_path)
cfg.run_name = os.environ["MODEL_RUN_NAME"]
cfg.training.filename = str(split_path)
cfg.testing.filename = str(split_path)

temp_config_path.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, temp_config_path)

print(f"Wrote testing config to {temp_config_path}")
print(f"Run name: {cfg.run_name}")
print(f"Split file: {split_path}")
PY

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m src.evaluation.test_eeg2mel_baseline \
  --config "$TEMP_CONFIG_PATH" \
  --checkpoint-path "$CHECKPOINT_PATH"

echo "Job finished at $(date)"
