#!/bin/bash
#SBATCH --job-name=music-eeg2mel-25fold
#SBATCH --output=eeg2mel_25fold_%A_%a.out
#SBATCH --error=eeg2mel_25fold_%A_%a.err
#SBATCH --partition=sdis
#SBATCH --account=sdis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-24

# Train and evaluate all five folds from all five split families on BlueHive.
# Each array task handles one split/fold from start to finish. Mel targets and
# split CSVs must be prepared once before submitting the array:
#
#   python -m src.data.create_mel_targets --config configs/eeg2mel_paper.yaml
#   python -m src.data.create_splits --config configs/eeg2mel_paper.yaml
#
# From the repository root on a BlueHive login node:
#   sbatch slurms/run_eeg2mel_25fold_pipeline.slurm
# Or, from inside slurms/:
#   sbatch run_eeg2mel_25fold_pipeline.slurm
#
# Common overrides:
#   sbatch \
#     --export=ALL,VENV_PATH=/scratch/<group>/<user>/venvs/music-eeg-reu,MERT_SPACE=true \
#     slurms/run_eeg2mel_25fold_pipeline.slurm
#
# The default evaluation is the EEG2Mel mel-space evaluation. Set
# MERT_SPACE=true to additionally run the reconstructed-audio MERT evaluation.
# Set SKIP_COMPLETED=true to reuse complete outputs. SKIP_TRAINING=true and
# SKIP_TESTING=true run only the requested phase.
#
# After the entire array succeeds, aggregate its per-fold test_metrics.csv files:
#
#   python -m scripts.average_test_results \
#     runs/final_results/eeg2mel-uv-25split \
#     --output-dir runs/final_results/eeg2mel-uv-25split/average_metrics \
#     --output-prefix eeg2mel-uv-25split

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -z "${REPO_DIR:-}" && "$(basename "$SUBMIT_DIR")" == "slurms" && -d "$SUBMIT_DIR/../configs" ]]; then
  REPO_DIR="$SUBMIT_DIR/.."
else
  REPO_DIR="${REPO_DIR:-$SUBMIT_DIR}"
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  exit 1
fi
REPO_DIR="$(cd "$REPO_DIR" && pwd -P)"

if command -v module >/dev/null 2>&1; then
  module purge
  module load git miniforge3/26.1.1
fi

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  ACTIVATE_PATH="$VENV_ACTIVATE"
elif [[ -n "${VENV_PATH:-}" ]]; then
  if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="$REPO_DIR/$VENV_PATH"
  fi
  if [[ -d "$VENV_PATH" ]]; then
    ACTIVATE_PATH="$VENV_PATH/bin/activate"
  else
    ACTIVATE_PATH="$VENV_PATH"
  fi
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
  ACTIVATE_PATH="$REPO_DIR/.venv/bin/activate"
elif [[ -f "/scratch/mcetin_lab/gkeeler/venvs/music-eeg-reu/bin/activate" ]]; then
  ACTIVATE_PATH="/scratch/mcetin_lab/gkeeler/venvs/music-eeg-reu/bin/activate"
else
  ACTIVATE_PATH=""
fi

if [[ -n "$ACTIVATE_PATH" ]]; then
  if [[ ! -f "$ACTIVATE_PATH" ]]; then
    echo "Virtual-environment activation file not found: $ACTIVATE_PATH" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$ACTIVATE_PATH"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found; set VENV_PATH, VENV_ACTIVATE, or PYTHON_BIN." >&2
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/configs/eeg2mel_paper.yaml}"
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_DIR/$CONFIG_PATH"
fi
SPLIT_RUN_NAME="${SPLIT_RUN_NAME:-eeg2mel_uv}"
OUTPUT_RUN_PREFIX="${OUTPUT_RUN_PREFIX:-eeg2mel-uv-25split}"
SKIP_COMPLETED="${SKIP_COMPLETED:-false}"
SKIP_TRAINING="${SKIP_TRAINING:-false}"
SKIP_TESTING="${SKIP_TESTING:-false}"
MERT_SPACE="${MERT_SPACE:-false}"
DRY_RUN="${DRY_RUN:-false}"

if [[ "$SKIP_TRAINING" == true && "$SKIP_TESTING" == true ]]; then
  echo "SKIP_TRAINING and SKIP_TESTING cannot both be true." >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
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
FOLD_INDEX=$((TASK_ID % 5))
SPLIT_PREFIX="${SPLIT_PREFIXES[$SPLIT_INDEX]}"
SPLIT_FILENAME="${SPLIT_PREFIX}_fold${FOLD_INDEX}.csv"
FOLD_RUN_NAME="${OUTPUT_RUN_PREFIX}-${SPLIT_PREFIX}-fold${FOLD_INDEX}"
RUN_NAME="final_results/${OUTPUT_RUN_PREFIX}/${FOLD_RUN_NAME}"

if [[ -n "${SPLIT_DIR:-}" ]]; then
  RESOLVED_SPLIT_DIR="$SPLIT_DIR"
  if [[ "$RESOLVED_SPLIT_DIR" != /* ]]; then
    RESOLVED_SPLIT_DIR="$REPO_DIR/$RESOLVED_SPLIT_DIR"
  fi
else
  RESOLVED_SPLIT_DIR="$REPO_DIR/runs/$SPLIT_RUN_NAME/splits"
fi

SPLIT_PATH="$RESOLVED_SPLIT_DIR/$SPLIT_FILENAME"
RUN_CONFIG_PATH="$REPO_DIR/runs/run_configs/final_results/$OUTPUT_RUN_PREFIX/$FOLD_RUN_NAME.yaml"
CHECKPOINT_DIR="$REPO_DIR/runs/checkpoints/$RUN_NAME"
TEST_METRICS_PATH="$REPO_DIR/runs/$RUN_NAME/test_metrics.csv"
RICH_METRICS_PATH="$REPO_DIR/runs/$RUN_NAME/benchmarks/eeg2mel_baseline_test_metrics.csv"
MERT_METRICS_PATH="$REPO_DIR/runs/$RUN_NAME/benchmarks/mert_space/song_search_summary.csv"

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local-dry-run}"
echo "Array task ID: $TASK_ID"
echo "Repository: $REPO_DIR"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "Split: $SPLIT_PATH"
echo "Run group: $OUTPUT_RUN_PREFIX"
echo "Run: $RUN_NAME"
echo "Optional MERT evaluation: $MERT_SPACE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
"$PYTHON_BIN" --version

if [[ "$DRY_RUN" == true ]]; then
  echo "Would write fold config: $RUN_CONFIG_PATH"
  echo "Would train: $PYTHON_BIN -m src.evaluation.train_eeg2mel_baseline --config $RUN_CONFIG_PATH"
  if [[ "$MERT_SPACE" == true ]]; then
    echo "Would test: $PYTHON_BIN -m src.evaluation.test_eeg2mel_baseline --config $RUN_CONFIG_PATH --checkpoint-path <best-or-final-checkpoint> --mert-space"
  else
    echo "Would test: $PYTHON_BIN -m src.evaluation.test_eeg2mel_baseline --config $RUN_CONFIG_PATH --checkpoint-path <best-or-final-checkpoint>"
  fi
  exit 0
fi

if [[ ! -f "$SPLIT_PATH" ]]; then
  echo "Split file not found: $SPLIT_PATH" >&2
  echo "Prepare mel targets and recreate the split files before submitting." >&2
  exit 1
fi

export SPLIT_PATH
"$PYTHON_BIN" - <<'PY'
import csv
import os

split_path = os.environ["SPLIT_PATH"]
with open(split_path, newline="", encoding="utf-8") as csv_file:
    fieldnames = csv.DictReader(csv_file).fieldnames or []
if "mel_path" not in fieldnames:
    raise SystemExit(
        f"Split file is missing the required mel_path column: {split_path}\n"
        "Prepare mel targets and recreate the split files before submitting."
    )
PY

cd "$REPO_DIR"
mkdir -p "$(dirname "$RUN_CONFIG_PATH")"

export CONFIG_PATH RUN_CONFIG_PATH RUN_NAME
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

from omegaconf import OmegaConf

config = OmegaConf.load(os.environ["CONFIG_PATH"])
split_path = Path(os.environ["SPLIT_PATH"]).resolve()
config.run_name = os.environ["RUN_NAME"]
config.training.filename = str(split_path)
config.testing.filename = str(split_path)

output_path = Path(os.environ["RUN_CONFIG_PATH"])
output_path.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config, output_path)

print(f"Wrote fold config to {output_path}")
print(f"Run name: {config.run_name}")
print(f"Split file: {split_path}")
PY

checkpoint_for_run() {
  if [[ -f "$CHECKPOINT_DIR/best.pt" ]]; then
    printf '%s\n' "$CHECKPOINT_DIR/best.pt"
  elif [[ -f "$CHECKPOINT_DIR/eeg2mel.pt" ]]; then
    printf '%s\n' "$CHECKPOINT_DIR/eeg2mel.pt"
  else
    return 1
  fi
}

if [[ "$SKIP_TRAINING" == true ]]; then
  echo "Skipping training; an existing checkpoint will be used."
elif [[ "$SKIP_COMPLETED" == true ]] && checkpoint_for_run >/dev/null; then
  echo "Reusing existing checkpoint in $CHECKPOINT_DIR."
else
  "$PYTHON_BIN" -m src.evaluation.train_eeg2mel_baseline \
    --config "$RUN_CONFIG_PATH"
fi

if [[ "$SKIP_TESTING" == true ]]; then
  echo "Skipping evaluation."
  echo "Job finished at $(date)"
  exit 0
fi

CHECKPOINT_PATH="$(checkpoint_for_run)" || {
  echo "No best.pt or eeg2mel.pt checkpoint found in $CHECKPOINT_DIR." >&2
  exit 1
}

if [[ "$SKIP_COMPLETED" == true && -f "$TEST_METRICS_PATH" && \
      -f "$RICH_METRICS_PATH" && \
      ( "$MERT_SPACE" == false || -f "$MERT_METRICS_PATH" ) ]]; then
  echo "Reusing existing evaluation: $TEST_METRICS_PATH"
else
  TEST_COMMAND=(
    "$PYTHON_BIN" -m src.evaluation.test_eeg2mel_baseline
    --config "$RUN_CONFIG_PATH"
    --checkpoint-path "$CHECKPOINT_PATH"
  )
  if [[ "$MERT_SPACE" == true ]]; then
    TEST_COMMAND+=(--mert-space)
  fi
  "${TEST_COMMAND[@]}"
fi

echo "Job finished at $(date)"
