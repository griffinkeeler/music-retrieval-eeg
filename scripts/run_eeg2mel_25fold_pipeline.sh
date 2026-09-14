#!/usr/bin/env bash

# Run the EEG2Mel experiment with UV inputs: prepare targets, create the five
# split families, train all 25 folds, and evaluate them after training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
CONFIG_PATH="$REPO_DIR/configs/eeg2mel_paper.yaml"
SPLIT_RUN_NAME="eeg2mel_uv"
OUTPUT_RUN_PREFIX="eeg2mel-uv-25split"
VENV_PATH=""
SKIP_MEL_TARGETS=false
SKIP_SPLITS=false
SKIP_TRAINING=false
SKIP_TESTING=false
SKIP_COMPLETED=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_eeg2mel_25fold_pipeline.sh [options]

Prepare data and run the complete 5-split x 5-fold EEG2Mel experiment locally.
All 25 folds are trained sequentially, then evaluated sequentially, and finally
averaged by split family. Outputs are written under:

  runs/final_results/<output-prefix>/
  runs/final_results/<output-prefix>/average_metrics/

Options:
  --config PATH          Config YAML (default: configs/eeg2mel_paper.yaml)
  --split-run-name NAME  Directory under runs/ containing splits
                         (default: eeg2mel_uv)
  --output-prefix NAME   Prefix for the 25 model run directories
                         (default: eeg2mel-uv-25split)
  --venv PATH            Python virtual environment directory, or its
                         bin/activate file
  --skip-mel-targets     Reuse existing mel targets and metadata mel_path
  --skip-splits          Reuse existing 25 split CSVs
  --skip-training        Test existing checkpoints without training
  --skip-testing         Train all folds without running evaluation
  --skip-completed       Reuse existing checkpoints and test_metrics.csv files
  --dry-run              Print all resolved commands without running them
  -h, --help             Show this help

Examples:
  scripts/run_eeg2mel_25fold_pipeline.sh
  scripts/run_eeg2mel_25fold_pipeline.sh --venv .venv
  scripts/run_eeg2mel_25fold_pipeline.sh --skip-mel-targets --skip-splits
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

while (($# > 0)); do
  case "$1" in
    --config)
      (($# >= 2)) || die "--config requires a path."
      CONFIG_PATH="$2"
      shift 2
      ;;
    --split-run-name)
      (($# >= 2)) || die "--split-run-name requires a name."
      SPLIT_RUN_NAME="$2"
      shift 2
      ;;
    --output-prefix)
      (($# >= 2)) || die "--output-prefix requires a name."
      OUTPUT_RUN_PREFIX="$2"
      shift 2
      ;;
    --venv)
      (($# >= 2)) || die "--venv requires a path."
      VENV_PATH="$2"
      shift 2
      ;;
    --skip-mel-targets)
      SKIP_MEL_TARGETS=true
      shift
      ;;
    --skip-splits)
      SKIP_SPLITS=true
      shift
      ;;
    --skip-training)
      SKIP_TRAINING=true
      shift
      ;;
    --skip-testing)
      SKIP_TESTING=true
      shift
      ;;
    --skip-completed)
      SKIP_COMPLETED=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1. Run with --help for usage."
      ;;
  esac
done

if [[ "$SKIP_TRAINING" == true && "$SKIP_TESTING" == true ]]; then
  die "--skip-training and --skip-testing cannot be used together."
fi
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_DIR/$CONFIG_PATH"
fi
[[ -f "$CONFIG_PATH" ]] || die "Config not found: $CONFIG_PATH"

VENV_ACTIVATE=""
if [[ -n "$VENV_PATH" ]]; then
  if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="$REPO_DIR/$VENV_PATH"
  fi
  if [[ -d "$VENV_PATH" ]]; then
    VENV_ACTIVATE="$VENV_PATH/bin/activate"
  else
    VENV_ACTIVATE="$VENV_PATH"
  fi
  [[ -f "$VENV_ACTIVATE" ]] || die "Virtual-environment activation file not found: $VENV_ACTIVATE"
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

PYTHON_BIN="$(command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || die "python is not available; activate the project environment or use --venv."

if [[ -z "$OUTPUT_RUN_PREFIX" || "$OUTPUT_RUN_PREFIX" == */* || \
      "$OUTPUT_RUN_PREFIX" == "." || "$OUTPUT_RUN_PREFIX" == ".." ]]; then
  die "Output prefix must be a non-blank directory name without slashes."
fi

OUTPUT_RUN_DIR="$REPO_DIR/runs/final_results/$OUTPUT_RUN_PREFIX"
MODEL_RUN_PREFIX="final_results/$OUTPUT_RUN_PREFIX/$OUTPUT_RUN_PREFIX"
RUN_CONFIG_DIR="$REPO_DIR/runs/run_configs/final_results/$OUTPUT_RUN_PREFIX"
SPLIT_DIR="$REPO_DIR/runs/$SPLIT_RUN_NAME/splits"

write_split_config() {
  local output_path="$1"

  "$PYTHON_BIN" - "$CONFIG_PATH" "$SPLIT_RUN_NAME" "$output_path" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

config_path, split_run_name, output_path = sys.argv[1:]
config = OmegaConf.load(config_path)
config.run_name = split_run_name
output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config, output_path)
PY
}

write_fold_config() {
  local split_path="$1"
  local run_name="$2"
  local output_path="$3"

  "$PYTHON_BIN" - "$CONFIG_PATH" "$split_path" "$run_name" "$output_path" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

config_path, split_path, run_name, output_path = sys.argv[1:]
config = OmegaConf.load(config_path)
config.run_name = run_name
config.training.filename = str(Path(split_path).resolve())
config.testing.filename = str(Path(split_path).resolve())
output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config, output_path)
PY
}

checkpoint_for_run() {
  local run_name="$1"
  local checkpoint_dir="$REPO_DIR/runs/checkpoints/$run_name"

  if [[ -f "$checkpoint_dir/best.pt" ]]; then
    printf '%s\n' "$checkpoint_dir/best.pt"
  elif [[ -f "$checkpoint_dir/eeg2mel.pt" ]]; then
    printf '%s\n' "$checkpoint_dir/eeg2mel.pt"
  else
    return 1
  fi
}

cd "$REPO_DIR"
mkdir -p "$RUN_CONFIG_DIR"

echo "EEG2Mel 25-fold pipeline"
echo "  Repository: $REPO_DIR"
echo "  Config: $CONFIG_PATH"
echo "  Python: $PYTHON_BIN"
echo "  Split run: $SPLIT_RUN_NAME"
echo "  Split directory: $SPLIT_DIR"
echo "  Output prefix: $OUTPUT_RUN_PREFIX"
echo "  Fold output directory: $OUTPUT_RUN_DIR"

if [[ "$SKIP_MEL_TARGETS" == false ]]; then
  echo "Generating mel targets..."
  mel_command=(
    "$PYTHON_BIN" -m src.data.create_mel_targets --config "$CONFIG_PATH"
  )
  print_command "${mel_command[@]}"
  if [[ "$DRY_RUN" == false ]]; then
    "${mel_command[@]}"
  fi
else
  echo "Reusing existing mel targets."
fi

if [[ "$SKIP_SPLITS" == false ]]; then
  echo "Generating all 25 split CSVs..."
  split_config="$RUN_CONFIG_DIR/split_generation.yaml"
  split_command=(
    "$PYTHON_BIN" -m src.data.create_splits --config "$split_config"
  )
  if [[ "$DRY_RUN" == true ]]; then
    echo "  would write split config: $split_config"
    print_command "${split_command[@]}"
  else
    write_split_config "$split_config"
    "${split_command[@]}"
  fi
else
  echo "Reusing existing split CSVs."
fi

SPLIT_PREFIXES=(
  chunk_out
  random_segment_out
  song_out
  subject_out
  subject_song_out
)

RUN_NAMES=()
SPLIT_PATHS=()
CONFIG_PATHS=()

for split_prefix in "${SPLIT_PREFIXES[@]}"; do
  for fold_index in 0 1 2 3 4; do
    split_path="$SPLIT_DIR/${split_prefix}_fold${fold_index}.csv"
    fold_name="${OUTPUT_RUN_PREFIX}-${split_prefix}-fold${fold_index}"
    run_name="${MODEL_RUN_PREFIX}-${split_prefix}-fold${fold_index}"
    run_config="$RUN_CONFIG_DIR/${fold_name}.yaml"

    if [[ "$DRY_RUN" == false && ! -f "$split_path" ]]; then
      die "Split file not found: $split_path"
    fi

    RUN_NAMES+=("$run_name")
    SPLIT_PATHS+=("$split_path")
    CONFIG_PATHS+=("$run_config")
  done
done

if [[ "$SKIP_TRAINING" == false ]]; then
  echo "Training phase (25 folds)"
  for task_index in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$task_index]}"
    split_path="${SPLIT_PATHS[$task_index]}"
    run_config="${CONFIG_PATHS[$task_index]}"
    train_command=(
      "$PYTHON_BIN" -m src.evaluation.train_eeg2mel_baseline
      --config "$run_config"
    )

    echo "[$((task_index + 1))/25] Train $run_name"
    if [[ "$DRY_RUN" == true ]]; then
      echo "  would write config: $run_config"
      echo "  split: $split_path"
      print_command "${train_command[@]}"
      continue
    fi

    if [[ "$SKIP_COMPLETED" == true ]] && checkpoint_for_run "$run_name" >/dev/null; then
      echo "  Reusing existing checkpoint."
      continue
    fi

    write_fold_config "$split_path" "$run_name" "$run_config"
    "${train_command[@]}"
  done
else
  echo "Skipping training; existing checkpoints will be used."
fi

if [[ "$SKIP_TESTING" == false ]]; then
  echo "Testing phase (25 folds)"
  for task_index in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$task_index]}"
    split_path="${SPLIT_PATHS[$task_index]}"
    run_config="${CONFIG_PATHS[$task_index]}"
    test_metrics="$REPO_DIR/runs/$run_name/test_metrics.csv"

    echo "[$((task_index + 1))/25] Test $run_name"
    if [[ "$DRY_RUN" == true ]]; then
      expected_checkpoint="$REPO_DIR/runs/checkpoints/$run_name/best.pt"
      test_command=(
        "$PYTHON_BIN" -m src.evaluation.test_eeg2mel_baseline
        --config "$run_config"
        --checkpoint-path "$expected_checkpoint"
      )
      echo "  expected checkpoint: $expected_checkpoint"
      print_command "${test_command[@]}"
      continue
    fi

    if [[ "$SKIP_COMPLETED" == true && -f "$test_metrics" ]]; then
      echo "  Reusing existing evaluation: $test_metrics"
      continue
    fi

    checkpoint_path="$(checkpoint_for_run "$run_name")" || {
      die "No best.pt or eeg2mel.pt checkpoint found for $run_name."
    }
    if [[ ! -f "$run_config" ]]; then
      write_fold_config "$split_path" "$run_name" "$run_config"
    fi
    test_command=(
      "$PYTHON_BIN" -m src.evaluation.test_eeg2mel_baseline
      --config "$run_config"
      --checkpoint-path "$checkpoint_path"
    )
    "${test_command[@]}"
  done

  average_command=(
    "$PYTHON_BIN" -m scripts.average_test_results
    "$OUTPUT_RUN_DIR"
    --output-dir "$OUTPUT_RUN_DIR/average_metrics"
    --output-prefix "$OUTPUT_RUN_PREFIX"
  )
  echo "Averaging evaluation metrics across folds"
  if [[ "$DRY_RUN" == true ]]; then
    print_command "${average_command[@]}"
  else
    "${average_command[@]}"
  fi
else
  echo "Skipping testing."
fi

echo "EEG2Mel 25-fold pipeline complete."
