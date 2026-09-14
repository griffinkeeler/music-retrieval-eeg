#!/usr/bin/env bash

# Train and evaluate the base contrastive model on five folds from each of the
# five paper split families with UV inputs. This is the sequential counterpart
# to slurms/run_contrastive_25fold_pipeline.slurm.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
CONFIG_PATH="$REPO_DIR/configs/infonce_paper.yaml"
VENV_PATH=""
SPLIT_RUN_NAME=""
OUTPUT_RUN_PREFIX=""
SKIP_TRAINING=false
SKIP_TESTING=false
SKIP_COMPLETED=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_contrastive_25fold_pipeline.sh [options]

Train and test the base contrastive model on all 25 split CSVs from one split
run. Fold outputs and their five aggregate metric files are written as:

  runs/final_results/<output-prefix>/<output-prefix>-<split-family>-fold<0-4>/
  runs/final_results/<output-prefix>/average_metrics/

Options:
  --split-run-name NAME  Read splits from runs/NAME/splits. If omitted, use
                         split_directory from the config.
  --output-prefix NAME   Override the selected config's run_name for all 25
                         model runs
  --run-name NAME        Alias for --output-prefix
  --config PATH          Experiment config (default: configs/infonce_paper.yaml)
  --venv PATH            Python virtual environment directory, or its
                         bin/activate file
  --skip-training        Test existing checkpoints without training
  --skip-testing         Train all folds without running evaluation
  --skip-completed       Reuse existing checkpoints and test_metrics.csv files
  --dry-run              Show all resolved train/test commands without running
  -h, --help             Show this help

Examples:
  scripts/run_contrastive_25fold_pipeline.sh \
    --venv .venv

  scripts/run_contrastive_25fold_pipeline.sh \
    --split-run-name another-split-run \
    --output-prefix base-seed1 \
    --skip-completed
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
    --split-run-name)
      (($# >= 2)) || die "--split-run-name requires a name."
      SPLIT_RUN_NAME="$2"
      shift 2
      ;;
    --output-prefix|--run-name)
      (($# >= 2)) || die "$1 requires a name."
      OUTPUT_RUN_PREFIX="$2"
      shift 2
      ;;
    --config)
      (($# >= 2)) || die "--config requires a path."
      CONFIG_PATH="$2"
      shift 2
      ;;
    --venv)
      (($# >= 2)) || die "--venv requires a path."
      VENV_PATH="$2"
      shift 2
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

if [[ -n "$VENV_PATH" ]]; then
  if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="$REPO_DIR/$VENV_PATH"
  fi
  if [[ -d "$VENV_PATH" ]]; then
    VENV_PATH="$VENV_PATH/bin/activate"
  fi
  [[ -f "$VENV_PATH" ]] || die "Virtual-environment activation file not found: $VENV_PATH"
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi

PYTHON_BIN="$(command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || die "python is not available; activate the project environment or use --venv."

cd "$REPO_DIR"

if [[ -z "$OUTPUT_RUN_PREFIX" ]]; then
  OUTPUT_RUN_PREFIX="$(
    "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys

from src.config import load_config

config = load_config(Path(sys.argv[1]))
run_name = str(config.get("run_name", "")).strip()
if not run_name:
    raise SystemExit(
        f"Config {sys.argv[1]} has no run_name; pass --output-prefix explicitly."
    )
print(run_name)
PY
  )"
fi

if [[ -z "$OUTPUT_RUN_PREFIX" || "$OUTPUT_RUN_PREFIX" == */* || \
      "$OUTPUT_RUN_PREFIX" == "." || "$OUTPUT_RUN_PREFIX" == ".." ]]; then
  die "Output prefix must be a non-blank directory name without slashes."
fi

OUTPUT_RUN_DIR="$REPO_DIR/runs/final_results/$OUTPUT_RUN_PREFIX"
MODEL_RUN_PREFIX="final_results/$OUTPUT_RUN_PREFIX/$OUTPUT_RUN_PREFIX"
RUN_CONFIG_DIR="$REPO_DIR/runs/run_configs/final_results/$OUTPUT_RUN_PREFIX"

if [[ -n "$SPLIT_RUN_NAME" ]]; then
  SPLIT_DIR="$REPO_DIR/runs/$SPLIT_RUN_NAME/splits"
else
  SPLIT_DIR="$(
    "$PYTHON_BIN" - "$CONFIG_PATH" "$REPO_DIR" <<'PY'
from pathlib import Path
import sys

from src.config import load_config, resolve_split_directory

config = load_config(Path(sys.argv[1]))
print(resolve_split_directory(Path(sys.argv[2]), config))
PY
  )"
fi

SPLIT_PREFIXES=(
  chunk_out
  random_segment_out
  song_out
  subject_out
  subject_song_out
)
RUN_LABELS=(
  chunkout
  randomsegmentout
  songout
  subjectout
  subjectsongout
)

RUN_NAMES=()
SPLIT_PATHS=()
CONFIG_PATHS=()

mkdir -p "$RUN_CONFIG_DIR"

for split_index in "${!SPLIT_PREFIXES[@]}"; do
  split_prefix="${SPLIT_PREFIXES[$split_index]}"
  run_label="${RUN_LABELS[$split_index]}"
  for fold_index in 0 1 2 3 4; do
    split_path="$SPLIT_DIR/${split_prefix}_fold${fold_index}.csv"
    fold_name="${OUTPUT_RUN_PREFIX}-${run_label}-fold${fold_index}"
    run_name="${MODEL_RUN_PREFIX}-${run_label}-fold${fold_index}"
    run_config="$RUN_CONFIG_DIR/${fold_name}.yaml"

    if [[ "$DRY_RUN" == false && ! -f "$split_path" ]]; then
      die "Split file not found: $split_path"
    fi

    RUN_NAMES+=("$run_name")
    SPLIT_PATHS+=("$split_path")
    CONFIG_PATHS+=("$run_config")
  done
done

write_fold_config() {
  local split_path="$1"
  local run_name="$2"
  local output_path="$3"

  "$PYTHON_BIN" - "$CONFIG_PATH" "$split_path" "$run_name" "$output_path" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

from src.config import load_config

config_path, split_path, run_name, output_path = sys.argv[1:]
config = load_config(Path(config_path))
config.run_name = run_name
config.training.filename = str(Path(split_path).resolve())
config.testing.filename = str(Path(split_path).resolve())
OmegaConf.save(config, Path(output_path))
PY
}

checkpoint_for_run() {
  local run_name="$1"
  local checkpoint_dir="$REPO_DIR/runs/checkpoints/$run_name"

  if [[ -f "$checkpoint_dir/best.pt" ]]; then
    printf '%s\n' "$checkpoint_dir/best.pt"
  elif [[ -f "$checkpoint_dir/eeg_encoder.pt" ]]; then
    printf '%s\n' "$checkpoint_dir/eeg_encoder.pt"
  else
    return 1
  fi
}

echo "Base contrastive 25-fold pipeline"
echo "  Repository: $REPO_DIR"
echo "  Config: $CONFIG_PATH"
echo "  Python: $PYTHON_BIN"
echo "  Split directory: $SPLIT_DIR"
echo "  Output prefix: $OUTPUT_RUN_PREFIX"
echo "  Fold output directory: $OUTPUT_RUN_DIR"

if [[ "$SKIP_TRAINING" == false ]]; then
  echo "Training phase (25 folds)"
  for task_index in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$task_index]}"
    split_path="${SPLIT_PATHS[$task_index]}"
    run_config="${CONFIG_PATHS[$task_index]}"
    train_command=(
      "$PYTHON_BIN" -m scripts.train --config "$run_config"
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
        "$PYTHON_BIN" -m scripts.test
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
      die "No best.pt or eeg_encoder.pt checkpoint found for $run_name."
    }
    if [[ ! -f "$run_config" ]]; then
      write_fold_config "$split_path" "$run_name" "$run_config"
    fi
    test_command=(
      "$PYTHON_BIN" -m scripts.test
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

echo "Contrastive 25-fold pipeline complete."
