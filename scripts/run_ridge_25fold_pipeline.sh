#!/usr/bin/env bash

# Train and evaluate ridge regression on five folds from each of the five
# paper split families. The Python sweep writes per-fold, combined, and
# five-fold average metrics.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
CONFIG_PATH="$REPO_DIR/configs/ridge_paper.yaml"
VENV_PATH=""
SPLIT_RUN_NAME=""
OUTPUT_RUN_PREFIX=""
OUTPUT_CSV=""
ALPHA=""
N_PERMS=""
N_WITHIN_SONG_SHUFFLES=""
N_SONG_SEARCH_PERMS=""
SONG_SEARCH_PERMUTATION_BATCH_SIZE=""
SONG_SEARCH_MARGINAL_TEMPERATURE=""
SKIP_COMPLETED=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_ridge_25fold_pipeline.sh [options]

Train and evaluate ridge regression on all 25 paper splits. By default, the
split directory, output run prefix, and evaluation settings come from
configs/ridge_paper.yaml. Outputs are written as:

  runs/final_results/<output-prefix>/<output-prefix>-<split-family>-fold<0-4>/
  runs/final_results/<output-prefix>/average_metrics/

Options:
  --config PATH          Config YAML (default: configs/ridge_paper.yaml)
  --venv PATH            Python virtual environment directory, or its
                         bin/activate file
  --split-run-name NAME  Override the config's split_directory with
                         runs/NAME/splits
  --output-prefix NAME   Override the config's run_name for fold run names
  --output-csv PATH      Override the combined metrics CSV path
  --alpha VALUE          Override ridge alpha
  --n-perms N            Override candidate-pool permutations
  --n-within-song-shuffles N
                         Override within-song shuffle count
  --n-song-search-perms N
                         Override song-identification permutations
  --song-search-permutation-batch-size N
                         Override song permutation batch size
  --song-search-marginal-temperature VALUE
                         Override marginalized song-identification temperature
  --skip-completed       Reuse fold results that already include song search
  --dry-run              Print the resolved command without running it
  -h, --help             Show this help

Examples:
  scripts/run_ridge_25fold_pipeline.sh
  scripts/run_ridge_25fold_pipeline.sh --venv .venv --skip-completed
  scripts/run_ridge_25fold_pipeline.sh --output-prefix ridge-alpha10 \
    --alpha 10
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
    --venv)
      (($# >= 2)) || die "--venv requires a path."
      VENV_PATH="$2"
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
    --output-csv)
      (($# >= 2)) || die "--output-csv requires a path."
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --alpha)
      (($# >= 2)) || die "--alpha requires a value."
      ALPHA="$2"
      shift 2
      ;;
    --n-perms)
      (($# >= 2)) || die "--n-perms requires a value."
      N_PERMS="$2"
      shift 2
      ;;
    --n-within-song-shuffles)
      (($# >= 2)) || die "--n-within-song-shuffles requires a value."
      N_WITHIN_SONG_SHUFFLES="$2"
      shift 2
      ;;
    --n-song-search-perms)
      (($# >= 2)) || die "--n-song-search-perms requires a value."
      N_SONG_SEARCH_PERMS="$2"
      shift 2
      ;;
    --song-search-permutation-batch-size)
      (($# >= 2)) || die "--song-search-permutation-batch-size requires a value."
      SONG_SEARCH_PERMUTATION_BATCH_SIZE="$2"
      shift 2
      ;;
    --song-search-marginal-temperature)
      (($# >= 2)) || die "--song-search-marginal-temperature requires a value."
      SONG_SEARCH_MARGINAL_TEMPERATURE="$2"
      shift 2
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

COMMAND=(
  "$PYTHON_BIN"
  -m src.evaluation.train_all_ridge_splits
  --config "$CONFIG_PATH"
)

if [[ -n "$SPLIT_RUN_NAME" ]]; then
  COMMAND+=(--split-run-name "$SPLIT_RUN_NAME")
fi
if [[ -n "$OUTPUT_RUN_PREFIX" ]]; then
  COMMAND+=(--output-run-prefix "$OUTPUT_RUN_PREFIX")
fi
if [[ -n "$OUTPUT_CSV" ]]; then
  COMMAND+=(--output-csv "$OUTPUT_CSV")
fi
if [[ -n "$ALPHA" ]]; then
  COMMAND+=(--alpha "$ALPHA")
fi
if [[ -n "$N_PERMS" ]]; then
  COMMAND+=(--n-perms "$N_PERMS")
fi
if [[ -n "$N_WITHIN_SONG_SHUFFLES" ]]; then
  COMMAND+=(--n-within-song-shuffles "$N_WITHIN_SONG_SHUFFLES")
fi
if [[ -n "$N_SONG_SEARCH_PERMS" ]]; then
  COMMAND+=(--n-song-search-perms "$N_SONG_SEARCH_PERMS")
fi
if [[ -n "$SONG_SEARCH_PERMUTATION_BATCH_SIZE" ]]; then
  COMMAND+=(
    --song-search-permutation-batch-size
    "$SONG_SEARCH_PERMUTATION_BATCH_SIZE"
  )
fi
if [[ -n "$SONG_SEARCH_MARGINAL_TEMPERATURE" ]]; then
  COMMAND+=(
    --song-search-marginal-temperature
    "$SONG_SEARCH_MARGINAL_TEMPERATURE"
  )
fi
if [[ "$SKIP_COMPLETED" == true ]]; then
  COMMAND+=(--skip-completed)
fi

cd "$REPO_DIR"
echo "Ridge 25-fold pipeline"
echo "  Repository: $REPO_DIR"
echo "  Config: $CONFIG_PATH"
echo "  Python: $PYTHON_BIN"
echo "  Command:"
print_command "${COMMAND[@]}"

if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

"${COMMAND[@]}"
