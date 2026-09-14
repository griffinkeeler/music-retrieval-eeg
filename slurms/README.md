# Running experiments with Slurm

These jobs run the paper experiments as 25-task arrays: one task for each of
the five split families and five folds. They are configured for the BlueHive
`sdis` partition and account and request one GPU per task. Submit them from the
repository root unless `REPO_DIR` is set explicitly.

The root `README.md` documents the experiment and local pipeline. This file
only covers cluster submission.

## Available jobs

| Script | Purpose | Default time |
| --- | --- | ---: |
| `run_contrastive_25fold_pipeline.slurm` | Train and evaluate one alignment configuration across all 25 folds | 8 hours per task |
| `train_eeg2mel_folds.slurm` | Train EEG2Mel across all 25 folds | 8 hours per task |
| `test_eeg2mel_folds.slurm` | Evaluate the 25 EEG2Mel checkpoints | 24 hours per task |

Array task IDs have the same meaning in all three jobs:

| Task IDs | Split family |
| --- | --- |
| 0–4 | Chunk out, folds 0–4 |
| 5–9 | Random segment out, folds 0–4 |
| 10–14 | Song out, folds 0–4 |
| 15–19 | Subject out, folds 0–4 |
| 20–24 | Subject and song out, folds 0–4 |

## Prerequisites

1. Clone the repository on a filesystem visible to BlueHive compute nodes.
2. Create the Python environment described in the root `README.md`.
3. Place the processed EEG and song audio in the documented `data/` layout.
4. Set a cluster-side activation path. Do not use a path from a local Mac or
   Windows machine.
5. Create the log directory before submitting EEG2Mel jobs; Slurm opens the
   output files before the job script starts.

For example:

```bash
cd /path/on/bluehive/music-eeg-reu
mkdir -p logs
export MUSIC_EEG_VENV=/path/on/bluehive/venvs/music-eeg-reu
```

If the account or partition differs from `sdis`, override the defaults when
submitting:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION [other options] JOB_FILE
```

The EEG2Mel scripts do not load environment modules themselves. Load any
modules required by the virtual environment before submission, or provide an
absolute `PYTHON_BIN` through `--export`.

## Prepare data once

Run preparation in an interactive compute allocation, not on a shared login
node. Generate the common five-second windows and alignment-model splits first:

```bash
source "$MUSIC_EEG_VENV/bin/activate"

python -m scripts.create_window_metadata \
  --window-length 5 \
  --output data/metadata/five_sec_windows.csv

python -m scripts.create_splits \
  --config configs/all_splits.yaml
```

Then add the EEG2Mel targets and generate its matching splits. This order keeps
the alignment splits independent of the extra `mel_path` column while ensuring
that the EEG2Mel splits contain it.

```bash
python -m src.data.create_mel_targets \
  --config configs/eeg2mel_paper.yaml

python -m src.data.create_splits \
  --config configs/eeg2mel_paper.yaml
```

Preparation should only be run once for a complete sweep. Do not start array
jobs while metadata, targets, or split CSVs are still being written.

## Run one alignment configuration

The alignment job trains and evaluates each fold within the same array task.
By default, it uses `configs/infonce_paper.yaml`. The config's `run_name`
becomes the parent directory, and all 25 fold directories are written beneath
`runs/<run_name>/`. Pass a different config through `--export`:

```bash
sbatch \
  --export=ALL,VENV_ACTIVATE="$MUSIC_EEG_VENV/bin/activate",CONFIG_PATH=configs/infonce_paper.yaml \
  slurms/run_contrastive_25fold_pipeline.slurm
```

The most useful overrides are:

- `CONFIG_PATH`: experiment configuration, relative to the repository or absolute.
- `OUTPUT_RUN_PREFIX`: parent directory and fold-name prefix; defaults to the config's `run_name` and must not contain slashes.
- `SPLIT_RUN_NAME`: read splits from `runs/<name>/splits`.
- `SPLIT_DIR`: explicit split directory; takes priority over `SPLIT_RUN_NAME`.
- `VENV_ACTIVATE`: absolute path to the virtual environment activation file.
- `PYTHON_BIN`: explicit Python executable.
- `SKIP_COMPLETED=true`: reuse existing checkpoints and `test_metrics.csv` files.
- `SKIP_TRAINING=true`: evaluate existing checkpoints only.
- `SKIP_TESTING=true`: train without evaluation.
- `DRY_RUN=true`: resolve one or more tasks without training or evaluation.

To limit the number of simultaneous GPUs, override the array specification:

```bash
sbatch \
  --array=0-24%5 \
  --export=ALL,VENV_ACTIVATE="$MUSIC_EEG_VENV/bin/activate" \
  slurms/run_contrastive_25fold_pipeline.slurm
```

To retry only task 13, for example:

```bash
sbatch \
  --array=13 \
  --export=ALL,VENV_ACTIVATE="$MUSIC_EEG_VENV/bin/activate",SKIP_COMPLETED=true \
  slurms/run_contrastive_25fold_pipeline.slurm
```

## Run the three Table I alignment conditions

Each config already has the desired Table I `run_name`, so a plain output
prefix places its 25 folds under the matching directory in `runs/`:

```bash
SUBJECT_ON_PREFIX=table1-infonce-subject-on
SUBJECT_OFF_PREFIX=table1-infonce-subject-off
COSINE_PREFIX=table1-cosine-regression

SUBJECT_ON_JOB=$(sbatch --parsable \
  --export=ALL,VENV_ACTIVATE="$MUSIC_EEG_VENV/bin/activate",CONFIG_PATH=configs/table1_infonce_subject_on.yaml,OUTPUT_RUN_PREFIX="$SUBJECT_ON_PREFIX" \
  slurms/run_contrastive_25fold_pipeline.slurm)

SUBJECT_OFF_JOB=$(sbatch --parsable \
  --export=ALL,VENV_ACTIVATE="$MUSIC_EEG_VENV/bin/activate",CONFIG_PATH=configs/table1_infonce_subject_off.yaml,OUTPUT_RUN_PREFIX="$SUBJECT_OFF_PREFIX" \
  slurms/run_contrastive_25fold_pipeline.slurm)

COSINE_JOB=$(sbatch --parsable \
  --export=ALL,VENV_ACTIVATE="$MUSIC_EEG_VENV/bin/activate",CONFIG_PATH=configs/table1_cosine_regression.yaml,OUTPUT_RUN_PREFIX="$COSINE_PREFIX" \
  slurms/run_contrastive_25fold_pipeline.slurm)

SUBJECT_ON_JOB=${SUBJECT_ON_JOB%%;*}
SUBJECT_OFF_JOB=${SUBJECT_OFF_JOB%%;*}
COSINE_JOB=${COSINE_JOB%%;*}

printf 'Subject on: %s\nSubject off: %s\nCosine regression: %s\n' \
  "$SUBJECT_ON_JOB" "$SUBJECT_OFF_JOB" "$COSINE_JOB"
```

Cosine Regression is a separate Table I condition; it does not use the removed
Ridge baseline.

## Run EEG2Mel

EEG2Mel training and evaluation are separate arrays. Submit evaluation with an
`afterok` dependency so it starts only after all 25 training tasks succeed.

For Table I, use the same plain prefix for both jobs. The 25 fold directories
and their checkpoints are grouped beneath directories with that name:

```bash
EEG2MEL_PREFIX=table1-eeg2mel

EEG2MEL_TRAIN_SUBMISSION=$(sbatch --parsable \
  --export=ALL,VENV_ACTIVATE="$MUSIC_EEG_VENV/bin/activate",OUTPUT_RUN_PREFIX="$EEG2MEL_PREFIX" \
  slurms/train_eeg2mel_folds.slurm)
EEG2MEL_TRAIN_JOB=${EEG2MEL_TRAIN_SUBMISSION%%;*}

EEG2MEL_TEST_SUBMISSION=$(sbatch --parsable \
  --dependency="afterok:$EEG2MEL_TRAIN_JOB" \
  --export=ALL,VENV_ACTIVATE="$MUSIC_EEG_VENV/bin/activate",MODEL_RUN_PREFIX="$EEG2MEL_PREFIX" \
  slurms/test_eeg2mel_folds.slurm)
EEG2MEL_TEST_JOB=${EEG2MEL_TEST_SUBMISSION%%;*}

printf 'EEG2Mel training: %s\nEEG2Mel evaluation: %s\n' \
  "$EEG2MEL_TRAIN_JOB" "$EEG2MEL_TEST_JOB"
```

The default EEG2Mel split source is `runs/eeg2mel_uv/splits`. Override it with
`SPLIT_RUN_NAME` if a different prepared split set is required. When changing
the training `OUTPUT_RUN_PREFIX`, pass the same value as `MODEL_RUN_PREFIX` to
the evaluation job.

## Monitor and retry jobs

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

Alignment logs use `alignment_25fold_<job>_<task>.out` and `.err` in the
repository root. EEG2Mel logs are written under `logs/`.

If one array task fails, inspect its error log and resubmit only that task with
`--array=<task-id>`. For alignment jobs, add `SKIP_COMPLETED=true` when
resuming a partially completed sweep. EEG2Mel training does not have a
skip-completed option, so retry only the failed task IDs rather than the whole
training array.

## Aggregate and format Table I

After all three alignment arrays and the EEG2Mel evaluation array finish
successfully, aggregate each method's five folds:

```bash
for METHOD in \
  table1-infonce-subject-on \
  table1-infonce-subject-off \
  table1-cosine-regression \
  table1-eeg2mel
do
  python -m scripts.average_test_results \
    "runs/$METHOD" \
    --output-dir "runs/$METHOD/average_metrics" \
    --output-prefix "$METHOD"
done
```

Then create the combined table:

```bash
python -m scripts.format_table1 \
  --format latex \
  --final-results-dir runs \
  --output runs/final_results/table1.tex

python -m scripts.format_table1 \
  --format markdown \
  --final-results-dir runs \
  --output runs/final_results/table1.md
```

The aggregation command validates that all five split families and all five
folds are present before writing summaries.
