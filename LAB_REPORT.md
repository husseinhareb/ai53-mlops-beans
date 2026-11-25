# AI53 MLOps Project Phase 1: Lab Report

Bean-leaf disease classification (3 classes, 1,295 images). Starter CNN in `src/model.py`.

## Milestone 0: Starter audit

Files read before any change: `params.yaml`, `pyproject.toml`, `.gitignore`, `src/reproducibility.py`, `src/validate_data.py`, `src/provenance.py`, `src/data.py`, `src/prepare.py`, `src/train.py`, `src/evaluate.py`, `scripts/check_environment.py`.

Baseline test run: `uv run --locked pytest -q` gives `11 failed, 8 passed`. The passing tests cover code that is already complete (`test_model.py`, `test_handoff.py`, `test_prepared_identity.py`). All failures come from the seeding, validation and provenance TODOs.

| # | Area | File / setting | Problem | Suggested fix |
|---|------|----------------|---------|---------------|
| 1 | Random seeds | `src/reproducibility.py`, `set_reproducibility()` | Only Python `random` is seeded. NumPy, PyTorch CPU, CUDA and the cuDNN flags are left as TODOs, so weight initialisation, dropout masks and augmentation differ on every run. | Seed `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all` when CUDA is available, set `cudnn.deterministic=True` and `cudnn.benchmark=False`. |
| 2 | Random seeds | `src/reproducibility.py`, `make_generator()` and `seed_worker()` | `make_generator()` returns an unseeded `torch.Generator()`, so the shuffle order of the training DataLoader changes between runs even with `random_state: 42`. `seed_worker()` is a no-op, so DataLoader workers would share or randomise NumPy/Python state as soon as `num_workers > 0`. | Return `torch.Generator().manual_seed(seed)`. In `seed_worker`, derive `torch.initial_seed() % 2**32` and seed NumPy and `random` with it. |
| 3 | Data checks | `src/validate_data.py`, `validate_dataset()` | The quality gate raises `NotImplementedError`. `src/prepare.py` only checks that `data/raw` exists, so a missing split, an extra class, a wrong count or a truncated JPEG would go straight into training. | Implement the six checks from Milestone 3 (splits, exact classes, exact counts, full pixel decode with `Image.load()`, no empty class, deterministic report) and raise on any failure so DVC stops. |
| 4 | Data | `dataset-archives/train.zip`, `train/healthy/healthy_train.120tore` | The train archive holds 1,035 files: 1,034 JPEGs plus one 6,148-byte non-image file with a garbled name (the size of a macOS `.DS_Store`). Both `validate_data.py` and `prepare.py` filter by extension, so this file is skipped silently and nobody is told the raw data contains junk. | Make the validator list (or reject) every non-image file it finds in a class folder instead of ignoring it silently, and document the decision in the report. |
| 5 | Data versioning | repository root (no `.dvc/`, no `data/raw.dvc`); `.gitignore` lines `data/raw/`, `/dataset-archives/` | Raw images are excluded from Git but are not tracked by anything else. There is no record of which image bytes a model was trained on, and no remote to restore them from. | `dvc init`, `dvc add data/raw`, configure a default DVC remote and commit `data/raw.dvc` (Milestone 2). |
| 6 | Pipeline | repository root (no `dvc.yaml`, no `dvc.lock`) | The stages are run by hand in any order. Nothing declares that PREPARE needs a passing quality report, nor which parameters, code files and lock files each stage depends on, so a stale `data/processed/` or `models/model.pt` can be reused without anyone noticing. | Write a four-stage `dvc.yaml` (validate, prepare, train, evaluate) with explicit `deps`, per-key `params`, `outs` and `metrics` (Milestone 4). |
| 7 | Provenance | `src/provenance.py`, `build_run_manifest()` | Not implemented. No Git commit SHA, dirty flag, library versions, platform, device or settings are recorded, so a model file cannot be traced back to the code and environment that produced it. | Collect Git SHA and `git status --porcelain` via `subprocess`, versions via `importlib.metadata`, platform via `platform`, plus `training_data_identity()` and the run ID. |
| 8 | Run history | `src/train.py`, TODO 1 to 7 | A MLflow Run is started but nothing is logged: no params, no dataset input, no per-epoch metrics, no artifacts and no model. Every Run in the UI would be empty, so runs cannot be compared, and `src/evaluate.py` would fail because it needs `image_size`, `training_data_identity` and the `bean-cnn` model in the Run. | Fill the TODOs following `docs/MLFLOW_API_REFERENCE.md`, writing the manifest first so the dirty flag reflects the state before TRAIN writes files. |
| 9 | Run history | `src/train.py`, `run_name` | The run name only encodes augmentation, dropout and learning rate. A run that changes `batch_size` (one of the Milestone 7 options) gets the same name as the baseline, so Runs A and C would look identical in the MLflow list. | Rely on the logged params and Run IDs, not on names, or add batch size, image size and seed to the run name. |
| 10 | Test set | `params.yaml`, `selection.run_id: ""`; `src/evaluate.py` | The code can evaluate any Run on the test set as often as one likes. No validation metric or tie-break rule is written down, so it would be easy to run EVALUATE on several candidates and keep the best test score, which leaks the test set into model selection. | Decide the primary validation metric and tie-break rule before training (Milestone 6), select one Run on validation evidence only, and run EVALUATE once. Keep EVALUATE independent of TRAIN in the DAG. |
| 11 | Settings | `params.yaml`: `train.epochs: 4`, `train.random_state: 42`, `compute.device: "auto"` | Four epochs, one seed and a 133-image validation set make score differences between runs noisy (one image is about 0.75 accuracy points). `device: auto` silently picks CUDA on a machine that has it, so the same params can give different numbers on different machines. | Record the device in the manifest and params, pin `device: cpu` for the course, and treat small differences as inconclusive (or repeat with several seeds if time allows). |
| 12 | Data processing | `src/prepare.py`, output name `f"{src.stem}.jpg"` and JPEG re-save | Every output is renamed to `<stem>.jpg` in a flat class folder, but input is collected with `rglob`. Two inputs with the same stem (different extension or sub-folder) would overwrite each other while still counting twice in the manifest. The current archives have no such collision, but nothing checks it. The lossy JPEG re-encode also means `processed_sha256`, and therefore `training_data_identity`, depends on the Pillow/libjpeg version. | Fail if a destination path already exists; keep Pillow pinned through `uv.lock` and record its version in the run manifest. |
| 13 | Environment | `scripts/check_environment.py`, line 19 | The check calls `Path(sys.executable).resolve()`, which follows symlinks. On Linux, uv creates `.venv/bin/python` as a symlink to the shared uv-managed interpreter, so the check reports `FAIL: Interpreter is outside the project .venv` even though `sys.prefix` is the project `.venv`. | Compare `Path(sys.prefix).resolve()` with `(root / ".venv").resolve()` instead of the resolved executable path. |
| 14 | Environment | `pyproject.toml` version ranges (for example `torch>=2.10,<2.12`) | The ranges alone allow different installs on different days. Reproducibility depends on everyone using `uv.lock`, which only happens if every command uses `--locked` (or `uv sync --locked`). | Always run through `uv run --locked`; list `uv.lock` and `pyproject.toml` as `deps` of every Python stage in `dvc.yaml`. |

### Engineering checkpoint 1

**Why is a working `model.pt` insufficient evidence that an experiment is reproducible?**

A `model.pt` only proves that some training run once finished and saved weights. It does not say which commit of the code produced it (or whether there were uncommitted changes), which exact image bytes were used for training and validation, which library versions and hardware were involved, which hyperparameters and seed were set, or what validation scores it reached. In this starter the seeds are incomplete and nothing is logged, so running the same command again would give different weights, and there is no recorded reference to compare against anyway. Reproducibility requires the inputs (code, data, environment, settings) to be pinned and recorded alongside the output, so that someone else can rerun the experiment and check that they get the same result.

## Milestone 1: Make training repeatable

### Environment files

| File | Purpose | In Git? |
|------|---------|---------|
| `pyproject.toml` | Declares the project and its direct dependencies with allowed version ranges (for example `torch>=2.10,<2.12`), the supported Python range (`>=3.11,<3.13`) and the CPU-only PyTorch index. It says what the project needs, not exactly what gets installed. | Yes |
| `uv.lock` | Generated by uv from `pyproject.toml`. Pins the exact version, source URL and hash of every package, including transitive ones (170 packages here), for all platforms. `uv sync --locked` installs exactly this set and `--locked` fails if the lock no longer matches `pyproject.toml`, so every machine gets the same environment. | Yes |
| `.venv/` | The local virtual environment that uv builds from `uv.lock`: an interpreter link plus the installed packages. It is machine-specific (paths, compiled wheels for this OS), large, and fully reproducible from the two files above, so it is not committed. | No (`.gitignore`) |

### 1.1 Starter tests

`uv run --locked pytest -q` before any change: `11 failed, 8 passed`.

### 1.2 Seeding (`src/reproducibility.py`)

| Generator | How it is seeded | What it controls in this project |
|-----------|------------------|----------------------------------|
| Python `random` | `random.seed(seed)` | Any Python-level randomness (some torchvision transforms) |
| NumPy | `np.random.seed(seed % 2**32)` | NumPy-based randomness; NumPy only accepts 32-bit seeds |
| PyTorch CPU | `torch.manual_seed(seed)` | Weight initialisation, dropout masks, `RandomHorizontalFlip` / `RandomRotation` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` when `torch.cuda.is_available()` | The same on every GPU |
| cuDNN | `cudnn.deterministic = True`, `cudnn.benchmark = False` | Stops cuDNN from auto-tuning and choosing different (non-deterministic) convolution kernels between runs |
| DataLoader shuffling | `make_generator(seed)` returns `torch.Generator().manual_seed(seed)`, passed to the train `DataLoader` | Order of training batches each epoch |
| DataLoader workers | `seed_worker` seeds NumPy and `random` from `torch.initial_seed() % 2**32` | PyTorch gives each worker a distinct torch seed derived from the generator, but not NumPy/Python ones. Not exercised yet because `num_workers=0`, but correct if it is raised. |

Result:

```
$ uv run --locked pytest -q tests/test_reproducibility.py
...                                                                      [100%]
3 passed in 1.30s
```

Extra check: building `BeanCNN` twice after `set_reproducibility(42)` gives identical weights, an identical shuffle order and an identical output with dropout active; seed 43 gives different weights and order.

### Engineering checkpoint 2

**Does `random_state=42` guarantee bit-for-bit identical results on every CPU/GPU/software stack?**

No. The seed fixes the sequence of random numbers, not the arithmetic done with them. Floating-point addition is not associative, so results depend on the order in which operations are executed, and that order changes with the hardware (CPU instruction set such as AVX2 vs AVX-512, GPU model), the number of threads, the library versions (PyTorch, MKL/oneDNN, cuDNN, CUDA) and the kernels they choose. Some GPU operations are non-deterministic even on the same machine unless `torch.use_deterministic_algorithms(True)` is set. A seed therefore gives repeatable results on the same software stack and hardware (which is what `uv.lock` and the run manifest help pin down), and only approximately equal results elsewhere.

## Milestone 2: Extract and track the images

### 2.1 Get the Beans images

Option A (included archives): `uv run --locked python scripts/download_dataset.py --archive-dir dataset-archives`

```
train.zip: SHA-256 OK
validation.zip: SHA-256 OK
test.zip: SHA-256 OK
train     : 1034 images OK
validation:  133 images OK
test      :  128 images OK
Dataset ready in data/raw/.
```

`uv run --locked python scripts/check_dataset.py`:

| Split | angular_leaf_spot | bean_rust | healthy | Total | Expected |
|-------|------------------:|----------:|--------:|------:|---------:|
| train | 345 | 348 | 341 | 1034 | 1034 |
| validation | 44 | 45 | 44 | 133 | 133 |
| test | 43 | 43 | 42 | 128 | 128 |

Each split contains `angular_leaf_spot/`, `bean_rust/` and `healthy/`. The class balance is close to uniform (about one third each), which matters for the choice of metric later.

Note: `data/raw/train/healthy/healthy_train.120tore` (6,148 bytes, not an image, see audit issue 4) is also extracted. It is part of the original archive, so it stays in `data/raw` and is versioned with it.

### 2.2 Track the images with DVC

```
uv run --locked dvc init
uv run --locked dvc add data/raw
uv run --locked dvc remote add -d lab-data ../ai53-phase1-dvc-storage
uv run --locked dvc remote list
lab-data   .../TD-Homework/ai53-phase1-dvc-storage   (default)
```

`data/raw.dvc` (the only thing Git stores about the images):

```yaml
outs:
- md5: 2b62d5a2e86898b281d0136cef5fdf1c.dir
  size: 179914877
  nfiles: 1296
  hash: md5
  path: raw
```

`nfiles: 1296` is the 1,295 images plus the stray non-image file, which is probably why the original iBean repository reports 1,296. The `.dir` hash identifies the whole folder content: changing, adding or removing any file changes it.

### Engineering checkpoint 3

- **Git remote (GitHub):** stores the code, configuration, report and history, plus the small DVC pointer files such as `data/raw.dvc` that record the hash of each data version.
- **DVC remote (`../ai53-phase1-dvc-storage`):** stores the actual file contents (the 1,296 raw files, later the processed data and model), addressed by their hashes, so any version referenced by a commit can be restored with `dvc pull`.
- **Does `git push` transfer the raw images?** No: `data/raw/` is git-ignored and `git push` only sends `data/raw.dvc`; the images reach the DVC remote only through `dvc push`.

## Milestone 3: Core data-quality gate

`validate_dataset()` in `src/validate_data.py` checks, in order, for each split:

| # | Check | Implementation | Failure |
|---|-------|----------------|---------|
| 1 | Split exists | `(raw_root / split).is_dir()` | `FileNotFoundError` |
| 2 | Exactly the three classes | sorted sub-folder names must equal `EXPECTED_CLASSES` (reports missing and unexpected ones) | `RuntimeError` |
| 5 | No empty class | each class must contain at least one image | `RuntimeError` |
| 4 | Full decode | `Image.open(path)` then `image.load()`, which decodes all pixel data, not only the header | `OSError` with the file path |
| 3 | Exact count | per-split total must equal `EXPECTED_SPLITS` (1034 / 133 / 128) | `RuntimeError` |
| 6 | Deterministic report | no timestamps, files and classes iterated in sorted order, `json.dumps(..., sort_keys=True)` | n/a |

Any exception makes `python -m src.validate_data` exit non-zero, so DVC stops before PREPARE. Images are recognised by extension (`.jpg`, `.jpeg`, `.png`). Other files are not counted but are listed in `ignored_files`, which makes the stray `train/healthy/healthy_train.120tore` (audit issue 4) visible in the report instead of silently skipped.

```
$ uv run --locked pytest -q tests/test_data_validation.py tests/test_validation_contract.py
.......                                                                  [100%]
7 passed in 0.19s
$ uv run --locked python -m src.validate_data
  "status": "PASS", "total_images": 1295,
  "ignored_files": ["train/healthy/healthy_train.120tore"]
  train 1034 (345 / 348 / 341), validation 133 (44 / 45 / 44), test 128 (43 / 43 / 42)
```

Running the validator twice produces a byte-identical `reports/data_quality.json` (checked with `cmp`), so DVC will only see a change when the data does.

Extra check (on a copy of `data/raw`, not committed): cutting a test JPEG in half leaves a valid header (`Image.open()` still reports `JPEG (500, 500)`), but the validator rejects it: `OSError - Cannot decode image .../healthy_test.0.jpg: image file is truncated`. A header-only check would have let it through.

## Milestone 4: DVC pipeline design

`dvc.yaml` defines four stages. Each `cmd` is the exact command from the guide (`uv run --locked python -m src.<module>`).

### Inputs and outputs traced from the code

| Stage | Data deps | Code deps (including indirect imports) | Params | Outputs |
|-------|-----------|----------------------------------------|--------|---------|
| validate | `data/raw` | `src/validate_data.py` | none (the script reads no setting) | `reports/data_quality.json` (cache: false) |
| prepare | `data/raw`, `reports/data_quality.json` | `src/prepare.py` | `data.image_size` | `data/processed/{train,validation,test}`, `data/processed/manifests/{train,validation,test}.csv`, `reports/data_summary.json` (cache: false) |
| train | `data/processed/{train,validation}`, `manifests/{train,validation}.csv` | `src/train.py`, `src/data.py`, `src/model.py`, `src/provenance.py`, `src/reproducibility.py` (via `data.py`) | `data.image_size`, `train.batch_size`, `train.epochs`, `train.learning_rate`, `train.dropout`, `train.random_state`, `augmentation.enabled`, `augmentation.horizontal_flip`, `augmentation.rotation_degrees`, `compute.device`, `tracking.experiment` | `models/model.pt` (DVC cache), `reports/training_curves.png`, `reports/run_manifest.json` (cache: false), metrics `reports/train_summary.json` (cache: false) |
| evaluate | `data/processed/test`, `manifests/{train,validation}.csv` | `src/evaluate.py`, `src/data.py`, `src/reproducibility.py` (via `data.py`), `src/provenance.py`, `src/model.py` (needed to unpickle the logged `BeanCNN`) | `selection.run_id`, `evaluation.batch_size`, `data.image_size`, `compute.device`, `tracking.experiment` | `reports/test_confusion_matrix.png` (cache: false), metrics `reports/test_metrics.json` (cache: false) |

Every stage also depends on `uv.lock` and `pyproject.toml`, since they define the Python environment that `uv run --locked` executes in. `src/__init__.py` is empty and was left out.

Design choices:

- **PREPARE outputs are declared per folder and per manifest**, not as one `data/processed` directory. That lets TRAIN depend only on train/validation data and EVALUATE only on test data (plus the two manifests it hashes), so the two branches stay separate.
- **EVALUATE reads `manifests/train.csv` and `validation.csv`** even though it scores test images: `validate_selected_run()` calls `training_data_identity()` on them to check that the selected Run was trained on the current prepared data.
- **`tracking.uri` is not a param of any stage.** It says where the MLflow server is, not what is trained or evaluated, so moving the server should not retrain or re-score anything. `tracking.experiment` is included: it decides which experiment the candidate is logged in, and EVALUATE rejects Runs from another experiment.
- **Side effects DVC cannot see:** TRAIN also writes a Run to MLflow (`mlflow.db`, `mlartifacts/`). DVC does not track these, which is why the run manifest and MLflow Run ID matter.

### Pipeline checks

```
$ uv run --locked dvc stage list
validate  Outputs reports/data_quality.json
prepare   Outputs data/processed/train, data/processed/validation, data/processed/test, ...
train     Outputs models/model.pt, reports/training_curves.png, reports/run_manifest.json ...
evaluate  Outputs reports/test_confusion_matrix.png; Reports reports/test_metrics.json

$ uv run --locked dvc dag
        +--------------+
        | data/raw.dvc |
        +--------------+
           **        **
+----------+           *
| validate |         **
+----------+       **
           **    **
           +---------+
           | prepare |
           +---------+
           **        **
  +-------+         +----------+
  | train |         | evaluate |
  +-------+         +----------+
```

All four stages appear, and there is no edge from `train` to `evaluate`: EVALUATE gets its model from MLflow through `selection.run_id`, not from `models/model.pt`.

### 4.2 Design questions

**Should PREPARE depend on `reports/data_quality.json`?** Yes. It is the only thing that connects VALIDATE to PREPARE in the DAG. Without it DVC could run PREPARE before (or without) the quality gate. Because VALIDATE raises on any failure, the report only exists after a PASS, so PREPARE never runs on data that failed the checks.

**Should PREPARE also depend directly on `data/raw`?** Yes. PREPARE reads the raw images, and the quality report only summarises counts and classes. If an image were replaced by a different valid image of the same class, the report would stay byte-identical and DVC would wrongly reuse the old processed data. Depending on `data/raw` catches any change in the pixels.

**Which changes in `params.yaml` should make TRAIN run again?** Only the settings TRAIN reads: `train.*` (batch size, epochs, learning rate, dropout, seed), `augmentation.*` (enabled, flip probability, rotation), `compute.device`, `data.image_size` and `tracking.experiment`. A change to `data.image_size` also reruns TRAIN indirectly, because PREPARE produces new images. Changes to `evaluation.batch_size`, `selection.run_id` or `tracking.uri` must not retrain. That is why params are listed key by key rather than as the whole file.

**Why should EVALUATE track `selection.run_id`?** The run ID is the only thing that says which model is being tested: the model lives in MLflow, not in a file DVC can hash. Tracking it makes DVC rerun EVALUATE when a different model is selected, and records in `dvc.lock` which Run the committed `test_metrics.json` belongs to. Otherwise the saved test results could silently describe a model that is no longer the selected one.

**Which file contains scores that DVC should track as metrics?** `reports/test_metrics.json`, which holds the final test accuracy and macro-F1 of the selected Run. I also declared `reports/train_summary.json` as a metric because it holds the final validation scores and the training time of each candidate, so `dvc metrics diff` can compare Runs A, B and C across commits using validation evidence only.

## Milestone 5: Record what each MLflow Run used

### 5.1 Run manifest (`src/provenance.py`)

`build_run_manifest()` returns five sections, written to `reports/run_manifest.json`:

| Section | Content | How it is collected |
|---------|---------|---------------------|
| `git` | `commit_sha`, `dirty`, `changed_paths`, `available` | `git rev-parse HEAD` and `git status --porcelain` through `subprocess.run`; the exit code is checked, so outside a repository the fields are `null` and `available` is `false` instead of a wrong "clean" answer |
| `data` | `training_data_identity`, manifest paths | supplied `training_data_identity()` (SHA-256 of the train and validation manifests, which contain the processed image hashes) |
| `environment` | OS, platform, machine, Python, `cuda_available`, CUDA runtime and cuDNN (when available), versions of torch, torchvision, dvc, mlflow, numpy, pandas, pillow, scikit-learn | `platform`, `importlib.metadata.version()`, `torch.version.cuda` |
| `configuration` | image size, batch size, epochs, learning rate, dropout, augmentation enabled, flip probability, rotation | `params.yaml` |
| `execution` | resolved device, requested device, seed, MLflow Run ID, experiment name | `train.py` arguments, `run.info.run_id` |

`changed_paths` lists what made the tree dirty, so a `dirty: true` can be interpreted. Expected case: when `dvc repro` runs VALIDATE or PREPARE before TRAIN, it rewrites `dvc.lock` and report files, so the tree is dirty with only pipeline outputs, not code.

Bug found while testing: the first version stripped the whole `git status` output, which removed the leading space of the first porcelain line (` M src/...`) and recorded `rc/provenance.py`. Only the commit SHA is stripped now.

### 5.2 MLflow logging (`src/train.py`)

In the required order inside one `mlflow.start_run()` in experiment `bean-image-classification`:

1. `write_run_manifest(params, str(device), run.info.run_id, ...)` before any training output is written;
2. `training_identity = run_manifest["data"]["training_data_identity"]`;
3. `mlflow.log_params()` with the 11 required parameters;
4. `mlflow.data.from_pandas(train_manifest_df, source=..., name="bean-training", targets="label")` then `mlflow.log_input(..., context="training")`;
5. `mlflow.log_metrics({train_loss, val_loss, val_accuracy, val_macro_f1}, step=epoch)` inside the epoch loop;
6. `mlflow.log_metric("training_seconds", ...)` after the loop;
7. `mlflow.log_artifact()` for `training_curves.png`, `train_summary.json` and `run_manifest.json` (under `reports/`);
8. model moved to CPU and `eval()`, two validation images (NCHW, normalized float32) as input example, logits computed under `torch.no_grad()`, `infer_signature(input, logits)`;
9. `mlflow.pytorch.log_model(name="bean-cnn", signature, input_example, metadata={class_names, image_size}, serialization_format="pickle", code_paths=["src"])`.

The test set is never touched in TRAIN.

### 5.3 Tests and smoke run

```
$ uv run --locked pytest -q
...................                                                      [100%]
19 passed
```

The tests do not execute `train.py`, so I also ran VALIDATE, PREPARE and a 1-epoch TRAIN in a throwaway copy of the project with its own SQLite MLflow store (deleted afterwards, nothing added to the real Run history). The logged Run had status `FINISHED`, the 11 params, the 4 per-epoch metrics plus `training_seconds`, the `bean-training` dataset input (context `training`, 1,034 rows), the 3 artifacts, and the `bean-cnn` model with metadata `{class_names: [angular_leaf_spot, bean_rust, healthy], image_size: 96}` and signature `Tensor(float32, (-1, 3, 96, 96)) -> Tensor(float32, (-1, 3))`. `src.evaluate.validate_selected_run()` accepted the Run and `mlflow.pytorch.load_model()` loaded it in eval mode. No test images were scored.

MLflow prints `UserWarning: The specified dataset source can be interpreted in multiple ways: LocalArtifactDatasetSource, LocalArtifactDatasetSource` when creating the dataset. It is harmless (the same source class is registered twice) and the source is resolved correctly.

### Engineering checkpoint 4

- **Git** records which version of the code, configuration (`params.yaml`, `dvc.yaml`) and lock files a Run was started from, identified by the commit SHA stored in the run manifest.
- **DVC** records which exact data and outputs belong to that commit: the hashes of the raw images, the processed data, the local model file and each stage's dependencies and params in `dvc.lock`, with the content stored in the DVC remote.
- **MLflow** records what happened in the Run itself: the parameters, per-epoch metrics, training time, input dataset, artifacts (including the run manifest that links back to Git and the data identity) and the logged model with its signature and metadata, under a Run ID.

## Milestone 6: Predict which stages need to run again

### 6.1 Predictions (written before running DVC)

Assumption: all outputs, including the test results, are up to date; one change at a time; `selection.run_id` fixed.

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
|--------|----------|---------|-------|----------|
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py`; the quality report stays the same | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

### Engineering checkpoint 5

**Setting change: `train.learning_rate`.** In `dvc.yaml` this key appears only under `train.params`. VALIDATE has no params and its deps (`data/raw`, `src/validate_data.py`, `uv.lock`, `pyproject.toml`) are unchanged, so it is REUSED. PREPARE only tracks `data.image_size` and its data and code deps are unchanged, so REUSED. TRAIN tracks `train.learning_rate`, so its params hash changes and it must RERUN. EVALUATE does not track the learning rate and none of its deps change (`data/processed/test`, the two manifests, its code); it scores the model from the fixed `selection.run_id`, which lives in MLflow and is not affected by a new TRAIN, so the saved test results still describe the selected model: REUSED.

**Setting change: `data.image_size`.** VALIDATE does not read it: REUSED. PREPARE tracks it, so it RERUNs and writes new resized images and manifests. TRAIN tracks it directly and its data deps change: RERUN. EVALUATE tracks `data.image_size` and depends on `data/processed/test` and the two manifests, which all change, so the saved results no longer match: NEEDS UPDATE. (Running it would actually fail: the selected Run was logged with `image_size=96` and its `training_data_identity` no longer matches, so a new Run would have to be selected.)

**File edit: `src/validate_data.py` with an unchanged report.** The file is a dep of VALIDATE only, so VALIDATE RERUNs. It writes `reports/data_quality.json` again, but the report is deterministic (sorted, no timestamps), so the file hash is identical. PREPARE depends on that report and on `data/raw`, both unchanged by hash, so it is REUSED, and TRAIN and EVALUATE, whose inputs come from PREPARE, are REUSED too. This only works because the report has no timestamp.

**File edit: `README.md`.** No stage lists it as a dep, so every stage is REUSED.

### Decisions made before training

- **Primary metric:** validation macro-F1 at the final epoch (`val_macro_f1`, the logged model is the last-epoch model).
- **Tie-break:** if two Runs are within 0.01 macro-F1 (about one validation image out of 133), choose the lower final `val_loss`; if still tied, the shorter `training_seconds`.
- **Expected effect of a lower learning rate (0.001 to 0.0003):** with Adam and only 4 epochs, each step moves the weights about three times less, so the model should converge more slowly: higher train and validation loss after 4 epochs and probably lower validation accuracy and macro-F1. The curves should be smoother. Training time should be about the same, since the amount of computation does not change.

### 6.2 MLflow server

Started from the project folder (in the background, in place of terminal 2):
`uv run --locked mlflow server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db`, UI at http://127.0.0.1:5000.

### 6.3 Run A (baseline, default settings)

`uv run --locked dvc repro train`:

```
'data/raw.dvc' didn't change, skipping
Running stage 'validate':
Running stage 'prepare':
Running stage 'train':
Epoch 01/4 | train_loss=1.0639 | val_loss=0.9811 | val_acc=0.5639 | val_macro_f1=0.5562
Epoch 02/4 | train_loss=0.9123 | val_loss=0.8819 | val_acc=0.6241 | val_macro_f1=0.6119
Epoch 03/4 | train_loss=0.8874 | val_loss=0.8472 | val_acc=0.6466 | val_macro_f1=0.6445
Epoch 04/4 | train_loss=0.8671 | val_loss=0.8328 | val_acc=0.6391 | val_macro_f1=0.6254
Training complete. MLflow Run ID: 899741c6b66f426584516a7f7c4b5e3d
```

**Run A ID: `899741c6b66f426584516a7f7c4b5e3d`**

Checked in MLflow (experiment `bean-image-classification`, status FINISHED):

- 11 params present (`image_size=96`, `batch_size=32`, `epochs=4`, `learning_rate=0.001`, `dropout=0.3`, `random_state=42`, `augmentation_enabled=False`, `horizontal_flip=0.5`, `rotation_degrees=10.0`, `device=cpu`, `training_data_identity=8fa943ff6c17a3c8...`);
- `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1` logged at steps 1, 2, 3, 4, plus `training_seconds=11.48`;
- dataset input `bean-training` with context `training`;
- artifacts `reports/run_manifest.json`, `reports/train_summary.json`, `reports/training_curves.png`;
- logged model `bean-cnn`;
- `run_manifest.json`: commit `ee7a0de`, `dirty: true` with `changed_paths: [dvc.lock, reports/data_summary.json]`. Only pipeline outputs written by VALIDATE and PREPARE in the same `dvc repro`, no code change.

Epoch 1 is bit-identical to the 1-epoch smoke run of Milestone 5 (`train_loss=1.0639`, `val_acc=0.5639`), which ran in a different folder and process: the seeding works.

**Check 1, rerun without any change:**

```
'data/raw.dvc' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Data and pipelines are up to date.
```

**Check 2, comment added to `src/validate_data.py`:**

```
'data/raw.dvc' didn't change, skipping
Running stage 'validate':
> uv run --locked python -m src.validate_data
Updating lock file 'dvc.lock'
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
$ sha256sum -c   (hash of reports/data_quality.json taken before the run)
reports/data_quality.json: OK
```

### 6.4 Run B (learning rate 0.0003)

`params.yaml` changed and committed alone (`a9b7b9d Lower the learning rate for Run B`), then `uv run --locked dvc repro train`:

```
'data/raw.dvc' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Running stage 'train':
Epoch 01/4 | train_loss=1.0903 | val_loss=1.0809 | val_acc=0.5038 | val_macro_f1=0.4723
Epoch 02/4 | train_loss=1.0495 | val_loss=1.0229 | val_acc=0.5714 | val_macro_f1=0.5763
Epoch 03/4 | train_loss=0.9785 | val_loss=0.9447 | val_acc=0.5865 | val_macro_f1=0.5864
Epoch 04/4 | train_loss=0.9198 | val_loss=0.8927 | val_acc=0.6165 | val_macro_f1=0.6091
Training complete. MLflow Run ID: e4c8dd28b6e44247a8561f226dd8ac7e
Updating lock file 'dvc.lock'
```

**Run B ID: `e4c8dd28b6e44247a8561f226dd8ac7e`** (MLflow shows `learning_rate=0.0003`, all other params identical to Run A).

Its manifest records commit `a9b7b9d` with `dirty: true` and `changed_paths: [reports/run_manifest.json, reports/train_summary.json, reports/training_curves.png]`: DVC deletes a stage's previous outputs before running it, so Git sees TRAIN's own reports as deleted when the manifest is written. The code and params in that commit are exactly what was run.

### Observed recomputation

| Check | VALIDATE | PREPARE | TRAIN | EVALUATE | Evidence |
|-------|----------|---------|-------|----------|----------|
| 1. Rerun, nothing changed | REUSED | REUSED | REUSED | NOT RUN | `Stage 'validate' didn't change, skipping` (same for prepare, train), `Data and pipelines are up to date.` |
| 2. Comment in `src/validate_data.py`, report unchanged | RERUN | REUSED | REUSED | NOT RUN | `Running stage 'validate'`, `Stage 'prepare' didn't change, skipping`, `Stage 'train' didn't change, skipping`; report hash unchanged (`sha256sum -c`: OK) |
| 3. `train.learning_rate` 0.001 to 0.0003 | REUSED | REUSED | RERUN | NOT RUN | `Stage 'validate' didn't change, skipping`, `Stage 'prepare' didn't change, skipping`, `Running stage 'train'` |

EVALUATE is NOT RUN in every check because `dvc repro train` only runs `train` and its upstream stages, and EVALUATE is on a separate branch of the DAG.

**Comparison with the predictions.** All observed results match the predictions for VALIDATE, PREPARE and TRAIN:

- Check 3 is the `train.learning_rate` row: only TRAIN reran, because the key is only in `train.params`.
- Check 2 is the validator-edit row: VALIDATE reran because its code dep changed, but the deterministic report kept the same hash, so DVC stopped there. If the report had contained a timestamp, PREPARE and TRAIN would have rerun for nothing.
- Check 1 has no change at all and behaves like the `README.md` row: nothing reran because no dep, param or output changed.

The only difference from the prediction table is the EVALUATE column: I predicted REUSED (the saved results would still be valid), and DVC reports it as NOT RUN because `dvc repro train` does not consider it at all. The `data.image_size` row was not executed. It would have retrained and invalidated the selected Run.

## Milestone 7: Compare three controlled runs

Run A: defaults. Run B: only `train.learning_rate` 0.001 to 0.0003. Both confirmed in MLflow (all other params identical).

### 7.1 Run C setup and prediction (written before training)

**Change:** `params.yaml` returns to `learning_rate: 0.001` and sets `augmentation.enabled: true` (random horizontal flip with p = 0.5 and random rotation up to 10 degrees on training images only). `git diff` against Run A's commit shows this is the only difference.

**Prediction:** little gain, or a small drop in validation macro-F1 compared with A. Augmentation is a regulariser: it makes each epoch harder by showing slightly different images, which reduces overfitting on long training, but with only 4 epochs Run A is not overfitting (validation loss still decreases at every epoch and stays close to the training loss), so there is little to gain yet and the model sees "harder" data for the same number of steps. Expect training loss at least as high as A's, a similar or slightly higher validation loss, and a slightly longer training time because of the extra transforms (computed on CPU, `num_workers=0`). With leaves photographed in the field, flips and small rotations do not change the class, so they should not hurt much either.
