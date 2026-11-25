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
