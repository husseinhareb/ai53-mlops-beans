from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import torch

TRAIN_MANIFEST = Path("data/processed/manifests/train.csv")
VALIDATION_MANIFEST = Path("data/processed/manifests/validation.csv")
TRACKED_PACKAGES = (
    "torch", "torchvision", "dvc", "mlflow",
    "numpy", "pandas", "pillow", "scikit-learn",
)


def training_data_identity(
    train_manifest: Path = TRAIN_MANIFEST,
    validation_manifest: Path = VALIDATION_MANIFEST,
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    """Run a git command; return its stdout, or None if git failed.

    A non-zero exit (for example outside a repository) must not be mistaken
    for an empty answer, so the exit status is checked explicitly.
    """
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
    except OSError:  # git not installed
        return None
    if result.returncode != 0:
        return None
    # Not stripped: porcelain lines start with a status column that may be a space.
    return result.stdout


def _git_info() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if commit is None or status is None:
        return {"available": False, "commit_sha": None, "dirty": None, "changed_paths": []}
    # Porcelain v1 format: "XY path", so the path starts at column 3.
    changed = sorted(line[3:] for line in status.splitlines() if line.strip())
    return {
        "available": True,
        "commit_sha": commit.strip(),
        "dirty": bool(changed),
        "changed_paths": changed,
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _environment() -> dict:
    cuda_available = torch.cuda.is_available()
    env = {
        "os": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda if cuda_available else None,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
    }
    for name in TRACKED_PACKAGES:
        env[name] = _package_version(name)
    return env


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata.

    Records which code (Git), data (prepared manifest identity), software and
    settings a training Run used, plus the MLflow Run ID that links them.
    """
    train = params["train"]
    aug = params["augmentation"]
    tracking = params.get("tracking", {})

    return {
        "git": _git_info(),
        "data": {
            "training_data_identity": training_data_identity(),
            "train_manifest": TRAIN_MANIFEST.as_posix(),
            "validation_manifest": VALIDATION_MANIFEST.as_posix(),
        },
        "environment": _environment(),
        "configuration": {
            "image_size": int(params["data"]["image_size"]),
            "batch_size": int(train["batch_size"]),
            "epochs": int(train["epochs"]),
            "learning_rate": float(train["learning_rate"]),
            "dropout": float(train["dropout"]),
            "augmentation_enabled": bool(aug["enabled"]),
            "horizontal_flip": float(aug["horizontal_flip"]),
            "rotation_degrees": float(aug["rotation_degrees"]),
        },
        "execution": {
            "device": str(device),
            "requested_device": params.get("compute", {}).get("device"),
            "random_state": int(train["random_state"]),
            "mlflow_run_id": mlflow_run_id,
            "mlflow_experiment": tracking.get("experiment"),
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
